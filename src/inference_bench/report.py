from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .models import canonical_json
from .statistics import (
    Estimate,
    block_proportion_interval,
    block_rate_interval,
    median_interval,
    qualified_p99,
    quantile,
    quantile_interval,
    wilson_interval,
)
from .validity import (
    MIN_DECODE_PROXY_CONTENT_EVENTS,
    MIN_DECODE_PROXY_SECONDS,
    MIN_DECODE_PROXY_TOKENS,
)

DECODE_PROVENANCE = (
    "provider-reported billed completion_tokens / (request_seconds - TTFT); client-observed "
    f"proxy; eligible only with >= {MIN_DECODE_PROXY_TOKENS} completion tokens, >= "
    f"{MIN_DECODE_PROXY_CONTENT_EVENTS} content events, and >= "
    f"{MIN_DECODE_PROXY_SECONDS:g} seconds after TTFT"
)
AGGREGATE_OUTPUT_PROVENANCE = "successful completion tokens / analysis-block wall-clock minute"


def _json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _estimate_columns(prefix: str, value: Estimate) -> dict[str, Any]:
    return {
        prefix: value.estimate,
        f"{prefix}_ci95_low": value.lower_95,
        f"{prefix}_ci95_high": value.upper_95,
        f"{prefix}_n": value.n,
        f"{prefix}_unit": value.unit,
        f"{prefix}_ci_method": value.method,
    }


def summarize_rows(rows: list[dict[str, Any]], *, seed: int = 1) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["state"] == "terminal":
            groups[(row["route_id"], row["suite"], row["cell_id"], row["cache_state"])].append(row)
    output: list[dict[str, Any]] = []
    for (route, suite, cell, cache_state), physical_items in sorted(groups.items()):
        # Retry attempts are conditional observations from the same logical request. Request-level
        # intervals therefore use one final terminal attempt per logical request; attempt-level
        # errors/costs remain separately counted and preserved in the audit.
        final_by_logical: dict[str, dict[str, Any]] = {}
        for row in physical_items:
            current = final_by_logical.get(row["logical_id"])
            if current is None or int(row["attempt_index"]) > int(current["attempt_index"]):
                final_by_logical[row["logical_id"]] = row
        items = list(final_by_logical.values())
        successes = [row for row in items if row["status"] == "success"]
        latency = [
            float(row["total_seconds"])
            for row in items
            if row["latency_eligible"] and row["total_seconds"] is not None
        ]
        ttft = [
            float(row["ttft_seconds"])
            for row in items
            if row["latency_eligible"] and row["ttft_seconds"] is not None
        ]
        decode_proxy = [
            float(row["output_tokens"]) / (float(row["total_seconds"]) - float(row["ttft_seconds"]))
            for row in items
            if row["decode_eligible"]
            and row["validity_class"] == "valid"
            and row["output_tokens"] is not None
            and row["ttft_seconds"] is not None
            and float(row["total_seconds"]) - float(row["ttft_seconds"]) > 0
        ]
        quality = [
            float(row["quality_score"])
            for row in items
            if row["quality_score"] is not None and row["quality_eligible"]
        ]
        input_usage = [
            float(row["input_tokens"])
            for row in items
            if row["usage_eligible"] and row["input_tokens"] is not None
        ]
        output_usage = [
            float(row["output_tokens"])
            for row in items
            if row["usage_eligible"] and row["output_tokens"] is not None
        ]
        success_estimate = wilson_interval(len(successes), len(items))
        settled_usd = sum(float(row["settled_usd"]) for row in physical_items)
        successful_usage_complete = all(row["usage_eligible"] for row in successes)
        successful_output_tokens = sum(
            int(row["output_tokens"] or 0) for row in successes if row["usage_eligible"]
        )
        record: dict[str, Any] = {
            "route_id": route,
            "suite": suite,
            "cell_id": cell,
            "cache_state": cache_state,
            "attempts_n": len(physical_items),
            "logical_requests_n": len(items),
            "successes_n": len(successes),
            "valid_n": sum(row["validity_class"] == "valid" for row in items),
            "anomalous_n": sum(row["validity_class"] == "anomalous" for row in items),
            "invalid_n": sum(row["validity_class"] == "invalid" for row in items),
            "censored_n": sum(row["validity_class"] == "censored" for row in items),
            "usage_complete_n": sum(row["usage_eligible"] for row in items),
            "decode_anomalous_excluded_n": sum(
                row["validity_class"] == "anomalous"
                and "decode_proxy_extreme_tokens_per_second"
                in _json(row.get("validity_reasons_json"), [])
                for row in items
            ),
            "cache_read_reported_n": sum(
                row["cache_read_input_tokens"] is not None for row in items
            ),
            "cache_read_unknown_n": sum(
                row["cache_read_input_tokens"] is None for row in items
            ),
            "cache_miss_n": sum(row["cache_read_input_tokens"] == 0 for row in items),
            "cache_hit_n": sum(
                row["cache_read_input_tokens"] is not None
                and int(row["cache_read_input_tokens"]) > 0
                for row in items
            ),
            "cache_read_tokens_sum": sum(
                int(row["cache_read_input_tokens"])
                for row in items
                if row["cache_read_input_tokens"] is not None
            ),
            "settled_usd_sum": settled_usd,
            "reserved_upper_bound_cost_n": sum(
                row["cost_basis"] == "reserved_upper_bound" for row in physical_items
            ),
            "cache_unknown_upper_bound_cost_n": sum(
                row["cost_basis"] == "provider_usage_cache_unknown_upper_bound"
                for row in physical_items
            ),
            "cost_per_successful_request_usd": (
                settled_usd / len(successes) if successes else None
            ),
            "cost_per_million_effective_output_tokens_usd": (
                settled_usd * 1_000_000 / successful_output_tokens
                if successful_output_tokens and successful_usage_complete
                else None
            ),
            "decode_metric_provenance": DECODE_PROVENANCE,
            "request_sampling_unit": "final terminal attempt per logical request",
            "retry_end_to_end_latency_scope": (
                "final physical attempt only; cumulative backoff plus prior-attempt latency is "
                "not reconstructed by this schema"
            ),
            "http_status_counts_json": canonical_json(
                dict(Counter(str(row["http_status"]) for row in physical_items))
            ),
            "final_http_status_counts_json": canonical_json(
                dict(Counter(str(row["http_status"]) for row in items))
            ),
            "finish_reason_counts_json": canonical_json(
                dict(Counter(str(row["finish_reason"]) for row in items))
            ),
            "realized_output_tokens_max": max(output_usage) if output_usage else None,
        }
        record.update(_estimate_columns("success_rate", success_estimate))
        record.update(
            _estimate_columns("latency_p50", median_interval(latency, unit="seconds", seed=seed))
        )
        record.update(
            _estimate_columns(
                "latency_p95", quantile_interval(latency, 0.95, unit="seconds", seed=seed)
            )
        )
        record.update(
            _estimate_columns("latency_p99", qualified_p99(latency, unit="seconds", seed=seed))
        )
        record.update(
            _estimate_columns("ttft_p50", median_interval(ttft, unit="seconds", seed=seed))
        )
        record.update(
            _estimate_columns(
                "decode_proxy_tps_p50",
                median_interval(decode_proxy, unit="tokens/second", seed=seed),
            )
        )
        record.update(_estimate_columns("quality_mean", _mean_interval(quality, seed=seed)))
        record.update(
            _estimate_columns(
                "input_tokens_p50", median_interval(input_usage, unit="tokens", seed=seed)
            )
        )
        record.update(
            _estimate_columns(
                "output_tokens_p50", median_interval(output_usage, unit="tokens", seed=seed)
            )
        )
        output.append(record)
    return output


def _load_epoch_usage(
    epoch: dict[str, Any], rows: list[dict[str, Any]] | None
) -> tuple[bool, int | None, int | None, str]:
    """Join an epoch to final per-logical-request rows to verify successful usage totals."""
    if rows is None:
        return False, None, None, "request_ledger_not_supplied"
    prefix = (
        f"load:{epoch['route_id']}:{epoch['shape']}:{epoch['epoch_id']}:"
    )
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        logical_id = str(row.get("logical_id") or "")
        if logical_id.startswith(prefix):
            attempts[logical_id].append(row)
    final_rows: list[dict[str, Any]] = []
    for logical_attempts in attempts.values():
        terminal = [row for row in logical_attempts if row.get("state") == "terminal"]
        if terminal:
            final_rows.append(
                max(terminal, key=lambda row: int(row.get("attempt_index") or 0))
            )
    if len(final_rows) != int(epoch["completed"]):
        return False, None, None, "ledger_join_completed_count_mismatch"
    successes = [row for row in final_rows if row.get("status") == "success"]
    if len(successes) != int(epoch["successful"]):
        return False, None, None, "ledger_join_success_count_mismatch"
    if any(
        not row.get("usage_eligible")
        or row.get("input_tokens") is None
        or row.get("output_tokens") is None
        for row in successes
    ):
        return False, None, None, "successful_request_usage_incomplete"
    return (
        True,
        sum(int(row["input_tokens"]) for row in successes),
        sum(int(row["output_tokens"]) for row in successes),
        "request_ledger_join",
    )


def summarize_load_events(
    events: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]] | None = None,
    seed: int = 1,
) -> list[dict[str, Any]]:
    epochs = [json.loads(row["payload_json"]) for row in events if row["kind"] == "load_epoch"]
    groups: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for epoch in epochs:
        key = (
            epoch["route_id"],
            epoch["shape"],
            epoch["phase"],
            float(epoch["offered_rps"]),
        )
        groups[key].append(epoch)
    result: list[dict[str, Any]] = []
    for (route, shape, phase, offered_rps), blocks in sorted(groups.items()):
        arrival_windows = [float(block["duration_seconds"]) for block in blocks]
        elapsed_wall = [
            float(
                block.get(
                    "actual_elapsed_seconds",
                    float(block["duration_seconds"])
                    + max(0.0, float(block.get("queue_end_seconds", 0))),
                )
            )
            for block in blocks
        ]
        offered = [float(block["scheduled"]) for block in blocks]
        launched_logical = [
            float(block.get("launched_logical", block["completed"])) for block in blocks
        ]
        completed = [float(block["completed"]) for block in blocks]
        successful = [float(block["successful"]) for block in blocks]
        total_completed = sum(int(block["completed"]) for block in blocks)
        total_successful = sum(int(block["successful"]) for block in blocks)
        physical_attempts = [float(block.get("physical_attempts", 0)) for block in blocks]
        physical_successes = [float(block.get("physical_successes", 0)) for block in blocks]
        physical_rate_limited = sum(int(block.get("rate_limited", 0)) for block in blocks)
        physical_server_errors = sum(int(block.get("server_errors", 0)) for block in blocks)
        physical_timeouts = sum(int(block.get("timeouts", 0)) for block in blocks)
        physical_transport_errors = sum(
            int(block.get("transport_errors", 0)) for block in blocks
        )
        physical_known = (
            sum(int(value) for value in physical_successes)
            + physical_rate_limited
            + physical_server_errors
            + physical_timeouts
            + physical_transport_errors
        )
        usage = [_load_epoch_usage(block, rows) for block in blocks]
        usage_complete_indexes = [index for index, item in enumerate(usage) if item[0]]
        input_tokens = [float(usage[index][1] or 0) for index in usage_complete_indexes]
        output_tokens = [float(usage[index][2] or 0) for index in usage_complete_indexes]
        usage_elapsed = [elapsed_wall[index] for index in usage_complete_indexes]
        usage_sources = Counter(item[3] for item in usage)
        if len(usage_complete_indexes) == len(blocks):
            tpm_state = "complete"
        elif usage_complete_indexes:
            tpm_state = "partial_complete_blocks_only"
        else:
            tpm_state = "censored_no_complete_usage_block"
        record: dict[str, Any] = {
            "route_id": route,
            "shape": shape,
            "phase": phase,
            "offered_rps_target": offered_rps,
            "blocks_n": len(blocks),
            "requests_completed_n": total_completed,
            "requests_successful_n": total_successful,
            "logical_requests_launched_n": sum(int(value) for value in launched_logical),
            "physical_attempts_n": sum(int(value) for value in physical_attempts),
            "physical_successes_n": sum(int(value) for value in physical_successes),
            "physical_rate_limited_n": physical_rate_limited,
            "physical_server_errors_n": physical_server_errors,
            "physical_timeouts_n": physical_timeouts,
            "physical_transport_errors_n": physical_transport_errors,
            "physical_other_outcomes_n": max(
                0, sum(int(value) for value in physical_attempts) - physical_known
            ),
            "physical_status_provenance": (
                "all terminal provider send attempts, including intermediate retries; "
                "other_outcomes is the residual not decomposed by the epoch event schema"
            ),
            "healthy_blocks_n": sum(bool(block["healthy"]) for block in blocks),
            "sampling_unit": (
                "load epoch/block; bootstrap assumes block-level exchangeability and does not "
                "remove temporal autocorrelation"
            ),
            "aggregate_output_metric_provenance": AGGREGATE_OUTPUT_PROVENANCE,
            "arrival_window_seconds_sum": sum(arrival_windows),
            "elapsed_wall_seconds_sum": sum(elapsed_wall),
            "post_arrival_drain_seconds_sum": sum(
                max(0.0, elapsed - window)
                for elapsed, window in zip(elapsed_wall, arrival_windows, strict=True)
            ),
            "early_termination_seconds_sum": sum(
                max(0.0, window - elapsed)
                for elapsed, window in zip(elapsed_wall, arrival_windows, strict=True)
            ),
            "tpm_complete_blocks_n": len(usage_complete_indexes),
            "tpm_censored_blocks_n": len(blocks) - len(usage_complete_indexes),
            "tpm_reporting_state": tpm_state,
            "usage_verification_counts_json": canonical_json(dict(usage_sources)),
            "offered_rate_denominator": "scheduled-arrival window",
            "achieved_rate_denominator": "full epoch wall time including response drain",
            "success_rate_estimand": "logical arrivals whose final retry outcome succeeded",
            "tpm_estimand": (
                "successful provider-reported tokens in complete-usage blocks / full block wall "
                "time; partial results may be biased if usage missingness is informative"
            ),
        }
        record.update(
            _estimate_columns(
                "offered_rpm",
                block_rate_interval(offered, arrival_windows, unit_name="requests", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "launched_logical_rpm",
                block_rate_interval(
                    launched_logical, elapsed_wall, unit_name="requests", seed=seed
                ),
            )
        )
        record.update(
            _estimate_columns(
                "completed_rpm",
                block_rate_interval(completed, elapsed_wall, unit_name="requests", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "successful_rpm",
                block_rate_interval(successful, elapsed_wall, unit_name="requests", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "physical_attempt_rpm",
                block_rate_interval(
                    physical_attempts, elapsed_wall, unit_name="attempts", seed=seed
                ),
            )
        )
        record.update(
            _estimate_columns(
                "successful_input_tpm",
                block_rate_interval(input_tokens, usage_elapsed, unit_name="tokens", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "successful_output_tpm",
                block_rate_interval(output_tokens, usage_elapsed, unit_name="tokens", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "success_rate", block_proportion_interval(successful, completed, seed=seed)
            )
        )
        record.update(
            _estimate_columns(
                "physical_attempt_success_rate",
                block_proportion_interval(physical_successes, physical_attempts, seed=seed),
            )
        )
        result.append(record)
    return result


def _mean_interval(values: list[float], *, seed: int) -> Estimate:
    from .statistics import mean_interval

    return mean_interval(values, unit="score [0,1]", seed=seed)


def _metric_values(row: dict[str, Any]) -> dict[str, float | None]:
    decode = None
    if (
        row["decode_eligible"]
        and row["output_tokens"] is not None
        and row["ttft_seconds"] is not None
        and row["total_seconds"] - row["ttft_seconds"] > 0
    ):
        decode = row["output_tokens"] / (row["total_seconds"] - row["ttft_seconds"])
    return {
        "total_seconds": row["total_seconds"],
        "ttft_seconds": row["ttft_seconds"],
        "decode_proxy_tokens_per_second": decode,
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "queue_delay_seconds": row["queue_delay_seconds"],
        "content_event_count": row["content_event_count"],
    }


def build_outlier_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for row in rows:
        classification = row.get("validity_class")
        if classification in {"invalid", "censored", "anomalous"}:
            excluded: list[str] = []
            if not row["latency_eligible"]:
                excluded.append("latency")
            if not row["usage_eligible"]:
                excluded.extend(["input_tpm", "output_tpm", "cost_per_token"])
            if not row["decode_eligible"]:
                excluded.append("decode_proxy_tokens_per_second")
            if not row["quality_eligible"]:
                excluded.append("quality")
            audit[row["request_id"]] = {
                "request_id": row["request_id"],
                "route_id": row["route_id"],
                "suite": row["suite"],
                "cell_id": row["cell_id"],
                "cache_state": row["cache_state"],
                "audit_class": classification,
                "reasons": _json(row.get("validity_reasons_json"), ["unspecified"]),
                "excluded_estimands": sorted(set(excluded)),
                "metric_values": _metric_values(row),
                "metric_provenance": {
                    "latency": "client monotonic request duration",
                    "ttft": "client monotonic start to first content-bearing SSE event",
                    "decode_proxy": DECODE_PROVENANCE,
                },
                "preserved": True,
            }

    # Valid extremes are discovered within matched cells, never across heterogeneous workloads.
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("validity_class") in {"valid", "anomalous"}:
            groups[(row["route_id"], row["suite"], row["cell_id"], row["cache_state"])].append(row)
    for items in groups.values():
        for metric in ("total_seconds", "ttft_seconds", "queue_delay_seconds"):
            values = [float(row[metric]) for row in items if row.get(metric) is not None]
            if len(values) < 5:
                continue
            q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
            lower, upper = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
            for row in items:
                value = row.get(metric)
                if value is None or lower <= float(value) <= upper:
                    continue
                entry = audit.setdefault(
                    row["request_id"],
                    {
                        "request_id": row["request_id"],
                        "route_id": row["route_id"],
                        "suite": row["suite"],
                        "cell_id": row["cell_id"],
                        "cache_state": row["cache_state"],
                        "audit_class": "valid_extreme",
                        "reasons": [],
                        "excluded_estimands": [],
                        "metric_values": _metric_values(row),
                        "metric_provenance": {
                            "latency": "client monotonic request duration",
                            "ttft": "client monotonic start to first content-bearing SSE event",
                            "decode_proxy": DECODE_PROVENANCE,
                        },
                        "preserved": True,
                    },
                )
                entry["reasons"].append(f"matched_cell_3xIQR_extreme:{metric}")
    return sorted(audit.values(), key=lambda item: item["request_id"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    override = os.environ.get("INFERENCE_BENCH_CODE_REVISION", "")
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", override):
        return override.lower()
    for parent in Path(__file__).resolve().parents:
        git_dir = parent / ".git"
        head = git_dir / "HEAD"
        if not head.is_file():
            continue
        value = head.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-fA-F]{40,64}", value):
            return value.lower()
        if value.startswith("ref: "):
            ref = git_dir / value[5:]
            if ref.is_file():
                revision = ref.read_text(encoding="utf-8").strip()
                if re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
                    return revision.lower()
    return "not_recorded"


def _distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "inference-endpoint-benchmark",
        "httpx",
        "matplotlib",
        "PyYAML",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not_installed_as_distribution"
    return versions


def _write_reproducibility_manifest(
    *,
    run_dir: Path,
    report_dir: Path,
    campaign_hash: str | None,
    campaign_started_at_utc: str | None,
    config_json: str | None,
    events: list[dict[str, Any]],
) -> Path:
    candidates = [
        run_dir / "ledger.sqlite3",
        run_dir / "events.jsonl",
        run_dir / "campaign.public.json",
        *sorted(path for path in report_dir.rglob("*") if path.is_file()),
    ]
    manifest_path = report_dir / "reproducibility-manifest.json"
    artifacts: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(run_dir).as_posix()
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "release_class": (
                    "private_source_review_required"
                    if relative in {"ledger.sqlite3", "events.jsonl"}
                    else "public_candidate_review_required"
                ),
            }
        )
    terminal_events = [row for row in events if row["kind"] == "campaign_terminal"]
    terminal_reason: Any = "not_recorded"
    if terminal_events:
        terminal_reason = _json(terminal_events[-1]["payload_json"], "unparseable")
    manifest = {
        "schema_version": "inference-benchmark-reproducibility/v1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "campaign": {
            "identity_hash": campaign_hash or "not_recorded",
            "started_at_utc": campaign_started_at_utc or "not_recorded",
            "sanitized_config_sha256": (
                hashlib.sha256(config_json.encode("utf-8")).hexdigest()
                if config_json is not None
                else "not_recorded"
            ),
            "terminal_event": terminal_reason,
        },
        "software": {
            "source_revision": _source_revision(),
            "source_tree_state": "not_assessed",
            "source_revision_scope": (
                "HEAD or explicit revision identifier only; the reporter does not detect or hash "
                "uncommitted source changes"
            ),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "operating_system": platform.system(),
            "machine_architecture": platform.machine(),
            "distributions": _distribution_versions(),
            "dependency_capture_scope": (
                "direct runtime distributions visible to the report process; this is not a "
                "transitive lockfile"
            ),
        },
        "artifacts": artifacts,
        "release_status": {
            "publication_gate": "not_implemented",
            "human_secret_and_claim_review_required": True,
            "pdf_generated": False,
            "report_format": "Markdown, CSV, JSON/JSONL, and PNG",
        },
    }
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest_path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:100]


def _plot_matched_cells(summary: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        grouped[(row["suite"], row["cell_id"], row["cache_state"])].append(row)
    created: list[str] = []
    for (suite, cell, cache_state), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda item: item["route_id"])
        metrics = [
            ("ttft_p50", "TTFT p50 (seconds)"),
            ("latency_p50", "End-to-end p50 (seconds)"),
            (
                "decode_proxy_tps_p50",
                "Billed completion-token decode proxy p50 (tokens/second)",
            ),
            ("success_rate", "Success probability"),
        ]
        eligible = [
            (key, label) for key, label in metrics if any(row.get(key) is not None for row in rows)
        ]
        if not eligible:
            continue
        fig, axes = plt.subplots(
            1, len(eligible), figsize=(4.5 * len(eligible), max(3, 0.5 * len(rows) + 1.5))
        )
        if len(eligible) == 1:
            axes = [axes]
        labels = [row["route_id"] for row in rows]
        for axis, (metric, label) in zip(axes, eligible, strict=True):
            for y, row in enumerate(rows):
                value = row.get(metric)
                if value is None:
                    continue
                low, high = row.get(f"{metric}_ci95_low"), row.get(f"{metric}_ci95_high")
                error = None if low is None or high is None else [[value - low], [high - value]]
                axis.errorbar(value, y, xerr=error, fmt="o", capsize=3, color="#176B87")
                axis.annotate(
                    f"n={row.get(f'{metric}_n', 0)}",
                    (value, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                )
            axis.set_yticks(range(len(labels)), labels if axis is axes[0] else [""] * len(labels))
            axis.set_xlabel(label)
            axis.grid(axis="x", alpha=0.25)
            positive = [
                float(row[metric])
                for row in rows
                if row.get(metric) is not None and row[metric] > 0
            ]
            if positive and max(positive) / min(positive) >= 20 and metric != "success_rate":
                axis.set_xscale("log")
                axis.set_xlabel(label + " · log scale")
        fig.suptitle(
            f"Matched cell: {suite} / {cell} / cache={cache_state}\n"
            "points are estimates; bars are 95% CIs"
        )
        fig.tight_layout()
        filename = f"matched-{_slug(suite)}-{_slug(cell)}-{_slug(cache_state)}.png"
        fig.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(filename)
    return created


def _plot_load_small_multiples(events: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = summarize_load_events(events)
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_shape[summary["shape"]].append(summary)
    created: list[str] = []
    for shape, items in sorted(by_shape.items()):
        route_ids = sorted({item["route_id"] for item in items})
        if not route_ids:
            continue
        columns = min(3, len(route_ids))
        rows_n = math.ceil(len(route_ids) / columns)
        fig, axes = plt.subplots(
            rows_n, columns, figsize=(5 * columns, 3.5 * rows_n), squeeze=False
        )
        for axis, route_id in zip(axes.flat, route_ids, strict=False):
            route_summaries = [item for item in items if item["route_id"] == route_id]
            for summary in route_summaries:
                estimate = summary["success_rate"]
                if estimate is None:
                    continue
                color = {
                    "aimd": "#176B87",
                    "confirmation": "#2E8B57",
                    "recovery": "#C65D21",
                    "soak_block": "#6A5ACD",
                }.get(summary["phase"], "#555555")
                yerr = None
                if summary["success_rate_ci95_low"] is not None:
                    yerr = [
                        [estimate - summary["success_rate_ci95_low"]],
                        [summary["success_rate_ci95_high"] - estimate],
                    ]
                axis.errorbar(
                    summary["offered_rps_target"],
                    estimate,
                    yerr=yerr,
                    fmt="o",
                    capsize=2,
                    color=color,
                )
                axis.annotate(
                    f"n={summary['success_rate_n']} blocks",
                    (summary["offered_rps_target"], estimate),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6,
                )
            axis.set_title(f"{route_id} · {len(route_summaries)} matched rate/phase cells")
            axis.set_xlabel("Offered requests/second")
            axis.set_ylabel("Successful-request proportion")
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.25)
        for axis in list(axes.flat)[len(route_ids) :]:
            axis.remove()
        fig.suptitle(
            f"{shape}: matched route × phase × offered-rate cells (unconnected)\n"
            "95% intervals resample epochs/blocks (exchangeability assumption)"
        )
        fig.tight_layout()
        filename = f"load-small-multiples-{_slug(shape)}.png"
        fig.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(filename)
    return created


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    ledger = Ledger(run_dir)
    rows = ledger.rows()
    events = ledger.event_rows()
    summary = summarize_rows(rows)
    load_summary = summarize_load_events(events, rows=rows)
    audit = build_outlier_audit(rows)
    report_dir = run_dir / "report"
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "matched-cell-summary.csv", summary)
    write_csv(report_dir / "load-block-summary.csv", load_summary)
    with (report_dir / "outlier-audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in audit:
            handle.write(canonical_json(item) + "\n")
    audit_counts = Counter(item["audit_class"] for item in audit)
    write_csv(
        report_dir / "outlier-audit-summary.csv",
        [{"audit_class": key, "n": value} for key, value in sorted(audit_counts.items())],
    )
    metric_contract = {
        "schema_version": "metric-contract/v1",
        "latency": {"unit": "seconds", "clock": "client monotonic"},
        "ttft": {"unit": "seconds", "definition": "start to first content-bearing SSE event"},
        "decode_proxy": {"unit": "tokens/second", "definition": DECODE_PROVENANCE},
        "aggregate_output_goodput": {
            "unit": "tokens/minute",
            "definition": AGGREGATE_OUTPUT_PROVENANCE,
            "denominator": "full epoch/block wall time including post-arrival drain",
            "usage_requirement": (
                "all successful requests in an included block have provider-reported prompt and "
                "completion usage; otherwise that block is censored from TPM"
            ),
        },
        "offered_request_rate": {
            "unit": "requests/minute",
            "denominator": "scheduled arrival window",
        },
        "achieved_request_rate": {
            "unit": "requests/minute",
            "denominator": "full epoch/block wall time including post-arrival drain",
        },
        "physical_attempt_rate": {
            "unit": "provider send attempts/minute",
            "denominator": "full epoch/block wall time including post-arrival drain",
            "retry_policy": "includes intermediate retry sends and their failures",
        },
        "load_confidence_intervals": {
            "sampling_unit": "epoch/block, not individual request",
            "method": "paired epoch/block bootstrap of ratio of sums",
            "limitation": (
                "intervals assume exchangeable blocks; adjacent or adaptively selected epochs may "
                "remain temporally dependent"
            ),
        },
        "cache_read_usage": {
            "unit": "tokens",
            "missing_value": "unknown/not reported",
            "explicit_zero": "provider-reported cache miss",
            "positive_value": "provider-reported cache-read token count",
            "pooling": "cached_trial, uncached_trial, and uncontrolled are separate cells",
        },
        "sse_event_span": {
            "eligible_for_token_rate": False,
            "reason": "events may batch arbitrary token counts",
        },
        "p99_minimum_n": 1000,
        "trimming": "none",
        "anomalous_decode_policy": (
            "exclude validity-class anomalous decode proxies from primary decode summaries; "
            "retain in the request-level audit"
        ),
    }
    (report_dir / "metric-contract.json").write_text(
        canonical_json(metric_contract) + "\n", encoding="utf-8"
    )
    plot_names = _plot_matched_cells(summary, figures) + _plot_load_small_multiples(events, figures)
    exposure = ledger.exposure()
    unknown = sum(row["state"] == "unknown" for row in rows)
    lines = [
        "# Inference endpoint benchmark report",
        "",
        "This report contains matched route × suite × workload estimands only. "
        "It is not a global model ranking.",
        "",
        "## Run integrity",
        "",
        f"- Terminal/recorded attempts: {len(rows):,}",
        f"- Unknown outcomes retained and never replayed: {unknown:,}",
        f"- Settled conservative exposure: ${exposure.settled_usd:.6f}",
        f"- Unknown/orphan reservation exposure: ${exposure.reserved_usd:.6f}",
        f"- Matched result cells: {len(summary):,}",
        f"- Outlier/validity audit rows: {len(audit):,}",
        "",
        "## Interpretation",
        "",
        "- `valid_extreme` observations remain in estimates; they are never silently trimmed.",
        "- Invalid observations are excluded only from incompatible estimands and remain "
        "in the audit.",
        "- Decode-proxy observations classified as anomalous are excluded from the primary "
        "decode summary; valid matched-cell extremes remain included and flagged.",
        "- Censored rows may support latency while remaining ineligible for usage/TPM.",
        "- Offered RPM uses the scheduled arrival window. Completed RPM and effective TPM use "
        "the full epoch wall time, including response drain.",
        "- TPM is calculated only from blocks whose successful-request usage is complete and "
        "ledger-verifiable; partial coverage is labelled and can be informatively missing.",
        "- Decode speed is a client-observed request proxy, not direct server compute.",
        "- Cached, uncached, and uncontrolled cells are never pooled.",
        "- Missing cache-read usage is kept distinct from an explicit provider-reported zero.",
        "- p99 is withheld when fewer than 1,000 eligible observations exist.",
        "",
        "## Artifacts",
        "",
        "- `matched-cell-summary.csv`: estimates, units, n, CI bounds, and methods.",
        "- `load-block-summary.csv`: epoch/block RPM and effective TPM with 95% intervals.",
        "- `outlier-audit.jsonl`: request-level validity and outlier evidence.",
        "- `metric-contract.json`: exact metric definitions and provenance.",
        "- `reproducibility-manifest.json`: code/environment identifiers and artifact hashes.",
        "",
        "This generator produces an evidence package, not a publication approval or PDF. "
        "Human claim, privacy, secret, and visual review remains required before release.",
    ]
    if plot_names:
        lines.extend(["", "## Figures", ""])
        lines.extend(f"- `figures/{name}`" for name in plot_names)
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    campaign_hash = ledger.meta("campaign_hash")
    campaign_started_at_utc = ledger.meta("started_at_utc")
    config_json = ledger.meta("config_json")
    ledger.close()
    _write_reproducibility_manifest(
        run_dir=run_dir,
        report_dir=report_dir,
        campaign_hash=campaign_hash,
        campaign_started_at_utc=campaign_started_at_utc,
        config_json=config_json,
        events=events,
    )
    return report_path
