from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .models import canonical_json
from .statistics import (
    Estimate,
    block_rate_interval,
    median_interval,
    qualified_p99,
    quantile,
    quantile_interval,
    wilson_interval,
)

DECODE_PROVENANCE = "completion_tokens / (request_seconds - TTFT); client-observed proxy"
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
    for (route, suite, cell, cache_state), items in sorted(groups.items()):
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
        settled_usd = sum(float(row["settled_usd"]) for row in items)
        successful_output_tokens = sum(
            int(row["output_tokens"] or 0) for row in successes if row["usage_eligible"]
        )
        record: dict[str, Any] = {
            "route_id": route,
            "suite": suite,
            "cell_id": cell,
            "cache_state": cache_state,
            "attempts_n": len(items),
            "successes_n": len(successes),
            "valid_n": sum(row["validity_class"] == "valid" for row in items),
            "anomalous_n": sum(row["validity_class"] == "anomalous" for row in items),
            "invalid_n": sum(row["validity_class"] == "invalid" for row in items),
            "censored_n": sum(row["validity_class"] == "censored" for row in items),
            "usage_complete_n": sum(row["usage_eligible"] for row in items),
            "cache_read_tokens_sum": sum(row["cache_read_input_tokens"] or 0 for row in items),
            "settled_usd_sum": settled_usd,
            "reserved_upper_bound_cost_n": sum(
                row["cost_basis"] == "reserved_upper_bound" for row in items
            ),
            "cost_per_successful_request_usd": (
                settled_usd / len(successes) if successes else None
            ),
            "cost_per_million_effective_output_tokens_usd": (
                settled_usd * 1_000_000 / successful_output_tokens
                if successful_output_tokens
                else None
            ),
            "decode_metric_provenance": DECODE_PROVENANCE,
            "request_sampling_unit": "independent terminal request attempt",
            "http_status_counts_json": canonical_json(
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


def summarize_load_events(events: list[dict[str, Any]], *, seed: int = 1) -> list[dict[str, Any]]:
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
        durations = [float(block["duration_seconds"]) for block in blocks]
        offered = [float(block["scheduled"]) for block in blocks]
        successful = [float(block["successful"]) for block in blocks]
        input_tokens = [float(block["successful_input_tokens"]) for block in blocks]
        output_tokens = [float(block["successful_output_tokens"]) for block in blocks]
        total_completed = sum(int(block["completed"]) for block in blocks)
        total_successful = sum(int(block["successful"]) for block in blocks)
        record: dict[str, Any] = {
            "route_id": route,
            "shape": shape,
            "phase": phase,
            "offered_rps_target": offered_rps,
            "blocks_n": len(blocks),
            "requests_completed_n": total_completed,
            "healthy_blocks_n": sum(bool(block["healthy"]) for block in blocks),
            "sampling_unit": "independent load epoch/block",
            "aggregate_output_metric_provenance": AGGREGATE_OUTPUT_PROVENANCE,
        }
        record.update(
            _estimate_columns(
                "offered_rpm",
                block_rate_interval(offered, durations, unit_name="requests", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "successful_rpm",
                block_rate_interval(successful, durations, unit_name="requests", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "successful_input_tpm",
                block_rate_interval(input_tokens, durations, unit_name="tokens", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "successful_output_tpm",
                block_rate_interval(output_tokens, durations, unit_name="tokens", seed=seed),
            )
        )
        record.update(
            _estimate_columns("success_rate", wilson_interval(total_successful, total_completed))
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
            ("decode_proxy_tps_p50", "Decode proxy p50 (tokens/second)"),
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

    epochs = [json.loads(row["payload_json"]) for row in events if row["kind"] == "load_epoch"]
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for epoch in epochs:
        by_shape[epoch["shape"]].append(epoch)
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
            route_epochs = [item for item in items if item["route_id"] == route_id]
            for epoch in route_epochs:
                total = epoch["completed"]
                success = epoch["successful"]
                estimate = wilson_interval(success, total)
                if estimate.estimate is None:
                    continue
                color = {
                    "aimd": "#176B87",
                    "confirmation": "#2E8B57",
                    "recovery": "#C65D21",
                    "soak_block": "#6A5ACD",
                }.get(epoch["phase"], "#555555")
                yerr = None
                if estimate.lower_95 is not None:
                    yerr = [
                        [estimate.estimate - estimate.lower_95],
                        [estimate.upper_95 - estimate.estimate],
                    ]
                axis.errorbar(
                    epoch["offered_rps"],
                    estimate.estimate,
                    yerr=yerr,
                    fmt="o",
                    capsize=2,
                    color=color,
                )
            axis.set_title(f"{route_id} · {len(route_epochs)} epochs")
            axis.set_xlabel("Offered requests/second")
            axis.set_ylabel("Successful-request proportion")
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.25)
        for axis in list(axes.flat)[len(route_ids) :]:
            axis.remove()
        fig.suptitle(f"{shape}: independent epochs (unconnected), Wilson 95% intervals")
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
    load_summary = summarize_load_events(events)
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
        },
        "sse_event_span": {
            "eligible_for_token_rate": False,
            "reason": "events may batch arbitrary token counts",
        },
        "p99_minimum_n": 1000,
        "trimming": "none",
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
        "- Censored rows may support latency while remaining ineligible for usage/TPM.",
        "- Decode speed is a client-observed request proxy, not direct server compute.",
        "- Cached, uncached, and uncontrolled cells are never pooled.",
        "- p99 is withheld when fewer than 1,000 eligible observations exist.",
        "",
        "## Artifacts",
        "",
        "- `matched-cell-summary.csv`: estimates, units, n, CI bounds, and methods.",
        "- `load-block-summary.csv`: epoch/block RPM and effective TPM with 95% intervals.",
        "- `outlier-audit.jsonl`: request-level validity and outlier evidence.",
        "- `metric-contract.json`: exact metric definitions and provenance.",
    ]
    if plot_names:
        lines.extend(["", "## Figures", ""])
        lines.extend(f"- `figures/{name}`" for name in plot_names)
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger.close()
    return report_path
