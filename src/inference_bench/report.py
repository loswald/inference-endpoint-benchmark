from __future__ import annotations

import csv
import hashlib
import math
import platform
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .environment import locked_distribution_versions, validate_run_directory_separation
from .json_contract import StrictJSONError, strict_json_loads
from .ledger import LEDGER_PRODUCER_SCHEMA_VERSION, Ledger
from .load import soak_rate_rps
from .models import TRANSPORT_HEADER_PROFILE, canonical_json, normalize_finish_reason
from .statistics import (
    Estimate,
    block_median_interval,
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
    f"{MIN_DECODE_PROXY_SECONDS:g} seconds after TTFT, with explicitly reported "
    "reasoning_tokens=0"
)
AGGREGATE_OUTPUT_PROVENANCE = "successful completion tokens / analysis-block wall-clock minute"


def _json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return strict_json_loads(value)
    except StrictJSONError:
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


def _reasoning_state(row: dict[str, Any]) -> str:
    value = row.get("reasoning_tokens")
    if value is None:
        return "unknown"
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return "invalid_reported"
    return "reported_zero" if value == 0 else "reported_positive"


def _warm_state(suite: str) -> str:
    return "standalone_diagnostic_only" if suite == "warmup" else "uncontrolled_not_paired"


def summarize_rows(rows: list[dict[str, Any]], *, seed: int = 1) -> list[dict[str, Any]]:
    final_outcome_by_logical: dict[str, dict[str, Any]] = {}
    physical_attempts_by_logical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["state"] not in {"terminal", "unknown"}:
            continue
        physical_attempts_by_logical[row["logical_id"]].append(row)
        if not bool(row.get("final_logical", 1)):
            continue
        current = final_outcome_by_logical.get(row["logical_id"])
        if current is None or int(row["attempt_index"]) > int(current["attempt_index"]):
            final_outcome_by_logical[row["logical_id"]] = row
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for logical_id, final_outcome in final_outcome_by_logical.items():
        # A retry sequence is one request-level observation. Choose every stratum from its final
        # outcome first, then charge all physical attempts (including failed retries) to that same
        # stratum. Grouping attempts before this collapse can duplicate a logical request when an
        # intermediate response has unknown usage/cache metadata and its retry reports them.
        groups[
            (
                final_outcome["route_id"],
                final_outcome["suite"],
                final_outcome["cell_id"],
                final_outcome["cache_state"],
            )
        ].extend(physical_attempts_by_logical[logical_id])
    incomplete_latest_by_logical: dict[str, dict[str, Any]] = {}
    for logical_id, physical_attempts in physical_attempts_by_logical.items():
        if logical_id in final_outcome_by_logical:
            continue
        # A guard can stop after a retryable response but before another attempt is legally
        # launchable. Its physical send, error, and cost remain evidence, but it has no final
        # request-level observation. Keep it in its declared base cell for physical costs and the
        # predeclared quality denominator; it remains excluded from success, latency, and usage.
        latest = max(physical_attempts, key=lambda row: int(row["attempt_index"]))
        incomplete_latest_by_logical[logical_id] = latest
        groups[
            (
                latest["route_id"],
                latest["suite"],
                latest["cell_id"],
                latest["cache_state"],
            )
        ].extend(physical_attempts)
    output: list[dict[str, Any]] = []
    for (route, suite, cell, cache_state), physical_items in sorted(groups.items()):
        physical_items = sorted(
            physical_items,
            key=lambda row: (
                str(row["logical_id"]),
                int(row["attempt_index"]),
                str(row.get("request_id", "")),
            ),
        )
        # Retry attempts are conditional observations from the same logical request. Request-level
        # intervals therefore use one final terminal attempt per logical request; attempt-level
        # errors/costs remain separately counted and preserved in the audit.
        final_by_logical = {
            logical_id: final_outcome_by_logical[logical_id]
            for logical_id in {str(row["logical_id"]) for row in physical_items}
            if logical_id in final_outcome_by_logical
        }
        items = [final_by_logical[key] for key in sorted(final_by_logical)]
        successes = [
            row for row in items if row["state"] == "terminal" and row["status"] == "success"
        ]
        latency = [
            float(row["arrival_to_completion_seconds"])
            for row in items
            if row["latency_eligible"] and row.get("arrival_to_completion_seconds") is not None
        ]
        service_latency = [
            float(row["total_seconds"])
            for row in items
            if row["latency_eligible"] and row.get("total_seconds") is not None
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
        incomplete_quality_population = [
            latest
            for latest in incomplete_latest_by_logical.values()
            if (
                str(latest["route_id"]),
                str(latest["suite"]),
                str(latest["cell_id"]),
                str(latest["cache_state"]),
            )
            == (route, suite, cell, cache_state)
        ]
        quality_population = [*items, *incomplete_quality_population]
        quality_trials = [
            row for row in quality_population if bool(row.get("quality_predeclared", 0))
        ]
        malformed_quality_trials = [
            row
            for row in quality_trials
            if row.get("quality_score") is None or not bool(row.get("quality_eligible", 0))
        ]
        if malformed_quality_trials:
            raise ValueError(
                "predeclared quality trial lacks a terminal deterministic score: "
                + ", ".join(str(row.get("request_id")) for row in malformed_quality_trials)
            )
        quality = [float(row["quality_score"]) for row in quality_trials]
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
        unknown_reserved_usd = sum(
            float(row["reserved_usd"]) for row in physical_items if row["state"] == "unknown"
        )
        conservative_exposure_usd = settled_usd + unknown_reserved_usd
        successful_usage_complete = all(row["usage_eligible"] for row in successes)
        successful_output_tokens = sum(
            int(row["output_tokens"] or 0) for row in successes if row["usage_eligible"]
        )
        record: dict[str, Any] = {
            "route_id": route,
            "suite": suite,
            "cell_id": cell,
            "cache_state": cache_state,
            "warm_state": _warm_state(suite),
            "reasoning_token_state": (
                "unconditional_base_cell" if items else "incomplete_retry_sequence"
            ),
            "decode_reasoning_scope": "explicit_reported_zero_only",
            "capability_evidence_scope": (
                "parameter_acceptance_only; feature_behavior_unverified"
                if suite == "capability" and cell.startswith("parameter_acceptance_only_")
                else "functional_or_validation_scope_requires_cell_specific_scorer_review"
                if suite == "capability"
                else "not_applicable"
            ),
            "attempts_n": len(physical_items),
            "physical_successes_n": sum(row["status"] == "success" for row in physical_items),
            "physical_rate_limited_n": sum(
                row["status"] == "rate_limited" for row in physical_items
            ),
            "physical_server_errors_n": sum(
                row["status"] == "server_error" for row in physical_items
            ),
            "physical_timeouts_n": sum(row["status"] == "timeout" for row in physical_items),
            "physical_transport_errors_n": sum(
                row["status"] == "transport_error" for row in physical_items
            ),
            "incomplete_retry_sequences_n": len(
                {
                    str(row["logical_id"])
                    for row in physical_items
                    if str(row["logical_id"]) not in final_outcome_by_logical
                }
            ),
            "logical_requests_n": len(items),
            "successes_n": len(successes),
            "unknown_outcomes_n": sum(row["state"] == "unknown" for row in items),
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
            "reasoning_tokens_sum": sum(
                int(row["reasoning_tokens"])
                for row in items
                if isinstance(row.get("reasoning_tokens"), int)
                and not isinstance(row.get("reasoning_tokens"), bool)
                and int(row["reasoning_tokens"]) >= 0
            ),
            "reasoning_reported_zero_n": sum(
                _reasoning_state(row) == "reported_zero" for row in items
            ),
            "reasoning_reported_positive_n": sum(
                _reasoning_state(row) == "reported_positive" for row in items
            ),
            "reasoning_unknown_n": sum(_reasoning_state(row) == "unknown" for row in items),
            "reasoning_invalid_reported_n": sum(
                _reasoning_state(row) == "invalid_reported" for row in items
            ),
            "quality_estimand": (
                "end_to_end_all_predeclared_trials_non_success_is_zero"
                if quality_trials
                else "not_applicable_no_predeclared_scorer"
            ),
            "quality_trials_n": len(quality_trials),
            "quality_successful_response_n": sum(
                row["status"] == "success" for row in quality_trials
            ),
            "quality_non_success_zero_n": sum(
                row["status"] != "success" and float(row["quality_score"]) == 0
                for row in quality_trials
            ),
            "quality_incomplete_retry_zero_n": sum(
                row in incomplete_quality_population and float(row["quality_score"]) == 0
                for row in quality_trials
            ),
            "quality_unscored_n": len(quality_population) - len(quality_trials),
            "cache_read_reported_n": sum(
                row["cache_read_input_tokens"] is not None for row in items
            ),
            "cache_read_unknown_n": sum(row["cache_read_input_tokens"] is None for row in items),
            "cache_miss_n": sum(row["cache_read_input_tokens"] == 0 for row in items),
            "cache_hit_n": sum(
                isinstance(row["cache_read_input_tokens"], int)
                and not isinstance(row["cache_read_input_tokens"], bool)
                and int(row["cache_read_input_tokens"]) > 0
                for row in items
            ),
            "cache_read_tokens_sum": sum(
                int(row["cache_read_input_tokens"])
                for row in items
                if isinstance(row["cache_read_input_tokens"], int)
                and not isinstance(row["cache_read_input_tokens"], bool)
                and int(row["cache_read_input_tokens"]) >= 0
            ),
            "settled_usd_sum": settled_usd,
            "unknown_reserved_usd_sum": unknown_reserved_usd,
            "conservative_exposure_usd_sum": conservative_exposure_usd,
            "reserved_upper_bound_cost_n": sum(
                row["cost_basis"] == "reserved_upper_bound" for row in physical_items
            ),
            "cache_unknown_upper_bound_cost_n": sum(
                row["cost_basis"] == "provider_usage_cache_unknown_upper_bound"
                for row in physical_items
            ),
            "conservative_exposure_per_successful_request_usd": (
                conservative_exposure_usd / len(successes) if successes else None
            ),
            "conservative_exposure_per_million_effective_output_tokens_usd": (
                conservative_exposure_usd * 1_000_000 / successful_output_tokens
                if successful_output_tokens and successful_usage_complete
                else None
            ),
            "cost_estimand": (
                "settled provider-priced exposure plus retained reservation for unknown outcomes"
            ),
            "decode_metric_provenance": DECODE_PROVENANCE,
            "request_sampling_unit": (
                "no final logical observation; physical attempts and cost only"
                if not items
                else "persisted final logical outcome per request; incomplete retry sequences "
                "remain excluded from non-quality request-level estimands"
            ),
            "retry_end_to_end_latency_scope": (
                "scheduled arrival through final completion, including queueing, prior attempts, "
                "retry backoff, and response drain within one uninterrupted execute call; "
                "process-resumed retries are explicitly censored"
            ),
            "service_latency_estimand": "successful_final_logical_attempts_only",
            "arrival_latency_estimand": "successful_final_logical_outcomes_only",
            "ttft_estimand": "successful_final_logical_outcomes_with_observed_ttft_only",
            "expected_probe_observed_validation_status_n": sum(
                "expected_probe_observed_validation_http_status"
                in _json(row.get("validity_reasons_json"), [])
                for row in items
            ),
            "expected_validation_observed_acceptance_n": sum(
                "expected_validation_rejection_not_enforced_observed_acceptance"
                in _json(row.get("validity_reasons_json"), [])
                for row in items
            ),
            "unexpected_client_error_n": sum(
                "unexpected_client_error" in _json(row.get("validity_reasons_json"), [])
                for row in items
            ),
            "parameter_acceptance_client_error_n": sum(
                "parameter_acceptance_probe_observed_client_error"
                in _json(row.get("validity_reasons_json"), [])
                for row in items
            ),
            "http_status_counts_json": canonical_json(
                dict(Counter(str(row["http_status"]) for row in physical_items))
            ),
            "final_http_status_counts_json": canonical_json(
                dict(Counter(str(row["http_status"]) for row in items))
            ),
            "finish_reason_counts_json": canonical_json(
                dict(
                    Counter(
                        normalize_finish_reason(row["finish_reason"]) or "not_reported"
                        for row in items
                    )
                )
            ),
            "realized_output_tokens_max": max(output_usage) if output_usage else None,
        }
        record.update(_estimate_columns("success_rate", success_estimate))
        record.update(
            _estimate_columns("latency_p50", median_interval(latency, unit="seconds", seed=seed))
        )
        record.update(
            _estimate_columns(
                "service_latency_p50",
                median_interval(service_latency, unit="seconds", seed=seed),
            )
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
        record.update(_estimate_columns("quality_mean", _binary_quality_interval(quality)))
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
        if suite == "load":
            # Concurrent requests in one epoch/block are correlated and are not independent
            # inferential sampling units. Preserve their descriptive point estimates and n, but
            # suppress request-level confidence bars; load-block-summary owns load inference.
            for key in list(record):
                if key.endswith("_ci95_low") or key.endswith("_ci95_high"):
                    record[key] = None
                elif key.endswith("_ci_method"):
                    record[key] = "descriptive_correlated_load_requests_no_CI"
            record["request_level_inference_scope"] = (
                "descriptive_only; use load-block-summary for epoch/block confidence intervals"
            )
        output.append(record)
    return output


def _load_epoch_usage(
    epoch: dict[str, Any], rows: list[dict[str, Any]] | None
) -> tuple[bool, int | None, int | None, str]:
    """Join an epoch to final per-logical-request rows to verify successful usage totals."""
    if rows is None:
        return False, None, None, "request_ledger_not_supplied"
    prefix = f"load:{epoch['route_id']}:{epoch['shape']}:{epoch['epoch_id']}:"
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        logical_id = str(row.get("logical_id") or "")
        if logical_id.startswith(prefix):
            attempts[logical_id].append(row)
    final_rows: list[dict[str, Any]] = []
    for logical_attempts in attempts.values():
        terminal = [row for row in logical_attempts if row.get("state") == "terminal"]
        if terminal:
            final_rows.append(max(terminal, key=lambda row: int(row.get("attempt_index") or 0)))
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


def _load_block_censor_reason(block: dict[str, Any]) -> str | None:
    launch_reason = block.get("launch_guard_reason")
    if block.get("launch_guard_triggered") or launch_reason:
        return str(launch_reason or "unspecified_launch_guard")
    scientific_reason = block.get("scientific_censor_reason")
    if scientific_reason:
        return str(scientific_reason)
    if block.get("controller_eligible") is False:
        return "controller_ineligible_unspecified"
    return None


def summarize_load_events(
    events: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]] | None = None,
    seed: int = 1,
) -> list[dict[str, Any]]:
    epoch_by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["kind"] != "load_epoch":
            continue
        epoch = strict_json_loads(event["payload_json"])
        if not isinstance(epoch, dict):
            raise ValueError("load_epoch payload must be a JSON object")
        existing = epoch_by_id.get(str(epoch["epoch_id"]))
        if existing is not None and canonical_json(existing) != canonical_json(epoch):
            raise ValueError(f"conflicting load summaries for epoch {epoch['epoch_id']}")
        epoch_by_id[str(epoch["epoch_id"])] = epoch
    epochs = list(epoch_by_id.values())
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
    for (route, shape, phase, offered_rps), grouped_blocks in sorted(groups.items()):
        observed_blocks = sorted(grouped_blocks, key=lambda block: str(block["epoch_id"]))
        censored_blocks = [
            block for block in observed_blocks if _load_block_censor_reason(block) is not None
        ]
        blocks = [block for block in observed_blocks if _load_block_censor_reason(block) is None]
        censor_reasons = Counter(str(_load_block_censor_reason(block)) for block in censored_blocks)
        arrival_windows = [float(block["duration_seconds"]) for block in blocks]
        raw_elapsed_wall = [
            float(
                block.get(
                    "actual_elapsed_seconds",
                    float(block["duration_seconds"])
                    + max(0.0, float(block.get("queue_end_seconds", 0))),
                )
            )
            for block in blocks
        ]
        # The registered open-loop experiment observes an arrival window even when the final
        # sampled Poisson arrival occurs early. Achieved rates use at least that full window, plus
        # any response drain beyond it; otherwise sparse blocks can report achieved > offered at
        # 100% success merely because the runner did not sleep through a quiet tail.
        elapsed_wall = [
            max(window, elapsed)
            for elapsed, window in zip(raw_elapsed_wall, arrival_windows, strict=True)
        ]
        block_drain_seconds = [
            max(0.0, elapsed - window)
            for elapsed, window in zip(raw_elapsed_wall, arrival_windows, strict=True)
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
        physical_transport_errors = sum(int(block.get("transport_errors", 0)) for block in blocks)
        unknown_outcomes = sum(int(block.get("unknown", 0)) for block in blocks)
        block_ttft_p95 = [
            float(block["p95_ttft_seconds"])
            for block in blocks
            if block.get("p95_ttft_seconds") is not None
        ]
        block_arrival_latency_p95 = [
            float(block["p95_total_seconds"])
            for block in blocks
            if block.get("p95_total_seconds") is not None
        ]
        block_service_latency_p95 = [
            float(block["p95_service_seconds"])
            for block in blocks
            if block.get("p95_service_seconds") is not None
        ]
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
        if not blocks:
            tpm_state = "censored_no_capacity_eligible_block"
        elif len(usage_complete_indexes) == len(blocks):
            tpm_state = "complete"
        elif usage_complete_indexes:
            tpm_state = "partial_complete_blocks_only"
        else:
            tpm_state = "censored_no_complete_usage_block"
        record: dict[str, Any] = {
            "route_id": route,
            "shape": shape,
            "phase": phase,
            "warm_state": "uncontrolled_not_paired",
            "offered_rps_target": offered_rps,
            "blocks_n": len(observed_blocks),
            "capacity_estimand_blocks_n": len(blocks),
            "censored_blocks_n": len(censored_blocks),
            "censored_block_reasons_json": canonical_json(dict(censor_reasons)),
            "capacity_estimand_state": (
                "eligible_blocks_only" if blocks else "censored_no_eligible_blocks"
            ),
            "requests_completed_n": total_completed,
            "requests_successful_n": total_successful,
            "logical_requests_launched_n": sum(int(value) for value in launched_logical),
            "physical_attempts_n": sum(int(value) for value in physical_attempts),
            "physical_successes_n": sum(int(value) for value in physical_successes),
            "physical_rate_limited_n": physical_rate_limited,
            "physical_server_errors_n": physical_server_errors,
            "physical_timeouts_n": physical_timeouts,
            "physical_transport_errors_n": physical_transport_errors,
            "unknown_outcomes_n": unknown_outcomes,
            "physical_other_outcomes_n": max(
                0, sum(int(value) for value in physical_attempts) - physical_known
            ),
            "physical_status_provenance": (
                "all terminal provider send attempts, including intermediate retries; "
                "other_outcomes is the residual not decomposed by the epoch event schema"
            ),
            "healthy_blocks_n": sum(bool(block["healthy"]) for block in blocks),
            "observed_blocks_healthy_n": sum(bool(block["healthy"]) for block in observed_blocks),
            "observed_requests_completed_n": sum(
                int(block["completed"]) for block in observed_blocks
            ),
            "observed_requests_successful_n": sum(
                int(block["successful"]) for block in observed_blocks
            ),
            "sampling_unit": (
                "capacity-eligible load epoch/block; launch-guarded and scientifically censored "
                "blocks are retained as audit counts but excluded from provider-capacity "
                "estimands; bootstrap assumes block-level exchangeability and does not remove "
                "temporal autocorrelation"
            ),
            "aggregate_output_metric_provenance": AGGREGATE_OUTPUT_PROVENANCE,
            "arrival_window_seconds_sum": sum(arrival_windows),
            "elapsed_wall_seconds_sum": sum(elapsed_wall),
            "raw_runner_elapsed_seconds_sum": sum(raw_elapsed_wall),
            "post_arrival_drain_seconds_sum": sum(
                max(0.0, elapsed - window)
                for elapsed, window in zip(raw_elapsed_wall, arrival_windows, strict=True)
            ),
            "early_termination_seconds_sum": sum(
                max(0.0, window - elapsed)
                for elapsed, window in zip(raw_elapsed_wall, arrival_windows, strict=True)
            ),
            "tpm_complete_blocks_n": len(usage_complete_indexes),
            "tpm_censored_blocks_n": len(blocks) - len(usage_complete_indexes),
            "tpm_reporting_state": tpm_state,
            "usage_verification_counts_json": canonical_json(dict(usage_sources)),
            "offered_rate_denominator": "scheduled-arrival window",
            "achieved_rate_denominator": (
                "at least the full registered arrival window, extended through response drain"
            ),
            "success_rate_estimand": (
                "successful final logical outcomes / all scheduled open-loop arrivals; "
                "unlaunched, unknown, and failed arrivals remain in the denominator"
            ),
            "tpm_estimand": (
                "successful provider-reported tokens in complete-usage blocks / at least the full "
                "registered arrival window, extended through response drain; partial results may "
                "be biased if usage missingness is informative"
            ),
            "arrival_latency_estimand": "successful_final_logical_outcomes_only",
            "service_latency_estimand": "successful_final_logical_outcomes_only",
            "ttft_estimand": "successful_final_logical_outcomes_with_observed_ttft_only",
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
                "success_rate", block_proportion_interval(successful, offered, seed=seed)
            )
        )
        record.update(
            _estimate_columns(
                "completion_rate", block_proportion_interval(completed, offered, seed=seed)
            )
        )
        record.update(
            _estimate_columns(
                "launch_rate", block_proportion_interval(launched_logical, offered, seed=seed)
            )
        )
        record.update(
            _estimate_columns(
                "physical_attempt_success_rate",
                block_proportion_interval(physical_successes, physical_attempts, seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "ttft_p95_across_blocks",
                block_median_interval(block_ttft_p95, unit="seconds", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "arrival_latency_p95_across_blocks",
                block_median_interval(block_arrival_latency_p95, unit="seconds", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "service_latency_p95_across_blocks",
                block_median_interval(block_service_latency_p95, unit="seconds", seed=seed),
            )
        )
        record.update(
            _estimate_columns(
                "post_arrival_drain_p50",
                block_median_interval(block_drain_seconds, unit="seconds", seed=seed),
            )
        )
        result.append(record)
    return result


_CONTROLLER_CENSOR_REASONS = frozenset(
    {
        "cost_guard",
        "time_guard",
        "http_402_latch",
        "reservation_overrun_latch",
        "launch_guard",
        "unhealthy_low_load_baseline",
        "zero_scheduled_poisson_arrivals",
        "interrupted_epoch_unknown_provider_outcomes_no_replay",
        "interrupted_epoch_incomplete_no_replay",
        "no_healthy_capacity_candidate_observed",
        "plan_completed",
        "other_controller_censor_reason",
    }
)


def _controller_reason(value: Any) -> str | None:
    if value is None:
        return None
    return (
        value
        if isinstance(value, str) and value in _CONTROLLER_CENSOR_REASONS
        else ("other_controller_censor_reason")
    )


_AIMD_COMPLETION_STATES = frozenset(
    {
        "campaign_guard_censored",
        "confirmations_inconclusive",
        "completed_confirmations_healthy",
        "completed_confirmations_unhealthy",
        "left_censored_no_healthy_candidate",
    }
)
_AIMD_BOUND_STATES = frozenset(
    {
        "bracketed_healthy_lower_unhealthy_upper",
        "nonmonotonic_overload_no_current_bracket",
        "right_censored_highest_tested_healthy_no_overload",
        "left_censored_no_healthy_candidate",
        "campaign_guard_censored_before_confirmation",
    }
)
_SOAK_COMPLETION_STATES = frozenset(
    {
        "campaign_guard_censored",
        "execution_complete_inconclusive",
        "completed_healthy",
        "completed_unhealthy",
        "partial_incomplete",
    }
)


def _strict_optional_bool(value: Any, field: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"controller field {field} must be Boolean or null")


def _strict_optional_nonnegative_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"controller field {field} must be numeric or null")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"controller field {field} must be finite and nonnegative")
    return number


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"controller field {field} must be a nonnegative integer")
    return value


def _strict_state(value: Any, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"controller field {field} has an invalid state")
    return value


def summarize_controller_events(
    events: list[dict[str, Any]],
    *,
    public_config: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit one explicit bound/completion record for every configured capacity controller."""

    routes = public_config.get("routes")
    suites = public_config.get("suites")
    if routes is None and suites is None:
        return []
    if not isinstance(routes, list) or not isinstance(suites, dict):
        raise ValueError("controller summary requires routes and suites in public configuration")
    route_ids = [route.get("id") for route in routes if isinstance(route, dict)]
    if any(not isinstance(route_id, str) for route_id in route_ids):
        raise ValueError("controller summary encountered an invalid route identity")
    default_shapes = ["short_short", "long_short", "short_long", "mixed"]
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for suite_name in ("aimd", "soak"):
        suite = suites.get(suite_name)
        if not isinstance(suite, dict) or not suite.get("enabled", True):
            continue
        shapes = suite.get("shapes", default_shapes)
        if not isinstance(shapes, list) or any(not isinstance(shape, str) for shape in shapes):
            raise ValueError("controller summary encountered invalid capacity shapes")
        for route_id in route_ids:
            for shape in shapes:
                expected[(suite_name, str(route_id), shape)] = {
                    "suite": suite_name,
                    "route_id": route_id,
                    "shape": shape,
                    "controller_completion_state": "missing_terminal_controller_event",
                    "censor_reason": None,
                    "capacity_bound_state": None,
                    "highest_observed_healthy_rps": None,
                    "healthy_lower_bound_rps": None,
                    "unhealthy_upper_bound_rps": None,
                    "overload_observed": None,
                    "nonmonotonic_overload_observed": None,
                    "confirmations_required": 3 if suite_name == "aimd" else None,
                    "confirmation_execution_complete": None,
                    "confirmation_complete": None,
                    "confirmation_all_healthy": None,
                    "confirmation_healthy_json": "[]",
                    "confirmation_eligible_json": "[]",
                    "confirmation_censor_reasons_json": "[]",
                    "recovery_run": None,
                    "recovery_healthy": None,
                    "recovery_eligible": None,
                    "recovery_censor_reason": None,
                    "tested_rate_rps": None,
                    "planned_blocks": None,
                    "completed_blocks": None,
                    "block_eligible_json": "[]",
                    "block_healthy_json": "[]",
                    "block_censor_reasons_json": "[]",
                    "execution_complete": None,
                    "scientifically_complete": None,
                    "all_blocks_healthy": None,
                }
    terminal_reason: str | None = None
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        kind = str(event.get("kind"))
        if kind not in {
            "aimd_complete",
            "aimd_controller_censored",
            "soak_complete",
            "soak_controller_censored",
            "campaign_terminal",
        }:
            continue
        payload = strict_json_loads(event["payload_json"])
        if not isinstance(payload, dict):
            raise ValueError("controller event payload must be an object")
        if kind == "campaign_terminal":
            terminal_reason = _controller_reason(payload.get("reason"))
            continue
        suite_name = "aimd" if kind.startswith("aimd_") else "soak"
        key = (suite_name, str(payload.get("route_id")), str(payload.get("shape")))
        if key not in expected:
            raise ValueError("controller event does not match configured route/suite/shape")
        if key in seen:
            raise ValueError("duplicate terminal controller event")
        seen.add(key)
        row = expected[key]
        if kind.endswith("controller_censored"):
            row.update(
                {
                    "controller_completion_state": "baseline_censored",
                    "censor_reason": _controller_reason(payload.get("reason")),
                }
            )
            continue
        row["censor_reason"] = _controller_reason(payload.get("censor_reason"))
        if suite_name == "aimd":
            confirmations = payload.get("confirmation_healthy", [])
            if not isinstance(confirmations, list) or any(
                value is not None and not isinstance(value, bool) for value in confirmations
            ):
                raise ValueError("AIMD confirmation health must be a Boolean-or-null array")
            eligible = payload.get("confirmation_eligible", [])
            if not isinstance(eligible, list) or any(
                not isinstance(value, bool) for value in eligible
            ):
                raise ValueError("AIMD confirmation eligibility must be a Boolean array")
            reasons = payload.get("confirmation_censor_reasons", [])
            if not isinstance(reasons, list):
                raise ValueError("AIMD confirmation censor reasons must be an array")
            reasons = [_controller_reason(reason) for reason in reasons]
            if not (len(confirmations) == len(eligible) == len(reasons)):
                raise ValueError("AIMD confirmation evidence arrays must have equal length")
            if any(
                (is_eligible and health is None)
                or (not is_eligible and health is not None)
                or (is_eligible and reason is not None)
                or (not is_eligible and reason is None)
                for health, is_eligible, reason in zip(
                    confirmations, eligible, reasons, strict=True
                )
            ):
                raise ValueError("AIMD confirmation eligibility/censor evidence is inconsistent")
            completion_state = _strict_state(
                payload.get("controller_completion_state"),
                _AIMD_COMPLETION_STATES,
                "controller_completion_state",
            )
            bound_state = _strict_state(
                payload.get("capacity_bound_state"),
                _AIMD_BOUND_STATES,
                "capacity_bound_state",
            )
            confirmations_required = _strict_nonnegative_int(
                payload.get("confirmations_required", 3), "confirmations_required"
            )
            if confirmations_required != 3:
                raise ValueError("AIMD contract requires exactly three confirmation epochs")
            confirmation_execution_complete = _strict_optional_bool(
                payload.get("confirmation_execution_complete"),
                "confirmation_execution_complete",
            )
            confirmation_complete = _strict_optional_bool(
                payload.get("confirmation_complete"), "confirmation_complete"
            )
            confirmation_all_healthy = _strict_optional_bool(
                payload.get("confirmation_all_healthy"), "confirmation_all_healthy"
            )
            expected_execution_complete = len(confirmations) == confirmations_required
            expected_scientific_complete = expected_execution_complete and all(eligible)
            if confirmation_execution_complete is not expected_execution_complete:
                raise ValueError("AIMD confirmation execution completeness is inconsistent")
            if confirmation_complete is not expected_scientific_complete:
                raise ValueError("AIMD scientific confirmation completeness is inconsistent")
            if expected_scientific_complete:
                if confirmation_all_healthy is not all(bool(value) for value in confirmations):
                    raise ValueError("AIMD aggregate confirmation health is inconsistent")
            elif confirmation_all_healthy is not None:
                raise ValueError("censored AIMD confirmations cannot have aggregate health")
            if row["censor_reason"] is not None:
                expected_completion_state = "campaign_guard_censored"
            elif bound_state == "left_censored_no_healthy_candidate":
                expected_completion_state = "left_censored_no_healthy_candidate"
            elif not expected_scientific_complete:
                expected_completion_state = "confirmations_inconclusive"
            elif confirmation_all_healthy:
                expected_completion_state = "completed_confirmations_healthy"
            else:
                expected_completion_state = "completed_confirmations_unhealthy"
            if completion_state != expected_completion_state:
                raise ValueError("AIMD controller completion state contradicts its evidence")
            recovery_run = _strict_optional_bool(payload.get("recovery_run", False), "recovery_run")
            recovery_healthy = _strict_optional_bool(
                payload.get("recovery_healthy"), "recovery_healthy"
            )
            recovery_eligible = _strict_optional_bool(
                payload.get("recovery_eligible"), "recovery_eligible"
            )
            recovery_censor_reason = _controller_reason(payload.get("recovery_censor_reason"))
            if recovery_run:
                if recovery_eligible is None:
                    raise ValueError("a run recovery must declare scientific eligibility")
                if recovery_eligible and (recovery_healthy is None or recovery_censor_reason):
                    raise ValueError("eligible recovery evidence is inconsistent")
                if not recovery_eligible and (
                    recovery_healthy is not None or recovery_censor_reason is None
                ):
                    raise ValueError("censored recovery evidence is inconsistent")
            elif any(
                value is not None
                for value in (recovery_healthy, recovery_eligible, recovery_censor_reason)
            ):
                raise ValueError("a recovery that did not run cannot contain recovery evidence")
            healthy_lower = _strict_optional_nonnegative_number(
                payload.get("healthy_lower_bound_rps"), "healthy_lower_bound_rps"
            )
            highest_healthy = _strict_optional_nonnegative_number(
                payload.get("highest_observed_healthy_rps"),
                "highest_observed_healthy_rps",
            )
            unhealthy_upper = _strict_optional_nonnegative_number(
                payload.get("unhealthy_upper_bound_rps"), "unhealthy_upper_bound_rps"
            )
            overload_observed = _strict_optional_bool(
                payload.get("overload_observed"), "overload_observed"
            )
            nonmonotonic_overload = _strict_optional_bool(
                payload.get("nonmonotonic_overload_observed"),
                "nonmonotonic_overload_observed",
            )
            if (highest_healthy is None) != (healthy_lower is None) or (
                highest_healthy is not None
                and (highest_healthy <= 0 or highest_healthy != healthy_lower)
            ):
                raise ValueError("AIMD highest healthy rate and healthy lower bound disagree")
            if nonmonotonic_overload is True and overload_observed is not True:
                raise ValueError("nonmonotonic overload evidence requires observed overload")
            if bound_state == "bracketed_healthy_lower_unhealthy_upper" and not (
                healthy_lower is not None
                and unhealthy_upper is not None
                and unhealthy_upper > healthy_lower
                and overload_observed is True
                and nonmonotonic_overload is False
            ):
                raise ValueError("AIMD bracket bounds are inconsistent")
            if bound_state == "nonmonotonic_overload_no_current_bracket" and not (
                overload_observed is True
                and nonmonotonic_overload is True
                and healthy_lower is not None
                and unhealthy_upper is None
            ):
                raise ValueError("nonmonotonic AIMD state is inconsistent")
            if bound_state == "right_censored_highest_tested_healthy_no_overload" and not (
                overload_observed is False
                and nonmonotonic_overload is False
                and healthy_lower is not None
                and unhealthy_upper is None
            ):
                raise ValueError("right-censored AIMD state is inconsistent")
            if bound_state == "left_censored_no_healthy_candidate" and (
                highest_healthy is not None or healthy_lower is not None
            ):
                raise ValueError("left-censored AIMD state cannot contain a healthy candidate")
            if bound_state == "campaign_guard_censored_before_confirmation" and (
                completion_state != "campaign_guard_censored" or row["censor_reason"] is None
            ):
                raise ValueError("guard-censored AIMD bound must be explicitly censored")
            if completion_state == "left_censored_no_healthy_candidate" and (
                bound_state != "left_censored_no_healthy_candidate"
                or confirmations
                or highest_healthy is not None
            ):
                raise ValueError("left-censored AIMD completion state is inconsistent")
            configured_aimd = suites.get("aimd")
            if not isinstance(configured_aimd, dict):
                raise ValueError("AIMD controller lacks immutable suite configuration")
            configured_max_rps = configured_aimd.get("max_rps")
            if configured_max_rps is not None:
                maximum = _strict_optional_nonnegative_number(
                    configured_max_rps, "configured AIMD max_rps"
                )
                if maximum is None or maximum <= 0:
                    raise ValueError("configured AIMD max_rps must be positive")
                if any(
                    value is not None and value > maximum
                    for value in (highest_healthy, healthy_lower, unhealthy_upper)
                ):
                    raise ValueError("AIMD controller evidence exceeds configured max_rps")
            row["controller_completion_state"] = completion_state
            row.update(
                {
                    "capacity_bound_state": bound_state,
                    "highest_observed_healthy_rps": highest_healthy,
                    "healthy_lower_bound_rps": healthy_lower,
                    "unhealthy_upper_bound_rps": unhealthy_upper,
                    "overload_observed": overload_observed,
                    "nonmonotonic_overload_observed": nonmonotonic_overload,
                    "confirmations_required": confirmations_required,
                    "confirmation_execution_complete": confirmation_execution_complete,
                    "confirmation_complete": confirmation_complete,
                    "confirmation_all_healthy": confirmation_all_healthy,
                    "confirmation_healthy_json": canonical_json(confirmations),
                    "confirmation_eligible_json": canonical_json(eligible),
                    "confirmation_censor_reasons_json": canonical_json(reasons),
                    "recovery_run": recovery_run,
                    "recovery_healthy": recovery_healthy,
                    "recovery_eligible": recovery_eligible,
                    "recovery_censor_reason": recovery_censor_reason,
                }
            )
        else:
            eligible = payload.get("block_eligible", [])
            health = payload.get("block_healthy", [])
            reasons = payload.get("block_censor_reasons", [])
            if not isinstance(eligible, list) or any(
                not isinstance(value, bool) for value in eligible
            ):
                raise ValueError("soak block eligibility must be a Boolean array")
            if not isinstance(health, list) or any(
                value is not None and not isinstance(value, bool) for value in health
            ):
                raise ValueError("soak block health must be a Boolean-or-null array")
            if not isinstance(reasons, list):
                raise ValueError("soak block censor reasons must be an array")
            reasons = [_controller_reason(reason) for reason in reasons]
            if not (len(eligible) == len(health) == len(reasons)):
                raise ValueError("soak block evidence arrays must have equal length")
            if any(
                (is_eligible and value is None)
                or (not is_eligible and value is not None)
                or (is_eligible and reason is not None)
                or (not is_eligible and reason is None)
                for value, is_eligible, reason in zip(health, eligible, reasons, strict=True)
            ):
                raise ValueError("soak block eligibility/censor evidence is inconsistent")
            completion_state = _strict_state(
                payload.get("controller_completion_state"),
                _SOAK_COMPLETION_STATES,
                "controller_completion_state",
            )
            planned_blocks = _strict_nonnegative_int(payload.get("blocks"), "blocks")
            completed_blocks = _strict_nonnegative_int(
                payload.get("completed_blocks"), "completed_blocks"
            )
            execution_complete = _strict_optional_bool(
                payload.get("execution_complete"), "execution_complete"
            )
            scientifically_complete = _strict_optional_bool(
                payload.get("scientifically_complete"), "scientifically_complete"
            )
            all_blocks_healthy = _strict_optional_bool(
                payload.get("all_blocks_healthy"), "all_blocks_healthy"
            )
            if completed_blocks != len(eligible) or completed_blocks > planned_blocks:
                raise ValueError("soak block counts are inconsistent")
            configured_soak = suites.get("soak")
            if not isinstance(configured_soak, dict):
                raise ValueError("soak controller lacks immutable suite configuration")
            configured_blocks = _strict_nonnegative_int(
                configured_soak.get("blocks", 4), "configured soak blocks"
            )
            if configured_blocks <= 0 or planned_blocks != configured_blocks:
                raise ValueError("soak block count contradicts immutable suite configuration")
            tested_rate_rps = _strict_optional_nonnegative_number(
                payload.get("rate_rps"), "rate_rps"
            )
            try:
                configured_rate_rps = soak_rate_rps(
                    configured_soak, str(payload.get("route_id")), str(payload.get("shape"))
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("configured soak rate cannot be resolved") from exc
            if (
                tested_rate_rps is None
                or tested_rate_rps <= 0
                or not math.isfinite(configured_rate_rps)
                or configured_rate_rps <= 0
                or tested_rate_rps != configured_rate_rps
            ):
                raise ValueError("soak tested rate contradicts immutable suite configuration")
            expected_execution_complete = completed_blocks == planned_blocks
            expected_scientific_complete = expected_execution_complete and all(eligible)
            if execution_complete is not expected_execution_complete:
                raise ValueError("soak execution completeness is inconsistent")
            if scientifically_complete is not expected_scientific_complete:
                raise ValueError("soak scientific completeness is inconsistent")
            if expected_scientific_complete:
                if all_blocks_healthy is not all(bool(value) for value in health):
                    raise ValueError("soak aggregate health is inconsistent")
            elif all_blocks_healthy is not None:
                raise ValueError("censored soak blocks cannot have aggregate health")
            if row["censor_reason"] is not None:
                expected_completion_state = "campaign_guard_censored"
            elif not expected_execution_complete:
                expected_completion_state = "partial_incomplete"
            elif not expected_scientific_complete:
                expected_completion_state = "execution_complete_inconclusive"
            elif all_blocks_healthy:
                expected_completion_state = "completed_healthy"
            else:
                expected_completion_state = "completed_unhealthy"
            if completion_state != expected_completion_state:
                raise ValueError("soak controller completion state contradicts its evidence")
            if completion_state == "campaign_guard_censored" and not any(
                reason == row["censor_reason"] for reason in reasons
            ):
                raise ValueError("guard-censored soak state lacks matching block evidence")
            row["controller_completion_state"] = completion_state
            row.update(
                {
                    "tested_rate_rps": _strict_optional_nonnegative_number(
                        tested_rate_rps, "rate_rps"
                    ),
                    "planned_blocks": planned_blocks,
                    "completed_blocks": completed_blocks,
                    "block_eligible_json": canonical_json(eligible),
                    "block_healthy_json": canonical_json(health),
                    "block_censor_reasons_json": canonical_json(reasons),
                    "execution_complete": execution_complete,
                    "scientifically_complete": scientifically_complete,
                    "all_blocks_healthy": all_blocks_healthy,
                }
            )
    for key, row in expected.items():
        prefix = f"load_epoch:{key[0]}-{key[1]}-{key[2]}-"
        states = Counter(
            str(item["state"])
            for item in coverage_rows
            if str(item.get("plan_cell_id", "")).startswith(prefix)
        )
        row["coverage_state_counts_json"] = canonical_json(dict(sorted(states.items())))
        if key not in seen:
            if terminal_reason == "plan_completed":
                raise ValueError("completed campaign is missing a terminal controller event")
            row["controller_completion_state"] = "campaign_censored_before_start"
            row["censor_reason"] = terminal_reason or "other_controller_censor_reason"
    return [expected[key] for key in sorted(expected)]


def _binary_quality_interval(values: list[float]) -> Estimate:
    if any(value not in {0.0, 1.0} for value in values):
        raise ValueError("deterministic quality scores must be binary")
    return wilson_interval(sum(value == 1.0 for value in values), len(values))


def _metric_values(row: dict[str, Any]) -> dict[str, float | None]:
    def finite(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    decode = None
    if (
        finite(row.get("output_tokens")) is not None
        and finite(row.get("ttft_seconds")) is not None
        and finite(row.get("total_seconds")) is not None
        and float(row["total_seconds"]) - float(row["ttft_seconds"]) > 0
    ):
        decode = float(row["output_tokens"]) / (
            float(row["total_seconds"]) - float(row["ttft_seconds"])
        )
    return {
        "total_seconds": finite(row.get("total_seconds")),
        "arrival_to_completion_seconds": finite(row.get("arrival_to_completion_seconds")),
        "ttft_seconds": finite(row.get("ttft_seconds")),
        "billed_output_proxy_tokens_per_second": finite(decode),
        "input_tokens": finite(row.get("input_tokens")),
        "output_tokens": finite(row.get("output_tokens")),
        "reasoning_tokens": finite(row.get("reasoning_tokens")),
        "queue_delay_seconds": finite(row.get("queue_delay_seconds")),
        "content_event_count": finite(row.get("content_event_count")),
    }


def build_outlier_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    completed_logicals = {
        str(row["logical_id"])
        for row in rows
        if row.get("state") in {"terminal", "unknown"} and bool(row.get("final_logical", 1))
    }
    for row in rows:
        incomplete_retry = str(row["logical_id"]) not in completed_logicals
        classification = (
            "censored"
            if row.get("state") == "unknown" or incomplete_retry
            else row.get("validity_class")
        )
        if classification in {"invalid", "censored", "anomalous"}:
            excluded: list[str] = []
            if incomplete_retry:
                excluded.extend(
                    [
                        "success_rate",
                        "latency",
                        "service_latency",
                        "ttft",
                        "input_tpm",
                        "output_tpm",
                        "cost_per_token",
                        "cost_per_successful_request",
                        "decode_proxy_tokens_per_second",
                        "realized_output_tokens",
                    ]
                )
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
                "warm_state": _warm_state(str(row["suite"])),
                "audit_class": classification,
                "reasons": (
                    ["incomplete_retry_sequence_guarded_before_final_outcome"]
                    if incomplete_retry
                    else ["unknown_provider_outcome_final_attempt"]
                    if row.get("state") == "unknown"
                    else _json(row.get("validity_reasons_json"), ["unspecified"])
                ),
                "excluded_estimands": sorted(set(excluded)),
                "metric_values": _metric_values(row),
                "metric_provenance": {
                    "latency": "client monotonic request duration",
                    "ttft": "client monotonic start to first content-bearing SSE event",
                    "decode_proxy": DECODE_PROVENANCE,
                },
                "preserved": True,
            }

    # Valid extremes are discovered within base matched cells, never across heterogeneous
    # workloads or response-derived reasoning states. The latter cannot condition the reliability,
    # latency, cost, or usage population; decode eligibility is already enforced separately.
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("validity_class") == "valid"
            and row.get("state") == "terminal"
            and str(row["logical_id"]) in completed_logicals
        ):
            groups[
                (
                    row["route_id"],
                    row["suite"],
                    row["cell_id"],
                    row["cache_state"],
                )
            ].append(row)
    for items in groups.values():
        metric_rows = {row["request_id"]: _metric_values(row) for row in items}
        for metric in (
            "arrival_to_completion_seconds",
            "total_seconds",
            "ttft_seconds",
            "queue_delay_seconds",
            "billed_output_proxy_tokens_per_second",
            "output_tokens",
        ):
            values = [
                float(metrics[metric])
                for metrics in metric_rows.values()
                if metrics.get(metric) is not None
            ]
            if len(values) < 5:
                continue
            q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
                criterion = "Tukey_outer_fence_3xIQR"
            elif min(values) < q1 or max(values) > q3:
                # A common deterministic baseline can collapse both quartiles. In that case the
                # classical fence is the single central value; preserve and flag deviations
                # instead of silently losing the audit just because IQR is exactly zero.
                lower = upper = q1
                criterion = "zero_IQR_central_value_deviation"
            else:
                continue
            for row in items:
                value = metric_rows[row["request_id"]].get(metric)
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
                        "warm_state": _warm_state(str(row["suite"])),
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
                entry["reasons"].append(f"matched_cell_{criterion}:{metric}:n={len(values)}")
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


def _report_source_snapshot(run_dir: Path) -> dict[str, Any]:
    """Bind the report generator's source tree without exposing local path names."""

    root: Path | None = None
    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if not (candidate / "pyproject.toml").is_file():
            continue
        try:
            resolved = Path(
                subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=candidate,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="surrogateescape",
                ).stdout.strip()
            ).resolve()
        except (OSError, subprocess.CalledProcessError):
            continue
        if resolved == candidate.resolve():
            root = resolved
            break
    if root is None:
        raise ValueError("report generation requires an accessible git source identity")

    def git(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="surrogateescape",
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    tracked = git("ls-files")
    if tracked is None:
        raise ValueError("report generation requires a complete tracked-source inventory")
    validate_run_directory_separation(root, run_dir, tracked.splitlines())
    pathspec = ["--", "."]
    try:
        run_relative = run_dir.resolve().relative_to(root).as_posix()
    except ValueError:
        run_relative = None
    if run_relative and run_relative != ".":
        pathspec.append(f":(exclude){run_relative}/**")
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all", *pathspec)
    diff = git("diff", "--binary", "HEAD", *pathspec)
    untracked = git("ls-files", "--others", "--exclude-standard", *pathspec)
    if None in {commit, status, diff, untracked}:
        raise ValueError("report generation could not bind the source tree state")
    if status:
        raise ValueError("report generation requires clean committed source")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("report generator source revision is invalid")
    # Reuse the source-state verification hash contract captured at inference time. Dirty source
    # was already refused, so this binds the observed clean state without replacing the commit.
    from .cli import _dirty_tree_hash

    dirty_hash = _dirty_tree_hash(root, status, diff, untracked)
    if dirty_hash is None or not re.fullmatch(r"[0-9a-f]{64}", dirty_hash):
        raise ValueError("report generator source tree hash is unavailable")
    return {
        "source_revision": commit,
        "source_tree_state": "dirty" if status else "clean",
        "source_dirty_tree_sha256": dirty_hash,
        "distributions": locked_distribution_versions(root / "requirements.lock"),
    }


def _write_reproducibility_manifest(
    *,
    run_dir: Path,
    report_dir: Path,
    campaign_hash: str | None,
    campaign_started_at_utc: str | None,
    config_json: str | None,
    run_manifest_json: str | None,
    events: list[dict[str, Any]],
    report_source: dict[str, Any],
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
    terminal_ended_at_utc: Any = "not_recorded"
    if terminal_events:
        terminal_reason = _json(terminal_events[-1]["payload_json"], "unparseable")
        terminal_ended_at_utc = terminal_events[-1].get("recorded_at_utc", "not_recorded")
    runtime = _json(run_manifest_json, {})
    manifest = {
        "schema_version": "inference-benchmark-reproducibility/v2",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "campaign": {
            "identity_hash": campaign_hash or "not_recorded",
            "started_at_utc": campaign_started_at_utc or "not_recorded",
            "ended_at_utc": terminal_ended_at_utc,
            "ledger_producer_schema_version": LEDGER_PRODUCER_SCHEMA_VERSION,
            "sanitized_config_sha256": (
                hashlib.sha256(config_json.encode("utf-8")).hexdigest()
                if config_json is not None
                else "not_recorded"
            ),
            "terminal_event": terminal_reason,
        },
        "software": {
            "run_source_revision": runtime.get("source_commit"),
            "run_source_tree_state": (
                "dirty"
                if runtime.get("source_dirty")
                else "clean"
                if runtime.get("source_dirty") is False
                else "not_recorded"
            ),
            "run_source_dirty_tree_sha256": runtime.get("source_dirty_tree_sha256"),
            "run_environment": runtime.get("execution_environment"),
            "report_generator": {
                **report_source,
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "operating_system": platform.system(),
                "operating_system_release": platform.release(),
                "machine_architecture": platform.machine(),
            },
            "dependency_lock_file": runtime.get("dependency_lock_file"),
            "dependency_lock_sha256": runtime.get("dependency_lock_sha256"),
            "dependency_capture_scope": "hash-bound requirements.lock plus observed versions",
        },
        "execution": {
            "normalized_exact_invocation": runtime.get(
                "normalized_exact_invocation", "not_recorded"
            ),
            "raw_invocation_sha256": runtime.get("raw_invocation_sha256", "not_recorded"),
            "client_location": runtime.get("client_location", "not_recorded"),
            "connection_reuse_by_route": runtime.get("connection_reuse_by_route", "not_recorded"),
            "http2_by_route": runtime.get("http2_by_route", "not_recorded"),
            "transport_max_connections_by_route": runtime.get(
                "transport_max_connections_by_route", "not_recorded"
            ),
            "request_timeout_seconds_by_route": runtime.get(
                "request_timeout_seconds_by_route", "not_recorded"
            ),
            "transport_trust_env": runtime.get("transport_trust_env", "not_recorded"),
        },
        "provider_documentation_declarations": runtime.get(
            "provider_documentation_declarations", "not_recorded"
        ),
        "artifacts": artifacts,
        "release_status": {
            "publication_gate": "not_implemented",
            "documentation_evidence_gate": (
                "declared bundle digest not byte-verified by this harness; external evidence "
                "bundle verification is required before publication"
            ),
            "human_secret_and_claim_review_required": True,
            "pdf_generated": False,
            "report_format": "Markdown, CSV, JSON/JSONL, and PNG",
        },
    }
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest_path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:100]


_TIME_PANEL_RE = re.compile(r":panel=(\d{3,})$")


def summarize_time_variation(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project matched-cell estimates onto an explicit route × shape × time panel table."""

    result: list[dict[str, Any]] = []
    for row in summary:
        if row.get("suite") != "time_variation":
            continue
        cell = str(row.get("cell_id") or "")
        match = _TIME_PANEL_RE.search(cell)
        if match is None:
            raise ValueError(f"time-variation cell lacks a panel identity: {cell}")
        shape = cell.split(":", 1)[0]
        projected = dict(row)
        projected["shape"] = shape
        projected["panel_index"] = int(match.group(1))
        result.append(projected)
    return sorted(result, key=lambda row: (row["route_id"], row["shape"], row["panel_index"]))


def _plot_time_variation(summary: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projected = summarize_time_variation(summary)
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in projected:
        by_route[str(row["route_id"])].append(row)
    created: list[str] = []
    metrics = (
        ("latency_p50", "End-to-end latency p50", "seconds"),
        ("ttft_p50", "Time to first token p50", "seconds"),
        ("success_rate", "Request success rate", "proportion"),
    )
    for route_id, route_rows in sorted(by_route.items()):
        shapes = sorted({str(row["shape"]) for row in route_rows})
        if not shapes:
            continue
        fig, axes = plt.subplots(
            len(shapes),
            len(metrics),
            figsize=(4.4 * len(metrics), 3.2 * len(shapes)),
            squeeze=False,
        )
        for shape_index, shape in enumerate(shapes):
            rows = sorted(
                (row for row in route_rows if row["shape"] == shape),
                key=lambda row: int(row["panel_index"]),
            )
            for metric_index, (metric, title, unit) in enumerate(metrics):
                axis = axes[shape_index][metric_index]
                eligible = [row for row in rows if row.get(metric) is not None]
                x = [int(row["panel_index"]) for row in eligible]
                y = [float(row[metric]) for row in eligible]
                lower = [float(row.get(f"{metric}_ci95_low") or row[metric]) for row in eligible]
                upper = [float(row.get(f"{metric}_ci95_high") or row[metric]) for row in eligible]
                if eligible:
                    yerr = [
                        [value - low for value, low in zip(y, lower, strict=True)],
                        [high - value for value, high in zip(y, upper, strict=True)],
                    ]
                    axis.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt="o-",
                        linewidth=1.4,
                        markersize=4,
                        capsize=2.5,
                        color="#176B87",
                    )
                axis.set_title(title)
                axis.set_xlabel("Fixed time panel (equal spacing)")
                axis.set_ylabel(unit)
                axis.grid(axis="y", alpha=0.2)
                if metric == "success_rate":
                    axis.set_ylim(-0.03, 1.03)
                if metric_index == 0:
                    axis.text(
                        -0.23,
                        0.5,
                        shape.replace("_", " / "),
                        transform=axis.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )
        fig.suptitle(
            f"{route_id}: matched low-load measurements across time\n"
            "Same tasks and offered load in every panel; whiskers are request-level 95% intervals"
        )
        fig.tight_layout()
        suffix = hashlib.sha256(route_id.encode()).hexdigest()[:10]
        filename = f"time-variation-{_slug(route_id)}-{suffix}.png"
        fig.savefig(output / filename, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(filename)
    return created


def _plot_matched_cells(summary: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        grouped[
            (
                row["suite"],
                row["cell_id"],
                row["cache_state"],
            )
        ].append(row)
    created: list[str] = []
    for (suite, cell, cache_state), rows in sorted(grouped.items()):
        if suite == "load":
            continue
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda item: item["route_id"])
        metrics = [
            ("ttft_p50", "Successful outcomes: TTFT p50 (seconds)"),
            (
                "latency_p50",
                "Successful outcomes: scheduled-arrival-to-completion p50 (seconds)",
            ),
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
        cell_identity = canonical_json([suite, cell, cache_state]).encode()
        filename = (
            f"matched-{_slug(suite)}-{_slug(cell)}-{_slug(cache_state)}-"
            f"{hashlib.sha256(cell_identity).hexdigest()[:10]}.png"
        )
        fig.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(filename)
    return created


def _plot_load_small_multiples(
    events: list[dict[str, Any]], output: Path, *, rows: list[dict[str, Any]] | None = None
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = summarize_load_events(events, rows=rows)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_cell[(summary["route_id"], summary["shape"])].append(summary)
    created: list[str] = []
    metrics = (
        ("success_rate", "Successful logical requests", "proportion"),
        ("successful_rpm", "Achieved successful RPM", "requests/minute"),
        ("physical_attempt_rpm", "Physical send-attempt RPM", "attempts/minute"),
        ("successful_input_tpm", "Successful billed input TPM", "tokens/minute"),
        ("successful_output_tpm", "Successful billed output TPM", "tokens/minute"),
        ("ttft_p95_across_blocks", "Successful outcomes: block p95 TTFT", "seconds"),
        (
            "arrival_latency_p95_across_blocks",
            "Successful outcomes: block p95 arrival-to-completion latency",
            "seconds",
        ),
        ("post_arrival_drain_p50", "Post-arrival drain p50 across blocks", "seconds"),
    )
    colors = {
        "aimd": "#176B87",
        "confirmation": "#2E8B57",
        "confirmation_separator": "#8AA399",
        "recovery_after_observed_overload": "#C65D21",
        "soak_baseline": "#A68A00",
        "soak_block": "#6A5ACD",
        "baseline": "#777777",
    }
    for (route_id, shape), items in sorted(by_cell.items()):
        eligible = [
            metric for metric in metrics if any(item.get(metric[0]) is not None for item in items)
        ]
        if not eligible:
            continue
        columns = min(3, len(eligible))
        rows_n = math.ceil(len(eligible) / columns)
        fig, axes = plt.subplots(
            rows_n, columns, figsize=(5 * columns, 3.6 * rows_n), squeeze=False
        )
        for axis, (metric, label, unit) in zip(axes.flat, eligible, strict=False):
            for summary in items:
                estimate = summary.get(metric)
                if estimate is None:
                    continue
                color = colors.get(summary["phase"], "#555555")
                yerr = None
                low = summary.get(f"{metric}_ci95_low")
                high = summary.get(f"{metric}_ci95_high")
                if low is not None and high is not None:
                    yerr = [
                        [estimate - low],
                        [high - estimate],
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
                    f"{summary['phase']} · n={summary.get(f'{metric}_n', summary['blocks_n'])}",
                    (summary["offered_rps_target"], estimate),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6,
                )
            axis.set_title(label)
            axis.set_xlabel("Offered requests/second")
            axis.set_ylabel(unit)
            if metric == "success_rate":
                axis.set_ylim(-0.03, 1.03)
            offered_values = [
                float(summary["offered_rps_target"])
                for summary in items
                if float(summary["offered_rps_target"]) > 0
            ]
            if offered_values and max(offered_values) / min(offered_values) >= 20:
                axis.set_xscale("log")
                axis.set_xlabel("Offered requests/second · log scale")
            metric_values = [
                float(summary[metric])
                for summary in items
                if summary.get(metric) is not None and float(summary[metric]) > 0
            ]
            if (
                metric != "success_rate"
                and metric_values
                and max(metric_values) / min(metric_values) >= 20
            ):
                axis.set_yscale("log")
                axis.set_ylabel(unit + " · log scale")
            axis.grid(alpha=0.25)
        for axis in list(axes.flat)[len(eligible) :]:
            axis.remove()
        fig.suptitle(
            f"{route_id} · {shape}: matched phase × offered-rate estimands (unconnected)\n"
            "95% intervals resample epochs/blocks (exchangeability assumption)"
        )
        fig.tight_layout()
        identity_suffix = hashlib.sha256(canonical_json([route_id, shape]).encode()).hexdigest()[
            :10
        ]
        filename = f"load-small-multiples-{_slug(route_id)}-{_slug(shape)}-{identity_suffix}.png"
        fig.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(filename)
    return created


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    source_snapshot = _report_source_snapshot(run_dir)
    ledger = Ledger(run_dir, exclusive_owner=True)
    try:
        return _generate_report_locked(run_dir, ledger, source_snapshot)
    finally:
        ledger.close()


def _generate_report_locked(run_dir: Path, ledger: Ledger, source_snapshot: dict[str, Any]) -> Path:
    if _report_source_snapshot(run_dir) != source_snapshot:
        raise ValueError("report generator source or locked environment changed before derivation")
    if ledger.meta("events_projection_state") == "dirty":
        ledger.rebuild_events_jsonl()
    ledger.checkpoint_for_export()
    rows = ledger.rows()
    events = ledger.event_rows()
    coverage_rows = ledger.coverage_rows()
    _assert_terminal_snapshot(ledger, events)
    summary = summarize_rows(rows)
    time_variation_summary = summarize_time_variation(summary)
    load_summary = summarize_load_events(events, rows=rows)
    config_value = strict_json_loads(ledger.meta("config_json") or "null")
    if not isinstance(config_value, dict):
        raise ValueError("stored public campaign configuration is not an object")
    controller_summary = summarize_controller_events(
        events,
        public_config=config_value,
        coverage_rows=coverage_rows,
    )
    audit = build_outlier_audit(rows)
    report_dir = run_dir / "report"
    if report_dir.exists():
        if report_dir.resolve().parent != run_dir.resolve() or report_dir.name != "report":
            raise ValueError("refusing to clean an unexpected report output path")
        shutil.rmtree(report_dir)
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "matched-cell-summary.csv", summary)
    write_csv(report_dir / "time-variation-summary.csv", time_variation_summary)
    write_csv(report_dir / "load-block-summary.csv", load_summary)
    write_csv(report_dir / "controller-summary.csv", controller_summary)
    (report_dir / "controller-summary.json").write_text(
        canonical_json(controller_summary) + "\n", encoding="utf-8"
    )
    write_csv(report_dir / "coverage-ledger.csv", coverage_rows)
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
        "latency": {
            "unit": "seconds",
            "clock": "client monotonic",
            "headline_definition": (
                "successful final logical outcomes only: scheduled arrival to completion, "
                "including queue delay, retries, backoff, and response drain"
            ),
            "sampling_population": "successful_final_logical_outcomes_only",
            "service_duration": (
                "successful final-logical attempt start to body/stream completion; intermediate "
                "retry and failed-attempt durations are not part of this request-level summary"
            ),
        },
        "ttft": {
            "unit": "seconds",
            "definition": "start to first content-bearing SSE event",
            "sampling_population": ("successful_final_logical_outcomes_with_observed_ttft_only"),
        },
        "decode_proxy": {"unit": "tokens/second", "definition": DECODE_PROVENANCE},
        "aggregate_output_goodput": {
            "unit": "tokens/minute",
            "definition": AGGREGATE_OUTPUT_PROVENANCE,
            "denominator": (
                "at least the full registered arrival window, extended through post-arrival drain"
            ),
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
            "denominator": (
                "at least the full registered arrival window, extended through post-arrival drain"
            ),
        },
        "physical_attempt_rate": {
            "unit": "provider send attempts/minute",
            "denominator": (
                "at least the full registered arrival window, extended through post-arrival drain"
            ),
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
        "warm_state": {
            "standalone_warmup_suite": (
                "diagnostic traffic only; it is not paired with a later measured block"
            ),
            "measured_cells": (
                "uncontrolled_not_paired; no warm- or cold-endpoint latency claim is made"
            ),
        },
        "quality": {
            "estimand": "mean score across every predeclared quality trial",
            "non_success_handling": "deterministic score zero, retained in denominator",
            "unknown_and_incomplete_retry_handling": (
                "claimed predeclared trials receive deterministic zero"
            ),
            "confidence_interval": "Wilson-95 binomial interval; logical trial is the unit",
            "unrelated_validity_handling": (
                "usage/timing validity cannot remove a deterministic quality score"
            ),
            "reported_counts": (
                "quality_trials_n, quality_successful_response_n, "
                "quality_non_success_zero_n, quality_incomplete_retry_zero_n, "
                "quality_unscored_n"
            ),
        },
        "matched_cell_cost": {
            "settled": "provider-priced settled amount only",
            "unknown_reserved": "reservation retained for an unknown provider outcome",
            "conservative_exposure": "settled plus unknown-reserved",
            "derived_costs": "explicit conservative-exposure denominators",
        },
        "sse_event_span": {
            "eligible_for_token_rate": False,
            "reason": "events may batch arbitrary token counts",
        },
        "reasoning_tokens": {
            "reported_positive": "separate stratum; excluded from visible post-TTFT proxy",
            "reported_zero": "eligible for proxy when all other validity gates pass",
            "unknown": "separate stratum; proxy censored",
            "invalid_reported": "separate invalid stratum; proxy and usage estimands censored",
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
    plot_names = (
        _plot_matched_cells(summary, figures)
        + _plot_load_small_multiples(events, figures, rows=rows)
        + _plot_time_variation(summary, figures)
    )
    exposure = ledger.exposure()
    unknown = sum(row["state"] == "unknown" for row in rows)
    coverage = ledger.coverage_rows()
    coverage_counts = Counter(row["state"] for row in coverage)
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
        f"- Planned coverage cells/requests: {len(coverage):,}",
        f"- Coverage dispositions: {canonical_json(dict(sorted(coverage_counts.items())))}",
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
        "at least the full registered arrival window, extended through response drain.",
        "- TPM is calculated only from blocks whose successful-request usage is complete and "
        "ledger-verifiable; partial coverage is labelled and can be informatively missing.",
        "- Decode speed is a client-observed request proxy, not direct server compute.",
        "- Reliability, cost, latency, and usage rows are unconditional on response-derived "
        "reasoning state. Decode proxy uses only explicit reasoning_tokens=0; positive, invalid, "
        "or unreported reasoning counts are retained as counts but censored from that metric.",
        "- Quality is end-to-end across every predeclared scored trial: non-success is zero, and "
        "unrelated usage/timing invalidity cannot remove a deterministic task score.",
        "- Cached, uncached, and uncontrolled cells are never pooled.",
        "- Missing cache-read usage is kept distinct from an explicit provider-reported zero.",
        "- The warmup suite is standalone diagnostic traffic. Measured cells are labelled "
        "uncontrolled_not_paired and support no warm/cold latency claim.",
        "- A validation-class 4xx on an expected-rejection probe is only an observed status, not "
        "proof that the intended boundary was enforced. Without a bound provider error reason "
        "and matched successful control, the boundary stays inconclusive.",
        "- Context probes are fixed-anchor acceptance/retrieval screens. This harness does not "
        "claim adaptive exact-boundary localization or isolate tool-schema/image contributions.",
        "- p99 is withheld when fewer than 1,000 eligible observations exist.",
        "",
        "## Artifacts",
        "",
        "- `matched-cell-summary.csv`: estimates, units, n, CI bounds, and methods.",
        "- `load-block-summary.csv`: epoch/block RPM and effective TPM with 95% intervals.",
        "- `time-variation-summary.csv`: matched low-load panels across the day with 95% "
        "request-level intervals.",
        "- `controller-summary.csv` / `.json`: AIMD bound/censor semantics, confirmations, "
        "recovery, and soak completion state for every configured endpoint × shape controller.",
        "- `coverage-ledger.csv`: every registered completed, untested, conditional, or "
        "cap/time-censored plan cell.",
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
    run_manifest_json = ledger.meta("run_manifest_json")
    ledger.checkpoint_for_export()
    if _report_source_snapshot(run_dir) != source_snapshot:
        shutil.rmtree(report_dir)
        raise ValueError("report generator source or locked environment changed during derivation")
    _write_reproducibility_manifest(
        run_dir=run_dir,
        report_dir=report_dir,
        campaign_hash=campaign_hash,
        campaign_started_at_utc=campaign_started_at_utc,
        config_json=config_json,
        run_manifest_json=run_manifest_json,
        events=events,
        report_source=source_snapshot,
    )
    if _report_source_snapshot(run_dir) != source_snapshot:
        shutil.rmtree(report_dir)
        raise ValueError("report generator source or locked environment changed during export")
    return report_path


def _assert_terminal_snapshot(ledger: Ledger, events: list[dict[str, Any]]) -> None:
    if ledger.meta("producer_schema_version") != LEDGER_PRODUCER_SCHEMA_VERSION:
        raise ValueError("report requires the current immutable ledger producer schema")
    in_flight = [row for row in ledger.rows() if row["state"] == "in_flight"]
    if in_flight:
        raise ValueError("report requires a terminal ledger snapshot with zero in-flight attempts")
    terminal = [row for row in events if row["kind"] == "campaign_terminal"]
    if len(terminal) != 1:
        raise ValueError("report requires exactly one canonical campaign_terminal event")
    keyed = [row for row in terminal if row.get("event_key") == "campaign_terminal"]
    if len(keyed) != 1:
        raise ValueError("campaign_terminal must use the canonical idempotency key")
    runtime = _json(ledger.meta("run_manifest_json"), None)
    required_runtime_fields = {
        "schema_version",
        "normalized_exact_invocation",
        "raw_invocation_sha256",
        "client_location",
        "connection_reuse_by_route",
        "http2_by_route",
        "transport_max_connections_by_route",
        "transport_header_profile_by_route",
        "request_timeout_seconds_by_route",
        "provider_documentation_declarations",
        "transport_trust_env",
        "source_commit",
        "source_dirty",
        "source_dirty_tree_sha256",
        "dependency_lock_sha256",
        "dependency_lock_file",
        "execution_environment",
    }
    if not isinstance(runtime, dict) or runtime.get("schema_version") != "run-manifest/v2":
        raise ValueError("report requires a canonical run-manifest/v2 snapshot")
    runtime_json = ledger.meta("run_manifest_json")
    assert runtime_json is not None
    runtime_digest = hashlib.sha256(runtime_json.encode("utf-8")).hexdigest()
    required_identity_stages = ["terminal"]
    if ledger.rows():
        required_identity_stages.append("pre_send")
    for stage in required_identity_stages:
        if ledger.meta(f"{stage}_run_manifest_sha256") != runtime_digest:
            raise ValueError(f"report requires matching {stage} runtime identity verification")
    if any(event["kind"] == "source_identity_drift" for event in events):
        raise ValueError("report refuses a campaign with source identity drift")
    terminal_timestamp = keyed[0].get("recorded_at_utc")
    try:
        parsed_terminal_timestamp = datetime.fromisoformat(
            str(terminal_timestamp).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("campaign_terminal recorded_at_utc is invalid") from exc
    if (
        parsed_terminal_timestamp.tzinfo is None
        or parsed_terminal_timestamp.utcoffset() != UTC.utcoffset(None)
    ):
        raise ValueError("campaign_terminal recorded_at_utc must be UTC")
    missing_runtime = sorted(required_runtime_fields - runtime.keys())
    if missing_runtime:
        raise ValueError("run manifest is missing fields: " + ", ".join(missing_runtime))
    if (
        not isinstance(runtime.get("source_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", runtime["source_commit"])
        or not isinstance(runtime.get("source_dirty"), bool)
        or runtime.get("source_dirty") is not False
        or not isinstance(runtime.get("source_dirty_tree_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", runtime["source_dirty_tree_sha256"])
        or not isinstance(runtime.get("dependency_lock_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", runtime["dependency_lock_sha256"])
        or runtime.get("dependency_lock_file") != "requirements.lock"
        or runtime.get("transport_trust_env") is not False
        or not isinstance(runtime.get("execution_environment"), dict)
        or not isinstance(runtime["execution_environment"].get("python"), str)
        or not isinstance(runtime["execution_environment"].get("distributions"), dict)
    ):
        raise ValueError("run manifest contains unresolved or invalid reproducibility identity")
    request_timeouts = runtime.get("request_timeout_seconds_by_route")
    transport_profiles = runtime.get("transport_header_profile_by_route")
    documentation_declarations = runtime.get("provider_documentation_declarations")
    if (
        not isinstance(request_timeouts, dict)
        or not request_timeouts
        or any(
            not isinstance(route_id, str)
            or not route_id
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for route_id, value in request_timeouts.items()
        )
        or not isinstance(transport_profiles, dict)
        or set(transport_profiles) != set(request_timeouts)
        or any(value != TRANSPORT_HEADER_PROFILE for value in transport_profiles.values())
        or not isinstance(documentation_declarations, list)
        or not documentation_declarations
    ):
        raise ValueError("run manifest has invalid route transport/documentation declarations")
    declared_route_ids: list[str] = []
    for declaration in documentation_declarations:
        if not isinstance(declaration, dict):
            raise ValueError("run manifest documentation declaration must be an object")
        route_id = declaration.get("route_id")
        declared_route_ids.append(str(route_id))
        try:
            retrieved_at = datetime.fromisoformat(
                str(declaration.get("evidence_retrieved_at_utc")).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("run manifest documentation retrieval time is invalid") from exc
        if (
            not isinstance(route_id, str)
            or not route_id
            or not isinstance(declaration.get("documentation_source_url"), str)
            or not str(declaration["documentation_source_url"]).startswith("https://")
            or not isinstance(declaration.get("pricing_source_url"), str)
            or not str(declaration["pricing_source_url"]).startswith("https://")
            or retrieved_at.tzinfo is None
            or retrieved_at.utcoffset() != UTC.utcoffset(None)
            or not isinstance(declaration.get("declared_evidence_bundle_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", declaration["declared_evidence_bundle_sha256"])
            or declaration.get("verification_status") != "declared_unverified_by_harness"
        ):
            raise ValueError("run manifest documentation declaration is invalid")
    if len(set(declared_route_ids)) != len(declared_route_ids) or set(declared_route_ids) != set(
        request_timeouts
    ):
        raise ValueError("run manifest route declaration identities disagree")
    config_json = ledger.meta("config_json")
    public_config_path = ledger.directory / "campaign.public.json"
    if config_json is None or not public_config_path.is_file():
        raise ValueError("report requires the authoritative sanitized campaign.public.json")
    try:
        public_config = strict_json_loads(public_config_path.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as exc:
        raise ValueError("campaign.public.json is missing or invalid JSON") from exc
    try:
        public_config_canonical = canonical_json(public_config)
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign.public.json is not canonicalizable finite JSON") from exc
    if public_config_canonical != config_json:
        raise ValueError(
            "campaign.public.json is not the exact canonical projection stored in SQLite"
        )
    configured_routes = public_config.get("routes")
    if isinstance(configured_routes, list):
        expected_timeouts: dict[str, Any] = {}
        expected_profiles: dict[str, str] = {}
        expected_declarations: list[dict[str, Any]] = []
        for configured_route in configured_routes:
            if not isinstance(configured_route, dict) or not isinstance(
                configured_route.get("id"), str
            ):
                raise ValueError("stored public route declaration is invalid")
            route_id = configured_route["id"]
            expected_timeouts[route_id] = configured_route.get("request_timeout_seconds")
            expected_profiles[route_id] = configured_route.get("transport_header_profile")
            expected_declarations.append(
                {
                    "route_id": route_id,
                    "documentation_source_url": configured_route.get("documentation_source_url"),
                    "pricing_source_url": configured_route.get("pricing_source_url"),
                    "evidence_retrieved_at_utc": configured_route.get("evidence_retrieved_at_utc"),
                    "declared_evidence_bundle_sha256": configured_route.get(
                        "evidence_bundle_sha256"
                    ),
                    "verification_status": "declared_unverified_by_harness",
                }
            )
        if (
            request_timeouts != expected_timeouts
            or transport_profiles != expected_profiles
            or documentation_declarations != expected_declarations
        ):
            raise ValueError(
                "run manifest route transport/documentation declarations disagree with config"
            )
    if not ledger.events_path.is_file():
        raise ValueError("report requires the durable events.jsonl projection")
    projected: list[dict[str, Any]] = []
    for line in ledger.events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                value = strict_json_loads(line)
            except StrictJSONError as exc:
                raise ValueError("events.jsonl contains invalid or duplicate-key JSON") from exc
            if not isinstance(value, dict):
                raise ValueError("events.jsonl contains a non-object row")
            projected.append(value)
    if projected != events:
        raise ValueError(
            "events.jsonl is not an exact complete ordered projection of SQLite events"
        )
