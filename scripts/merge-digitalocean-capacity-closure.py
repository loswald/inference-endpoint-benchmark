from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SHAPE_MAP = {
    "short_short": "short_short",
    "long_short": "input100k_short",
    "short_long": "short_long",
    "mixed": "mixed",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def number(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def interval(row: dict[str, str], field: str) -> str:
    estimate = number(row.get(field))
    low = number(row.get(f"{field}_ci95_low"))
    high = number(row.get(f"{field}_ci95_high"))
    n = int(float(row.get(f"{field}_n") or 0))
    return json.dumps(
        {
            "bootstrap_replicates": 0,
            "ci95_high": high,
            "ci95_low": low,
            "estimate": estimate,
            "method": row.get(f"{field}_ci_method") or None,
            "n_units": n,
            "qualified": low is not None and high is not None and n >= 2,
            "sampling_unit": row.get("sampling_unit") or "load epoch/block",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def claim(bound_state: str) -> str:
    if bound_state.startswith("right_censored_"):
        return "confirmed_right_censored_lower_bound"
    if bound_state.startswith("bracketed_"):
        return "confirmed_bracketed_interval"
    if bound_state.startswith("left_censored_"):
        return "censored_no_valid_healthy_epoch"
    if bound_state.startswith("nonmonotonic_"):
        return "censored_nonmonotonic_overload"
    return "measured_capacity_state_without_numeric_bound"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--load-blocks", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--controller-source-id",
        required=True,
        help="Exact campaign that produced --controllers and --load-blocks.",
    )
    parser.add_argument("--fallback-source-id", required=True)
    parser.add_argument("--campaign-sha256s", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--long-input-tokens", type=int, default=100_000)
    parser.add_argument("--minimax-long-input-tokens", type=int, default=50_000)
    args = parser.parse_args()

    fields, base_rows = read_csv(args.base)
    _, controllers = read_csv(args.controllers)
    _, load_rows = read_csv(args.load_blocks)
    load_by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in load_rows:
        load_by_cell[(row["route_id"], row["shape"])].append(row)
    fallback = {
        (row["endpoint_id"], row["shape"]): row
        for row in base_rows
        if row.get("source_id") == args.fallback_source_id
    }

    def fallback_row(endpoint: str, runner_shape: str) -> dict[str, str]:
        """Return prior evidence without silently changing its registered recipe.

        The current closure calls its long-input recipe ``long_short`` and registers a 100K
        prompt (50K for Minimax).  Older evidence used a 32K prompt.  A fallback row may be
        relabelled as 100K only when its own provenance is the 2026-08-28 100K closure; a row
        sourced from the historical 32K campaign keeps its 32K identity.
        """

        public_shape = SHAPE_MAP[runner_shape]
        exact = fallback.get((endpoint, public_shape))
        if exact is not None:
            return dict(exact)
        if runner_shape != "long_short":
            return dict(fallback[(endpoint, public_shape)])
        historical = dict(fallback[(endpoint, "input32k_short")])
        if historical.get("provenance_source_id") == "do-capacity-20260828-r2":
            historical["shape"] = "input100k_short"
            historical["workload_input_target_tokens"] = str(
                args.minimax_long_input_tokens
                if endpoint == "minimax-m2.5"
                else args.long_input_tokens
            )
            historical["workload_recipe_identity"] = (
                "input50k_short" if endpoint == "minimax-m2.5" else "input100k_short"
            )
        return historical

    combined: list[dict[str, str]] = []
    provenance_counts: dict[str, int] = defaultdict(int)
    for controller in controllers:
        endpoint = controller["route_id"]
        runner_shape = controller["shape"]
        public_shape = SHAPE_MAP[runner_shape]
        state = controller["controller_completion_state"]
        bound_state = controller["capacity_bound_state"]
        if not bound_state or state == "campaign_censored_before_start":
            row = fallback_row(endpoint, runner_shape)
            row["source_id"] = args.source_id
            row["provenance_source_id"] = args.fallback_source_id
            row["provenance_controller_state"] = state
            row["provenance_capacity_bound_state"] = bound_state
            row["provenance_campaign_sha256s"] = args.campaign_sha256s
            combined.append(row)
            provenance_counts[args.fallback_source_id] += 1
            continue

        row = {field: "" for field in fields}
        lower = number(controller.get("healthy_lower_bound_rps"))
        upper = number(controller.get("unhealthy_upper_bound_rps"))
        highest = number(controller.get("highest_observed_healthy_rps"))
        blocks = load_by_cell[(endpoint, runner_shape)]
        tested = [number(item.get("offered_rps_target")) for item in blocks]
        tested = [value for value in tested if value is not None]
        candidate: dict[str, str] | None = None
        if lower is not None:
            matches = [
                item
                for item in blocks
                if number(item.get("offered_rps_target")) == lower
                and item.get("phase") == "confirmation"
            ]
            if matches:
                candidate = max(matches, key=lambda item: int(float(item.get("blocks_n") or 0)))
        row.update(
            {
                "source_id": args.source_id,
                "endpoint_id": endpoint,
                "shape": public_shape,
                "capacity_claim": claim(bound_state),
                "capacity_lower_bound_rps": "" if lower is None else str(lower),
                "capacity_lower_bound_rpm": "" if lower is None else str(lower * 60),
                "capacity_upper_bound_rps": "" if upper is None else str(upper),
                "capacity_upper_bound_rpm": "" if upper is None else str(upper * 60),
                "confirmed_healthy_offered_rps": "" if lower is None else str(lower),
                "confirmed_healthy_offered_rpm": "" if lower is None else str(lower * 60),
                "confirmed_healthy_offered_upper_rps": "" if upper is None else str(upper),
                "confirmed_healthy_offered_upper_rpm": "" if upper is None else str(upper * 60),
                "highest_observed_healthy_rps": "" if highest is None else str(highest),
                "highest_observed_healthy_rpm": "" if highest is None else str(highest * 60),
                "right_censored": str(bound_state.startswith("right_censored_")),
                "tested_min_offered_rps": "" if not tested else str(min(tested)),
                "tested_max_offered_rps": "" if not tested else str(max(tested)),
                "epoch_count": str(sum(int(float(item.get("blocks_n") or 0)) for item in blocks)),
                "valid_epoch_count": str(
                    sum(int(float(item.get("capacity_estimand_blocks_n") or 0)) for item in blocks)
                ),
                "healthy_epoch_count": str(
                    sum(int(float(item.get("healthy_blocks_n") or 0)) for item in blocks)
                ),
                "capacity_metric_kind": (
                    "registered open-loop AIMD bound with three separated confirmations when "
                    "a numeric healthy lower bound is present"
                ),
                "provenance_source_id": args.controller_source_id,
                "provenance_controller_state": state,
                "provenance_capacity_bound_state": bound_state,
                "provenance_campaign_sha256s": args.campaign_sha256s,
                "workload_input_target_tokens": (
                    str(
                        args.minimax_long_input_tokens
                        if endpoint == "minimax-m2.5"
                        else args.long_input_tokens
                    )
                    if runner_shape == "long_short"
                    else ""
                ),
                "workload_recipe_identity": (
                    "input100k_short"
                    if runner_shape == "long_short" and endpoint != "minimax-m2.5"
                    else "input50k_short"
                    if runner_shape == "long_short"
                    else public_shape
                ),
            }
        )
        if candidate is not None:
            row.update(
                {
                    "candidate_rate_confirmation_epoch_count": candidate.get("blocks_n", ""),
                    "candidate_successful_request_count": candidate.get(
                        "observed_requests_successful_n", ""
                    ),
                    "candidate_completed_request_count": candidate.get(
                        "observed_requests_completed_n", ""
                    ),
                    "candidate_scheduled_request_count": candidate.get(
                        "logical_requests_launched_n", ""
                    ),
                    "candidate_request_row_count": candidate.get("physical_attempts_n", ""),
                    "candidate_rate_limit_count": candidate.get("physical_rate_limited_n", ""),
                    "candidate_server_error_count": candidate.get("physical_server_errors_n", ""),
                    "candidate_timeout_count": candidate.get("physical_timeouts_n", ""),
                    "achieved_rpm": candidate.get("offered_rpm", ""),
                    "achieved_rpm_ci95": interval(candidate, "offered_rpm"),
                    "completed_rpm": candidate.get("completed_rpm", ""),
                    "completed_rpm_ci95": interval(candidate, "completed_rpm"),
                    "effective_input_tpm": candidate.get("successful_input_tpm", ""),
                    "effective_input_tpm_ci95": interval(candidate, "successful_input_tpm"),
                    "effective_output_tpm": candidate.get("successful_output_tpm", ""),
                    "effective_output_tpm_ci95": interval(candidate, "successful_output_tpm"),
                    "latency_p95_seconds": candidate.get("service_latency_p95_across_blocks", ""),
                    "latency_p95_seconds_ci95": interval(
                        candidate, "service_latency_p95_across_blocks"
                    ),
                }
            )
        combined.append(row)
        provenance_counts[args.controller_source_id] += 1

    retained = [row for row in base_rows if row.get("source_id") != args.source_id]
    fields = sorted(
        {
            *fields,
            *(key for row in combined for key in row),
            "provenance_source_id",
            "provenance_controller_state",
            "provenance_capacity_bound_state",
            "provenance_campaign_sha256s",
            "workload_input_target_tokens",
            "workload_recipe_identity",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + combined)
    manifest = {
        "combined_source_id": args.source_id,
        "controller_source_id": args.controller_source_id,
        "fallback_source_id": args.fallback_source_id,
        "campaign_sha256s": args.campaign_sha256s,
        "cells": len(combined),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "controller_summary_sha256": hashlib.sha256(args.controllers.read_bytes()).hexdigest(),
        "load_block_summary_sha256": hashlib.sha256(args.load_blocks.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
