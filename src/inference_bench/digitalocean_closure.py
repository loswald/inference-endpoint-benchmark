from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

from .config import load_config
from .payload import reserved_input_tokens
from .plan import build_plan
from .workloads import plan_static_suites

_CONFIRMED_CAPACITY_PREFIX = "confirmed_"
_RUNNER_SHAPE = {
    "short_short": "short_short",
    "input32k_short": "long_short",
    "input100k_short": "long_short",
    "short_long": "short_long",
    "mixed": "mixed",
}
_PUBLIC_SHAPE = {
    "short_short": "short_short",
    "long_short": "input100k_short",
    "short_long": "short_long",
    "mixed": "mixed",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    """Hash the repository's canonical UTF-8/LF representation of a text artifact."""

    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _optional_bool(value: str | None) -> bool | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _routes_with_status(
    coverage: list[dict[str, str]], dimension: str, status: str = "inconclusive"
) -> list[str]:
    return sorted(
        row["endpoint_id"]
        for row in coverage
        if row.get("coverage_dimension") == dimension and row.get("status") == status
    )


def build_digitalocean_closure_package(
    base_config_path: str | Path,
    summary_dir: str | Path,
    output_dir: str | Path,
    *,
    capacity_source: str = "do-combined-capacity-20260828",
    fixed_rate_source: str = "do-direct-soak-20260823-r1",
) -> tuple[Path, Path]:
    """Compile one byte-stable, credential-free six-hour gap-closure plan.

    This function never loads provider credentials and never opens a network connection. The live
    runner remains a separate explicit command protected by ``--confirm-live``.
    """

    base_path = Path(base_config_path).resolve()
    summary = Path(summary_dir).resolve()
    destination = Path(output_dir).resolve()
    source_document = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(source_document, dict):
        raise ValueError("base configuration must be a mapping")

    capacity_path = summary / "capacity-summary.csv"
    fixed_rate_path = summary / "soak-cell-summary.csv"
    cache_path = summary / "cache-verification-pairs.csv"
    coverage_path = summary / "coverage-matrix.csv"
    limits_path = summary / "observed-limits.csv"
    for path in (capacity_path, fixed_rate_path, cache_path, coverage_path, limits_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    route_ids = [str(route["id"]) for route in source_document["routes"]]
    route_id_set = set(route_ids)
    capacity_rows = [
        row
        for row in _read_csv(capacity_path)
        if row.get("source_id") == capacity_source
    ]
    if len(capacity_rows) != len(route_ids) * 4:
        raise ValueError("capacity source must contain exactly one row per endpoint/workload cell")
    missing_capacity = [
        row
        for row in capacity_rows
        if not str(row.get("capacity_claim") or "").startswith(_CONFIRMED_CAPACITY_PREFIX)
    ]
    capacity_cells = sorted(
        f"{row['endpoint_id']}:{_RUNNER_SHAPE[row['shape']]}" for row in missing_capacity
    )

    fixed_rate_rows = [
        row
        for row in _read_csv(fixed_rate_path)
        if row.get("source_id") == fixed_rate_source
    ]
    if len(fixed_rate_rows) != len(route_ids) * 4:
        raise ValueError("fixed-rate source must contain exactly one row per endpoint/workload")
    fixed_rate_gaps = [
        row
        for row in fixed_rate_rows
        if _optional_bool(row.get("soak_acceptance_pass")) is not True
    ]
    cache_verified = {
        row["endpoint_id"] for row in _read_csv(cache_path) if row.get("endpoint_id")
    }
    cache_routes = sorted(route_id_set - cache_verified)
    coverage = _read_csv(coverage_path)
    context_routes = _routes_with_status(coverage, "input_context")
    output_routes = _routes_with_status(coverage, "output_length")
    tools_routes = _routes_with_status(coverage, "tool_calling")
    structured_routes = _routes_with_status(coverage, "structured_output")
    vision_routes = set(_routes_with_status(coverage, "vision"))

    limits = _read_csv(limits_path)
    declared_vision = {
        str(route["id"]): bool((route.get("capabilities") or {}).get("vision", False))
        for route in source_document["routes"]
    }
    for row in limits:
        if row.get("dimension") != "vision" or row.get("endpoint_id") not in route_id_set:
            continue
        finding = str(row.get("finding") or "")
        endpoint = row["endpoint_id"]
        if finding == "inconclusive" or (
            finding == "observed_functional" and not declared_vision[endpoint]
        ):
            vision_routes.add(endpoint)

    probe_groups: dict[str, set[str]] = defaultdict(set)
    for endpoint in tools_routes:
        probe_groups[endpoint].add("tool_calling")
    for endpoint in structured_routes:
        probe_groups[endpoint].add("structured_output")
    for endpoint in vision_routes:
        probe_groups[endpoint].add("vision")
    for endpoint in probe_groups:
        probe_groups[endpoint].add("transport_baseline")

    document = dict(source_document)
    campaign = dict(document["campaign"])
    campaign.update(
        {
            "name": "digitalocean-hosted-six-hour-gap-closure",
            "seed": 20260828,
            # Seven hourly panel starts span exactly six hours. The additional 15 minutes
            # are a bounded drain/finalization tail for the last panel; they are not an
            # eighth observation window.
            "max_wall_seconds": 22_500,
            "max_cost_usd": 5_000,
            "launch_reserve_seconds": 60,
            "launch_reserve_usd": 1,
            "concurrency": 128,
            "retries": 0,
        }
    )
    document["campaign"] = campaign
    document["suites"] = {
        "static": {"enabled": True, "offered_rps": 1.0},
        "time_variation": {
            "enabled": True,
            "interleave_gap_work": True,
            "panels": 7,
            "interval_minutes": 60,
            "samples_per_route_shape": 4,
            "stable_exact_prompt_repeats": 2,
            "panel_unique_cache_cold_repeats": 2,
            "shapes": ["short_short", "long_short", "short_long", "mixed"],
            "offered_rps": 1.0,
            "concurrency": 176,
            "long_input_tokens": 100_000,
            "long_input_tokens_by_route": {"minimax-m2.5": 50_000},
            "long_input_overflow": "fail",
            "long_output_tokens": 4_096,
            "long_output_overflow": "fail",
            "panel_guard_seconds": 420,
            "panel_deadline_seconds": 600,
            "send_cutoff_seconds": 21_840,
        },
        "aimd": {
            "enabled": True,
            "shapes": ["short_short", "long_short", "short_long", "mixed"],
            "cells": capacity_cells,
            "long_input_tokens": 100_000,
            "long_input_tokens_by_route": {"minimax-m2.5": 50_000},
            "long_input_overflow": "fail",
            "long_output_tokens": 4_096,
            "long_output_overflow": "fail",
            "initial_rps": 0.125,
            "additive_rps": 0.25,
            "multiplicative_decrease": 0.5,
            "bracket_epochs": 7,
            "bracket_multiplier": 2,
            "max_rps": 64,
            "max_rps_by_shape": {
                "short_short": 64,
                "long_short": 0.5,
                "short_long": 1,
                "mixed": 2,
            },
            "epochs": 9,
            "epoch_seconds": 5,
            "concurrency": 128,
            "baseline_rps": 0.5,
            "baseline_samples": 20,
            "baseline_attempts": 1,
            "baseline_multiplicative_decrease": 0.5,
            "confirmation_max_stages": 1,
            "confirmation_multiplicative_decrease": 0.5,
            "confirmation_separator_samples": 5,
            "minimum_rps": 0.03125,
        },
        # The historical fixed-rate rows were generated with an older, exact workload recipe.
        # They remain evidence, but are deliberately not scheduled under the corrected 100K/newer
        # recipe because doing so would create a different estimand and silently relabel it.
        "soak": {"enabled": False},
        "cache": {
            "enabled": True,
            "route_ids": cache_routes,
            "repeats": 12,
            "prefix_tokens": 8_192,
        },
        "context": {
            "enabled": True,
            "route_ids": context_routes,
            "percentages": [10, 50, 75, 90, 95, 99],
            "fixed_tokens": [1_024, 8_192, 32_768, 65_536, 100_000, 131_072, 262_144],
        },
        "output": {
            "enabled": True,
            "route_ids": output_routes,
            "fallback_max_output_tokens": 8_192,
            "realized_generation_ceiling": 8_192,
        },
        "capability": {
            "enabled": True,
            "route_ids": sorted(probe_groups),
            "probe_groups_by_route": {
                endpoint: sorted(groups) for endpoint, groups in sorted(probe_groups.items())
            },
            "tool_counts": [1, 8, 32, 64, 128],
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "digitalocean-six-hour-gap-closure.yaml"
    config_text = yaml.safe_dump(document, sort_keys=False, width=100)
    config_path.write_text(config_text, encoding="utf-8", newline="\n")
    validated = load_config(config_path)
    plan = build_plan(validated).to_dict()
    route_by_id = {route.id: route for route in validated.routes}
    panel_specs = [
        spec
        for spec in plan_static_suites(validated.routes, validated.suites, seed=validated.seed)
        if spec.suite == "time_variation"
    ]
    panel_worst_case_usd = round(
        sum(
            route_by_id[spec.route_id].worst_case_cost(
                reserved_input_tokens(
                    route_by_id[spec.route_id],
                    spec,
                    validated.input_token_reservation_factor,
                ),
                spec.max_output_tokens,
            )
            for spec in panel_specs
        ),
        9,
    )
    panel_count = 7
    panel_samples_per_route_shape = 4
    panel_shapes = 4
    panel_requests_per_panel = (
        len(route_ids) * panel_shapes * panel_samples_per_route_shape
    )
    static_panel_requests = panel_count * panel_requests_per_panel
    panel_launch_span_seconds = (
        panel_requests_per_panel - 1
    ) / float(document["suites"]["time_variation"]["offered_rps"])
    maximum_timeout_seconds = float(plan["max_single_request_timeout_seconds"])
    panel_drain_bound_seconds = panel_launch_span_seconds + maximum_timeout_seconds
    panel_deadline_seconds = float(
        document["suites"]["time_variation"]["panel_deadline_seconds"]
    )
    selection = {
        "schema": "digitalocean-six-hour-gap-closure-plan/v1",
        "claim_boundary": (
            "six-hour variation study; no 24-hour, diurnal, or production-sustainability claim"
        ),
        "input_identity": {
            "runner_shape": "long_short",
            "public_shape": "input100k_short",
            "target_tokens": 100_000,
            "historical_input32k_short_is_distinct": True,
            "historical_fixed_rate_rows_are_not_rerun": True,
            "historical_fixed_rate_exclusion_reason": (
                "exact workload recipe identity differs from the corrected live recipes"
            ),
        },
        "sources": {
            str(path.name): _sha256(path)
            for path in (capacity_path, fixed_rate_path, cache_path, coverage_path, limits_path)
        },
        "capacity_source": capacity_source,
        "fixed_rate_source": fixed_rate_source,
        "selected": {
            "capacity_cells": [
                {
                    "endpoint_id": row["endpoint_id"],
                    "historical_shape": row["shape"],
                    "closure_shape": _PUBLIC_SHAPE[_RUNNER_SHAPE[row["shape"]]],
                    "prior_claim": row["capacity_claim"],
                }
                for row in sorted(
                    missing_capacity,
                    key=lambda item: (item["endpoint_id"], item["shape"]),
                )
            ],
            "historical_fixed_rate_evidence_not_scheduled": [
                {
                    "endpoint_id": row["endpoint_id"],
                    "historical_shape": row["shape"],
                    "prior_result": row.get("status"),
                    "prior_acceptance": _optional_bool(row.get("soak_acceptance_pass")),
                    "live_disposition": "not_scheduled_exact_recipe_identity_differs",
                }
                for row in sorted(
                    fixed_rate_gaps,
                    key=lambda item: (item["endpoint_id"], item["shape"]),
                )
            ],
            "cache_routes": cache_routes,
            "context_routes": context_routes,
            "output_routes": output_routes,
            "capability_probe_groups_by_route": {
                endpoint: sorted(groups) for endpoint, groups in sorted(probe_groups.items())
            },
        },
        "counts": {
            "missing_capacity_cells": len(capacity_cells),
            "historical_failed_fixed_rate_cells": sum(
                _optional_bool(row.get("soak_acceptance_pass")) is False
                for row in fixed_rate_gaps
            ),
            "historical_transport_gated_fixed_rate_cells": sum(
                row.get("status") == "baseline_transport_gate_failed"
                for row in fixed_rate_gaps
            ),
            "scheduled_fixed_rate_cells": 0,
            "cache_gap_endpoints": len(cache_routes),
            "time_panel_requests": static_panel_requests,
        },
        "schedule": {
            "measurement_span_seconds": 21_600,
            "max_wall_seconds": 22_500,
            "panel_offsets_seconds": [index * 3_600 for index in range(7)],
            "panel_guard_seconds": 420,
            "panel_deadline_seconds": 600,
            "hard_send_cutoff_seconds": 21_840,
            "finalization_reserve_seconds": 660,
            "gap_work_is_serial_and_never_overlaps_a_time_panel": True,
        },
        "timing_proof": {
            "guaranteed_core": "seven matched low-load panels",
            "panel_prompt_design": (
                "two stable exact-prompt repeats plus two panel-unique cache-cold repeats per cell"
            ),
            "optional_work": (
                "gap closure runs only when its conservative bound fits before the next guard"
            ),
            "panels": panel_count,
            "requests_per_panel": panel_requests_per_panel,
            "observations_per_endpoint_shape": (
                panel_count * panel_samples_per_route_shape
            ),
            "stable_exact_prompt_observations_per_endpoint_shape": panel_count * 2,
            "panel_unique_cache_cold_observations_per_endpoint_shape": panel_count * 2,
            "maximum_panel_in_flight": panel_requests_per_panel,
            "configured_panel_concurrency": int(
                document["suites"]["time_variation"]["concurrency"]
            ),
            "core_worst_case_usd": panel_worst_case_usd,
            "global_offered_rps": float(
                document["suites"]["time_variation"]["offered_rps"]
            ),
            "panel_launch_span_seconds": panel_launch_span_seconds,
            "maximum_request_timeout_seconds": maximum_timeout_seconds,
            "panel_launch_plus_timeout_bound_seconds": panel_drain_bound_seconds,
            "panel_deadline_seconds": panel_deadline_seconds,
            "panel_deadline_slack_seconds": panel_deadline_seconds
            - panel_drain_bound_seconds,
            "load_arrival_window_seconds_sequential_upper_path": plan[
                "load_arrival_window_seconds_sequential_upper_path"
            ],
            "success_reason_after_all_panels": "six_hour_window_completed",
        },
        "plan": plan,
    }
    manifest_path = destination / "plan.json"
    manifest_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return config_path, manifest_path
