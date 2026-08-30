from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .config import load_config

CAPACITY_SHAPES = ("short_short", "long_short", "short_long", "mixed")
ROUTE_PROFILE_SCHEMA = "inference-bench-route-profile-overrides/v1"
ROUTE_PROFILE_FIELDS = frozenset(
    {
        "output_limit_scope",
        "output_limit_tolerance_tokens",
        "reasoning_reservation_tokens",
    }
)


def _positive_number(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _csv_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_route_profile_overrides(
    path: Path | None, route_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != ROUTE_PROFILE_SCHEMA:
        raise ValueError(f"route profile overrides require schema={ROUTE_PROFILE_SCHEMA}")
    unknown_top_level = set(raw) - {"schema", "routes"}
    if unknown_top_level:
        raise ValueError(
            "route profile overrides contain unknown top-level fields: "
            + ", ".join(sorted(unknown_top_level))
        )
    routes = raw.get("routes")
    if not isinstance(routes, dict) or not routes:
        raise ValueError("route profile overrides require a nonempty routes mapping")
    unknown_routes = set(routes) - route_ids
    if unknown_routes:
        raise ValueError(
            "route profile overrides name unknown routes: "
            + ", ".join(sorted(unknown_routes))
        )
    overrides: dict[str, dict[str, Any]] = {}
    for route_id, values in routes.items():
        if not isinstance(values, dict) or not values:
            raise ValueError(f"route profile override for {route_id} must be a nonempty mapping")
        unknown_fields = set(values) - ROUTE_PROFILE_FIELDS
        if unknown_fields:
            raise ValueError(
                f"route profile override for {route_id} contains unsupported fields: "
                + ", ".join(sorted(unknown_fields))
            )
        overrides[str(route_id)] = dict(values)
    return overrides


def derive_soak_config(
    source_config: Path,
    controller_summary: Path,
    output: Path,
    *,
    fallback_rps: float | None = None,
    route_profile_overrides: Path | None = None,
    censor_incomplete: bool = False,
) -> Path:
    """Create a soak-only campaign from observed AIMD endpoint/workload bounds.

    A contract-complete confirmed healthy lower bound is copied exactly: this command does not
    invent production headroom or silently turn a right-censored maximum into a recommendation.
    A caller may explicitly censor incomplete cells from confirmation or provide one exploratory
    fallback; either disposition is named in the adjacent provenance JSON.
    """

    if fallback_rps is not None and (not math.isfinite(fallback_rps) or fallback_rps <= 0):
        raise ValueError("fallback_rps must be a finite positive number")
    if fallback_rps is not None and censor_incomplete:
        raise ValueError("fallback_rps and censor_incomplete are mutually exclusive")
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source configuration must contain a mapping")
    validated = load_config(source_config)
    overrides = _load_route_profile_overrides(
        route_profile_overrides, {route.id for route in validated.routes}
    )
    aimd = validated.suites.get("aimd")
    if not aimd or not aimd.get("enabled", True):
        raise ValueError("source configuration does not enable AIMD")
    shapes = tuple(aimd.get("shapes", CAPACITY_SHAPES))

    with controller_summary.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {
        (str(row.get("route_id")), str(row.get("shape"))): row
        for row in rows
        if row.get("suite") == "aimd"
    }

    rates: dict[str, dict[str, float]] = {}
    provenance: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    missing: list[str] = []
    for route in validated.routes:
        for shape in shapes:
            row = indexed.get((route.id, shape))
            confirmation_complete = _csv_true(
                row.get("confirmation_complete") if row else None
            )
            confirmation_all_healthy = _csv_true(
                row.get("confirmation_all_healthy") if row else None
            )
            measured = (
                _positive_number(row.get("healthy_lower_bound_rps") if row else None)
                if confirmation_complete and confirmation_all_healthy
                else None
            )
            if measured is not None:
                rate = measured
                basis = "observed_confirmed_healthy_aimd_lower_bound"
            elif fallback_rps is not None:
                rate = fallback_rps
                basis = "explicit_exploratory_fallback_no_confirmed_healthy_aimd_bound"
            elif censor_incomplete:
                if row is None:
                    reason = "no_controller_row"
                elif not confirmation_complete:
                    reason = "confirmation_incomplete"
                elif not confirmation_all_healthy:
                    reason = "confirmation_not_all_healthy"
                else:
                    reason = "no_positive_healthy_bound"
                excluded.append(
                    {
                        "route_id": route.id,
                        "shape": shape,
                        "disposition": "censored_not_scheduled_for_fixed_rate_confirmation",
                        "reason": reason,
                        "aimd_controller_completion_state": (
                            row.get("controller_completion_state") if row else None
                        ),
                        "aimd_capacity_bound_state": (
                            row.get("capacity_bound_state") if row else None
                        ),
                        "aimd_confirmation_complete": (
                            row.get("confirmation_complete") if row else None
                        ),
                        "aimd_confirmation_all_healthy": (
                            row.get("confirmation_all_healthy") if row else None
                        ),
                    }
                )
                continue
            else:
                missing.append(f"{route.id}:{shape}")
                continue
            rates.setdefault(route.id, {})[shape] = rate
            provenance.append(
                {
                    "route_id": route.id,
                    "shape": shape,
                    "candidate_rate_rps": rate,
                    "basis": basis,
                    "aimd_controller_completion_state": (
                        row.get("controller_completion_state") if row else None
                    ),
                    "aimd_capacity_bound_state": row.get("capacity_bound_state") if row else None,
                    "aimd_confirmation_complete": (
                        row.get("confirmation_complete") if row else None
                    ),
                    "aimd_confirmation_all_healthy": (
                        row.get("confirmation_all_healthy") if row else None
                    ),
                }
            )
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "AIMD has no confirmed positive healthy bound for these cells: "
            f"{joined}. Re-run AIMD, pass --censor-incomplete to omit them from fixed-rate "
            "confirmation, or pass an explicit --fallback-rps for an exploratory soak."
        )
    if not provenance:
        raise ValueError("AIMD produced no cells eligible for fixed-rate confirmation")

    suites = raw.get("suites")
    if not isinstance(suites, dict):
        raise ValueError("source configuration must contain suites")
    for _name, values in suites.items():
        if isinstance(values, dict):
            values["enabled"] = False
    source_aimd = suites.get("aimd") if isinstance(suites.get("aimd"), dict) else {}
    suites["soak"] = {
        "enabled": True,
        "shapes": list(shapes),
        "cells": [f"{cell['route_id']}:{cell['shape']}" for cell in provenance],
        "rate_rps_by_route_shape": rates,
        "blocks": 4,
        "block_seconds": 30,
        "concurrency": int(source_aimd.get("concurrency", validated.concurrency)),
        "baseline_rps": float(source_aimd.get("baseline_rps", 0.5)),
        "baseline_samples": int(source_aimd.get("baseline_samples", 20)),
        "long_input_tokens": int(source_aimd.get("long_input_tokens", 32768)),
        "long_input_overflow": source_aimd.get("long_input_overflow", "clip"),
        "long_output_tokens": int(source_aimd.get("long_output_tokens", 4096)),
        "long_output_overflow": source_aimd.get("long_output_overflow", "clip"),
    }
    campaign = raw.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("source configuration must contain campaign")
    campaign["name"] = f"{campaign.get('name', 'campaign')}-soak"

    raw_routes = raw.get("routes")
    if not isinstance(raw_routes, list):
        raise ValueError("source configuration must contain a routes list")
    for route in raw_routes:
        if not isinstance(route, dict) or not isinstance(route.get("id"), str):
            raise ValueError("source configuration contains an invalid route mapping")
        route.update(overrides.get(route["id"], {}))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    derived = load_config(output)
    provenance_path = output.with_suffix(".rates.json")
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "inference-bench-soak-rates/v1",
                "source_config": str(source_config),
                "source_config_sha256": _sha256_file(source_config),
                "source_campaign_identity_sha256": validated.identity_hash,
                "controller_summary": str(controller_summary),
                "controller_summary_sha256": _sha256_file(controller_summary),
                "route_profile_overrides": overrides,
                "route_profile_overrides_sha256": (
                    _sha256_file(route_profile_overrides)
                    if route_profile_overrides is not None
                    else None
                ),
                "incomplete_policy": "censor" if censor_incomplete else "error",
                "derived_campaign_identity_sha256": derived.identity_hash,
                "cells": provenance,
                "excluded_cells": excluded,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
