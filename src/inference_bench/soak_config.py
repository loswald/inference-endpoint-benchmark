from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .config import load_config

CAPACITY_SHAPES = ("short_short", "long_short", "short_long", "mixed")


def _positive_number(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def derive_soak_config(
    source_config: Path,
    controller_summary: Path,
    output: Path,
    *,
    fallback_rps: float | None = None,
) -> Path:
    """Create a soak-only campaign from observed AIMD endpoint/workload bounds.

    The measured healthy lower bound is copied exactly: this command does not invent production
    headroom or silently turn a right-censored maximum into a recommendation.  A caller may provide
    one explicit exploratory fallback for cells where AIMD found no healthy candidate; those cells
    are named as such in the adjacent provenance JSON.
    """

    if fallback_rps is not None and (not math.isfinite(fallback_rps) or fallback_rps <= 0):
        raise ValueError("fallback_rps must be a finite positive number")
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source configuration must contain a mapping")
    validated = load_config(source_config)
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
    missing: list[str] = []
    for route in validated.routes:
        for shape in shapes:
            row = indexed.get((route.id, shape))
            measured = _positive_number(row.get("healthy_lower_bound_rps") if row else None)
            if measured is not None:
                rate = measured
                basis = "observed_aimd_healthy_lower_bound"
            elif fallback_rps is not None:
                rate = fallback_rps
                basis = "explicit_exploratory_fallback_no_healthy_aimd_bound"
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
            "AIMD has no positive healthy bound for these cells: "
            f"{joined}. Re-run AIMD or pass an explicit --fallback-rps for an exploratory soak."
        )

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

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_config(output)
    provenance_path = output.with_suffix(".rates.json")
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "inference-bench-soak-rates/v1",
                "source_config": str(source_config),
                "controller_summary": str(controller_summary),
                "cells": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
