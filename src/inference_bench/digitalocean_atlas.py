from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

SHAPES = ("short_short", "input32k_short", "input100k_short", "short_long", "mixed")
SHAPE_LABELS = {
    "short_short": "short prompt / short answer",
    "input32k_short": "32K-token prompt / short answer",
    "input100k_short": "100K-token prompt / short answer",
    "short_long": "short prompt / long answer",
    "mixed": "seeded multi-workload mix",
}

# The two capacity sources used different registered recipes.  Keeping the recipes attached to
# their exact source prevents a 32K result from being relabelled as 100K (or vice versa) merely
# because both were once stored under the internal ``long_short`` shape name.
CAPACITY_RECIPES = {
    "do-capacity-20260828-r2": {
        "short_short": "256-token prompt -> 128-token answer target",
        "input100k_short": (
            "100,000-token prompt -> 128-token answer target "
            "(50,000-token prompt for Minimax M2.5)"
        ),
        "short_long": "256-token prompt -> 4,096-token answer target",
        "mixed": (
            "seeded four-way mix: 256->128, 100K/50K->128, 256->4,096, "
            "and 1,024->512 JSON"
        ),
    },
    "do-sixhour-aimd-20260824-r1": {
        "short_short": "short exact-answer prompt -> 64-token answer ceiling",
        "input32k_short": "32,000-token prompt -> 64-token answer ceiling",
        "short_long": "short prompt -> 1,024-word target / 2,048-token ceiling",
        "mixed": (
            "seeded five-way mix: short exact answer, 4,096-token context, "
            "512-word answer, JSON, and tool call; 1,024-token ceiling"
        ),
    },
}

FIXED_RATE_RECIPES = {
    "short_short": "short exact-answer prompt -> 64-token answer ceiling",
    "input32k_short": "32,000-token prompt -> 64-token answer ceiling",
    "short_long": "short prompt -> 1,024-word target / 2,048-token ceiling",
    "mixed": (
        "seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, "
        "JSON, and tool call; 1,024-token ceiling"
    ),
}
CAPABILITY_DIMENSIONS = (
    "capability_smoke",
    "response_format",
    "tools",
    "parallel_tool_calls",
    "vision",
    "seed",
    "stop",
    "temperature",
    "top_p",
    "top_logprobs",
    "automatic_prompt_cache",
    "batch_open_models",
)

FIXED_RATE_INTERVAL_NOTE = (
    "Whiskers are exploratory 95% Student-t intervals across four contiguous 30-second "
    "analysis blocks; serial correlation is not modeled."
)

_FIXED_RATE_PRESENTATION = {
    "passed": ("passed", "#0F766E", "o"),
    "failed": ("failed", "#B91C1C", "X"),
    "could_not_start": ("could not start", "#64748B", "s"),
    "not_measured": ("not measured", "#64748B", "s"),
}


def _merge_platform_capabilities(
    rows: list[dict[str, str]],
    cache_rows: list[dict[str, str]],
    endpoints: list[str],
) -> list[dict[str, str]]:
    """Replace the obsolete cache-option probe with the actual DigitalOcean contracts.

    DigitalOcean-hosted open models cache exact prefixes automatically; there is no request option
    to enable or disable that behaviour. Batch inference is a different service and explicitly
    excludes open-source/DigitalOcean-hosted models. These platform contracts are kept distinct
    from endpoint transport probes so an invalid parameter test cannot erase a supported feature.
    """

    merged = [row for row in rows if row.get("capability_dimension") != "caching_option"]
    by_endpoint: dict[str, list[dict[str, str]]] = {endpoint: [] for endpoint in endpoints}
    for row in cache_rows:
        endpoint = str(row.get("endpoint_id") or "")
        if endpoint in by_endpoint:
            by_endpoint[endpoint].append(row)
    for endpoint in endpoints:
        observations = by_endpoint[endpoint]
        hit_count = sum(
            int(_number(row.get("request_count")) or 0)
            for row in observations
            if row.get("cache_state") == "cache_hit_observed"
        )
        observed_count = sum(int(_number(row.get("request_count")) or 0) for row in observations)
        merged.append(
            {
                "endpoint_id": endpoint,
                "capability_dimension": "automatic_prompt_cache",
                "transport_status": "documented_supported",
                "functional_status": "passed" if hit_count else "degraded",
                "functional_pass_count": str(hit_count),
                "functional_scored_count": str(observed_count),
                "sampling_unit": "request_id",
                "measurement_note": (
                    "automatic exact-prefix cache hit observed"
                    if hit_count
                    else "documented automatic best-effort cache; no hit observed in retained rows"
                ),
            }
        )
        merged.append(
            {
                "endpoint_id": endpoint,
                "capability_dimension": "batch_open_models",
                "transport_status": "documented_unavailable",
                "functional_status": "documented_unavailable",
                "sampling_unit": "product_contract",
                "measurement_note": (
                    "DigitalOcean batch inference excludes open-source and "
                    "DigitalOcean-hosted models"
                ),
            }
        )
    return merged


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_interval(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(parsed, dict):
        low, high = _number(parsed.get("ci95_low")), _number(parsed.get("ci95_high"))
    elif isinstance(parsed, list) and len(parsed) == 2:
        low, high = _number(parsed[0]), _number(parsed[1])
    else:
        return None
    return (low, high) if low is not None and high is not None and low <= high else None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _capacity_workload(row: dict[str, Any]) -> str:
    """Return the evidence-bearing workload identity without merging 32K and 100K rows."""

    shape = str(row.get("shape") or "")
    if shape in {"long_short", "input100k_short"}:
        return "input100k_short"
    if shape == "input32k_short" and row.get("provenance_source_id") == "do-capacity-20260828-r2":
        # Compatibility for summaries created before the merge-layer identity fix.  New summaries
        # emit input100k_short directly; old summaries can still be rendered without lying.
        return "input100k_short"
    return shape


def _capacity_result(row: dict[str, Any]) -> str:
    claim = str(row.get("capacity_claim") or "")
    lower = _number(row.get("capacity_lower_bound_rps"))
    if claim.startswith("confirmed_") and lower is not None:
        return "confirmed"
    if claim == "unconfirmed_healthy_observation_only":
        return "exploratory"
    if claim == "censored_no_valid_healthy_epoch":
        return "no_healthy_epoch"
    if claim == "measured_capacity_state_without_numeric_bound":
        return "no_numeric_bound"
    return "not_measured"


def _capacity_result_text(row: dict[str, Any]) -> str:
    state = _capacity_result(row)
    lower = _number(row.get("capacity_lower_bound_rps"))
    upper = _number(row.get("capacity_upper_bound_rps"))
    observed = _number(row.get("highest_observed_healthy_rps"))
    if state == "confirmed" and lower is not None and upper is not None and upper > lower:
        return f"repeatedly passed at {lower:g} RPS; degraded by {upper:g} RPS"
    if state == "confirmed" and lower is not None:
        return f"repeatedly passed through at least {lower:g} RPS; ceiling not found"
    if state == "exploratory":
        return (
            f"healthy at {observed:g} RPS once; not repeated"
            if observed is not None
            else "healthy behavior observed, but no repeat-confirmed numeric rate"
        )
    if state == "no_healthy_epoch":
        floor = _number(row.get("tested_min_offered_rps"))
        return (
            f"no healthy result at the lowest tested rate ({floor:g} RPS)"
            if floor is not None
            else "no valid healthy result"
        )
    if state == "no_numeric_bound":
        return "test ran, but no defensible numeric rate was established"
    return "not measured"


def _capacity_recipe(row: dict[str, Any]) -> str:
    source = str(row.get("provenance_source_id") or "")
    workload = _capacity_workload(row)
    return CAPACITY_RECIPES.get(source, {}).get(
        workload,
        "recipe unavailable in the retained public evidence",
    )


_FAILURE_REASON_LABELS = {
    "paired_low_load_quality_failure": "low-load reference answer failed its quality check",
    "paired_near_load_quality_failure": "loaded answer failed its quality check",
    "paired_quality_regression_near_vs_low": "quality fell under load",
    "ttft_p95_above_2x_paired_low_load_phase": (
        "p95 time to first token exceeded 2x the low-load reference"
    ),
    "latency_p95_above_2x_paired_low_load_phase": (
        "p95 end-to-end latency exceeded 2x the low-load reference"
    ),
    "success_rate_below_0.99": "success rate fell below 99%",
    "rate_limit_rate_above_0.01": "more than 1% of requests were rate limited",
    "arrival_queue_growth": "the request queue kept growing",
    "combined_timeout_5xx_rate_above_0.01": "timeouts plus server errors exceeded 1%",
    "recovery_deterministic_quality_pass_rate_below_1.0": (
        "post-load recovery answers did not all pass quality checks"
    ),
    "recovery_latency_p95_above_2x_low_load": (
        "post-load p95 latency remained above 2x the low-load reference"
    ),
    "recovery_ttft_p95_above_2x_low_load": (
        "post-load p95 time to first token remained above 2x the low-load reference"
    ),
    "recovery_quality_drop_from_low_load_above_0.05": (
        "post-load quality remained more than 5 percentage points below low load"
    ),
}


def _fixed_rate_failure_reasons(
    row: dict[str, Any],
    block_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> list[str]:
    result = _fixed_rate_result(row)
    if result == "passed":
        return []
    if result == "could_not_start":
        return ["a reliable low-load baseline could not be established"]
    cell_id = str(row.get("cell_id") or "")
    reason_codes: list[str] = []
    for block in block_rows:
        if str(block.get("cell_id") or "") == cell_id:
            reason_codes.extend(_json_list(block.get("acceptance_reasons")))
    for recovery in recovery_rows:
        if str(recovery.get("cell_id") or "") == cell_id:
            reason_codes.extend(_json_list(recovery.get("recovery_acceptance_reasons")))
    ordered_codes = list(dict.fromkeys(reason_codes))
    reasons = [_FAILURE_REASON_LABELS.get(code, code.replace("_", " ")) for code in ordered_codes]
    return reasons or ["the registered acceptance checks did not pass"]


def _evidence_snapshot(
    capacity: list[dict[str, Any]],
    fixed_rate: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> dict[str, int]:
    capacity_rows = [row for row in capacity if _capacity_workload(row) in SHAPES]
    fixed_rate_results = [_fixed_rate_result(row) for row in fixed_rate]
    coverage_counts = {
        state: sum(str(row.get("status") or "") == state for row in coverage)
        for state in ("completed", "inconclusive", "unsupported")
    }
    return {
        "capacity_total": len(capacity_rows),
        "capacity_confirmed": sum(_capacity_result(row) == "confirmed" for row in capacity_rows),
        "capacity_exploratory": sum(
            _capacity_result(row) == "exploratory" for row in capacity_rows
        ),
        "capacity_no_healthy_epoch": sum(
            _capacity_result(row) == "no_healthy_epoch" for row in capacity_rows
        ),
        "capacity_no_numeric_bound": sum(
            _capacity_result(row) == "no_numeric_bound" for row in capacity_rows
        ),
        "fixed_rate_total": len(fixed_rate_results),
        "fixed_rate_passed": fixed_rate_results.count("passed"),
        "fixed_rate_failed": fixed_rate_results.count("failed"),
        "fixed_rate_could_not_start": fixed_rate_results.count("could_not_start"),
        "coverage_total": len(coverage),
        "coverage_completed": coverage_counts["completed"],
        "coverage_inconclusive": coverage_counts["inconclusive"],
        "coverage_unsupported": coverage_counts["unsupported"],
    }


def _fixed_rate_result(row: dict[str, Any]) -> str:
    """Return the publication status for a 120-second fixed-rate stability test.

    Finishing the request schedule is not evidence that the test passed. A pass or failure comes
    only from the registered ``soak_acceptance_pass`` field. The transport gate is a distinct
    state because no fixed-rate test could start.
    """

    if row.get("status") == "baseline_transport_gate_failed":
        return "could_not_start"
    acceptance = _optional_bool(row.get("soak_acceptance_pass"))
    if acceptance is True:
        return "passed"
    if acceptance is False:
        return "failed"
    return "not_measured"


def _fixed_rate_presentation(row: dict[str, Any]) -> tuple[str, str, str]:
    return _FIXED_RATE_PRESENTATION[_fixed_rate_result(row)]


def _accepted_fixed_rate_test_count(rows: list[dict[str, Any]]) -> int:
    return sum(_fixed_rate_result(row) == "passed" for row in rows)


def _style_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D8DEE8", linewidth=0.7, alpha=0.85)
    axis.set_axisbelow(True)


def _plot_capacity(rows: list[dict[str, str]], source_id: str, destination: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    created: list[Path] = []
    combined = [row for row in rows if row.get("source_id") == source_id]
    sources = (
        (
            "do-sixhour-aimd-20260824-r1",
            ("short_short", "input32k_short", "short_long", "mixed"),
            "32K-era adaptive load search (2026-08-24 source)",
            "digitalocean-adaptive-load-32k-source.png",
        ),
        (
            "do-capacity-20260828-r2",
            ("short_short", "input100k_short", "short_long", "mixed"),
            "100K-era adaptive load search (2026-08-28 source)",
            "digitalocean-adaptive-load-100k-source.png",
        ),
    )
    state_value = {
        "not_measured": 0,
        "no_numeric_bound": 1,
        "no_healthy_epoch": 2,
        "exploratory": 3,
        "confirmed": 4,
    }
    colors_by_state = ["#E2E8F0", "#7C3AED", "#DC2626", "#D97706", "#0F766E"]
    labels_by_state = {
        "not_measured": "not sourced from this campaign",
        "no_numeric_bound": "ran; no numeric bound",
        "no_healthy_epoch": "no healthy result",
        "exploratory": "observed once; not confirmed",
        "confirmed": "three separated confirmations",
    }
    for provenance_source, workloads, title, filename in sources:
        selected = [
            row for row in combined if row.get("provenance_source_id") == provenance_source
        ]
        if not selected:
            continue
        endpoints = sorted({str(row.get("endpoint_id")) for row in combined})
        index = {
            (str(row.get("endpoint_id")), _capacity_workload(row)): row for row in selected
        }
        matrix = np.zeros((len(endpoints), len(workloads)))
        annotations: list[list[str]] = []
        for endpoint in endpoints:
            annotation_row: list[str] = []
            for column, workload in enumerate(workloads):
                row = index.get((endpoint, workload), {})
                state = _capacity_result(row)
                matrix[endpoints.index(endpoint), column] = state_value[state]
                lower = _number(row.get("capacity_lower_bound_rps"))
                upper = _number(row.get("capacity_upper_bound_rps"))
                observed = _number(row.get("highest_observed_healthy_rps"))
                if state == "confirmed" and lower is not None and upper is not None:
                    annotation = f"{lower:g}-{upper:g}\nRPS"
                elif state == "confirmed" and lower is not None:
                    annotation = f">={lower:g}\nRPS"
                elif state == "exploratory" and observed is not None:
                    annotation = f"once @\n{observed:g} RPS"
                elif state == "no_healthy_epoch":
                    floor = _number(row.get("tested_min_offered_rps"))
                    annotation = "none" if floor is None else f"none @\n{floor:g} RPS"
                elif state == "no_numeric_bound":
                    annotation = "no numeric\nbound"
                else:
                    annotation = "-"
                annotation_row.append(annotation)
            annotations.append(annotation_row)
        figure, axis = plt.subplots(figsize=(13.2, max(6.4, 0.42 * len(endpoints) + 1.5)))
        axis.imshow(
            matrix,
            aspect="auto",
            vmin=-0.5,
            vmax=4.5,
            cmap=ListedColormap(colors_by_state),
        )
        for row_index, annotation_row in enumerate(annotations):
            for column, annotation in enumerate(annotation_row):
                state = int(matrix[row_index, column])
                axis.text(
                    column,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    color="white" if state in {1, 2, 3, 4} else "#334155",
                    fontsize=10,
                    fontweight="bold" if state == 4 else "normal",
                )
        axis.set_yticks(range(len(endpoints)), endpoints)
        axis.invert_yaxis()
        axis.set_xticks(
            range(len(workloads)),
            [SHAPE_LABELS[workload] for workload in workloads],
            rotation=14,
            ha="right",
        )
        axis.tick_params(length=0, labelsize=10)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=15, pad=12)
        figure.legend(
            handles=[
                Patch(facecolor=colors_by_state[state_value[state]], label=label)
                for state, label in labels_by_state.items()
            ],
            frameon=False,
            ncol=5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            fontsize=9.2,
        )
        figure.tight_layout(rect=(0.04, 0.08, 1, 0.98))
        path = destination / filename
        figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(path)
    return created


def _plot_fixed_rate_tests(
    rows: list[dict[str, str]],
    source_id: str,
    destination: Path,
    *,
    block_rows: list[dict[str, str]],
    recovery_rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    selected = [row for row in rows if row.get("source_id") == source_id]
    endpoints = sorted({str(row.get("endpoint_id")) for row in selected})
    workloads = ("short_short", "input32k_short", "short_long", "mixed")
    index = {(str(row.get("endpoint_id")), str(row.get("shape"))): row for row in selected}
    state_value = {"not_measured": 0, "could_not_start": 1, "failed": 2, "passed": 3}
    state_colors = ["#E2E8F0", "#64748B", "#DC2626", "#0F766E"]
    matrix = np.zeros((len(endpoints), len(workloads)))
    annotations: list[list[str]] = []
    for endpoint in endpoints:
        annotation_row: list[str] = []
        for column, workload in enumerate(workloads):
            row = index.get((endpoint, workload), {})
            result = _fixed_rate_result(row)
            matrix[endpoints.index(endpoint), column] = state_value[result]
            rate = _number(row.get("candidate_rate_rps")) or _number(
                row.get("two_minute_soak_observed_rps")
            )
            label = {
                "passed": "PASS",
                "failed": "FAIL",
                "could_not_start": "NO START",
                "not_measured": "-",
            }[result]
            annotation_row.append(label if rate is None else f"{label}\n{rate:g} RPS")
        annotations.append(annotation_row)
    figure, axis = plt.subplots(figsize=(13.2, max(6.4, 0.42 * len(endpoints) + 1.5)))
    axis.imshow(
        matrix,
        aspect="auto",
        vmin=-0.5,
        vmax=3.5,
        cmap=ListedColormap(state_colors),
    )
    for row_index, annotation_row in enumerate(annotations):
        for column, annotation in enumerate(annotation_row):
            state = int(matrix[row_index, column])
            axis.text(
                column,
                row_index,
                annotation,
                ha="center",
                va="center",
                color="white" if state else "#334155",
                fontsize=9.5,
                fontweight="bold" if state == 3 else "normal",
            )
    axis.set_yticks(range(len(endpoints)), endpoints)
    axis.invert_yaxis()
    axis.set_xticks(
        range(len(workloads)),
        [SHAPE_LABELS[workload] for workload in workloads],
        rotation=14,
        ha="right",
    )
    axis.tick_params(length=0, labelsize=10)
    axis.set_title(
        "120-second fixed-rate stability results",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )
    figure.legend(
        handles=[
            Patch(facecolor=state_colors[3], label="passed every acceptance check"),
            Patch(facecolor=state_colors[2], label="test ran; one or more checks failed"),
            Patch(facecolor=state_colors[1], label="reliable baseline could not be established"),
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        fontsize=9.2,
    )
    figure.tight_layout(rect=(0.04, 0.08, 1, 0.98))
    status_path = destination / "digitalocean-fixed-rate-status.png"
    figure.savefig(status_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    reason_counts: dict[str, int] = {}
    for row in selected:
        if _fixed_rate_result(row) not in {"failed", "could_not_start"}:
            continue
        for reason in _fixed_rate_failure_reasons(row, block_rows, recovery_rows):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    ordered = sorted(reason_counts.items(), key=lambda item: (item[1], item[0]))
    reason_figure, reason_axis = plt.subplots(
        figsize=(13.2, max(6.4, 0.36 * len(ordered) + 1.6))
    )
    reason_axis.barh(
        [reason for reason, _ in ordered],
        [count for _, count in ordered],
        color="#B91C1C",
    )
    for index_value, (_, count) in enumerate(ordered):
        reason_axis.text(count + 0.25, index_value, str(count), va="center", fontsize=10)
    reason_axis.set_xlabel(
        "Failing endpoint-workload cells mentioning this reason", fontsize=10.5
    )
    reason_axis.set_title(
        "Why the fixed-rate tests failed", loc="left", fontweight="bold", fontsize=15, pad=12
    )
    reason_axis.tick_params(axis="both", labelsize=9.5)
    _style_axis(reason_axis)
    reason_figure.tight_layout(rect=(0.04, 0.04, 0.99, 0.97))
    reason_path = destination / "digitalocean-fixed-rate-failure-reasons.png"
    reason_figure.savefig(reason_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(reason_figure)
    return [status_path, reason_path]


def _plot_capabilities(rows: list[dict[str, str]], destination: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    dimensions = CAPABILITY_DIMENSIONS
    endpoints = sorted({str(row.get("endpoint_id")) for row in rows})
    matrix = np.zeros((len(endpoints), len(dimensions)))
    symbols = np.full((len(endpoints), len(dimensions)), "N", dtype=object)
    status_value = {
        "not_measured": (0, "N"),
        "documented_unavailable": (1, "U"),
        "failed": (2, "F"),
        "degraded": (3, "D"),
        "passed": (4, "P"),
    }
    for row in rows:
        dimension = str(row.get("capability_dimension"))
        endpoint = str(row.get("endpoint_id"))
        if dimension not in dimensions or endpoint not in endpoints:
            continue
        functional = str(row.get("functional_status"))
        transport = str(row.get("transport_status"))
        key = "documented_unavailable" if transport == "documented_unavailable" else functional
        if key not in status_value:
            key = (
                "degraded"
                if transport == "observed_transport_degraded"
                else "not_measured"
                if transport in {"", "not_measured", "not_tested"}
                else "failed"
            )
        value, symbol = status_value[key]
        row_index, column = endpoints.index(endpoint), dimensions.index(dimension)
        matrix[row_index, column] = value
        symbols[row_index, column] = symbol
    figure, axis = plt.subplots(figsize=(13.5, max(5.2, 0.4 * len(endpoints) + 2)))
    cmap = ListedColormap(["#94A3B8", "#7C3AED", "#DC2626", "#D97706", "#0F766E"])
    image = axis.imshow(matrix, aspect="auto", vmin=-0.5, vmax=4.5, cmap=cmap)
    for row_index in range(len(endpoints)):
        for column in range(len(dimensions)):
            if symbols[row_index, column]:
                axis.text(
                    column,
                    row_index,
                    symbols[row_index, column],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9.5,
                    fontweight="bold",
                )
    axis.set_xticks(range(len(dimensions)), [value.replace("_", " ") for value in dimensions])
    axis.tick_params(axis="x", labelrotation=32, labelsize=10.5)
    axis.tick_params(axis="y", labelsize=10.5)
    axis.set_yticks(range(len(endpoints)), endpoints)
    axis.set_title(
        "DigitalOcean functional capability evidence",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, ticks=range(5))
    colorbar.ax.set_yticklabels(
        ["not measured", "documented unavailable", "failed", "degraded", "passed"]
    )
    colorbar.ax.tick_params(labelsize=10)
    figure.tight_layout(pad=1.2)
    path = destination / "digitalocean-capabilities.png"
    figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _format(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:,.{digits}g}"


def _markdown_cell(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def _figure_explainer(path: Path) -> tuple[str, str, str, str]:
    name = path.name
    if "adaptive-load-32k" in name:
        return (
            "What was run",
            "Five-second open-loop epochs, then three separated confirmations where possible. "
            "Exact recipes: short exact answer -> 64-token ceiling; 32,000-token prompt -> "
            "64-token ceiling; short prompt -> 1,024-word target / 2,048-token ceiling; seeded "
            "five-way exact-answer, 4,096-context, long-answer, JSON, and tool-call mix.",
            "What it proves",
            "Green cells establish a repeatedly passing lower bound for that exact endpoint and "
            "recipe. Other colors are explicit non-results or weaker evidence.",
        )
    if "adaptive-load-100k" in name:
        return (
            "What was run",
            "Five-second open-loop epochs, then three separated confirmations where possible. "
            "Exact recipes: 256 -> 128 tokens; 100,000 -> 128 tokens (50,000 for Minimax); "
            "256 -> 4,096 tokens; seeded four-way short, 100K/50K, long-answer, and JSON mix.",
            "What it proves",
            "Only green cells establish a repeat-confirmed lower bound. These results are separate "
            "from the 32K source and cannot be substituted for it.",
        )
    if "fixed-rate-status" in name:
        return (
            "What was run",
            "One printed rate per endpoint and historical 32K-era workload for four adjacent "
            "30-second blocks, plus recovery. The four recipes match the 32K chart above.",
            "What it proves",
            "PASS means every registered check passed at that exact rate for 120 seconds. It does "
            "not prove six-hour stability or a higher rate.",
        )
    if "failure-reasons" in name:
        return (
            "What was counted",
            "Every registered reason attached to a failed or could-not-start fixed-rate cell. One "
            "cell can contribute to several bars, so the bars do not sum to 41.",
            "How to use it",
            "The bars show which engineering failure modes dominated; they are not independent "
            "events and do not sum to the number of failed cells.",
        )
    return (
        "What was tested",
        "Valid and malformed capability probes were scored independently where evidence existed.",
        "What it proves",
        "Every cell carries an explicit evidence state: passed, degraded, failed, documented "
        "unavailable, or not measured.",
    )


def _build_interim_markdown(
    inventory: list[dict[str, str]],
    capacity: list[dict[str, str]],
    fixed_rate: list[dict[str, str]],
    coverage: list[dict[str, str]],
    block_rows: list[dict[str, str]],
    recovery_rows: list[dict[str, str]],
    *,
    capacity_source: str,
    fixed_rate_source: str,
) -> str:
    """Build the human-readable report before PDF layout is applied."""

    capacity_rows = [row for row in capacity if row.get("source_id") == capacity_source]
    fixed_rate_rows = [row for row in fixed_rate if row.get("source_id") == fixed_rate_source]
    endpoint_ids = {str(row.get("endpoint_id")) for row in inventory}
    coverage_rows = [row for row in coverage if str(row.get("endpoint_id")) in endpoint_ids]
    snapshot = _evidence_snapshot(capacity_rows, fixed_rate_rows, coverage_rows)
    lines = [
        "# DigitalOcean hosted inference: interim technical evidence",
        "",
        "> **Earlier DigitalOcean report editions are withdrawn.** Do not use their PDFs, plots, "
        "or summary claims for engineering decisions. This evidence rebuild is the only current "
        "report draft.",
        "",
        "> **Not complete. Not a production qualification.** This report describes only the",
        "> experiments that produced auditable evidence. A completed request schedule is never",
        "> counted as a passing experiment.",
        "",
        "## Executive truth",
        "",
        f"- **{snapshot['capacity_confirmed']}/{snapshot['capacity_total']}** endpoint-workload "
        "cells have a repeat-confirmed adaptive-load rate bound.",
        f"- **{snapshot['fixed_rate_passed']}/{snapshot['fixed_rate_total']}** 120-second "
        "fixed-rate tests passed every registered acceptance check; "
        f"**{snapshot['fixed_rate_failed']} failed** and "
        f"**{snapshot['fixed_rate_could_not_start']} could not establish a reliable baseline**.",
        f"- **{snapshot['coverage_completed']}/{snapshot['coverage_total']}** endpoint-capability "
        "checks produced complete evidence; "
        f"**{snapshot['coverage_inconclusive']} were inconclusive** and "
        f"**{snapshot['coverage_unsupported']} were documented as unsupported**.",
        "- **Six-hour time-of-day variation has not been measured yet.** A reserved section below "
        "defines the required matched panel; this report makes no full-day or diurnal claim.",
        "",
        "## What the tests mean",
        "",
        "- **Adaptive load search** raises offered request rate while the endpoint is healthy, "
        "reduces it after degradation, and requires three separated healthy confirmations before "
        "publishing a numeric bound. A confirmed lower bound is not a theoretical maximum.",
        "- **120-second fixed-rate stability test** holds one candidate rate for four adjacent "
        "30-second analysis blocks and then checks recovery. A pass requires every registered "
        "reliability, latency, queueing, usage, quality, and recovery condition to pass.",
        "- **95% intervals** use the sampling unit printed with each result. The four stability "
        "blocks are adjacent in time, so their Student-t intervals are exploratory and do not "
        "model serial correlation.",
        "",
        "## Exact workload recipes",
        "",
        "The combined adaptive-load table contains two separately identified source recipes. "
        "They are shown separately and must not be compared as if 32K and 100K prompts were the "
        "same workload.",
        "",
        "| Source | Workload | Registered recipe |",
        "|---|---|---|",
    ]
    for source, recipes in CAPACITY_RECIPES.items():
        for workload, recipe in recipes.items():
            lines.append(
                f"| `{source}` | {_markdown_cell(SHAPE_LABELS[workload])} | "
                f"{_markdown_cell(recipe)} |"
            )
    lines.extend(
        [
            "",
            "The fixed-rate campaign used the historical 32K-era recipes:",
            "",
            "| Workload | Registered recipe |",
            "|---|---|",
        ]
    )
    for workload, recipe in FIXED_RATE_RECIPES.items():
        lines.append(f"| {_markdown_cell(SHAPE_LABELS[workload])} | {_markdown_cell(recipe)} |")

    lines.extend(
        [
            "",
            "## Adaptive-load results",
            "",
            "| Endpoint | Workload | Exact recipe | Result | Evidence source |",
            "|---|---|---|---|---|",
        ]
    )
    for row in sorted(
        capacity_rows,
        key=lambda value: (
            str(value.get("endpoint_id")),
            SHAPES.index(_capacity_workload(value)),
        ),
    ):
        lines.append(
            f"| `{_markdown_cell(row.get('endpoint_id'))}` | "
            f"{_markdown_cell(SHAPE_LABELS[_capacity_workload(row)])} | "
            f"{_markdown_cell(_capacity_recipe(row))} | "
            f"{_markdown_cell(_capacity_result_text(row))} | "
            f"`{_markdown_cell(row.get('provenance_source_id'))}` |"
        )

    lines.extend(
        [
            "",
            "## 120-second fixed-rate stability results",
            "",
            "| Endpoint | Workload | Candidate rate | Outcome | Why it failed |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in sorted(
        fixed_rate_rows,
        key=lambda value: (str(value.get("endpoint_id")), SHAPES.index(str(value.get("shape")))),
    ):
        result = _fixed_rate_result(row)
        rate = _number(row.get("candidate_rate_rps")) or _number(
            row.get("two_minute_soak_observed_rps")
        )
        reason_text = "; ".join(_fixed_rate_failure_reasons(row, block_rows, recovery_rows))
        lines.append(
            f"| `{_markdown_cell(row.get('endpoint_id'))}` | "
            f"{_markdown_cell(SHAPE_LABELS[str(row.get('shape'))])} | "
            f"{'-' if rate is None else f'{rate:g} RPS'} | "
            f"**{_markdown_cell(result.replace('_', ' '))}** | "
            f"{_markdown_cell(reason_text) if result != 'passed' else '-'} |"
        )

    lines.extend(
        [
            "",
            "## Six-hour matched variation panel — pending",
            "",
            "No six-hour panel is present in the retained evidence. The closure experiment must "
            "repeat the same endpoint, workload recipe, offered rate, region, and acceptance "
            "checks at predeclared times across a six-hour window. Results belong here only after "
            "all matched panels and their request-level receipts are verified.",
            "",
            "This six-hour panel can support a six-hour within-run variation statement. It cannot "
            "support a 24-hour, daily, or diurnal claim.",
            "",
            "## Engineering use",
            "",
            "The current evidence is useful for reproducing observed behavior and selecting cells "
            "for follow-up. It is not sufficient for a provider-wide production approval. Treat "
            "every unconfirmed, failed, could-not-start, inconclusive, or unmeasured cell as an "
            "open engineering risk—not as zero performance and not as implicit support.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_pdf(
    path: Path,
    inventory: list[dict[str, str]],
    capacity: list[dict[str, str]],
    soak: list[dict[str, str]],
    capabilities: list[dict[str, str]],
    limits: list[dict[str, str]],
    coverage: list[dict[str, str]],
    soak_blocks: list[dict[str, str]],
    recovery: list[dict[str, str]],
    figures: list[Path],
    *,
    static_verification: list[dict[str, str]],
    cache_verification: list[dict[str, str]],
    verification_manifest: dict[str, Any],
    capacity_manifest: dict[str, Any],
    capacity_source: str,
    soak_source: str,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        KeepTogether,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    navy = colors.HexColor("#0F172A")
    teal = colors.HexColor("#0F766E")
    pale = colors.HexColor("#F1F5F9")
    slate = colors.HexColor("#475569")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="DoTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=29,
            leading=34,
            textColor=navy,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=navy,
            spaceAfter=8,
            keepWithNext=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=12.7,
            textColor=navy,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.5,
            textColor=slate,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoKpiNumber",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=22,
            alignment=TA_CENTER,
            textColor=navy,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoKpiLabel",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.3,
            alignment=TA_CENTER,
            textColor=navy,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoTable",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.2,
            textColor=navy,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoTableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.2,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoEndpointTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15.5,
            leading=18,
            textColor=navy,
            spaceAfter=6,
            keepWithNext=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoEndpointCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.1,
            textColor=slate,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoEndpointHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.1,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DoPanelH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=navy,
            spaceBefore=5,
            spaceAfter=5,
            keepWithNext=0,
        )
    )

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        page_width, _ = canvas._pagesize
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(18 * mm, 12 * mm, page_width - 18 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(slate)
        canvas.drawString(18 * mm, 7.5 * mm, "DigitalOcean hosted inference - interim evidence")
        canvas.drawRightString(page_width - 18 * mm, 7.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    landscape_a4 = landscape(A4)
    document = BaseDocTemplate(
        str(path),
        pagesize=A4,
        pageTemplates=[
            PageTemplate(
                id="portrait",
                pagesize=A4,
                frames=[
                    Frame(
                        18 * mm,
                        17 * mm,
                        A4[0] - 36 * mm,
                        A4[1] - 34 * mm,
                        id="portrait-frame",
                    )
                ],
                onPage=footer,
            ),
            PageTemplate(
                id="landscape",
                pagesize=landscape_a4,
                frames=[
                    Frame(
                        18 * mm,
                        17 * mm,
                        landscape_a4[0] - 36 * mm,
                        landscape_a4[1] - 34 * mm,
                        id="landscape-frame",
                    )
                ],
                onPage=footer,
            ),
        ],
        title="DigitalOcean Hosted Inference - Interim Technical Evidence",
        author="Sqwish Labs",
    )
    capacity_index = {
        (row.get("endpoint_id"), _capacity_workload(row)): row
        for row in capacity
        if row.get("source_id") == capacity_source
    }
    soak_index = {
        (row.get("endpoint_id"), row.get("shape")): row
        for row in soak
        if row.get("source_id") == soak_source
    }
    capability_index = {
        (row.get("endpoint_id"), row.get("capability_dimension")): row for row in capabilities
    }
    selected_capacity = [row for row in capacity if row.get("source_id") == capacity_source]
    selected_fixed_rate = [row for row in soak if row.get("source_id") == soak_source]
    inventory_endpoints = {str(row.get("endpoint_id")) for row in inventory}
    selected_coverage = [
        row for row in coverage if str(row.get("endpoint_id")) in inventory_endpoints
    ]
    snapshot = _evidence_snapshot(selected_capacity, selected_fixed_rate, selected_coverage)
    provenance_counts = capacity_manifest.get("provenance_counts") or {}
    current_capacity_cells = int(provenance_counts.get("do-capacity-20260828-r2") or 0)
    fallback_capacity_cells = int(provenance_counts.get("do-sixhour-aimd-20260824-r1") or 0)
    capacity_provenance_note = (
        f"The combined capacity table uses {current_capacity_cells} exact endpoint-by-workload "
        f"controllers from the corrected 2026-08-28 closure and {fallback_capacity_cells} "
        "matching cells from the earlier verified six-hour campaign. The corrected closure's "
        "four-hour guard censored those fallback cells before start; none is presented as new "
        "evidence."
        if current_capacity_cells or fallback_capacity_cells
        else "Capacity provenance is recorded per exact endpoint-by-workload cell."
    )
    story: list[Any] = [
        Spacer(1, 14 * mm),
        Paragraph("DigitalOcean hosted inference", styles["DoTitle"]),
        Paragraph("Interim evidence rebuild - 28 August 2026", styles["DoH1"]),
        Table(
            [["EARLIER REPORT WITHDRAWN", "Do not use prior PDFs or plots for decisions"]],
            colWidths=[63 * mm, 97 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#7F1D1D")),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FFF7ED")),
                    ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                    ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#7C2D12")),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            ),
        ),
        Spacer(1, 2.5 * mm),
        Table(
            [["NOT COMPLETE", "NOT A PRODUCTION QUALIFICATION"]],
            colWidths=[43 * mm, 117 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#B91C1C")),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FEF2F2")),
                    ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                    ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#7F1D1D")),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            ),
        ),
        Spacer(1, 6 * mm),
        Table(
            [
                [
                    Paragraph(
                        f"{snapshot['capacity_confirmed']}/{snapshot['capacity_total']}",
                        styles["DoKpiNumber"],
                    ),
                    Paragraph(
                        f"{snapshot['fixed_rate_passed']}/{snapshot['fixed_rate_total']}",
                        styles["DoKpiNumber"],
                    ),
                    Paragraph(
                        f"{snapshot['coverage_completed']}/{snapshot['coverage_total']}",
                        styles["DoKpiNumber"],
                    ),
                ],
                [
                    Paragraph(
                        "adaptive-load cells with a repeat-confirmed rate",
                        styles["DoKpiLabel"],
                    ),
                    Paragraph(
                        "120-second fixed-rate tests that passed",
                        styles["DoKpiLabel"],
                    ),
                    Paragraph(
                        "endpoint-capability checks with complete evidence",
                        styles["DoKpiLabel"],
                    ),
                ],
            ],
            colWidths=[54 * mm, 54 * mm, 54 * mm],
            rowHeights=[12 * mm, 16 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), pale),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                    ("PADDING", (0, 0), (-1, -1), 3),
                ]
            ),
        ),
        Spacer(1, 6 * mm),
        Paragraph(
            f"The fixed-rate result is {snapshot['fixed_rate_passed']} passes, "
            f"{snapshot['fixed_rate_failed']} measured "
            f"failures, and {snapshot['fixed_rate_could_not_start']} tests that could not "
            "establish "
            "a reliable baseline. Finishing a request schedule does not count as a pass.",
            styles["DoBody"],
        ),
        Paragraph(
            "Six-hour time-of-day variation has not been measured yet. This edition reserves a "
            "section for that matched panel and makes no 24-hour, daily, or diurnal claim.",
            styles["DoBody"],
        ),
        Paragraph(
            f"Scope: {len(inventory)} exact DigitalOcean-hosted open-model endpoints. Commercial "
            "pass-through routes are excluded. Capacity and fixed-rate source identifiers are "
            "printed in the provenance section.",
            styles["DoSmall"],
        ),
        PageBreak(),
        Paragraph("How to read the experiments", styles["DoH1"]),
        Paragraph(
            "Adaptive load search raises offered traffic while the endpoint is healthy, reduces "
            "traffic after degradation, and requires three separated healthy confirmations "
            "before publishing a numeric rate. A confirmed lower bound is not a maximum.",
            styles["DoBody"],
        ),
        Paragraph(
            "The 120-second fixed-rate stability test holds one rate for four adjacent 30-second "
            "blocks, then checks recovery. A pass requires every registered reliability, latency, "
            "queueing, usage, quality, and recovery condition to pass. Its four-block intervals "
            "are exploratory because serial correlation is not modeled.",
            styles["DoBody"],
        ),
        Paragraph(
            "The capacity sources used different recipes. The 2026-08-24 source used a 32K long "
            "prompt; the 2026-08-28 source used a 100K long prompt (50K for Minimax). They remain "
            "separate in every table and chart.",
            styles["DoBody"],
        ),
    ]
    recipe_rows: list[list[Any]] = [
        [
            Paragraph("Source", styles["DoTableHeader"]),
            Paragraph("Workload", styles["DoTableHeader"]),
            Paragraph("Exact registered recipe", styles["DoTableHeader"]),
        ]
    ]
    for source, recipes in CAPACITY_RECIPES.items():
        for workload, recipe in recipes.items():
            recipe_rows.append(
                [
                    Paragraph(source, styles["DoTable"]),
                    Paragraph(SHAPE_LABELS[workload], styles["DoTable"]),
                    Paragraph(recipe, styles["DoTable"]),
                ]
            )
    story.extend(
        [
            Table(
                recipe_rows,
                colWidths=[43 * mm, 41 * mm, 90 * mm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), navy),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 3),
                    ]
                ),
            ),
            Spacer(1, 4 * mm),
            Paragraph(capacity_provenance_note, styles["DoSmall"]),
            PageBreak(),
            Paragraph("Endpoint evidence map", styles["DoH1"]),
            Paragraph(
                "Counts below are evidence counts, not a provider ranking. A production decision "
                "must match the application's exact workload and required capability.",
                styles["DoBody"],
            ),
        ]
    )
    decision_rows: list[list[Any]] = [
        [
            Paragraph("Exact endpoint", styles["DoTableHeader"]),
            Paragraph("Documented context", styles["DoTableHeader"]),
            Paragraph("Repeat-confirmed load rates", styles["DoTableHeader"]),
            Paragraph("Fixed-rate stability passes", styles["DoTableHeader"]),
            Paragraph("Capability checks passed", styles["DoTableHeader"]),
        ]
    ]
    for endpoint in sorted(inventory, key=lambda row: str(row.get("endpoint_id"))):
        endpoint_id = str(endpoint.get("endpoint_id"))
        confirmed = sum(
            str(
                capacity_index.get((endpoint_id, shape), {}).get("capacity_claim") or ""
            ).startswith("confirmed_")
            for shape in SHAPES
        )
        endpoint_capacity_total = sum(
            1
            for row in selected_capacity
            if str(row.get("endpoint_id")) == endpoint_id
        )
        accepted_fixed_rate_tests = _accepted_fixed_rate_test_count(
            [soak_index.get((endpoint_id, shape), {}) for shape in SHAPES]
        )
        endpoint_capabilities = [
            row for row in capabilities if row.get("endpoint_id") == endpoint_id
        ]
        passed = sum(row.get("functional_status") == "passed" for row in endpoint_capabilities)
        decision_rows.append(
            [
                Paragraph(endpoint_id, styles["DoTable"]),
                Paragraph(
                    str(endpoint.get("context_window") or "not documented"), styles["DoTable"]
                ),
                Paragraph(f"{confirmed}/{endpoint_capacity_total}", styles["DoTable"]),
                Paragraph(f"{accepted_fixed_rate_tests}/4", styles["DoTable"]),
                Paragraph(f"{passed}/{len(endpoint_capabilities)}", styles["DoTable"]),
            ]
        )
    story.append(
        Table(
            decision_rows,
            colWidths=[56 * mm, 28 * mm, 31 * mm, 28 * mm, 31 * mm],
            repeatRows=1,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 3),
                ]
            ),
        )
    )
    verification_by_endpoint: dict[str, list[dict[str, str]]] = {}
    for row in static_verification:
        verification_by_endpoint.setdefault(str(row.get("endpoint_id")), []).append(row)
    cache_verification_index = {
        str(row.get("endpoint_id")): row for row in cache_verification
    }
    manifest_campaign = str(verification_manifest.get("campaign_id") or "not provided")
    manifest_sha = str(verification_manifest.get("source_bundle_sha256") or "not provided")
    story.extend(
        [
            PageBreak(),
            Paragraph("Separate static verification", styles["DoH1"]),
            Paragraph(
                f"Campaign {manifest_campaign} is a separate, hash-bound verification layer "
                f"({manifest_sha}). It contributes static, caching, capability, context, output, "
                "quality, and latency evidence only. It contributes no adaptive-load or "
                "sustained-capacity claim. Cells not reached before its wall-time guard remain "
                "explicitly labelled.",
                styles["DoBody"],
            ),
        ]
    )
    verification_rows = [
        ["Exact endpoint", "Suites reached", "Attempts", "Successes", "Observed state"]
    ]
    for endpoint in sorted(inventory, key=lambda row: str(row.get("endpoint_id"))):
        endpoint_id = str(endpoint.get("endpoint_id"))
        rows = verification_by_endpoint.get(endpoint_id, [])
        attempts = sum(int(row.get("attempts_n") or 0) for row in rows)
        successes = sum(int(row.get("successes_n") or 0) for row in rows)
        suites = ", ".join(sorted({str(row.get("suite")) for row in rows}))
        verification_rows.append(
            [
                endpoint_id,
                suites or "not reached",
                str(attempts) if rows else "not measured",
                str(successes) if rows else "not measured",
                "measured" if rows else "wall-time censored before start",
            ]
        )
    story.append(
        Table(
            verification_rows,
            colWidths=[55 * mm, 51 * mm, 20 * mm, 20 * mm, 34 * mm],
            repeatRows=1,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 6.6),
                    ("PADDING", (0, 0), (-1, -1), 3.2),
                ]
            ),
        )
    )
    if cache_verification:
        story.extend(
            [
                Spacer(1, 5 * mm),
                Paragraph("Automatic exact-prefix cache verification", styles["DoH1"]),
                Paragraph(
                    "DigitalOcean-hosted open models apply best-effort exact-prefix caching "
                    "automatically. The matched pairs below report observed cached-token counters, "
                    "TTFT when the transport exposed it, and settled-cost ratios. A missing row "
                    "means the verification campaign did not reach that endpoint; it is not a "
                    "claim that caching is unavailable.",
                    styles["DoBody"],
                ),
            ]
        )
        cache_rows = [
            [
                "Endpoint",
                "Pairs",
                "Hits / misses",
                "Cached tokens",
                "TTFT cached / uncached",
                "Cost ratio",
            ]
        ]
        for endpoint_id in sorted(cache_verification_index):
            row = cache_verification_index[endpoint_id]
            cached_ttft = _number(row.get("cached_ttft_p50_seconds"))
            uncached_ttft = _number(row.get("uncached_ttft_p50_seconds"))
            ratio = _number(row.get("observed_cost_ratio_cached_over_uncached"))
            cache_rows.append(
                [
                    endpoint_id,
                    f"{row.get('cached_requests_n')}/{row.get('uncached_requests_n')}",
                    f"{row.get('cached_token_hits_n')}/{row.get('cached_token_misses_n')}",
                    str(row.get("cache_read_tokens_sum") or "0"),
                    (
                        "not reported"
                        if cached_ttft is None or uncached_ttft is None
                        else f"{_format(cached_ttft)} / {_format(uncached_ttft)} s"
                    ),
                    "not established" if ratio is None else f"{ratio:.2f}x",
                ]
            )
        story.append(
            Table(
                cache_rows,
                colWidths=[48 * mm, 19 * mm, 24 * mm, 24 * mm, 40 * mm, 24 * mm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), teal),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                        ("PADDING", (0, 0), (-1, -1), 3.1),
                    ]
                ),
            )
        )
    first_figure = True
    for figure in figures:
        from PIL import Image as PILImage

        with PILImage.open(figure) as source:
            width, height = source.size
        # The failure-reason chart has long labels and a correspondingly taller image. Keep
        # additional room below it so the three-row explainer remains fully inside the frame.
        max_chart_height = 88 * mm if "failure-reasons" in figure.name else 105 * mm
        scale = min(246 * mm / width, max_chart_height / height)
        chart = Image(str(figure), width=width * scale, height=height * scale)
        chart.hAlign = "CENTER"
        page_start: list[Any] = [PageBreak()]
        if first_figure:
            page_start = [NextPageTemplate("landscape"), PageBreak()]
            first_figure = False
        run_heading, run_text, proof_heading, proof_text = _figure_explainer(figure)
        story.extend(
            page_start
            + [
                chart,
                Spacer(1, 2 * mm),
                Table(
                    [
                        [
                            Paragraph(run_heading, styles["DoTable"]),
                            Paragraph(run_text, styles["DoSmall"]),
                        ],
                        [
                            Paragraph(proof_heading, styles["DoTable"]),
                            Paragraph(proof_text, styles["DoSmall"]),
                        ],
                        [
                            Paragraph("What it does not prove", styles["DoTable"]),
                            Paragraph(
                                "No chart on this page establishes a provider-wide maximum, "
                                "production approval, 24-hour behavior, or diurnal behavior.",
                                styles["DoSmall"],
                            ),
                        ],
                    ],
                    colWidths=[38 * mm, 207 * mm],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), pale),
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                            ("PADDING", (0, 0), (-1, -1), 3),
                        ]
                    ),
                ),
            ]
        )
    story.extend(
        [
            NextPageTemplate("portrait"),
            PageBreak(),
            Paragraph("Six-hour matched variation panel - pending", styles["DoH1"]),
            Table(
                [["EVIDENCE STATUS", "PENDING - no verified matched six-hour panel"]],
                colWidths=[44 * mm, 124 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#B91C1C")),
                        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FEF2F2")),
                        ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                        ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#7F1D1D")),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                        ("PADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            ),
            Spacer(1, 5 * mm),
            Paragraph(
                "No six-hour time-of-day panel exists in the retained evidence. The closure "
                "experiment must repeat the same endpoint, workload recipe, offered rate, region, "
                "and acceptance checks at predeclared times across a six-hour window.",
                styles["DoBody"],
            ),
            Paragraph("Matched timeline", styles["DoPanelH2"]),
            Table(
                [
                    [
                        Paragraph("Hour 0", styles["DoTableHeader"]),
                        Paragraph("Registered checkpoints", styles["DoTableHeader"]),
                        Paragraph("Hour 6", styles["DoTableHeader"]),
                    ],
                    [
                        Paragraph(
                            "Lock endpoint + recipe + rate + region; record the low-load "
                            "reference.",
                            styles["DoTable"],
                        ),
                        Paragraph(
                            "Repeat the identical matched panel only at times predeclared in the "
                            "closure plan; preserve request-level receipts.",
                            styles["DoTable"],
                        ),
                        Paragraph(
                            "Run the final matched panel and recovery check; close the evidence "
                            "set.",
                            styles["DoTable"],
                        ),
                    ],
                ],
                colWidths=[49 * mm, 70 * mm, 49 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), navy),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
            Paragraph("Measurements required at every panel", styles["DoPanelH2"]),
            Table(
                [
                    [
                        Paragraph("Reliability", styles["DoTable"]),
                        Paragraph(
                            "success rate; rate-limit rate; timeout plus server-error rate",
                            styles["DoTable"],
                        ),
                    ],
                    [
                        Paragraph("Responsiveness", styles["DoTable"]),
                        Paragraph(
                            "time to first token; end-to-end latency; achieved request rate; "
                            "queue growth",
                            styles["DoTable"],
                        ),
                    ],
                    [
                        Paragraph("Efficiency", styles["DoTable"]),
                        Paragraph(
                            "completion throughput; prompt and completion usage completeness; "
                            "cache counters where exposed",
                            styles["DoTable"],
                        ),
                    ],
                    [
                        Paragraph("Quality and recovery", styles["DoTable"]),
                        Paragraph(
                            "matched quality result during load; latency and quality after load",
                            styles["DoTable"],
                        ),
                    ],
                ],
                colWidths=[44 * mm, 124 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), pale),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            ),
            Paragraph("Claims boundary", styles["DoPanelH2"]),
            Paragraph(
                "After verification, this panel can support only a six-hour within-run variation "
                "statement. It cannot support a 24-hour, daily, diurnal, provider-wide maximum, "
                "or production-approval claim.",
                styles["DoBody"],
            ),
        ]
    )
    limits_by_endpoint: dict[str, list[dict[str, str]]] = {}
    for row in limits:
        limits_by_endpoint.setdefault(str(row.get("endpoint_id")), []).append(row)
    for endpoint in sorted(inventory, key=lambda row: str(row.get("endpoint_id"))):
        endpoint_id = str(endpoint.get("endpoint_id"))
        story.extend([NextPageTemplate("portrait"), PageBreak()])
        endpoint_page: list[Any] = [Paragraph(endpoint_id, styles["DoEndpointTitle"])]
        facts: list[list[Any]] = [
            [
                Paragraph("<b>Model / API</b>", styles["DoEndpointCell"]),
                Paragraph(
                    f"{endpoint_id} / {endpoint.get('api_surface') or '-'}",
                    styles["DoEndpointCell"],
                ),
            ],
            [
                Paragraph("<b>Region / API version</b>", styles["DoEndpointCell"]),
                Paragraph(
                    f"{endpoint.get('server_region') or 'not reported'} / "
                    f"{endpoint.get('api_version') or 'not reported'}",
                    styles["DoEndpointCell"],
                ),
            ],
            [
                Paragraph("<b>Context / max output</b>", styles["DoEndpointCell"]),
                Paragraph(
                    f"{endpoint.get('context_window') or 'not documented'} / "
                    f"{endpoint.get('max_output_tokens') or 'not documented'} tokens",
                    styles["DoEndpointCell"],
                ),
            ],
            [
                Paragraph("<b>Input / output price</b>", styles["DoEndpointCell"]),
                Paragraph(
                    f"${endpoint.get('input_usd_per_million')} / "
                    f"${endpoint.get('output_usd_per_million')} per million tokens",
                    styles["DoEndpointCell"],
                ),
            ],
        ]
        endpoint_page.append(
            Table(
                facts,
                colWidths=[48 * mm, 120 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), pale),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("PADDING", (0, 0), (-1, -1), 2.6),
                    ]
                ),
            )
        )
        capacity_rows: list[list[Any]] = [
            [
                Paragraph("Workload", styles["DoEndpointHeader"]),
                Paragraph("Exact recipe and source", styles["DoEndpointHeader"]),
                Paragraph("Adaptive-load result", styles["DoEndpointHeader"]),
                Paragraph("120 s fixed-rate result", styles["DoEndpointHeader"]),
            ]
        ]
        endpoint_capacity_rows = sorted(
            [
                row
                for row in selected_capacity
                if str(row.get("endpoint_id")) == endpoint_id
            ],
            key=lambda row: SHAPES.index(_capacity_workload(row)),
        )
        for aimd in endpoint_capacity_rows:
            shape = _capacity_workload(aimd)
            sustained = soak_index.get((endpoint_id, shape), {})
            fixed_state = _fixed_rate_result(sustained)
            fixed_rate = _number(sustained.get("candidate_rate_rps")) or _number(
                sustained.get("two_minute_soak_observed_rps")
            )
            if shape == "input100k_short":
                fixed_text = "not run for the 100K recipe"
            elif fixed_state == "passed":
                fixed_text = f"PASS at {fixed_rate:g} RPS" if fixed_rate is not None else "PASS"
            elif fixed_state in {"failed", "could_not_start"}:
                reasons = _fixed_rate_failure_reasons(sustained, soak_blocks, recovery)
                rate_text = "" if fixed_rate is None else f" at {fixed_rate:g} RPS"
                fixed_text = (
                    f"{fixed_state.replace('_', ' ')}{rate_text}: "
                    + "; ".join(reasons[:2])
                )
            else:
                fixed_text = "not measured"
            capacity_rows.append(
                [
                    Paragraph(SHAPE_LABELS[shape], styles["DoEndpointCell"]),
                    Paragraph(
                        _capacity_recipe(aimd)
                        + f"<br/><font color='#64748B'>{aimd.get('provenance_source_id')}</font>",
                        styles["DoEndpointCell"],
                    ),
                    Paragraph(_capacity_result_text(aimd), styles["DoEndpointCell"]),
                    Paragraph(fixed_text, styles["DoEndpointCell"]),
                ]
            )
        endpoint_page.extend(
            [
                Spacer(1, 4 * mm),
                Table(
                    capacity_rows,
                    colWidths=[34 * mm, 52 * mm, 45 * mm, 43 * mm],
                    repeatRows=1,
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), navy),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("PADDING", (0, 0), (-1, -1), 2.7),
                        ]
                    ),
                ),
                Spacer(1, 4 * mm),
            ]
        )
        capabilities_rows: list[list[Any]] = [
            [
                Paragraph("Capability", styles["DoEndpointHeader"]),
                Paragraph("Transport", styles["DoEndpointHeader"]),
                Paragraph("Functional", styles["DoEndpointHeader"]),
            ]
        ]
        for dimension in (
            "response_format",
            "tools",
            "parallel_tool_calls",
            "vision",
            "automatic_prompt_cache",
            "batch_open_models",
        ):
            row = capability_index.get((endpoint_id, dimension), {})
            capabilities_rows.append(
                [
                    Paragraph(dimension.replace("_", " "), styles["DoEndpointCell"]),
                    Paragraph(
                        str(row.get("transport_status") or "not measured").replace("_", " "),
                        styles["DoEndpointCell"],
                    ),
                    Paragraph(
                        str(row.get("functional_status") or "not measured").replace("_", " "),
                        styles["DoEndpointCell"],
                    ),
                ]
            )
        endpoint_page.append(
            Table(
                capabilities_rows,
                colWidths=[45 * mm, 62 * mm, 62 * mm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), teal),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("PADDING", (0, 0), (-1, -1), 2.6),
                    ]
                ),
            )
        )
        findings = limits_by_endpoint.get(endpoint_id, [])
        if findings:
            endpoint_page.extend(
                [
                    Spacer(1, 4 * mm),
                    Paragraph(
                        "Observed boundaries: "
                        + "; ".join(
                            f"{str(row.get('dimension') or '').replace('_', ' ')}: "
                            f"{str(row.get('finding') or '').replace('_', ' ')}"
                            for row in findings
                        ),
                        styles["DoEndpointCell"],
                    ),
                ]
            )
        # Treat an endpoint sheet as one bounded unit. Every retained endpoint sheet is well
        # below the portrait frame height, so this prevents a previous flowable from leaving a
        # partial facts table or workload header at a page boundary.
        story.append(KeepTogether(endpoint_page))

    def invariant_canvas(*args: Any, **kwargs: Any) -> Any:
        kwargs["invariant"] = 1
        return pdf_canvas.Canvas(*args, **kwargs)

    document.build(story, canvasmaker=invariant_canvas)


def generate_digitalocean_atlas(
    summary_dir: str | Path,
    output_dir: str | Path,
    *,
    capacity_source: str,
    soak_source: str,
    exclude_endpoints: tuple[str, ...] = (),
) -> Path:
    source = Path(summary_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        if output.name != "digitalocean-atlas":
            raise ValueError("refusing to replace an output directory not named digitalocean-atlas")
        shutil.rmtree(output)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True)
    excluded = set(exclude_endpoints)

    def included(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return [row for row in rows if row.get("endpoint_id") not in excluded]

    inventory = included(_read_csv(source / "endpoint-inventory.csv"))
    capacity = included(_read_csv(source / "capacity-summary.csv"))
    soak = included(_read_csv(source / "soak-cell-summary.csv"))
    soak_blocks = included(_read_csv(source / "soak-block-summary.csv"))
    recovery = included(_read_csv(source / "recovery-summary.csv"))
    coverage = included(_read_csv(source / "coverage-matrix.csv"))
    capabilities = included(_read_csv(source / "capability-evidence.csv"))
    cache_rows = included(_read_csv(source / "cache-state-metrics.csv"))
    static_verification_path = source / "static-verification-summary.csv"
    cache_verification_path = source / "cache-verification-pairs.csv"
    verification_manifest_path = source / "static-verification-manifest.json"
    capacity_manifest_path = source / "capacity-provenance-manifest.json"
    static_verification = (
        included(_read_csv(static_verification_path))
        if static_verification_path.is_file()
        else []
    )
    cache_verification = (
        included(_read_csv(cache_verification_path))
        if cache_verification_path.is_file()
        else []
    )
    verification_manifest = (
        json.loads(verification_manifest_path.read_text(encoding="utf-8"))
        if verification_manifest_path.is_file()
        else {}
    )
    capacity_manifest = (
        json.loads(capacity_manifest_path.read_text(encoding="utf-8"))
        if capacity_manifest_path.is_file()
        else {}
    )
    endpoints = [str(row.get("endpoint_id")) for row in inventory]
    capabilities = _merge_platform_capabilities(capabilities, cache_rows, endpoints)
    limits = included(_read_csv(source / "observed-limits.csv"))
    interim_markdown = _build_interim_markdown(
        inventory,
        capacity,
        soak,
        coverage,
        soak_blocks,
        recovery,
        capacity_source=capacity_source,
        fixed_rate_source=soak_source,
    )
    (output / "INTERIM-REPORT.md").write_text(interim_markdown, encoding="utf-8")
    figures = _plot_capacity(capacity, capacity_source, figures_dir)
    figures.extend(
        _plot_fixed_rate_tests(
            soak,
            soak_source,
            figures_dir,
            block_rows=soak_blocks,
            recovery_rows=recovery,
        )
    )
    figures.append(_plot_capabilities(capabilities, figures_dir))
    report = output / "digitalocean-hosted-inference-evidence-atlas.pdf"
    _build_pdf(
        report,
        inventory,
        capacity,
        soak,
        capabilities,
        limits,
        coverage,
        soak_blocks,
        recovery,
        figures,
        static_verification=static_verification,
        cache_verification=cache_verification,
        verification_manifest=verification_manifest,
        capacity_manifest=capacity_manifest,
        capacity_source=capacity_source,
        soak_source=soak_source,
    )
    return report
