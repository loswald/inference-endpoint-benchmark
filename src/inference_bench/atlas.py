from __future__ import annotations

import csv
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .matrix import CampaignMatrix
from .report import generate_report, write_csv


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _workload_label(value: str) -> str:
    labels = {
        "short_short": "short input / short output",
        "long_short": "100K input / short output",
        "short_long": "short input / 4K output",
        "mixed": "heterogeneous production mix",
        "mixed:short_short": "production mix: short / short",
        "mixed:long_short": "production mix: 100K / short",
        "mixed:short_long": "production mix: short / 4K",
        "mixed:structured": "production mix: structured task",
    }
    return labels.get(value, value.replace("_", " ").replace(":", ": "))


def _provider_label(value: str) -> str:
    return {
        "amazon-bedrock": "Amazon Bedrock",
        "azure-ai-foundry": "Azure AI Foundry",
        "google-vertex-ai": "Google Vertex AI",
        "openrouter": "OpenRouter",
    }.get(value, value.replace("_", " ").replace("-", " ").title())


def _load_family(row: dict[str, Any]) -> str:
    phase = str(row.get("phase") or "")
    if phase == "soak_block":
        return "soak"
    if phase in {"baseline", "bracket", "aimd", "confirmation", "recovery"}:
        return "aimd"
    return phase or "other"


def _latest_run_groups(rows: list[dict[str, Any]], key: Any) -> list[dict[str, Any]]:
    """Keep complete evidence groups from the latest supplied run root.

    A corrected capacity campaign must supersede the whole earlier trajectory, not merely
    points that happen to share an offered rate. This prevents incompatible controller
    criteria from appearing in one chart.
    """

    latest: dict[tuple[Any, ...], int] = {}
    for row in rows:
        group = tuple(key(row))
        latest[group] = max(latest.get(group, -1), _integer(row.get("run_index")))
    return [row for row in rows if _integer(row.get("run_index")) == latest[tuple(key(row))]]


def _collect(
    matrix: CampaignMatrix, run_roots: Iterable[Path]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_provider: dict[str, str] = {}
    for campaign in matrix.campaigns:
        for route in campaign.config.routes:
            route_provider[route.id] = campaign.provider
    names = (
        "matched-cell-summary.csv",
        "load-block-summary.csv",
        "controller-summary.csv",
        "time-variation-summary.csv",
        "coverage-ledger.csv",
        "outlier-audit-summary.csv",
    )
    for root_index, root in enumerate(run_roots):
        for campaign in matrix.campaigns:
            run_dir = root / campaign.output_name
            if not run_dir.exists():
                continue
            report = generate_report(run_dir).parent
            for name in names:
                for row in _read_csv(report / name):
                    row.update(
                        {
                            "provider": campaign.provider,
                            "campaign": campaign.name,
                            "run_index": root_index,
                        }
                    )
                    tables[name].append(row)
    if not tables["coverage-ledger.csv"]:
        raise ValueError("no terminal provider reports were found under the supplied run roots")

    # Later roots intentionally supersede earlier evidence for the same experimental family.
    # Static capability/latency/context rows stay intact while corrected capacity and soak
    # campaigns replace only their own families.
    tables["controller-summary.csv"] = _latest_run_groups(
        tables["controller-summary.csv"],
        lambda row: (
            row.get("campaign"),
            row.get("route_id"),
            row.get("suite"),
            row.get("shape"),
        ),
    )
    tables["load-block-summary.csv"] = _latest_run_groups(
        tables["load-block-summary.csv"],
        lambda row: (
            row.get("campaign"),
            row.get("route_id"),
            row.get("shape"),
            _load_family(row),
        ),
    )
    tables["matched-cell-summary.csv"] = _latest_run_groups(
        tables["matched-cell-summary.csv"],
        lambda row: (
            row.get("campaign"),
            row.get("route_id"),
            row.get("suite"),
            row.get("cell_id"),
            row.get("cache_state"),
            row.get("reasoning_token_state"),
        ),
    )
    tables["time-variation-summary.csv"] = _latest_run_groups(
        tables["time-variation-summary.csv"],
        lambda row: (
            row.get("campaign"),
            row.get("route_id"),
            row.get("shape"),
            row.get("panel_index"),
        ),
    )
    tables["coverage-ledger.csv"] = _latest_run_groups(
        tables["coverage-ledger.csv"],
        lambda row: (
            row.get("campaign"),
            row.get("route_id"),
            row.get("suite"),
            (
                "soak"
                if "soak" in str(row.get("plan_cell_id") or "")
                else "aimd"
                if str(row.get("suite") or "") in {"aimd", "load"}
                else "static"
            ),
        ),
    )
    return dict(tables), route_provider


def _style_axes(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D8DEE8", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _plot_coverage(
    rows: list[dict[str, Any]], route_provider: dict[str, str], destination: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = ("completed", "unsupported", "inconclusive", "untested", "censored")
    colors = {
        "completed": "#0F766E",
        "unsupported": "#7C3AED",
        "inconclusive": "#D97706",
        "untested": "#94A3B8",
        "censored": "#DC2626",
    }
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(row.get("route_id") or "unknown")][str(row.get("state") or "untested")] += 1
    routes = sorted(grouped, key=lambda route: (route_provider.get(route, ""), route))
    figure, axis = plt.subplots(figsize=(12.5, max(5.0, 0.36 * len(routes) + 1.8)))
    left = [0] * len(routes)
    for state in states:
        values = [grouped[route][state] for route in routes]
        axis.barh(routes, values, left=left, label=state, color=colors[state], height=0.68)
        left = [prior + value for prior, value in zip(left, values, strict=True)]
    axis.invert_yaxis()
    axis.set_xlabel("Planned experiment cells")
    axis.set_title("Evidence coverage by endpoint", loc="left", fontweight="bold")
    axis.legend(ncol=len(states), frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    _style_axes(axis)
    figure.tight_layout()
    path = destination / "coverage-by-endpoint.png"
    figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _shape_from_cell(cell_id: str) -> str | None:
    prefix = cell_id.split(":", 1)[0]
    return prefix if prefix in {"short_short", "long_short", "short_long", "mixed"} else None


def _latency_panel(cell_id: str) -> str | None:
    shape = _shape_from_cell(cell_id)
    if shape != "mixed":
        return shape
    parts = cell_id.split(":")
    return f"mixed:{parts[1]}" if len(parts) > 1 else "mixed"


def _plot_latency(rows: list[dict[str, Any]], destination: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[Path] = []
    panels = (
        "short_short",
        "long_short",
        "short_long",
        "mixed:short_short",
        "mixed:long_short",
        "mixed:short_long",
        "mixed:structured",
    )
    palette = {
        "azure": "#2563EB",
        "bedrock": "#EA580C",
        "vertex": "#DC2626",
        "openrouter": "#7C3AED",
    }
    for shape in panels:
        selected = [
            row
            for row in rows
            if row.get("suite") == "latency" and _latency_panel(str(row.get("cell_id"))) == shape
        ]
        selected.sort(key=lambda row: (str(row.get("provider")), str(row.get("route_id"))))
        if not selected:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(13, max(5.0, 0.36 * len(selected) + 1.6)))
        labels = [str(row.get("route_id")) for row in selected]
        for axis, metric, label in (
            (axes[0], "ttft_p50", "TTFT p50 (seconds)"),
            (axes[1], "latency_p50", "End-to-end latency p50 (seconds)"),
        ):
            for index, row in enumerate(selected):
                value = _number(row.get(metric))
                if value is None:
                    continue
                low = _number(row.get(f"{metric}_ci95_low"))
                high = _number(row.get(f"{metric}_ci95_high"))
                xerr = None if low is None or high is None else [[value - low], [high - value]]
                axis.errorbar(
                    value,
                    index,
                    xerr=xerr,
                    fmt="o",
                    color=palette.get(str(row.get("provider")), "#0F766E"),
                    ecolor="#64748B",
                    capsize=2.5,
                    markersize=5,
                )
            axis.set_yticks(range(len(labels)))
            if axis is axes[0]:
                axis.set_yticklabels(labels, fontsize=8)
            else:
                axis.set_yticklabels([])
            axis.set_xlabel(label)
            axis.invert_yaxis()
            _style_axes(axis)
        figure.suptitle(
            f"Low-load latency — {_workload_label(shape)}",
            x=0.08,
            ha="left",
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        path = destination / f"latency-{_slug(shape)}.png"
        figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(path)
    return created


def _plot_aimd(rows: list[dict[str, Any]], destination: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[Path] = []
    for shape in ("short_short", "long_short", "short_long", "mixed"):
        selected = [row for row in rows if row.get("suite") == "aimd" and row.get("shape") == shape]
        selected.sort(key=lambda row: (str(row.get("provider")), str(row.get("route_id"))))
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(10.8, max(4.8, 0.34 * len(selected) + 1.8)))
        labels = [str(row.get("route_id")) for row in selected]
        positive_bounds = [
            value
            for row in selected
            if (value := _number(row.get("healthy_lower_bound_rps"))) is not None and value > 0
        ]
        for index, row in enumerate(selected):
            lower = _number(row.get("healthy_lower_bound_rps"))
            upper = _number(row.get("unhealthy_upper_bound_rps"))
            if lower is None:
                axis.text(
                    0.01,
                    index,
                    "not established",
                    transform=axis.get_yaxis_transform(),
                    color="#B91C1C",
                    fontsize=7.5,
                    va="center",
                )
                continue
            confirmed = row.get("confirmation_all_healthy") in (True, "True", "true", "1")
            axis.scatter(
                lower,
                index,
                marker="o" if confirmed else "s",
                facecolor="#0F766E" if confirmed else "white",
                edgecolor="#0F766E",
                s=44,
                zorder=3,
            )
            if upper is not None and upper > lower:
                axis.plot([lower, upper], [index, index], color="#D97706", linewidth=1.4)
                axis.scatter(upper, index, marker="|", color="#D97706", s=70)
            elif str(row.get("capacity_bound_state", "")).startswith("right_censored"):
                axis.annotate(
                    "tested lower bound",
                    (lower, index),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                )
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Offered requests/second")
        if positive_bounds and max(positive_bounds) / min(positive_bounds) >= 10:
            axis.set_xscale("log")
        figure.suptitle(
            f"AIMD capacity bounds — {_workload_label(shape)}",
            x=0.10,
            y=0.985,
            ha="left",
            fontweight="bold",
        )
        figure.text(
            0.10,
            0.935,
            "filled circle: three healthy confirmations; square: exploratory; "
            "orange tick: unhealthy upper bound",
            fontsize=8,
            color="#475569",
        )
        _style_axes(axis)
        figure.tight_layout(rect=(0, 0, 1, 0.89))
        path = destination / f"aimd-capacity-{shape}.png"
        figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(path)
    return created


def _plot_soak(rows: list[dict[str, Any]], destination: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[Path] = []
    metric_groups = (
        (
            ("successful_rpm", "Successful requests/minute"),
            ("quality_adjusted_rpm", "Correct tasks/minute"),
        ),
        (
            ("successful_input_tpm", "Effective input tokens/minute"),
            ("successful_output_tpm", "Effective output tokens/minute"),
        ),
    )
    for shape in ("short_short", "long_short", "short_long", "mixed"):
        selected = [
            row for row in rows if row.get("phase") == "soak_block" and row.get("shape") == shape
        ]
        selected.sort(key=lambda row: (str(row.get("provider")), str(row.get("route_id"))))
        if not selected:
            continue
        labels = [str(row.get("route_id")) for row in selected]
        for group_index, metrics in enumerate(metric_groups, start=1):
            figure, axes = plt.subplots(1, 2, figsize=(13, max(5.0, 0.36 * len(selected) + 1.6)))
            for axis, (metric, label) in zip(axes, metrics, strict=True):
                for index, row in enumerate(selected):
                    value = _number(row.get(metric))
                    if value is None:
                        continue
                    low = _number(row.get(f"{metric}_ci95_low"))
                    high = _number(row.get(f"{metric}_ci95_high"))
                    xerr = (
                        None
                        if low is None or high is None
                        else [[max(0, value - low)], [max(0, high - value)]]
                    )
                    axis.errorbar(
                        value,
                        index,
                        xerr=xerr,
                        fmt="o",
                        color="#0F766E",
                        ecolor="#64748B",
                        capsize=2.5,
                    )
                axis.set_yticks(range(len(labels)))
                if axis is axes[0]:
                    axis.set_yticklabels(labels, fontsize=8)
                else:
                    axis.set_yticklabels([])
                axis.invert_yaxis()
                axis.set_xlabel(label)
                _style_axes(axis)
            figure.suptitle(
                f"Two-minute sustained workload — {_workload_label(shape)}",
                x=0.08,
                ha="left",
                fontweight="bold",
            )
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            path = destination / f"soak-{shape}-part-{group_index}.png"
            figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
            plt.close(figure)
            created.append(path)
    return created


def _plot_load_response(
    rows: list[dict[str, Any]], route_provider: dict[str, str], destination: Path
) -> list[Path]:
    """Plot unconnected endpoint-specific quality and latency response points under load."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    phase_colors = {
        "baseline": "#64748B",
        "bracket": "#2563EB",
        "aimd": "#2563EB",
        "confirmation": "#0F766E",
        "recovery": "#D97706",
        "soak_block": "#7C3AED",
    }
    created: list[Path] = []
    providers = sorted(set(route_provider.values()))
    for provider in providers:
        provider_routes = sorted(
            route for route, owner in route_provider.items() if owner == provider
        )
        for shape in ("short_short", "long_short", "short_long", "mixed"):
            selected = [
                row
                for row in rows
                if row.get("route_id") in provider_routes
                and row.get("shape") == shape
                and _number(row.get("offered_rps_target")) is not None
            ]
            routes = [
                route
                for route in provider_routes
                if any(r.get("route_id") == route for r in selected)
            ]
            if not routes:
                continue
            chunks = [routes[index : index + 3] for index in range(0, len(routes), 3)]
            for part, route_chunk in enumerate(chunks, start=1):
                figure, axes = plt.subplots(
                    len(route_chunk),
                    2,
                    figsize=(13.5, max(5.3, 2.55 * len(route_chunk) + 1.4)),
                    squeeze=False,
                )
                observed_phases: set[str] = set()
                for row_index, route in enumerate(route_chunk):
                    route_rows = [row for row in selected if row.get("route_id") == route]
                    route_rows.sort(key=lambda row: _number(row.get("offered_rps_target")) or 0)
                    for column, (metric, label) in enumerate(
                        (
                            ("quality_mean", "Task quality (0–1)"),
                            ("arrival_latency_p95_across_blocks", "Arrival-to-finish p95 (s)"),
                        )
                    ):
                        axis = axes[row_index][column]
                        positive_rates: list[float] = []
                        for row in route_rows:
                            rate = _number(row.get("offered_rps_target"))
                            value = _number(row.get(metric))
                            if rate is None or rate <= 0 or value is None:
                                continue
                            phase = str(row.get("phase") or "other")
                            observed_phases.add(phase)
                            color = phase_colors.get(phase, "#475569")
                            low = _number(row.get(f"{metric}_ci95_low"))
                            high = _number(row.get(f"{metric}_ci95_high"))
                            yerr = None
                            if low is not None and high is not None and low <= value <= high:
                                yerr = [[value - low], [high - value]]
                            eligible = _integer(row.get("capacity_estimand_blocks_n"))
                            healthy = _integer(row.get("healthy_blocks_n"))
                            filled = eligible > 0 and healthy == eligible
                            axis.errorbar(
                                rate,
                                value,
                                yerr=yerr,
                                fmt="o",
                                markerfacecolor=color if filled else "white",
                                markeredgecolor=color,
                                ecolor=color,
                                capsize=2,
                                markersize=5,
                                alpha=0.9,
                            )
                            positive_rates.append(rate)
                        if positive_rates and max(positive_rates) / min(positive_rates) >= 10:
                            axis.set_xscale("log")
                        if metric == "quality_mean":
                            axis.set_ylim(-0.04, 1.04)
                        axis.set_ylabel(label)
                        axis.set_xlabel("Offered requests/second")
                        axis.set_title(route, loc="left", fontsize=9, fontweight="bold")
                        _style_axes(axis)
                legend = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=color,
                        markeredgecolor=color,
                        label=phase.replace("_", " "),
                    )
                    for phase, color in phase_colors.items()
                    if phase in observed_phases
                ]
                if legend:
                    figure.legend(
                        handles=legend,
                        frameon=False,
                        ncol=len(legend),
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.955),
                    )
                figure.suptitle(
                    f"Load response — {provider} — {_workload_label(shape)}",
                    x=0.06,
                    y=0.995,
                    ha="left",
                    fontweight="bold",
                )
                figure.text(
                    0.06,
                    0.005,
                    "Filled markers passed the healthy-block rule; hollow markers did not. "
                    "Points are unconnected because AIMD revisits rates.",
                    fontsize=7.5,
                    color="#475569",
                )
                figure.tight_layout(rect=(0, 0.03, 1, 0.90))
                path = destination / (f"load-response-{_slug(provider)}-{shape}-part-{part}.png")
                figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
                plt.close(figure)
                created.append(path)
    return created


def _context_percentage(cell_id: str) -> float | None:
    if not cell_id.startswith("context_") or "pct" not in cell_id:
        return None
    return _number(cell_id.removeprefix("context_").split("pct", 1)[0])


def _plot_context(rows: list[dict[str, Any]], destination: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fixed = [
        row
        for row in rows
        if row.get("suite") == "context"
        and _context_percentage(str(row.get("cell_id"))) is not None
    ]
    if not fixed:
        return None
    routes = sorted({str(row.get("route_id")) for row in fixed})
    percentages = sorted({_context_percentage(str(row.get("cell_id"))) for row in fixed} - {None})
    matrix = np.full((len(routes), len(percentages)), np.nan)
    for row in fixed:
        route = str(row.get("route_id"))
        percentage = _context_percentage(str(row.get("cell_id")))
        value = _number(row.get("quality_mean"))
        if percentage is not None and value is not None:
            matrix[routes.index(route), percentages.index(percentage)] = value
    figure, axis = plt.subplots(figsize=(11.5, max(5, 0.35 * len(routes) + 1.8)))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(percentages)), [f"{value:g}%" for value in percentages])
    axis.set_yticks(range(len(routes)), routes)
    axis.set_xlabel("Advertised context-window anchor")
    axis.set_title("Long-context retrieval success", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean deterministic retrieval score")
    figure.tight_layout()
    path = destination / "context-retrieval.png"
    figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _plot_time_variation(rows: list[dict[str, Any]], destination: Path) -> list[Path]:
    """Plot matched sequential panels as endpoint-specific small multiples.

    Lines connect only repeated observations of the same endpoint and workload. This preserves
    the time sequence without producing a cross-endpoint spaghetti chart.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[Path] = []
    providers = sorted({str(row.get("provider")) for row in rows if row.get("provider")})
    for provider in providers:
        provider_rows = [row for row in rows if row.get("provider") == provider]
        for shape in ("short_short", "long_short", "short_long", "mixed"):
            shape_rows = [row for row in provider_rows if row.get("shape") == shape]
            routes = sorted({str(row.get("route_id")) for row in shape_rows})
            for part, route_chunk in enumerate(
                (routes[index : index + 3] for index in range(0, len(routes), 3)),
                start=1,
            ):
                if not route_chunk:
                    continue
                figure, axes = plt.subplots(
                    len(route_chunk),
                    2,
                    figsize=(13.2, max(5.2, 2.45 * len(route_chunk) + 1.5)),
                    squeeze=False,
                )
                for row_index, route in enumerate(route_chunk):
                    route_rows = sorted(
                        (row for row in shape_rows if row.get("route_id") == route),
                        key=lambda row: _integer(row.get("panel_index")),
                    )
                    for column, (metric, label) in enumerate(
                        (("ttft_p50", "TTFT p50 (s)"), ("latency_p50", "Latency p50 (s)"))
                    ):
                        axis = axes[row_index][column]
                        eligible = [
                            row for row in route_rows if _number(row.get(metric)) is not None
                        ]
                        x = [_integer(row.get("panel_index")) + 1 for row in eligible]
                        y = [_number(row.get(metric)) or 0 for row in eligible]
                        low = [
                            _number(row.get(f"{metric}_ci95_low"))
                            if _number(row.get(f"{metric}_ci95_low")) is not None
                            else value
                            for row, value in zip(eligible, y, strict=True)
                        ]
                        high = [
                            _number(row.get(f"{metric}_ci95_high"))
                            if _number(row.get(f"{metric}_ci95_high")) is not None
                            else value
                            for row, value in zip(eligible, y, strict=True)
                        ]
                        if eligible:
                            axis.errorbar(
                                x,
                                y,
                                yerr=[
                                    [
                                        max(0, value - bound)
                                        for value, bound in zip(y, low, strict=True)
                                    ],
                                    [
                                        max(0, bound - value)
                                        for value, bound in zip(y, high, strict=True)
                                    ],
                                ],
                                fmt="o-",
                                color="#0F766E",
                                ecolor="#64748B",
                                linewidth=1.4,
                                capsize=2.5,
                            )
                            axis.set_xticks(sorted(set(x)))
                        axis.set_xlabel("Matched panel (equal spacing)")
                        axis.set_ylabel(label)
                        axis.set_title(route, loc="left", fontsize=9, fontweight="bold")
                        _style_axes(axis)
                figure.suptitle(
                    f"Intra-session variation — {provider} — {_workload_label(shape)}",
                    x=0.06,
                    y=0.995,
                    ha="left",
                    fontweight="bold",
                )
                figure.text(
                    0.06,
                    0.006,
                    "Same prompts and offered load in every panel; whiskers are request-level "
                    "95% intervals. This is an intra-session panel, not a 24-hour claim.",
                    fontsize=7.5,
                    color="#475569",
                )
                figure.tight_layout(rect=(0, 0.035, 1, 0.95))
                path = destination / f"time-{_slug(provider)}-{shape}-part-{part}.png"
                figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
                plt.close(figure)
                created.append(path)
    return created


_CAPABILITY_PREFIXES = (
    ("baseline_stream", "stream"),
    ("baseline_nonstream", "non-stream"),
    ("structured_json:", "JSON"),
    ("structured_json_schema", "JSON schema"),
    ("tool_call:", "tool"),
    ("parallel_tool_calls", "parallel tools"),
    ("nested_tool_schema", "nested schema"),
    ("tool_count_64", "64 tools"),
    ("vision_small_png", "vision"),
    ("vision_two_png_images", "two images"),
    ("stop_sequence_triggered", "stop"),
    ("logprobs_presence", "logprobs"),
)


def _plot_capabilities(rows: list[dict[str, Any]], destination: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    selected = [row for row in rows if row.get("suite") == "capability"]
    if not selected:
        return None
    routes = sorted({str(row.get("route_id")) for row in selected})
    matrix = np.full((len(routes), len(_CAPABILITY_PREFIXES)), np.nan)
    for row in selected:
        cell = str(row.get("cell_id") or "")
        for column, (prefix, _) in enumerate(_CAPABILITY_PREFIXES):
            if not cell.startswith(prefix):
                continue
            success = _number(row.get("success_rate"))
            quality = _number(row.get("quality_mean"))
            value = success if quality is None else min(success or 0, quality)
            if value is not None:
                matrix[routes.index(str(row.get("route_id"))), column] = value
            break
    figure, axis = plt.subplots(figsize=(13.5, max(5, 0.35 * len(routes) + 2)))
    display = np.where(
        np.isnan(matrix), -1, np.where(matrix >= 0.999, 2, np.where(matrix > 0, 1, 0))
    )
    cmap = ListedColormap(["#E2E8F0", "#DC2626", "#D97706", "#0F766E"])
    axis.imshow(display + 1, aspect="auto", vmin=0, vmax=3, cmap=cmap)
    axis.set_xticks(
        range(len(_CAPABILITY_PREFIXES)),
        [label for _, label in _CAPABILITY_PREFIXES],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(routes)), routes)
    axis.set_title("Functional capability checks", loc="left", fontweight="bold")
    axis.set_xlabel("teal: passed | amber: partial | red: failed | grey: no measured result")
    figure.tight_layout()
    path = destination / "capability-matrix.png"
    figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _format_metric(value: Any, *, digits: int = 2) -> str:
    number = _number(value)
    if number is None:
        return "not measured"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _format_interval(row: dict[str, Any], metric: str, *, digits: int = 2) -> str:
    value = _number(row.get(metric))
    if value is None:
        return "not measured"
    low = _number(row.get(f"{metric}_ci95_low"))
    high = _number(row.get(f"{metric}_ci95_high"))
    if low is None or high is None:
        return _format_metric(value, digits=digits)
    return (
        f"{_format_metric(value, digits=digits)} "
        f"[{_format_metric(low, digits=digits)}, {_format_metric(high, digits=digits)}]"
    )


def _figure_caption(name: str) -> str:
    if name.startswith("latency-"):
        return "Matched low-load requests; whiskers are request-level 95% intervals."
    if name.startswith("aimd-"):
        return (
            "AIMD reports tested healthy and unhealthy bounds. A missing bound is labelled "
            "rather than drawn as zero."
        )
    if name.startswith("load-response-"):
        return (
            "Unconnected points preserve the AIMD experiment structure. Whiskers use whole "
            "load blocks as the uncertainty unit."
        )
    if name.startswith("soak-"):
        return "Two-minute endpoint-isolated soak; whiskers are block-level 95% intervals."
    if name.startswith("time-"):
        return (
            "Matched sequential panels show intra-session variation only; they are not evidence "
            "of a full-day or diurnal pattern."
        )
    if name.startswith("context-"):
        return "Acceptance counts only when the deterministic retrieval check also succeeds."
    if name.startswith("capability-"):
        return "Capability states come from functional task scorers, not model-card claims."
    return "Measured evidence states; unavailable values remain blank rather than becoming zero."


def _build_pdf(
    path: Path,
    matrix: CampaignMatrix,
    tables: dict[str, list[dict[str, Any]]],
    figures: list[Path],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
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
    slate = colors.HexColor("#475569")
    pale = colors.HexColor("#F1F5F9")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="AtlasTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=31,
            textColor=navy,
            alignment=TA_LEFT,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AtlasH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=navy,
            spaceBefore=8,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AtlasH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=teal,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AtlasBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            textColor=navy,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AtlasSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10,
            textColor=slate,
        )
    )

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        page_width, _ = canvas._pagesize
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(18 * mm, 12 * mm, page_width - 18 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(slate)
        canvas.drawString(18 * mm, 7.5 * mm, "Inference Endpoint Benchmark - evidence atlas")
        canvas.drawRightString(page_width - 18 * mm, 7.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    landscape_a4 = landscape(A4)
    portrait_frame = Frame(
        18 * mm,
        17 * mm,
        A4[0] - 36 * mm,
        A4[1] - 34 * mm,
        id="portrait-frame",
    )
    landscape_frame = Frame(
        18 * mm,
        17 * mm,
        landscape_a4[0] - 36 * mm,
        landscape_a4[1] - 34 * mm,
        id="landscape-frame",
    )
    document = BaseDocTemplate(
        str(path),
        pagesize=A4,
        pageTemplates=[
            PageTemplate(id="portrait", pagesize=A4, frames=[portrait_frame], onPage=footer),
            PageTemplate(
                id="landscape",
                pagesize=landscape_a4,
                frames=[landscape_frame],
                onPage=footer,
            ),
        ],
        title="Inference Endpoint Benchmark - Evidence Atlas",
        author="Sqwish Labs",
    )
    endpoints = [route for campaign in matrix.campaigns for route in campaign.config.routes]
    coverage = tables.get("coverage-ledger.csv", [])
    matched = tables.get("matched-cell-summary.csv", [])
    controllers = tables.get("controller-summary.csv", [])
    load = tables.get("load-block-summary.csv", [])
    time_rows = tables.get("time-variation-summary.csv", [])
    coverage_counts = Counter(str(row.get("state")) for row in coverage)
    story: list[Any] = [
        Spacer(1, 18 * mm),
        Paragraph("Inference Endpoint Benchmark", styles["AtlasTitle"]),
        Paragraph("Technical performance and capability evidence atlas", styles["AtlasH1"]),
        Paragraph(
            f"{len(matrix.campaigns)} providers, {len(endpoints)} exact hosted routes, "
            f"{len(coverage):,} planned evidence cells. Regenerated from the terminal, "
            "hash-bound evidence ledgers.",
            styles["AtlasBody"],
        ),
        Spacer(1, 6 * mm),
        Paragraph(
            "This atlas separates endpoint, workload, phase, and measurement unit. It does not "
            "average unrelated workloads into a global score. Whiskers are 95% intervals; the "
            "sampling unit is the request for low-load cells and the epoch or soak block for load "
            "capacity.",
            styles["AtlasBody"],
        ),
        Spacer(1, 8 * mm),
        Table(
            [
                ["Evidence state", "Cells"],
                *[
                    [state.replace("_", " ").title(), f"{count:,}"]
                    for state, count in sorted(coverage_counts.items())
                ],
            ],
            colWidths=[90 * mm, 35 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), pale),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            ),
        ),
        PageBreak(),
        Paragraph("How to read the evidence", styles["AtlasH1"]),
        Paragraph(
            "TTFT is request start to first visible streamed content. End-to-end latency includes "
            "queueing, retries, backoff, and response drain. Effective TPM uses successful "
            "provider-reported tokens over the full arrival window plus drain. Decode proxy is a "
            "client-observed visible-token rate and is withheld when timing, usage, or hidden "
            "reasoning makes it incomparable.",
            styles["AtlasBody"],
        ),
        Paragraph("Plain-language glossary", styles["AtlasH2"]),
        Table(
            [
                ["Term", "Meaning in this report"],
                [
                    "Offered RPS / RPM",
                    Paragraph(
                        "Requests the test tried to start per second / minute. This is demand, "
                        "not completed work.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "Successful RPM",
                    Paragraph(
                        "Requests completed successfully per minute, measured over the arrival "
                        "window plus response drain.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "Effective input / output TPM",
                    Paragraph(
                        "Provider-reported tokens from successful requests per wall-clock minute. "
                        "A blank means usage reporting was insufficient.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "TTFT",
                    Paragraph(
                        "Time to first token: request start to the first visible streamed content.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "End-to-end latency",
                    Paragraph(
                        "Scheduled arrival to final completion, including local queueing, retries, "
                        "backoff, and response drain.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "Decode proxy",
                    Paragraph(
                        "Visible output tokens divided by client-observed decoding time. It is not "
                        "claimed as internal GPU speed and is withheld when incomparable.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "Quality-adjusted RPM",
                    Paragraph(
                        "Correctly completed predeclared tasks per minute. Transport success alone "
                        "does not count as correct.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "p50 / p95",
                    Paragraph(
                        "Median / 95th-percentile observation. p95 is a slow-tail measure; p99 is "
                        "not reported as reliable without enough samples.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "95% interval",
                    Paragraph(
                        "An uncertainty interval from independent requests at low load or whole "
                        "epochs/blocks under load. Tokens are never treated as independent "
                        "samples.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "Right-censored bound",
                    Paragraph(
                        "The endpoint stayed healthy at the highest tested rate; its true ceiling "
                        "may be higher. This is a lower bound, not a discovered maximum.",
                        styles["AtlasSmall"],
                    ),
                ],
                [
                    "Inconclusive",
                    Paragraph(
                        "The request ran, but throttling, timeout, missing usage, or another "
                        "measurement condition prevents the intended claim.",
                        styles["AtlasSmall"],
                    ),
                ],
            ],
            colWidths=[45 * mm, 123 * mm],
            repeatRows=1,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            ),
        ),
        Paragraph("AIMD and sustained load", styles["AtlasH2"]),
        Paragraph(
            "AIMD raises offered load while healthy and halves it after congestion. Its result is "
            "a tested healthy lower bound and, when observed, an unhealthy upper bound. "
            "A sustained "
            "claim requires the separate multi-block soak. Quality-adjusted goodput counts only "
            "predeclared tasks completed correctly per minute.",
            styles["AtlasBody"],
        ),
        Paragraph("Context and capabilities", styles["AtlasH2"]),
        Paragraph(
            "Context acceptance is not enough: retrieval markers must also be returned exactly. "
            "Capability cells use functional scorers for JSON, tools, parallel calls, nested "
            "schemas, vision, stop sequences, and other controls. Rejection without a bound error "
            "reason is not silently promoted to documented unsupported behavior.",
            styles["AtlasBody"],
        ),
    ]
    story.extend(
        [
            PageBreak(),
            Paragraph("Engineering decision map", styles["AtlasH1"]),
            Paragraph(
                "This is a route inventory, not a global ranking. Compare rows only for the "
                "workload you will actually run; the endpoint pages later in the atlas retain "
                "all four workload shapes and their uncertainty intervals.",
                styles["AtlasBody"],
            ),
        ]
    )
    for campaign in matrix.campaigns:
        provider_rows: list[list[Any]] = [
            [
                "Exact route",
                "TTFT p50",
                "Retrieval anchor",
                "Capabilities passed",
                "Soak workloads",
            ]
        ]
        for route in campaign.config.routes:
            route_id = route.id
            short_latency = next(
                (
                    row
                    for row in matched
                    if row.get("route_id") == route_id
                    and row.get("suite") == "latency"
                    and _shape_from_cell(str(row.get("cell_id"))) == "short_short"
                ),
                {},
            )
            context_rows = [
                row
                for row in matched
                if row.get("route_id") == route_id
                and row.get("suite") == "context"
                and (_number(row.get("quality_mean")) or 0) >= 0.999
                and _context_percentage(str(row.get("cell_id"))) is not None
            ]
            highest_context = max(
                (_context_percentage(str(row.get("cell_id"))) or 0 for row in context_rows),
                default=None,
            )
            capability_rows = [
                row
                for row in matched
                if row.get("route_id") == route_id and row.get("suite") == "capability"
            ]
            capability_passed = sum(
                (
                    _number(row.get("quality_mean"))
                    if _number(row.get("quality_mean")) is not None
                    else _number(row.get("success_rate")) or 0
                )
                >= 0.999
                for row in capability_rows
            )
            soak_rows = [
                row
                for row in load
                if row.get("route_id") == route_id
                and row.get("phase") == "soak_block"
                and _number(row.get("successful_rpm")) is not None
            ]
            provider_rows.append(
                [
                    route_id,
                    f"{_format_metric(short_latency.get('ttft_p50'))} s",
                    "not established"
                    if highest_context is None
                    else f"{highest_context:g}% of advertised",
                    f"{capability_passed}/{len(capability_rows)}",
                    f"{len({str(row.get('shape')) for row in soak_rows})}/4",
                ]
            )
        story.extend(
            [
                Paragraph(_provider_label(campaign.provider), styles["AtlasH2"]),
                Table(
                    provider_rows,
                    colWidths=[47 * mm, 27 * mm, 37 * mm, 31 * mm, 30 * mm],
                    repeatRows=1,
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), navy),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                            ("PADDING", (0, 0), (-1, -1), 4),
                        ]
                    ),
                ),
                Spacer(1, 3 * mm),
            ]
        )
    first_figure = True
    for figure in figures:
        from PIL import Image as PILImage

        with PILImage.open(figure) as source:
            width, height = source.size
        max_width = 258 * mm
        max_height = 160 * mm
        scale = min(max_width / width, max_height / height)
        chart = Image(str(figure), width=width * scale, height=height * scale)
        chart.hAlign = "CENTER"
        page_start: list[Any] = [PageBreak()]
        if first_figure:
            page_start = [NextPageTemplate("landscape"), PageBreak()]
            first_figure = False
        story.extend(
            page_start
            + [
                chart,
                Spacer(1, 2 * mm),
                Paragraph(_figure_caption(figure.stem), styles["AtlasSmall"]),
            ]
        )
    story.append(NextPageTemplate("portrait"))
    for route in endpoints:
        route_id = route.id
        latency = next(
            (
                row
                for row in matched
                if row.get("route_id") == route_id
                and row.get("suite") == "latency"
                and _shape_from_cell(str(row.get("cell_id"))) == "short_short"
            ),
            {},
        )
        endpoint_controllers = [
            row
            for row in controllers
            if row.get("route_id") == route_id and row.get("suite") == "aimd"
        ]
        endpoint_soaks = [
            row
            for row in load
            if row.get("route_id") == route_id and row.get("phase") == "soak_block"
        ]
        endpoint_capabilities = [
            row
            for row in matched
            if row.get("route_id") == route_id and row.get("suite") == "capability"
        ]
        functional = sum(
            (
                _number(row.get("quality_mean"))
                if _number(row.get("quality_mean")) is not None
                else _number(row.get("success_rate")) or 0
            )
            >= 0.999
            for row in endpoint_capabilities
        )
        endpoint_time = [
            row
            for row in time_rows
            if row.get("route_id") == route_id and row.get("shape") == "short_short"
        ]
        time_ttft = [
            value for row in endpoint_time if (value := _number(row.get("ttft_p50"))) is not None
        ]
        rows_data = [
            ["Route", route_id],
            ["Provider / region", f"{route.provider} / {route.region}"],
            ["Exact model / API", f"{route.model} / {route.api_family}"],
            ["Model version", route.model_version],
            [
                "Context / output",
                f"{route.context_tokens or 'not documented'} / "
                f"{route.max_output_tokens or 'not documented'} tokens",
            ],
            [
                "Input / output price",
                f"${route.input_usd_per_million:g} / ${route.output_usd_per_million:g} "
                "per million tokens",
            ],
            [
                "Short TTFT p50 / p95",
                f"{_format_metric(latency.get('ttft_p50'))} / "
                f"{_format_metric(latency.get('ttft_p95'))} s",
            ],
            [
                "Short latency p50 / p95",
                f"{_format_metric(latency.get('latency_p50'))} / "
                f"{_format_metric(latency.get('latency_p95'))} s",
            ],
            [
                "Functional capability probes",
                f"{functional}/{len(endpoint_capabilities)} measured cells",
            ],
            [
                "Time panels / short TTFT range",
                (
                    "not measured"
                    if not time_ttft
                    else f"{len(time_ttft)} panels / {min(time_ttft):.2f}–{max(time_ttft):.2f} s"
                ),
            ],
        ]
        story.extend(
            [
                PageBreak(),
                Paragraph(route_id, styles["AtlasH1"]),
                Table(
                    rows_data,
                    colWidths=[48 * mm, 120 * mm],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), pale),
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                            ("PADDING", (0, 0), (-1, -1), 5),
                        ]
                    ),
                ),
                Spacer(1, 4 * mm),
                Paragraph("Capacity by workload", styles["AtlasH2"]),
            ]
        )
        capacity_rows = [
            [
                "Workload",
                "AIMD healthy lower bound",
                "Confirmation state",
                "Soak successful RPM [95% CI]",
            ]
        ]
        for shape in ("short_short", "long_short", "short_long", "mixed"):
            controller = next(
                (row for row in endpoint_controllers if row.get("shape") == shape), {}
            )
            soak = next((row for row in endpoint_soaks if row.get("shape") == shape), {})
            capacity_rows.append(
                [
                    shape.replace("_", " / "),
                    _format_metric(controller.get("healthy_lower_bound_rps")),
                    str(controller.get("controller_completion_state") or "not measured").replace(
                        "_", " "
                    ),
                    _format_interval(soak, "successful_rpm"),
                ]
            )
        story.append(
            Table(
                capacity_rows,
                colWidths=[36 * mm, 40 * mm, 51 * mm, 45 * mm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), navy),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            )
        )
        story.extend(
            [
                Spacer(1, 4 * mm),
                Paragraph("Interpretation boundary", styles["AtlasH2"]),
                Paragraph(
                    "Use only the workload cells shown. An AIMD lower bound is not a sustained "
                    "capacity claim unless the corresponding soak cell is present. Missing metrics "
                    "remain unverified for this exact route, API, region, and observation window.",
                    styles["AtlasBody"],
                ),
            ]
        )
    def invariant_canvas(*args: Any, **kwargs: Any) -> Any:
        kwargs["invariant"] = 1
        return pdf_canvas.Canvas(*args, **kwargs)

    document.build(story, canvasmaker=invariant_canvas)


def generate_atlas(
    matrix: CampaignMatrix,
    run_roots: Iterable[str | Path],
    output_dir: str | Path,
) -> Path:
    roots = [Path(root).resolve() for root in run_roots]
    if not roots:
        raise ValueError("at least one run root is required")
    output = Path(output_dir).resolve()
    if output.exists():
        if output.name != "atlas":
            raise ValueError("refusing to replace an output directory not named atlas")
        shutil.rmtree(output)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True)
    tables, route_provider = _collect(matrix, roots)
    for name, rows in tables.items():
        write_csv(output / name, rows)
    figures: list[Path] = [
        _plot_coverage(tables["coverage-ledger.csv"], route_provider, figures_dir)
    ]
    figures.extend(_plot_latency(tables.get("matched-cell-summary.csv", []), figures_dir))
    figures.extend(_plot_aimd(tables.get("controller-summary.csv", []), figures_dir))
    figures.extend(
        _plot_load_response(tables.get("load-block-summary.csv", []), route_provider, figures_dir)
    )
    figures.extend(_plot_soak(tables.get("load-block-summary.csv", []), figures_dir))
    figures.extend(_plot_time_variation(tables.get("time-variation-summary.csv", []), figures_dir))
    context = _plot_context(tables.get("matched-cell-summary.csv", []), figures_dir)
    capabilities = _plot_capabilities(tables.get("matched-cell-summary.csv", []), figures_dir)
    figures.extend(item for item in (context, capabilities) if item is not None)
    report = output / "inference-endpoint-evidence-atlas.pdf"
    _build_pdf(report, matrix, tables, figures)
    (output / "README.md").write_text(
        "# Inference endpoint evidence atlas\n\n"
        "The PDF is the readable decision guide. CSV files preserve every combined table used "
        "by the figures. Missing values are evidence states, never implicit zeroes.\n",
        encoding="utf-8",
    )
    return report
