"""Small, reusable chart factories for inference evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any

from .models import EvidenceCell, EvidenceState, ProviderReportProfile
from .states import state_presentation


@dataclass(frozen=True)
class ChartStyle:
    text_color: str = "#172033"
    grid_color: str = "#CBD5E1"
    background_color: str = "#FFFFFF"
    reference_color: str = "#64748B"
    font_size: float = 10.0
    dpi: int = 180


DEFAULT_CHART_STYLE = ChartStyle()


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _selected_cells(
    cells: Iterable[EvidenceCell],
    experiment_id: str,
    where: dict[str, str] | None = None,
) -> dict[tuple[str, str], EvidenceCell]:
    selected: dict[tuple[str, str], EvidenceCell] = {}
    for cell in cells:
        if cell.experiment_id != experiment_id:
            continue
        if where and any(cell.dimensions.get(key) != value for key, value in where.items()):
            continue
        key = (cell.endpoint_id, cell.workload_id)
        if key in selected:
            raise ValueError(f"duplicate evidence cell for {key[0]} × {key[1]}")
        selected[key] = cell
    return selected


def evidence_matrix(
    cells: Iterable[EvidenceCell],
    profile: ProviderReportProfile,
    *,
    experiment_id: str,
    title: str | None = None,
    where: dict[str, str] | None = None,
    style: ChartStyle = DEFAULT_CHART_STYLE,
) -> Any:
    """Render a directly labelled endpoint × workload state matrix.

    Missing input rows are rendered as explicit "not run" cells rather than blank
    chart area.  Color is redundant with text, so the chart remains interpretable
    when printed or viewed with impaired color perception.
    """

    import matplotlib.colors as colors
    from matplotlib.patches import Patch

    plt = _pyplot()
    selected = _selected_cells(cells, experiment_id, where)
    endpoints = profile.endpoints
    workloads = profile.workloads
    states = list(EvidenceState)
    state_index = {state: index for index, state in enumerate(states)}
    values: list[list[int]] = []
    labels: list[list[str]] = []
    for endpoint in endpoints:
        value_row: list[int] = []
        label_row: list[str] = []
        for workload in workloads:
            cell = selected.get((endpoint.id, workload.id))
            state = cell.state if cell else EvidenceState.NOT_RUN
            presentation = state_presentation(state)
            value_row.append(state_index[state])
            label_row.append(fill(presentation.short_label, width=10))
        values.append(value_row)
        labels.append(label_row)

    palette = [state_presentation(state).color for state in states]
    cmap = colors.ListedColormap(palette)
    norm = colors.BoundaryNorm(range(len(states) + 1), cmap.N)
    figure, axis = plt.subplots(
        figsize=(max(11.5, 2.05 * len(workloads) + 3.3), max(5.0, 0.55 * len(endpoints) + 2.0))
    )
    figure.patch.set_facecolor(style.background_color)
    axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks([index - 0.5 for index in range(1, len(workloads))], minor=True)
    axis.set_yticks([index - 0.5 for index in range(1, len(endpoints))], minor=True)
    axis.grid(which="minor", color="#FFFFFF", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xticks(range(len(workloads)), [item.label for item in workloads])
    axis.set_yticks(range(len(endpoints)), [item.label for item in endpoints])
    axis.tick_params(axis="x", rotation=25, labelsize=style.font_size)
    axis.tick_params(axis="y", labelsize=style.font_size)
    for row_index, row in enumerate(labels):
        for column_index, label in enumerate(row):
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="#FFFFFF",
                fontsize=max(7.0, style.font_size - 2),
                fontweight="bold",
                wrap=True,
            )
    experiment = profile.experiment(experiment_id)
    axis.set_title(title or experiment.label, loc="left", color=style.text_color, pad=34)
    axis.text(
        0,
        1.015,
        fill(experiment.question, width=100),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=style.reference_color,
        fontsize=max(8.0, style.font_size - 1),
    )
    used_states = sorted(
        {cell.state for cell in selected.values()} | {EvidenceState.NOT_RUN},
        key=lambda item: item.value,
    )
    handles = [
        Patch(color=state_presentation(state).color, label=state_presentation(state).short_label)
        for state in used_states
    ]
    axis.legend(
        handles=handles,
        title="Evidence state",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        frameon=False,
        fontsize=max(7.0, style.font_size - 1),
    )
    figure.tight_layout(rect=(0, 0, 0.80, 1))
    return figure


def interval_forest(
    cells: Iterable[EvidenceCell],
    profile: ProviderReportProfile,
    *,
    experiment_id: str,
    metric_id: str,
    workload_id: str,
    title: str | None = None,
    reference_value: float | None = None,
    where: dict[str, str] | None = None,
    style: ChartStyle = DEFAULT_CHART_STYLE,
) -> Any:
    """Render estimates and 95% intervals without hiding non-numeric states."""

    from matplotlib.transforms import blended_transform_factory

    plt = _pyplot()
    selected = _selected_cells(cells, experiment_id, where)
    rows = [selected.get((endpoint.id, workload_id)) for endpoint in profile.endpoints]
    figure, axis = plt.subplots(figsize=(10.5, max(4.8, 0.52 * len(profile.endpoints) + 1.8)))
    figure.patch.set_facecolor(style.background_color)
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    unit = ""
    for y, (_endpoint, cell) in enumerate(zip(profile.endpoints, rows, strict=True)):
        if cell is None:
            presentation = state_presentation(EvidenceState.NOT_RUN)
            axis.text(
                1.02,
                y,
                presentation.short_label,
                transform=transform,
                va="center",
                color=presentation.color,
            )
            continue
        metric = cell.metrics.get(metric_id)
        presentation = state_presentation(cell.state)
        if metric is None or metric.estimate is None:
            axis.text(
                1.02,
                y,
                presentation.short_label,
                transform=transform,
                va="center",
                color=presentation.color,
            )
            continue
        unit = metric.unit or unit
        if metric.low is not None and metric.high is not None:
            axis.errorbar(
                metric.estimate,
                y,
                xerr=[[metric.estimate - metric.low], [metric.high - metric.estimate]],
                fmt=presentation.marker,
                color=presentation.color,
                capsize=3,
                markersize=6,
            )
        else:
            axis.plot(
                metric.estimate, y, presentation.marker, color=presentation.color, markersize=6
            )
        if (
            metric.interval_kind == "evidence_bounds"
            and metric.low is not None
            and metric.high is not None
        ):
            direct_label = f" {metric.low:g}–{metric.high:g}"
            label_x = metric.high
        else:
            direct_label = (
                f" {'≥' if cell.state is EvidenceState.CONFIRMED_LOWER_BOUND else ''}"
                f"{metric.estimate:g}"
            )
            label_x = metric.estimate
        axis.annotate(
            direct_label,
            (label_x, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=style.text_color,
            fontsize=style.font_size - 1,
        )
        axis.text(
            1.02,
            y,
            presentation.short_label,
            transform=transform,
            va="center",
            color=presentation.color,
        )
    if reference_value is not None:
        axis.axvline(reference_value, color=style.reference_color, linestyle="--", linewidth=1)
    axis.set_yticks(range(len(profile.endpoints)), [item.label for item in profile.endpoints])
    axis.invert_yaxis()
    axis.grid(axis="x", color=style.grid_color, linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    workload = profile.workload(workload_id)
    experiment = profile.experiment(experiment_id)
    axis.set_title(title or f"{experiment.label}: {workload.label}", loc="left", pad=14)
    metric_profile = profile.metric(experiment_id, metric_id)
    axis.set_xlabel(f"{metric_profile.label} ({unit})" if unit else metric_profile.label)
    axis.text(
        1.02,
        1.015,
        "Evidence state",
        transform=axis.transAxes,
        va="bottom",
        color=style.reference_color,
        fontsize=max(8.0, style.font_size - 1),
        fontweight="bold",
    )
    axis.margins(x=0.08)
    figure.tight_layout(rect=(0, 0, 0.82, 1))
    return figure


def save_figure(
    figure: Any,
    path: str | Path,
    *,
    style: ChartStyle = DEFAULT_CHART_STYLE,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=style.dpi, bbox_inches="tight", facecolor=style.background_color)
    return output
