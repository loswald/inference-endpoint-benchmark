"""Provider-neutral, explicit-state evidence tables."""

from __future__ import annotations

from collections.abc import Iterable

from .models import EvidenceCell, ProviderReportProfile
from .states import state_message, state_presentation


def evidence_table(
    cells: Iterable[EvidenceCell],
    profile: ProviderReportProfile,
    *,
    metric_id: str | None = None,
) -> list[dict[str, str]]:
    """Build display rows with no unexplained empty result cells."""

    rows: list[dict[str, str]] = []
    for cell in cells:
        presentation = state_presentation(cell.state)
        metric = cell.metrics.get(metric_id) if metric_id else None
        estimate = ""
        interval = ""
        sample_size = ""
        if metric is not None:
            if metric.estimate is not None:
                estimate = f"{metric.estimate:g} {metric.unit}".strip()
            if metric.low is not None and metric.high is not None:
                prefix = "95% CI" if metric.interval_kind == "confidence" else "range"
                interval = (f"{prefix} [{metric.low:g}, {metric.high:g}] {metric.unit}").strip()
            elif metric.low is not None and metric.interval_kind == "evidence_bounds":
                interval = f"confirmed lower bound ≥ {metric.low:g} {metric.unit}".strip()
            if metric.sample_size is not None:
                sample_size = f"{metric.sample_size} {metric.sampling_unit}(s)"
        rows.append(
            {
                "endpoint": profile.endpoint(cell.endpoint_id).label,
                "workload": profile.workload(cell.workload_id).label,
                "experiment": profile.experiment(cell.experiment_id).label,
                "state": presentation.label,
                "result": estimate or state_message(cell),
                "uncertainty / bounds": interval or "Not applicable to this state",
                "sample": sample_size or "See experiment provenance",
                "source": cell.source_id or "Source not named",
            }
        )
    return rows


def render_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No evidence rows were supplied._\n"
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            str(row.get(column) or "Not supplied").replace("|", "\\|") for column in columns
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"
