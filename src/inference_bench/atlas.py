from __future__ import annotations

import csv
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
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


def _plot_latency(rows: list[dict[str, Any]], destination: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[Path] = []
    for shape in ("short_short", "long_short", "short_long", "mixed"):
        selected = [
            row
            for row in rows
            if row.get("suite") == "latency" and _shape_from_cell(str(row.get("cell_id"))) == shape
        ]
        selected.sort(key=lambda row: (str(row.get("provider")), str(row.get("route_id"))))
        if not selected:
            continue
        figure, axes = plt.subplots(
            1, 2, figsize=(13, max(5.0, 0.34 * len(selected) + 1.8)), sharey=True
        )
        for axis, metric, label in (
            (axes[0], "ttft_p50", "TTFT p50 (seconds)"),
            (axes[1], "latency_p50", "End-to-end latency p50 (seconds)"),
        ):
            labels = [str(row.get("route_id")) for row in selected]
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
                    color="#0F766E",
                    ecolor="#64748B",
                    capsize=2.5,
                    markersize=5,
                )
            axis.set_yticks(range(len(labels)), labels if axis is axes[0] else [""] * len(labels))
            axis.set_xlabel(label)
            axis.invert_yaxis()
            _style_axes(axis)
        figure.suptitle(
            f"Low-load latency - {shape.replace('_', ' / ')}",
            x=0.08,
            ha="left",
            fontweight="bold",
        )
        figure.tight_layout()
        path = destination / f"latency-{shape}.png"
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
        for index, row in enumerate(selected):
            lower = _number(row.get("healthy_lower_bound_rps"))
            upper = _number(row.get("unhealthy_upper_bound_rps"))
            if lower is None:
                axis.scatter(0, index, marker="x", color="#DC2626", s=35)
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
        axis.set_title(
            f"AIMD capacity bounds - {shape.replace('_', ' / ')}",
            loc="left",
            fontweight="bold",
        )
        axis.text(
            0,
            1.02,
            "filled circle: three healthy confirmations; square: exploratory; "
            "orange tick: unhealthy upper bound",
            transform=axis.transAxes,
            fontsize=8,
            color="#475569",
        )
        _style_axes(axis)
        figure.tight_layout()
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
    metrics = (
        ("successful_rpm", "Successful RPM"),
        ("successful_input_tpm", "Effective input TPM"),
        ("successful_output_tpm", "Effective output TPM"),
        ("quality_adjusted_rpm", "Quality-adjusted tasks/minute"),
    )
    for shape in ("short_short", "long_short", "short_long", "mixed"):
        selected = [
            row for row in rows if row.get("phase") == "soak_block" and row.get("shape") == shape
        ]
        selected.sort(key=lambda row: (str(row.get("provider")), str(row.get("route_id"))))
        if not selected:
            continue
        figure, axes = plt.subplots(
            1, 4, figsize=(17, max(5.0, 0.34 * len(selected) + 1.8)), sharey=True
        )
        labels = [str(row.get("route_id")) for row in selected]
        for axis, (metric, label) in zip(axes, metrics, strict=True):
            for index, row in enumerate(selected):
                value = _number(row.get(metric))
                if value is None:
                    continue
                low = _number(row.get(f"{metric}_ci95_low"))
                high = _number(row.get(f"{metric}_ci95_high"))
                xerr = None if low is None or high is None else [[value - low], [high - value]]
                axis.errorbar(
                    value, index, xerr=xerr, fmt="o", color="#0F766E", ecolor="#64748B", capsize=2.5
                )
            axis.set_yticks(range(len(labels)), labels if axis is axes[0] else [""] * len(labels))
            axis.invert_yaxis()
            axis.set_xlabel(label)
            _style_axes(axis)
        figure.suptitle(
            f"Sustained workload - {shape.replace('_', ' / ')}",
            x=0.06,
            ha="left",
            fontweight="bold",
        )
        figure.tight_layout()
        path = destination / f"soak-{shape}.png"
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
            figure, axes = plt.subplots(
                len(routes),
                2,
                figsize=(13.5, max(4.2, 2.15 * len(routes) + 1.4)),
                squeeze=False,
            )
            observed_phases: set[str] = set()
            for row_index, route in enumerate(routes):
                route_rows = [row for row in selected if row.get("route_id") == route]
                route_rows.sort(key=lambda row: _number(row.get("offered_rps_target")) or 0)
                for column, (metric, label) in enumerate(
                    (
                        ("quality_mean", "Task quality (0-1)"),
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
                    axis.set_title(route, loc="left", fontsize=8.5, fontweight="bold")
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
                figure.legend(handles=legend, frameon=False, ncol=len(legend), loc="upper right")
            figure.suptitle(
                f"Load response - {provider} - {shape.replace('_', ' / ')}",
                x=0.06,
                ha="left",
                fontweight="bold",
            )
            figure.text(
                0.06,
                0.005,
                "Filled markers passed the registered healthy-block rule; hollow markers did not. "
                "Points are not connected because AIMD revisits rates over time.",
                fontsize=7.5,
                color="#475569",
            )
            figure.tight_layout(rect=(0, 0.025, 1, 0.965))
            path = destination / f"load-response-{_slug(provider)}-{shape}.png"
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


def _build_pdf(
    path: Path,
    matrix: CampaignMatrix,
    tables: dict[str, list[dict[str, Any]]],
    figures: list[Path],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
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
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(18 * mm, 12 * mm, A4[0] - 18 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(slate)
        canvas.drawString(18 * mm, 7.5 * mm, "Inference Endpoint Benchmark - evidence atlas")
        canvas.drawRightString(A4[0] - 18 * mm, 7.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="Inference Endpoint Benchmark - Evidence Atlas",
        author="Sqwish Labs",
    )
    endpoints = [route for campaign in matrix.campaigns for route in campaign.config.routes]
    coverage = tables.get("coverage-ledger.csv", [])
    coverage_counts = Counter(str(row.get("state")) for row in coverage)
    story: list[Any] = [
        Spacer(1, 18 * mm),
        Paragraph("Inference Endpoint Benchmark", styles["AtlasTitle"]),
        Paragraph("Technical performance and capability evidence atlas", styles["AtlasH1"]),
        Paragraph(
            f"{len(matrix.campaigns)} providers, {len(endpoints)} exact hosted routes, "
            f"{len(coverage):,} planned evidence cells. Generated "
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}.",
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
    for figure in figures:
        from PIL import Image as PILImage

        with PILImage.open(figure) as source:
            width, height = source.size
        max_width = 174 * mm
        max_height = 238 * mm
        scale = min(max_width / width, max_height / height)
        story.extend(
            [
                PageBreak(),
                Paragraph(figure.stem.replace("-", " ").title(), styles["AtlasH1"]),
                Image(str(figure), width=width * scale, height=height * scale),
                Paragraph(
                    "Points are measured estimates. Missing values remain blank rather than being "
                    "drawn as zero.",
                    styles["AtlasSmall"],
                ),
            ]
        )
    matched = tables.get("matched-cell-summary.csv", [])
    controllers = tables.get("controller-summary.csv", [])
    load = tables.get("load-block-summary.csv", [])
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
        rows_data = [
            ["Route", route_id],
            ["Provider / region", f"{route.provider} / {route.region}"],
            ["Model version", route.model_version],
            [
                "Context / output",
                f"{route.context_tokens or 'not documented'} / "
                f"{route.max_output_tokens or 'not documented'} tokens",
            ],
            ["Short TTFT p50", f"{_format_metric(latency.get('ttft_p50'))} s"],
            ["Short latency p50", f"{_format_metric(latency.get('latency_p50'))} s"],
            [
                "Functional capability probes",
                f"{functional}/{len(endpoint_capabilities)} measured cells",
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
            ["Workload", "AIMD healthy lower bound", "Confirmation state", "Soak successful RPM"]
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
                    _format_metric(soak.get("successful_rpm")),
                ]
            )
        story.append(
            Table(
                capacity_rows,
                colWidths=[38 * mm, 42 * mm, 58 * mm, 34 * mm],
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
    document.build(story, onFirstPage=footer, onLaterPages=footer)


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
