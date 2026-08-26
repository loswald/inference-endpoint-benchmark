from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

SHAPES = ("short_short", "input32k_short", "short_long", "mixed")
SHAPE_LABELS = {
    "short_short": "short input / short output",
    "input32k_short": "32K input / short output",
    "short_long": "short input / long output",
    "mixed": "heterogeneous production mix",
}


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


def _style_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D8DEE8", linewidth=0.7, alpha=0.85)
    axis.set_axisbelow(True)


def _plot_capacity(rows: list[dict[str, str]], source_id: str, destination: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    created: list[Path] = []
    for shape in SHAPES:
        selected = [
            row for row in rows if row.get("source_id") == source_id and row.get("shape") == shape
        ]
        selected.sort(key=lambda row: str(row.get("endpoint_id")))
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(11.8, max(5.2, 0.4 * len(selected) + 1.8)))
        positive: list[float] = []
        for index, row in enumerate(selected):
            claim = str(row.get("capacity_claim") or "")
            confirmed = claim.startswith("confirmed_")
            value = _number(row.get("capacity_lower_bound_rps"))
            if value is None and not confirmed:
                value = _number(row.get("highest_observed_healthy_rps"))
            upper = _number(row.get("capacity_upper_bound_rps"))
            if value is None:
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
            positive.append(value)
            axis.scatter(
                value,
                index,
                marker="o" if confirmed else "s",
                facecolor="#0F766E" if confirmed else "white",
                edgecolor="#0F766E",
                s=46,
                zorder=3,
            )
            if upper is not None and upper > value:
                axis.plot([value, upper], [index, index], color="#D97706", linewidth=1.4)
                axis.scatter(upper, index, marker="|", color="#D97706", s=75)
        labels = [str(row.get("endpoint_id")) for row in selected]
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        if positive and max(positive) / min(positive) >= 20:
            axis.set_xscale("log")
        axis.set_xlabel("Offered requests/second")
        axis.set_title(
            f"DigitalOcean AIMD capacity - {SHAPE_LABELS[shape]}",
            loc="left",
            fontweight="bold",
        )
        axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#0F766E",
                    markeredgecolor="#0F766E",
                    label="confirmed healthy lower bound",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="none",
                    markerfacecolor="white",
                    markeredgecolor="#0F766E",
                    label="exploratory healthy observation",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    linestyle="none",
                    color="#D97706",
                    label="unhealthy upper bound",
                ),
            ],
            frameon=False,
            ncol=3,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
        )
        _style_axis(axis)
        figure.tight_layout()
        path = destination / f"digitalocean-capacity-{shape}.png"
        figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(path)
    return created


def _plot_soak(rows: list[dict[str, str]], source_id: str, destination: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_groups = (
        (
            (
                "successful_rpm_block_mean",
                "successful_rpm_block_mean_ci95_student_t",
                "Successful requests/minute",
            ),
            (
                "paired_quality_delta_mean",
                "paired_quality_delta_mean_ci95_student_t",
                "Quality change vs low load",
            ),
        ),
        (
            (
                "effective_input_tpm_block_mean",
                "effective_input_tpm_block_mean_ci95_student_t",
                "Effective input tokens/minute",
            ),
            (
                "effective_output_tpm_block_mean",
                "effective_output_tpm_block_mean_ci95_student_t",
                "Effective output tokens/minute",
            ),
        ),
    )
    created: list[Path] = []
    for shape in SHAPES:
        selected = [
            row for row in rows if row.get("source_id") == source_id and row.get("shape") == shape
        ]
        selected.sort(key=lambda row: str(row.get("endpoint_id")))
        if not selected:
            continue
        labels = [str(row.get("endpoint_id")) for row in selected]
        for group_index, metrics in enumerate(metric_groups, start=1):
            figure, axes = plt.subplots(1, 2, figsize=(13, max(5.2, 0.4 * len(selected) + 1.8)))
            for axis, (metric, interval_field, label) in zip(axes, metrics, strict=True):
                for index, row in enumerate(selected):
                    value = _number(row.get(metric))
                    if value is None:
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
                    interval = _json_interval(row.get(interval_field))
                    xerr = None
                    if interval is not None and interval[0] <= value <= interval[1]:
                        xerr = [[value - interval[0]], [interval[1] - value]]
                    complete = row.get("status") == "complete"
                    axis.errorbar(
                        value,
                        index,
                        xerr=xerr,
                        fmt="o",
                        markerfacecolor="#0F766E" if complete else "white",
                        markeredgecolor="#0F766E",
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
                _style_axis(axis)
            figure.suptitle(
                f"DigitalOcean two-minute soak — {SHAPE_LABELS[shape]}",
                x=0.055,
                ha="left",
                fontweight="bold",
            )
            figure.text(
                0.055,
                0.005,
                "Whiskers are 95% Student-t intervals across four 30-second blocks. "
                "Missing estimates are labelled, never plotted as zero.",
                fontsize=7.5,
                color="#475569",
            )
            figure.tight_layout(rect=(0, 0.025, 1, 0.97))
            path = destination / f"digitalocean-soak-{shape}-part-{group_index}.png"
            figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
            plt.close(figure)
            created.append(path)
    return created


def _plot_capabilities(rows: list[dict[str, str]], destination: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    dimensions = (
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
        "caching_option",
    )
    endpoints = sorted({str(row.get("endpoint_id")) for row in rows})
    matrix = np.full((len(endpoints), len(dimensions)), np.nan)
    symbols = np.full((len(endpoints), len(dimensions)), "", dtype=object)
    status_value = {
        "documented_unavailable": (0, "U"),
        "failed": (1, "F"),
        "degraded": (2, "D"),
        "passed": (3, "P"),
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
            key = "degraded" if transport == "observed_transport_degraded" else "failed"
        value, symbol = status_value[key]
        row_index, column = endpoints.index(endpoint), dimensions.index(dimension)
        matrix[row_index, column] = value
        symbols[row_index, column] = symbol
    figure, axis = plt.subplots(figsize=(13.5, max(5.2, 0.4 * len(endpoints) + 2)))
    cmap = ListedColormap(["#7C3AED", "#DC2626", "#D97706", "#0F766E"])
    image = axis.imshow(matrix, aspect="auto", vmin=-0.5, vmax=3.5, cmap=cmap)
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
                    fontsize=7.5,
                    fontweight="bold",
                )
    axis.set_xticks(range(len(dimensions)), [value.replace("_", " ") for value in dimensions])
    axis.tick_params(axis="x", labelrotation=35)
    axis.set_yticks(range(len(endpoints)), endpoints)
    axis.set_title("DigitalOcean functional capability evidence", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, ticks=range(4))
    colorbar.ax.set_yticklabels(["documented unavailable", "failed", "degraded", "passed"])
    figure.tight_layout()
    path = destination / "digitalocean-capabilities.png"
    figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _format(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:,.{digits}g}"


def _build_pdf(
    path: Path,
    inventory: list[dict[str, str]],
    capacity: list[dict[str, str]],
    soak: list[dict[str, str]],
    capabilities: list[dict[str, str]],
    limits: list[dict[str, str]],
    figures: list[Path],
    *,
    capacity_source: str,
    soak_source: str,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
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

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        page_width, _ = canvas._pagesize
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(18 * mm, 12 * mm, page_width - 18 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(slate)
        canvas.drawString(18 * mm, 7.5 * mm, "DigitalOcean hosted inference evidence atlas")
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
        title="DigitalOcean Hosted Inference - Evidence Atlas",
        author="Sqwish Labs",
    )
    capacity_index = {
        (row.get("endpoint_id"), row.get("shape")): row
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
    story: list[Any] = [
        Spacer(1, 19 * mm),
        Paragraph("DigitalOcean hosted inference", styles["DoTitle"]),
        Paragraph("Technical evidence atlas", styles["DoH1"]),
        Paragraph(
            f"{len(inventory)} exact hosted endpoints. Capacity source: {capacity_source}. "
            f"Sustained-load source: {soak_source}.",
            styles["DoBody"],
        ),
        Paragraph(
            "The AIMD pages show tested operational bounds, not fitted theoretical maxima. The "
            "soak pages report four independent 30-second blocks at one candidate rate. Missing "
            "values remain blank; singleton boundary/capability probes do not receive invented "
            "confidence intervals.",
            styles["DoBody"],
        ),
        PageBreak(),
        Paragraph("Engineering decision map", styles["DoH1"]),
        Paragraph(
            "This inventory reports evidence states for each exact hosted endpoint. It is not a "
            "single global ranking: capacity and capability must be read for the workload and "
            "feature the application actually uses.",
            styles["DoBody"],
        ),
    ]
    decision_rows = [
        ["Exact endpoint", "Context", "Confirmed AIMD", "Complete soaks", "Capabilities passed"]
    ]
    for endpoint in sorted(inventory, key=lambda row: str(row.get("endpoint_id"))):
        endpoint_id = str(endpoint.get("endpoint_id"))
        confirmed = sum(
            str(
                capacity_index.get((endpoint_id, shape), {}).get("capacity_claim") or ""
            ).startswith("confirmed_")
            for shape in SHAPES
        )
        complete_soaks = sum(
            soak_index.get((endpoint_id, shape), {}).get("status") == "complete" for shape in SHAPES
        )
        endpoint_capabilities = [
            row for row in capabilities if row.get("endpoint_id") == endpoint_id
        ]
        passed = sum(row.get("functional_status") == "passed" for row in endpoint_capabilities)
        decision_rows.append(
            [
                endpoint_id,
                str(endpoint.get("context_window") or "not documented"),
                f"{confirmed}/4",
                f"{complete_soaks}/4",
                f"{passed}/{len(endpoint_capabilities)}",
            ]
        )
    story.append(
        Table(
            decision_rows,
            colWidths=[58 * mm, 31 * mm, 28 * mm, 27 * mm, 30 * mm],
            repeatRows=1,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.3),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            ),
        )
    )
    first_figure = True
    for figure in figures:
        from PIL import Image as PILImage

        with PILImage.open(figure) as source:
            width, height = source.size
        scale = min(258 * mm / width, 160 * mm / height)
        chart = Image(str(figure), width=width * scale, height=height * scale)
        chart.hAlign = "CENTER"
        page_start: list[Any] = [PageBreak()]
        if first_figure:
            page_start = [NextPageTemplate("landscape"), PageBreak()]
            first_figure = False
        story.extend(page_start + [chart])
    story.append(NextPageTemplate("portrait"))
    limits_by_endpoint: dict[str, list[dict[str, str]]] = {}
    for row in limits:
        limits_by_endpoint.setdefault(str(row.get("endpoint_id")), []).append(row)
    for endpoint in sorted(inventory, key=lambda row: str(row.get("endpoint_id"))):
        endpoint_id = str(endpoint.get("endpoint_id"))
        story.extend([PageBreak(), Paragraph(endpoint_id, styles["DoH1"])])
        facts = [
            ["Model / API", f"{endpoint_id} / {endpoint.get('api_surface') or '-'}"],
            [
                "Region / API version",
                f"{endpoint.get('server_region')} / {endpoint.get('api_version')}",
            ],
            [
                "Context / max output",
                f"{endpoint.get('context_window')} / {endpoint.get('max_output_tokens')} tokens",
            ],
            [
                "Input / output price",
                f"${endpoint.get('input_usd_per_million')} / "
                f"${endpoint.get('output_usd_per_million')} per million tokens",
            ],
        ]
        story.append(
            Table(
                facts,
                colWidths=[48 * mm, 120 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), pale),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            )
        )
        capacity_rows = [["Workload", "AIMD evidence", "Healthy bound", "Soak rate / state"]]
        for shape in SHAPES:
            aimd = capacity_index.get((endpoint_id, shape), {})
            sustained = soak_index.get((endpoint_id, shape), {})
            capacity_rows.append(
                [
                    SHAPE_LABELS[shape],
                    str(aimd.get("capacity_claim") or "not measured").replace("_", " "),
                    f"{_format(aimd.get('capacity_lower_bound_rps'))} RPS",
                    f"{_format(sustained.get('two_minute_soak_observed_rps'))} RPS / "
                    f"{str(sustained.get('status') or 'not measured').replace('_', ' ')}",
                ]
            )
        story.extend(
            [
                Spacer(1, 4 * mm),
                Table(
                    capacity_rows,
                    colWidths=[45 * mm, 62 * mm, 27 * mm, 38 * mm],
                    repeatRows=1,
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), navy),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTSIZE", (0, 0), (-1, -1), 6.9),
                            ("PADDING", (0, 0), (-1, -1), 3.5),
                        ]
                    ),
                ),
                Spacer(1, 4 * mm),
            ]
        )
        capabilities_rows = [["Capability", "Transport", "Functional"]]
        for dimension in ("response_format", "tools", "parallel_tool_calls", "vision"):
            row = capability_index.get((endpoint_id, dimension), {})
            capabilities_rows.append(
                [
                    dimension.replace("_", " "),
                    str(row.get("transport_status") or "not measured").replace("_", " "),
                    str(row.get("functional_status") or "not measured").replace("_", " "),
                ]
            )
        story.append(
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
                        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                        ("PADDING", (0, 0), (-1, -1), 3.5),
                    ]
                ),
            )
        )
        findings = limits_by_endpoint.get(endpoint_id, [])
        if findings:
            story.extend(
                [
                    Spacer(1, 4 * mm),
                    Paragraph(
                        "Observed boundaries: "
                        + "; ".join(
                            f"{row.get('dimension')}: {row.get('finding')}" for row in findings
                        ),
                        styles["DoSmall"],
                    ),
                ]
            )
    document.build(story)


def generate_digitalocean_atlas(
    summary_dir: str | Path,
    output_dir: str | Path,
    *,
    capacity_source: str,
    soak_source: str,
) -> Path:
    source = Path(summary_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        if output.name != "digitalocean-atlas":
            raise ValueError("refusing to replace an output directory not named digitalocean-atlas")
        shutil.rmtree(output)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True)
    inventory = _read_csv(source / "endpoint-inventory.csv")
    capacity = _read_csv(source / "capacity-summary.csv")
    soak = _read_csv(source / "soak-cell-summary.csv")
    capabilities = _read_csv(source / "capability-evidence.csv")
    limits = _read_csv(source / "observed-limits.csv")
    figures = _plot_capacity(capacity, capacity_source, figures_dir)
    figures.extend(_plot_soak(soak, soak_source, figures_dir))
    figures.append(_plot_capabilities(capabilities, figures_dir))
    report = output / "digitalocean-hosted-inference-evidence-atlas.pdf"
    _build_pdf(
        report,
        inventory,
        capacity,
        soak,
        capabilities,
        limits,
        figures,
        capacity_source=capacity_source,
        soak_source=soak_source,
    )
    return report
