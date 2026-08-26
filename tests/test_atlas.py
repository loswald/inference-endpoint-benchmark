from pathlib import Path

from inference_bench.atlas import (
    _build_pdf,
    _latest_run_groups,
    _plot_coverage,
    _plot_latency,
    _plot_load_response,
    _plot_time_variation,
)
from inference_bench.matrix import CampaignMatrix, MatrixCampaign


def _matrix(campaign, tmp_path: Path) -> CampaignMatrix:
    provider = campaign.routes[0].provider
    return CampaignMatrix(
        path=tmp_path / "matrix.yaml",
        max_parallel_providers=1,
        campaigns=(
            MatrixCampaign(
                name="fixture",
                provider=provider,
                config_path=tmp_path / "fixture.yaml",
                output_name="fixture",
                config=campaign,
            ),
        ),
    )


def test_atlas_pdf_and_coverage_figure_are_readable(tmp_path, campaign) -> None:
    matrix = _matrix(campaign, tmp_path)
    route_id = campaign.routes[0].id
    figures = tmp_path / "figures"
    figures.mkdir()
    coverage = [
        {"route_id": route_id, "state": "completed"},
        {"route_id": route_id, "state": "inconclusive"},
    ]
    figure = _plot_coverage(coverage, {route_id: campaign.routes[0].provider}, figures)
    assert figure.read_bytes().startswith(b"\x89PNG")
    pdf = tmp_path / "atlas.pdf"
    _build_pdf(
        pdf,
        matrix,
        {
            "coverage-ledger.csv": coverage,
            "matched-cell-summary.csv": [],
            "controller-summary.csv": [],
            "load-block-summary.csv": [],
        },
        [figure],
    )
    material = pdf.read_bytes()
    assert material.startswith(b"%PDF-")
    assert len(material) > 10_000


def test_load_response_uses_unconnected_matched_endpoint_panels(tmp_path, campaign) -> None:
    route_id = campaign.routes[0].id
    figures = tmp_path / "figures"
    figures.mkdir()
    rows = []
    for rate, quality, latency, phase in (
        (0.25, 1.0, 0.8, "baseline"),
        (1.0, 0.95, 1.2, "bracket"),
        (4.0, 0.7, 2.8, "confirmation"),
    ):
        rows.append(
            {
                "route_id": route_id,
                "shape": "short_short",
                "phase": phase,
                "offered_rps_target": rate,
                "capacity_estimand_blocks_n": 3,
                "healthy_blocks_n": 3 if rate < 4 else 1,
                "quality_mean": quality,
                "quality_mean_ci95_low": max(0, quality - 0.05),
                "quality_mean_ci95_high": min(1, quality + 0.05),
                "arrival_latency_p95_across_blocks": latency,
                "arrival_latency_p95_across_blocks_ci95_low": latency * 0.9,
                "arrival_latency_p95_across_blocks_ci95_high": latency * 1.1,
            }
        )
    created = _plot_load_response(rows, {route_id: campaign.routes[0].provider}, figures)
    assert len(created) == 1
    assert created[0].read_bytes().startswith(b"\x89PNG")


def test_corrected_run_supersedes_whole_capacity_family() -> None:
    rows = [
        {"route_id": "r", "shape": "mixed", "phase": "aimd", "run_index": 0, "rate": 1},
        {"route_id": "r", "shape": "mixed", "phase": "aimd", "run_index": 0, "rate": 2},
        {"route_id": "r", "shape": "mixed", "phase": "aimd", "run_index": 1, "rate": 3},
    ]
    selected = _latest_run_groups(rows, lambda row: (row["route_id"], row["shape"], row["phase"]))
    assert [row["rate"] for row in selected] == [3]


def test_mixed_latency_is_split_into_task_specific_figures(tmp_path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    rows = []
    for task in ("short_short", "long_short", "short_long", "structured"):
        rows.append(
            {
                "provider": "vertex",
                "route_id": "route-a",
                "suite": "latency",
                "cell_id": f"mixed:{task}:in256:out128",
                "ttft_p50": 0.4,
                "latency_p50": 1.0,
            }
        )
    created = _plot_latency(rows, figures)
    assert len(created) == 4
    assert {path.name for path in created} == {
        "latency-mixed-short-short.png",
        "latency-mixed-long-short.png",
        "latency-mixed-short-long.png",
        "latency-mixed-structured.png",
    }


def test_time_variation_uses_endpoint_small_multiples(tmp_path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    rows = [
        {
            "provider": "azure",
            "route_id": "route-a",
            "shape": "short_short",
            "panel_index": panel,
            "ttft_p50": 0.4 + panel * 0.1,
            "latency_p50": 1.0 + panel * 0.2,
        }
        for panel in range(4)
    ]
    created = _plot_time_variation(rows, figures)
    assert len(created) == 1
    assert created[0].read_bytes().startswith(b"\x89PNG")
