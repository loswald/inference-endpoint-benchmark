from pathlib import Path

from inference_bench.atlas import _build_pdf, _plot_coverage, _plot_load_response
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
