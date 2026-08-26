from pathlib import Path

from inference_bench.atlas import _build_pdf, _plot_coverage
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
