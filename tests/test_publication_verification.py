from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from inference_bench.publication_verification import (
    ExpectedScientificCell,
    PublicationExpectation,
    ScientificCellResult,
    verify_publication,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(tmp_path: Path, *, pages: int = 2) -> tuple[Path, Path]:
    package = tmp_path / "publication"
    package.mkdir()
    (package / "results.csv").write_text(
        "route,cell,state\nroute-a,latency-a,measured\nroute-b,latency-b,unsupported\n",
        encoding="utf-8",
    )
    pdf = package / "report.pdf"
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with pdf.open("wb") as handle:
        writer.write(handle)
    return package, pdf


def _expectation() -> PublicationExpectation:
    return PublicationExpectation(
        publication_id="portable-provider-atlas",
        route_ids=("route-a", "route-b"),
        cells=(
            ExpectedScientificCell("latency-a", "route-a", 4),
            ExpectedScientificCell("latency-b", "route-b", 0),
        ),
        expected_requests=4,
        pdf_paths=("report.pdf",),
    )


def _cells() -> list[ScientificCellResult]:
    return [
        ScientificCellResult("latency-a", "route-a", 4, "measured_with_interval"),
        ScientificCellResult(
            "latency-b",
            "route-b",
            0,
            "unsupported",
            "The exact route contract documents this operation as unsupported.",
        ),
    ]


def _qa_receipt(tmp_path: Path, pdf: Path, *, pages: int = 2) -> Path:
    qa_root = tmp_path / "qa"
    renders = qa_root / "renders"
    renders.mkdir(parents=True)
    page_entries = []
    for page_number in range(1, pages + 1):
        render = renders / f"page-{page_number:03d}.png"
        render.write_bytes(b"rendered-page-" + str(page_number).encode("ascii"))
        page_entries.append(
            {
                "page_number": page_number,
                "render_path": f"renders/{render.name}",
                "render_sha256": _sha256(render),
                "inspected": True,
                "passed": True,
            }
        )
    receipt = qa_root / "visual-qa.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "pdf-visual-qa-receipt/v1",
                "inspector": "independent-page-review",
                "pdfs": [
                    {
                        "path": "report.pdf",
                        "sha256": _sha256(pdf),
                        "page_count": pages,
                        "pages": page_entries,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return receipt


def test_verification_requires_exact_coverage_safety_manifest_and_every_pdf_page(
    tmp_path: Path,
) -> None:
    package, pdf = _package(tmp_path)
    qa = _qa_receipt(tmp_path, pdf)
    receipt_path = tmp_path / "receipts" / "publication-verification.json"

    result_path = verify_publication(
        package,
        _expectation(),
        _cells(),
        visual_qa_receipt_path=qa,
        receipt_path=receipt_path,
    )

    first_receipt = result_path.read_bytes()
    result = json.loads(first_receipt)
    assert result["passed"] is True
    assert all(check["passed"] is True for check in result["checks"].values())
    assert result["expected"] == {"routes": 2, "cells": 2, "requests": 4, "pdfs": 1}
    assert result["observed"]["pdf_pages"] == 2
    assert result["observed"]["inspected_pdf_pages"] == 2
    assert result["hashes"]["pdf_sha256"] == {"report.pdf": _sha256(pdf)}
    assert len(result["hashes"]["page_render_sha256"]) == 2
    manifest = json.loads((package / "publication-manifest.json").read_text(encoding="utf-8"))
    assert {item["path"] for item in manifest["files"]} == {"report.pdf", "results.csv"}
    assert not receipt_path.is_relative_to(package)

    verify_publication(
        package,
        _expectation(),
        list(reversed(_cells())),
        visual_qa_receipt_path=qa,
        receipt_path=receipt_path,
    )
    assert receipt_path.read_bytes() == first_receipt


def test_verification_emits_failure_receipt_for_missing_blank_and_unexplained_cells(
    tmp_path: Path,
) -> None:
    package, pdf = _package(tmp_path)
    qa = _qa_receipt(tmp_path, pdf)
    cells = [ScientificCellResult("latency-a", "route-a", 3, "")]
    receipt_path = tmp_path / "failed-verification.json"

    verify_publication(
        package,
        _expectation(),
        cells,
        visual_qa_receipt_path=qa,
        receipt_path=receipt_path,
    )

    result = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
    codes = {finding["code"] for finding in result["findings"]}
    assert {
        "missing_cell",
        "missing_route",
        "request_count_mismatch",
        "total_request_count_mismatch",
        "blank_state",
    } <= codes
    assert result["checks"]["scientific_coverage"]["passed"] is False
    assert result["hashes"]["publication_manifest_sha256"]


def test_explicit_non_numeric_state_needs_plain_language_explanation(tmp_path: Path) -> None:
    package, pdf = _package(tmp_path)
    qa = _qa_receipt(tmp_path, pdf)
    cells = [
        _cells()[0],
        ScientificCellResult("latency-b", "route-b", 0, "transport_gated"),
    ]
    receipt_path = tmp_path / "failed-verification.json"

    verify_publication(
        package,
        _expectation(),
        cells,
        visual_qa_receipt_path=qa,
        receipt_path=receipt_path,
    )

    result = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert "missing_state_explanation" in {finding["code"] for finding in result["findings"]}


def test_visual_qa_must_cover_and_pass_every_current_pdf_page(tmp_path: Path) -> None:
    package, pdf = _package(tmp_path)
    qa = _qa_receipt(tmp_path, pdf)
    value = json.loads(qa.read_text(encoding="utf-8"))
    value["pdfs"][0]["pages"] = value["pdfs"][0]["pages"][:1]
    qa.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    receipt_path = tmp_path / "failed-verification.json"

    verify_publication(
        package,
        _expectation(),
        _cells(),
        visual_qa_receipt_path=qa,
        receipt_path=receipt_path,
    )

    result = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["observed"]["pdf_pages"] == 2
    assert result["observed"]["inspected_pdf_pages"] == 1
    assert "page_set_mismatch" in {finding["code"] for finding in result["findings"]}


def test_recursive_safety_finding_prevents_publication_success(tmp_path: Path) -> None:
    package, pdf = _package(tmp_path)
    qa = _qa_receipt(tmp_path, pdf)
    nested = package / "private" / "events.jsonl"
    nested.parent.mkdir()
    nested.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "failed-verification.json"

    verify_publication(
        package,
        _expectation(),
        _cells(),
        visual_qa_receipt_path=qa,
        receipt_path=receipt_path,
    )

    result = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["checks"]["publication_safety"]["passed"] is False
    assert any(
        finding["check"] == "publication_safety" and finding["code"] == "forbidden_raw_artifact"
        for finding in result["findings"]
    )


def test_expectation_is_internally_exact_and_receipt_cannot_enter_package(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="sum of expected cell requests"):
        PublicationExpectation(
            publication_id="bad-contract",
            route_ids=("route-a",),
            cells=(ExpectedScientificCell("cell-a", "route-a", 2),),
            expected_requests=3,
            pdf_paths=("report.pdf",),
        )

    package, pdf = _package(tmp_path)
    qa = _qa_receipt(tmp_path, pdf)
    with pytest.raises(ValueError, match="outside the publication root"):
        verify_publication(
            package,
            _expectation(),
            _cells(),
            visual_qa_receipt_path=qa,
            receipt_path=package / "verification.json",
        )


def test_scientific_cell_identity_is_compound_route_and_cell(tmp_path: Path) -> None:
    package, pdf = _package(tmp_path)
    qa = _qa_receipt(tmp_path, pdf)
    expectation = PublicationExpectation(
        publication_id="shared-workload-cell-labels",
        route_ids=("route-a", "route-b"),
        cells=(
            ExpectedScientificCell("short_short", "route-a", 2),
            ExpectedScientificCell("short_short", "route-b", 2),
        ),
        expected_requests=4,
        pdf_paths=("report.pdf",),
    )
    cells = [
        ScientificCellResult("short_short", "route-a", 2, "measured"),
        ScientificCellResult("short_short", "route-b", 2, "measured"),
    ]
    receipt_path = tmp_path / "compound-pass.json"

    verify_publication(
        package,
        expectation,
        cells,
        visual_qa_receipt_path=qa,
        receipt_path=receipt_path,
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["passed"] is True

    duplicate_receipt = tmp_path / "compound-duplicate.json"
    verify_publication(
        package,
        expectation,
        [*cells, ScientificCellResult("short_short", "route-a", 2, "measured")],
        visual_qa_receipt_path=qa,
        receipt_path=duplicate_receipt,
    )
    failed = json.loads(duplicate_receipt.read_text(encoding="utf-8"))
    assert failed["passed"] is False
    assert "duplicate_cell" in {finding["code"] for finding in failed["findings"]}
