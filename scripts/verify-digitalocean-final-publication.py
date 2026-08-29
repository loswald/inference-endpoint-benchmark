from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from inference_bench.digitalocean_final import (
    CAPACITY_SOURCE,
    ENDPOINT_LABELS,
    ENDPOINTS,
    FIXED_RATE_SOURCE,
    _validate_variation_tables,
)
from inference_bench.publication_manifest import publication_manifest
from inference_bench.publication_safety import scan_publication

FIXED_RATE_SHAPES = {"short_short", "input32k_short", "short_long", "mixed"}
EXPECTED_FIGURES = {
    "adaptive-load-input100k_short.png",
    "adaptive-load-mixed.png",
    "adaptive-load-short_long.png",
    "adaptive-load-short_short.png",
    "fixed-rate-stability-matrix.png",
    "prompt-reuse-effect.png",
    "six-hour-panel-outcomes.png",
    "six-hour-reliability-forest.png",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _exact_cells(
    rows: list[dict[str, str]], source_id: str, shapes: set[str]
) -> set[tuple[str, str]]:
    return {
        (row["endpoint_id"], row["shape"])
        for row in rows
        if row.get("source_id") == source_id
    }


def verify_publication(root: str | Path) -> dict[str, object]:
    from PIL import Image
    from pypdf import PdfReader

    package = Path(root).resolve()
    data = package / "data"
    stored_manifest = json.loads(
        (package / "publication-manifest.json").read_text(encoding="utf-8")
    )
    assert stored_manifest == publication_manifest(package), "publication manifest mismatch"

    scan_result = scan_publication(package)
    assert scan_result["passed"], "recursive public-safety scan failed"
    stored_scan = json.loads((package / "public-safety-scan.json").read_text(encoding="utf-8"))
    assert stored_scan == scan_result, "stored public-safety receipt is stale"

    endpoints = set(ENDPOINTS)
    capacity = _read_csv(data / "capacity-summary.csv")
    current_capacity = [row for row in capacity if row.get("source_id") == CAPACITY_SOURCE]
    current_capacity_cells = {(row["endpoint_id"], row["shape"]) for row in current_capacity}
    assert len(current_capacity) == len(current_capacity_cells) == 44
    for endpoint in endpoints:
        endpoint_shapes = {
            row["shape"] for row in current_capacity if row["endpoint_id"] == endpoint
        }
        assert {"short_short", "short_long", "mixed"} <= endpoint_shapes
        assert len(endpoint_shapes & {"input100k_short", "input32k_short"}) == 1
        assert len(endpoint_shapes) == 4

    fixed_rate = _read_csv(data / "soak-cell-summary.csv")
    expected_fixed = {(endpoint, shape) for endpoint in endpoints for shape in FIXED_RATE_SHAPES}
    assert _exact_cells(fixed_rate, FIXED_RATE_SOURCE, FIXED_RATE_SHAPES) == expected_fixed

    panel = _read_csv(data / "variation-panel-summary.csv")
    across = _read_csv(data / "variation-across-panel-summary.csv")
    paired = _read_csv(data / "variation-paired-cache-effects.csv")
    _validate_variation_tables(panel, across, paired)

    coverage = _read_csv(data / "coverage-matrix.csv")
    coverage_cells = {(row["endpoint_id"], row["coverage_dimension"]) for row in coverage}
    assert len(coverage) == len(coverage_cells) == 176, "coverage matrix is not exact"
    assert {row["endpoint_id"] for row in coverage} == endpoints
    assert all(row.get("status", "").strip() for row in coverage), "blank coverage state"

    limits = _read_csv(data / "observed-limits.csv")
    limit_cells = {(row["endpoint_id"], row["dimension"]) for row in limits}
    assert len(limits) == len(limit_cells) == 66, "endpoint limit matrix is not exact"
    assert {row["endpoint_id"] for row in limits} == endpoints
    assert all(row.get("finding", "").strip() for row in limits), "blank limit finding"

    pdf_path = package / "digitalocean-inference-endpoints-technical-benchmark-2026-08-29.pdf"
    reader = PdfReader(pdf_path)
    assert len(reader.pages) == 28, "unexpected PDF page count"
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert all(label in pdf_text for label in ENDPOINT_LABELS.values()), "endpoint missing from PDF"

    figure_paths = {path.name: path for path in (package / "figures").glob("*.png")}
    assert set(figure_paths) == EXPECTED_FIGURES, "figure inventory mismatch"
    for path in figure_paths.values():
        with Image.open(path) as image:
            assert image.width >= 800 and image.height >= 400, f"undersized figure: {path.name}"

    return {
        "schema_version": "digitalocean-final-publication-verification/v1",
        "passed": True,
        "endpoint_count": len(endpoints),
        "capacity_cells": len(current_capacity),
        "fixed_rate_cells": len(expected_fixed),
        "variation_panel_rows": len(panel),
        "variation_across_panel_rows": len(across),
        "variation_paired_rows": len(paired),
        "coverage_cells": len(coverage),
        "limit_cells": len(limits),
        "pdf_pages": len(reader.pages),
        "figures": len(figure_paths),
        "files_scanned": scan_result["files_scanned"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the final DigitalOcean public package.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_publication(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
