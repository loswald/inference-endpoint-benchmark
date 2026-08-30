"""Provider-neutral, fail-closed verification for publication packages.

Creating a PDF is not evidence that a benchmark publication is complete or safe.  This
module composes the existing recursive safety scan and deterministic byte manifest with
three additional contracts:

* the exact routes, scientific cells, and request counts expected in the publication;
* explicit, non-blank scientific states for every expected cell; and
* a visual-QA receipt binding every PDF page to an inspected page render.

The verification receipt is deliberately written outside the publication directory.  That
lets it bind the final publication manifest without making either artifact hash itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pypdf import PdfReader

from .models import canonical_json, sha256_json
from .publication_manifest import DEFAULT_MANIFEST_NAME, build_publication_manifest
from .publication_safety import scan_publication
from .reporting.models import EvidenceState

EXPECTATION_SCHEMA = "publication-expectation/v1"
VISUAL_QA_SCHEMA = "pdf-visual-qa-receipt/v1"
VERIFICATION_SCHEMA = "publication-verification-receipt/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REJECTED_STATES = frozenset({EvidenceState.NOT_RUN.value, EvidenceState.UNKNOWN.value})
_EXPLANATION_REQUIRED_STATES = frozenset(
    {
        EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR.value,
        EvidenceState.TRANSPORT_GATED.value,
        EvidenceState.UNSUPPORTED.value,
        EvidenceState.CENSORED.value,
    }
)
_KNOWN_STATES = frozenset(state.value for state in EvidenceState)


@dataclass(frozen=True)
class ExpectedScientificCell:
    """One exact cell and the number of request outcomes it must contain."""

    cell_id: str
    route_id: str
    expected_requests: int

    def __post_init__(self) -> None:
        if not self.cell_id.strip() or not self.route_id.strip():
            raise ValueError("expected cell_id and route_id must be non-empty")
        if self.expected_requests < 0:
            raise ValueError("expected_requests cannot be negative")


@dataclass(frozen=True)
class PublicationExpectation:
    """Exact coverage and PDF contract for one publication package."""

    publication_id: str
    route_ids: tuple[str, ...]
    cells: tuple[ExpectedScientificCell, ...]
    expected_requests: int
    pdf_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.publication_id.strip():
            raise ValueError("publication_id must be non-empty")
        _require_unique_nonempty(self.route_ids, "route_ids")
        _require_unique_nonempty(self.pdf_paths, "pdf_paths")
        for path in self.pdf_paths:
            _safe_relative_path(path, "pdf path")
            if Path(path).suffix.casefold() != ".pdf":
                raise ValueError(f"expected PDF path must end in .pdf: {path}")
        cell_keys = [(cell.route_id, cell.cell_id) for cell in self.cells]
        seen_cell_keys: set[tuple[str, str]] = set()
        duplicate_cell_keys: set[tuple[str, str]] = set()
        for key in cell_keys:
            if key in seen_cell_keys:
                duplicate_cell_keys.add(key)
            seen_cell_keys.add(key)
        if duplicate_cell_keys:
            rendered = ", ".join(
                f"{route_id}/{cell_id}" for route_id, cell_id in sorted(duplicate_cell_keys)
            )
            raise ValueError(f"duplicate route/cell identities: {rendered}")
        expected_routes = set(self.route_ids)
        cell_routes = {cell.route_id for cell in self.cells}
        undeclared = sorted(cell_routes - expected_routes)
        unused = sorted(expected_routes - cell_routes)
        if undeclared:
            raise ValueError("cells reference undeclared routes: " + ", ".join(undeclared))
        if unused:
            raise ValueError("routes have no expected cells: " + ", ".join(unused))
        if self.expected_requests < 0:
            raise ValueError("expected_requests cannot be negative")
        cell_request_sum = sum(cell.expected_requests for cell in self.cells)
        if cell_request_sum != self.expected_requests:
            raise ValueError(
                "expected_requests must equal the sum of expected cell requests "
                f"({cell_request_sum})"
            )

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPECTATION_SCHEMA,
            "publication_id": self.publication_id,
            "route_ids": list(self.route_ids),
            "cells": [asdict(cell) for cell in self.cells],
            "expected_requests": self.expected_requests,
            "pdf_paths": list(self.pdf_paths),
        }


@dataclass(frozen=True)
class ScientificCellResult:
    """Observed publication state for one contracted scientific cell."""

    cell_id: str
    route_id: str
    observed_requests: int
    state: str
    explanation: str = ""

    def __post_init__(self) -> None:
        if not self.cell_id.strip() or not self.route_id.strip():
            raise ValueError("scientific cell_id and route_id must be non-empty")
        if self.observed_requests < 0:
            raise ValueError("observed_requests cannot be negative")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_publication(
    publication_root: str | Path,
    expectation: PublicationExpectation,
    scientific_cells: list[ScientificCellResult],
    *,
    visual_qa_receipt_path: str | Path,
    receipt_path: str | Path,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> Path:
    """Verify a package and atomically write a deterministic pass/fail receipt.

    Scientific or publication failures are recorded in the receipt instead of raised.  Invalid
    API usage (for example, placing the receipt inside the package it verifies) raises before any
    output is written.
    """

    package = Path(publication_root).resolve()
    if not package.is_dir():
        raise ValueError("publication root must be an existing directory")
    destination = Path(receipt_path).resolve()
    if destination == package or destination.is_relative_to(package):
        raise ValueError("verification receipt must be outside the publication root")

    findings: list[dict[str, str]] = []
    coverage = _verify_scientific_coverage(expectation, scientific_cells, findings)
    pdf_qa, qa_hashes = _verify_visual_qa(
        package,
        expectation,
        Path(visual_qa_receipt_path).resolve(),
        findings,
    )

    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    try:
        manifest_path = build_publication_manifest(package, manifest_name=manifest_name)
        manifest_sha256 = _sha256_file(manifest_path)
        manifest_check = {"passed": True, "file_count": _manifest_file_count(manifest_path)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _finding(
            findings,
            "deterministic_manifest",
            "manifest_build_failed",
            f"manifest could not be built ({type(exc).__name__})",
        )
        manifest_check = {"passed": False, "file_count": 0}

    try:
        safety = scan_publication(package)
        safety_passed = safety.get("passed") is True
        for item in safety.get("findings", []):
            if isinstance(item, dict):
                _finding(
                    findings,
                    "publication_safety",
                    str(item.get("rule") or "safety_finding"),
                    str(item.get("file") or "publication artifact"),
                )
    except Exception as exc:  # Safety scanning is a fail-closed external-artifact boundary.
        safety = {
            "schema_version": "public-artifact-safety-scan/v1",
            "passed": False,
            "findings": [],
        }
        safety_passed = False
        _finding(
            findings,
            "publication_safety",
            "safety_scan_failed",
            f"publication could not be scanned ({type(exc).__name__})",
        )

    expectation_dict = expectation.public_dict()
    scientific_dicts = [
        cell.public_dict()
        for cell in sorted(scientific_cells, key=lambda item: (item.route_id, item.cell_id))
    ]
    hashes: dict[str, Any] = {
        "expectation_sha256": sha256_json(expectation_dict),
        "scientific_cells_sha256": sha256_json(scientific_dicts),
        "safety_scan_sha256": sha256_json(safety),
        "visual_qa_receipt_sha256": qa_hashes["receipt"],
        "publication_manifest_sha256": manifest_sha256,
        "pdf_sha256": qa_hashes["pdfs"],
        "page_render_sha256": qa_hashes["renders"],
    }
    checks = {
        "expectation_contract": {"passed": True},
        "scientific_coverage": coverage,
        "pdf_visual_qa": pdf_qa,
        "deterministic_manifest": manifest_check,
        "publication_safety": {
            "passed": safety_passed,
            "files_scanned": int(safety.get("files_scanned") or 0),
        },
    }
    passed = not findings and all(check["passed"] is True for check in checks.values())
    identity = sha256_json(
        {
            "schema_version": VERIFICATION_SCHEMA,
            "publication_id": expectation.publication_id,
            "checks": checks,
            "hashes": hashes,
            "findings": findings,
        }
    )
    receipt = {
        "schema_version": VERIFICATION_SCHEMA,
        "publication_id": expectation.publication_id,
        "verification_id_sha256": identity,
        "passed": passed,
        "contract": expectation_dict,
        "expected": {
            "routes": len(expectation.route_ids),
            "cells": len(expectation.cells),
            "requests": expectation.expected_requests,
            "pdfs": len(expectation.pdf_paths),
        },
        "observed": {
            "routes": coverage["observed_routes"],
            "cells": coverage["observed_cells"],
            "requests": coverage["observed_requests"],
            "pdfs": pdf_qa["observed_pdfs"],
            "pdf_pages": pdf_qa["observed_pages"],
            "inspected_pdf_pages": pdf_qa["inspected_pages"],
        },
        "checks": checks,
        "hashes": hashes,
        "findings": findings,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(canonical_json(receipt) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination


def _verify_scientific_coverage(
    expectation: PublicationExpectation,
    scientific_cells: list[ScientificCellResult],
    findings: list[dict[str, str]],
) -> dict[str, Any]:
    observed_by_key: dict[tuple[str, str], ScientificCellResult] = {}
    for cell in scientific_cells:
        key = (cell.route_id, cell.cell_id)
        if key in observed_by_key:
            _finding(
                findings,
                "scientific_coverage",
                "duplicate_cell",
                f"scientific cell {cell.route_id!r}/{cell.cell_id!r} appears more than once",
            )
            continue
        observed_by_key[key] = cell

    expected_by_key = {(cell.route_id, cell.cell_id): cell for cell in expectation.cells}
    for route_id, cell_id in sorted(expected_by_key.keys() - observed_by_key.keys()):
        _finding(
            findings,
            "scientific_coverage",
            "missing_cell",
            f"expected scientific cell {route_id!r}/{cell_id!r} is absent",
        )
    for route_id, cell_id in sorted(observed_by_key.keys() - expected_by_key.keys()):
        _finding(
            findings,
            "scientific_coverage",
            "unexpected_cell",
            f"uncontracted scientific cell {route_id!r}/{cell_id!r} is present",
        )

    for route_id, cell_id in sorted(expected_by_key.keys() & observed_by_key.keys()):
        key = (route_id, cell_id)
        expected = expected_by_key[key]
        observed = observed_by_key[key]
        if observed.observed_requests != expected.expected_requests:
            _finding(
                findings,
                "scientific_coverage",
                "request_count_mismatch",
                f"cell {route_id!r}/{cell_id!r} has "
                f"{observed.observed_requests} request outcomes; "
                f"expected {expected.expected_requests}",
            )
        state = observed.state.strip()
        if not state:
            _finding(
                findings,
                "scientific_coverage",
                "blank_state",
                f"cell {route_id!r}/{cell_id!r} has no scientific state",
            )
        elif state not in _KNOWN_STATES:
            _finding(
                findings,
                "scientific_coverage",
                "unknown_state",
                f"cell {route_id!r}/{cell_id!r} has unrecognized scientific state {state!r}",
            )
        elif state in _REJECTED_STATES:
            _finding(
                findings,
                "scientific_coverage",
                "unpublishable_state",
                f"cell {route_id!r}/{cell_id!r} remains {state!r}",
            )
        elif state in _EXPLANATION_REQUIRED_STATES and not observed.explanation.strip():
            _finding(
                findings,
                "scientific_coverage",
                "missing_state_explanation",
                f"cell {route_id!r}/{cell_id!r} state {state!r} "
                "requires a plain-language explanation",
            )

    observed_routes = {cell.route_id for cell in scientific_cells}
    expected_routes = set(expectation.route_ids)
    for route_id in sorted(expected_routes - observed_routes):
        _finding(
            findings,
            "scientific_coverage",
            "missing_route",
            f"expected route {route_id!r} has no scientific result",
        )
    for route_id in sorted(observed_routes - expected_routes):
        _finding(
            findings,
            "scientific_coverage",
            "unexpected_route",
            f"uncontracted route {route_id!r} has a scientific result",
        )

    observed_requests = sum(cell.observed_requests for cell in scientific_cells)
    if observed_requests != expectation.expected_requests:
        _finding(
            findings,
            "scientific_coverage",
            "total_request_count_mismatch",
            f"scientific results contain {observed_requests} request outcomes; "
            f"expected {expectation.expected_requests}",
        )
    coverage_findings = [item for item in findings if item["check"] == "scientific_coverage"]
    return {
        "passed": not coverage_findings,
        "observed_routes": len(observed_routes),
        "observed_cells": len(observed_by_key),
        "observed_requests": observed_requests,
    }


def _verify_visual_qa(
    package: Path,
    expectation: PublicationExpectation,
    receipt_path: Path,
    findings: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    hashes: dict[str, Any] = {"receipt": None, "pdfs": {}, "renders": {}}
    qa: dict[str, Any] | None = None
    try:
        raw_receipt = receipt_path.read_bytes()
        hashes["receipt"] = hashlib.sha256(raw_receipt).hexdigest()
        parsed = json.loads(raw_receipt)
        if not isinstance(parsed, dict):
            raise ValueError("visual-QA receipt must be a JSON object")
        qa = parsed
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _finding(
            findings,
            "pdf_visual_qa",
            "invalid_qa_receipt",
            f"visual-QA receipt could not be read or decoded ({type(exc).__name__})",
        )

    actual_pdf_paths = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".pdf"
    )
    expected_pdf_paths = sorted(expectation.pdf_paths)
    if actual_pdf_paths != expected_pdf_paths:
        _finding(
            findings,
            "pdf_visual_qa",
            "pdf_contract_mismatch",
            f"package PDFs {actual_pdf_paths!r} do not equal expected PDFs {expected_pdf_paths!r}",
        )

    observed_pages = 0
    inspected_pages = 0
    pdf_page_counts: dict[str, int] = {}
    for relative in actual_pdf_paths:
        pdf_path = package / Path(PurePosixPath(relative))
        hashes["pdfs"][relative] = _sha256_file(pdf_path)
        try:
            page_count = len(PdfReader(pdf_path).pages)
        except Exception as exc:  # pypdf exposes several parser-specific exception types
            _finding(
                findings,
                "pdf_visual_qa",
                "unreadable_pdf",
                f"cannot read {relative!r} ({type(exc).__name__})",
            )
            continue
        pdf_page_counts[relative] = page_count
        observed_pages += page_count

    if qa is not None:
        if qa.get("schema_version") != VISUAL_QA_SCHEMA:
            _finding(
                findings,
                "pdf_visual_qa",
                "qa_schema_mismatch",
                f"visual-QA receipt must use {VISUAL_QA_SCHEMA}",
            )
        if not isinstance(qa.get("inspector"), str) or not qa["inspector"].strip():
            _finding(
                findings,
                "pdf_visual_qa",
                "missing_inspector",
                "visual-QA receipt must identify its inspector",
            )
        entries = qa.get("pdfs")
        if not isinstance(entries, list):
            _finding(
                findings,
                "pdf_visual_qa",
                "invalid_pdf_entries",
                "visual-QA receipt pdfs must be a list",
            )
            entries = []
        by_path: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                _finding(
                    findings,
                    "pdf_visual_qa",
                    "invalid_pdf_entry",
                    "each visual-QA PDF entry needs a relative path",
                )
                continue
            relative = entry["path"]
            try:
                _safe_relative_path(relative, "visual-QA PDF path")
            except ValueError as exc:
                _finding(findings, "pdf_visual_qa", "invalid_pdf_path", str(exc))
                continue
            if relative in by_path:
                _finding(
                    findings,
                    "pdf_visual_qa",
                    "duplicate_pdf_entry",
                    f"visual-QA receipt repeats {relative!r}",
                )
                continue
            by_path[relative] = entry
        if sorted(by_path) != actual_pdf_paths:
            _finding(
                findings,
                "pdf_visual_qa",
                "qa_pdf_set_mismatch",
                f"visual-QA PDFs {sorted(by_path)!r} do not equal "
                f"package PDFs {actual_pdf_paths!r}",
            )
        for relative in actual_pdf_paths:
            entry = by_path.get(relative)
            pdf_sha = hashes["pdfs"][relative]
            if entry is None:
                continue
            if entry.get("sha256") != pdf_sha:
                _finding(
                    findings,
                    "pdf_visual_qa",
                    "pdf_hash_mismatch",
                    f"visual-QA receipt does not bind the current bytes of {relative!r}",
                )
            page_count = pdf_page_counts.get(relative)
            if page_count is None:
                continue
            if entry.get("page_count") != page_count:
                _finding(
                    findings,
                    "pdf_visual_qa",
                    "pdf_page_count_mismatch",
                    f"{relative!r} has {page_count} pages",
                )
            inspected_pages += _verify_pdf_pages(
                relative,
                page_count,
                entry.get("pages"),
                receipt_path.parent,
                findings,
                hashes["renders"],
            )

    qa_findings = [item for item in findings if item["check"] == "pdf_visual_qa"]
    return (
        {
            "passed": not qa_findings,
            "observed_pdfs": len(actual_pdf_paths),
            "observed_pages": observed_pages,
            "inspected_pages": inspected_pages,
        },
        hashes,
    )


def _verify_pdf_pages(
    pdf_relative: str,
    page_count: int,
    raw_pages: object,
    qa_root: Path,
    findings: list[dict[str, str]],
    render_hashes: dict[str, str],
) -> int:
    if not isinstance(raw_pages, list):
        _finding(
            findings,
            "pdf_visual_qa",
            "invalid_page_entries",
            f"{pdf_relative!r} visual-QA pages must be a list",
        )
        return 0
    by_number: dict[int, dict[str, Any]] = {}
    for page in raw_pages:
        if not isinstance(page, dict) or not isinstance(page.get("page_number"), int):
            _finding(
                findings,
                "pdf_visual_qa",
                "invalid_page_entry",
                f"{pdf_relative!r} has a page entry without an integer page_number",
            )
            continue
        page_number = page["page_number"]
        if page_number in by_number:
            _finding(
                findings,
                "pdf_visual_qa",
                "duplicate_page_entry",
                f"{pdf_relative!r} repeats page {page_number}",
            )
            continue
        by_number[page_number] = page
    expected_numbers = set(range(1, page_count + 1))
    if set(by_number) != expected_numbers:
        _finding(
            findings,
            "pdf_visual_qa",
            "page_set_mismatch",
            f"{pdf_relative!r} QA pages {sorted(by_number)!r} do not equal "
            f"rendered pages {sorted(expected_numbers)!r}",
        )

    inspected = 0
    for page_number in sorted(expected_numbers & set(by_number)):
        page = by_number[page_number]
        if page.get("inspected") is not True:
            _finding(
                findings,
                "pdf_visual_qa",
                "page_not_inspected",
                f"{pdf_relative!r} page {page_number} was not inspected",
            )
        elif page.get("passed") is not True:
            _finding(
                findings,
                "pdf_visual_qa",
                "page_visual_qa_failed",
                f"{pdf_relative!r} page {page_number} did not pass visual QA",
            )
        else:
            inspected += 1
        render_path = page.get("render_path")
        render_sha = page.get("render_sha256")
        if not isinstance(render_path, str):
            _finding(
                findings,
                "pdf_visual_qa",
                "missing_page_render",
                f"{pdf_relative!r} page {page_number} has no render path",
            )
            continue
        try:
            safe_render = _safe_relative_path(render_path, "page render path")
        except ValueError as exc:
            _finding(findings, "pdf_visual_qa", "invalid_page_render_path", str(exc))
            continue
        resolved_render = qa_root / Path(PurePosixPath(safe_render))
        if not resolved_render.is_file() or resolved_render.is_symlink():
            _finding(
                findings,
                "pdf_visual_qa",
                "missing_page_render",
                f"{pdf_relative!r} page {page_number} render is absent or is a symlink",
            )
            continue
        observed_sha = _sha256_file(resolved_render)
        render_hashes[f"{pdf_relative}#page-{page_number}"] = observed_sha
        if not isinstance(render_sha, str) or not _SHA256_RE.fullmatch(render_sha):
            _finding(
                findings,
                "pdf_visual_qa",
                "invalid_page_render_hash",
                f"{pdf_relative!r} page {page_number} has no valid SHA-256",
            )
        elif render_sha != observed_sha:
            _finding(
                findings,
                "pdf_visual_qa",
                "page_render_hash_mismatch",
                f"{pdf_relative!r} page {page_number} render hash does not match",
            )
    return inspected


def _safe_relative_path(value: str, label: str) -> str:
    candidate = PurePosixPath(str(value).replace("\\", "/"))
    if (
        not value
        or candidate.is_absolute()
        or candidate.name in {"", ".", ".."}
        or candidate.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return candidate.as_posix()


def _require_unique_nonempty(values: tuple[str, ...] | list[str], label: str) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty values")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file_count(path: Path) -> int:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    files = parsed.get("files")
    if not isinstance(files, list):
        raise ValueError("publication manifest has no files list")
    return len(files)


def _finding(findings: list[dict[str, str]], check: str, code: str, detail: str) -> None:
    findings.append({"check": check, "code": code, "detail": detail})
