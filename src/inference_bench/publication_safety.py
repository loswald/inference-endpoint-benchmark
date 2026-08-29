from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".jsonl", ".log", ".wal", ".shm"}
FORBIDDEN_FILENAMES = {"ledger.sqlite3", "events.jsonl", "worker.log", "lock"}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".py"}
FORBIDDEN_PATTERNS = {
    "credentials": re.compile(r"(?i)(authorization\s*[:=]\s*bearer|api[_-]?key\s*[:=])"),
    "raw_request_identifier": re.compile(r"(?i)\b(request_id|logical_id|reservation_id)\b"),
    "raw_content_field": re.compile(
        r"(?i)\b(prompt_text|prompt_content|response_body|response_content|raw_headers?)\b"
    ),
    "private_path": re.compile(
        r"(?i)([A-Z]:\\Users\\|/home/sqwish/private|/home/sqwish/deployments)"
    ),
    "excluded_route": re.compile(r"(?i)\barcee\b"),
}


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    metadata = "\n".join(str(value) for value in (reader.metadata or {}).values())
    pages = "\n".join(page.extract_text() or "" for page in reader.pages)
    return metadata + "\n" + pages


def scan_publication(root: str | Path) -> dict[str, object]:
    """Recursively scan a publication directory for private or excluded material."""
    package = Path(root).resolve()
    findings: list[dict[str, str]] = []
    files = sorted(path for path in package.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(package).as_posix()
        if path.name.lower() in FORBIDDEN_FILENAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"file": relative, "rule": "forbidden_raw_artifact"})
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif path.suffix.lower() == ".pdf":
            text = _pdf_text(path)
        else:
            continue
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                findings.append({"file": relative, "rule": name})
    return {
        "schema_version": "public-artifact-safety-scan/v1",
        "root_name": package.name,
        "files_scanned": len(files),
        "passed": not findings,
        "findings": findings,
    }
