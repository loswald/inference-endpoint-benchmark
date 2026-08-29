from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MANIFEST_NAME = "publication-manifest.json"


def publication_manifest(
    root: str | Path, *, manifest_name: str = DEFAULT_MANIFEST_NAME
) -> dict[str, Any]:
    """Return a deterministic byte manifest for a public artifact directory.

    Paths are relative POSIX paths. The manifest itself and its temporary write path are excluded,
    making repeated builds byte-for-byte stable when the other package files are unchanged.
    """

    package = Path(root).resolve()
    if not package.is_dir():
        raise ValueError("publication root must be an existing directory")
    relative_manifest = _safe_manifest_path(manifest_name)
    manifest_path = PurePosixPath(relative_manifest)
    temporary_path = manifest_path.parent / f".{manifest_path.name}.tmp"
    excluded = {relative_manifest, temporary_path.as_posix()}
    files: list[dict[str, Any]] = []
    for path in sorted(package.rglob("*"), key=lambda item: item.relative_to(package).as_posix()):
        relative = path.relative_to(package).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"publication package cannot contain symlink: {relative}")
        if not path.is_file():
            continue
        content = path.read_bytes()
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    return {
        "schema_version": "publication-manifest/v1",
        "hash_algorithm": "sha256",
        "path_format": "relative-posix",
        "files": files,
    }


def build_publication_manifest(
    root: str | Path, *, manifest_name: str = DEFAULT_MANIFEST_NAME
) -> Path:
    """Atomically write and return a deterministic publication manifest."""

    package = Path(root).resolve()
    relative_manifest = _safe_manifest_path(manifest_name)
    result = publication_manifest(package, manifest_name=relative_manifest)
    destination = package / Path(PurePosixPath(relative_manifest))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination


def _safe_manifest_path(value: str) -> str:
    candidate = PurePosixPath(str(value).replace("\\", "/"))
    if (
        not value
        or candidate.is_absolute()
        or candidate.name in {"", ".", ".."}
        or candidate.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("manifest_name must be a safe relative path")
    return candidate.as_posix()
