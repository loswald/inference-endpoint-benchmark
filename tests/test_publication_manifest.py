from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inference_bench.publication_manifest import build_publication_manifest


def test_publication_manifest_is_complete_self_excluding_and_deterministic(
    tmp_path: Path,
) -> None:
    package = tmp_path / "public-package"
    (package / "data").mkdir(parents=True)
    (package / "README.md").write_bytes(b"benchmark\r\n")
    (package / "data" / "measurements.csv").write_bytes(b"model,n\na,2\n")
    (package / "figure.bin").write_bytes(bytes([0, 1, 2, 255]))
    # A stale manifest must never hash itself into the next build.
    (package / "publication-manifest.json").write_text("stale", encoding="utf-8")
    (package / ".publication-manifest.json.tmp").write_text("stale-temp", encoding="utf-8")

    first = build_publication_manifest(package)
    first_bytes = first.read_bytes()
    parsed = json.loads(first_bytes)
    assert [entry["path"] for entry in parsed["files"]] == [
        "README.md",
        "data/measurements.csv",
        "figure.bin",
    ]
    assert all(entry["path"] != "publication-manifest.json" for entry in parsed["files"])
    expected = {
        path.relative_to(package).as_posix(): {
            "bytes": len(path.read_bytes()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (
            package / "README.md",
            package / "data" / "measurements.csv",
            package / "figure.bin",
        )
    }
    assert {
        entry["path"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        for entry in parsed["files"]
    } == expected

    second = build_publication_manifest(package)
    assert second.read_bytes() == first_bytes
