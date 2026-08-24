from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path


def locked_distribution_versions(lock_path: Path) -> dict[str, str]:
    """Verify and return only the public, hash-bound runtime dependency closure."""

    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        lock_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(f"unsupported requirements.lock line {line_number}")
        raw_name, version = line.split("==", 1)
        name = re.sub(r"[-_.]+", "-", raw_name.strip()).casefold()
        if not name or not version or name in expected:
            raise RuntimeError(f"invalid or duplicate requirements.lock line {line_number}")
        expected[name] = version
    if not expected:
        raise RuntimeError("requirements.lock contains no pinned distributions")
    observed: dict[str, str] = {}
    for name, expected_version in sorted(expected.items()):
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"locked distribution is not installed: {name}") from exc
        if installed != expected_version:
            raise RuntimeError(
                f"locked distribution version mismatch: {name}={installed}, "
                f"expected {expected_version}"
            )
        observed[name] = installed
    try:
        observed["inference-endpoint-benchmark"] = importlib.metadata.version(
            "inference-endpoint-benchmark"
        )
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("benchmark package must be installed as a distribution") from exc
    return dict(sorted(observed.items()))


def validate_run_directory_separation(
    source_root: Path, run_directory: Path, tracked_files: list[str]
) -> None:
    """Prevent an artifact exclusion from hiding source or repository metadata changes."""

    root = source_root.resolve()
    run = run_directory.resolve()
    critical = (
        root,
        root / ".git",
        root / "pyproject.toml",
        root / "requirements.lock",
        Path(__file__).resolve(),
    )
    if any(path.resolve().is_relative_to(run) for path in critical):
        raise ValueError("run directory cannot contain source or repository identity files")
    git_directory = (root / ".git").resolve()
    if run.is_relative_to(git_directory):
        raise ValueError("run directory cannot be inside repository metadata")
    try:
        relative = run.relative_to(root)
    except ValueError:
        return
    if not relative.parts:
        raise ValueError("run directory must be separate from the source root")
    tracked_top_levels = {
        Path(value).parts[0] for value in tracked_files if value.strip() and Path(value).parts
    }
    if relative.parts[0] in tracked_top_levels:
        raise ValueError(
            "run directory must use a dedicated non-source top-level path with no tracked files"
        )
