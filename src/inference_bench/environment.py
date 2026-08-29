from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    """A public build identity plus the private paths needed to verify it."""

    kind: str
    revision: str
    tree_sha256: str
    tree_state: str
    source_root: Path
    package_root: Path
    dependency_lock: Path
    tracked_files: tuple[str, ...]

    def public_dict(self) -> dict[str, str]:
        return {
            "schema_version": "build-identity/v1",
            "kind": self.kind,
            "revision": self.revision,
            "tree_sha256": self.tree_sha256,
            "tree_state": self.tree_state,
        }


def source_tree_state_hash(
    root: Path,
    status: str | None,
    diff: str | None,
    untracked: str | None,
) -> str | None:
    """Bind tracked changes and untracked bytes without publishing local paths."""

    if status is None or diff is None or untracked is None:
        return None
    untracked_digests: list[dict[str, str]] = []
    for relative in sorted(line for line in untracked.splitlines() if line):
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except (OSError, ValueError):
            digest = "unreadable"
        untracked_digests.append(
            {
                "path_sha256": hashlib.sha256(
                    relative.encode("utf-8", errors="surrogateescape")
                ).hexdigest(),
                "content_sha256": digest,
            }
        )
    material = {
        "status_sha256": hashlib.sha256(
            status.encode("utf-8", errors="surrogateescape")
        ).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(
            diff.encode("utf-8", errors="surrogateescape")
        ).hexdigest(),
        "untracked": untracked_digests,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def find_source_root(module_path: Path | None = None) -> Path:
    """Find a source archive root, or fall back to the installed package directory."""

    module = (module_path or Path(__file__)).resolve()
    for candidate in module.parents:
        source_packages = (
            candidate / "src" / "inference_bench",
            candidate / "inference_bench",
        )
        if (candidate / "pyproject.toml").is_file() and any(
            module.is_relative_to(package.resolve()) for package in source_packages
        ):
            return candidate.resolve()
    return module.parent.resolve()


def _package_root(source_root: Path) -> Path:
    candidates = (
        source_root / "src" / "inference_bench",
        source_root / "inference_bench",
        source_root,
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "__init__.py").is_file():
            return candidate.resolve()
    raise RuntimeError("cannot locate the installed inference_bench package")


def _dependency_lock(source_root: Path, package_root: Path) -> Path:
    for candidate in (source_root / "requirements.lock", package_root / "requirements.lock"):
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "benchmark build requires requirements.lock in the source archive or installed package"
    )


def _project_version(source_root: Path) -> str:
    pyproject = source_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            value = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
        except (KeyError, OSError, tomllib.TOMLDecodeError) as exc:
            raise RuntimeError("cannot read benchmark version from pyproject.toml") from exc
        if not isinstance(value, str) or not value:
            raise RuntimeError("benchmark version in pyproject.toml is invalid")
        return value
    try:
        return importlib.metadata.version("inference-endpoint-benchmark")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("benchmark package must be installed as a distribution") from exc


def _git(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _content_addressed_identity(source_root: Path, package_root: Path) -> BuildIdentity:
    lock = _dependency_lock(source_root, package_root)
    files: list[dict[str, str]] = []
    tracked_files: list[str] = []
    for path in sorted(candidate for candidate in package_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.as_posix() == "requirements.lock":
            continue
        label = f"inference_bench/{relative.as_posix()}"
        files.append({"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        try:
            tracked_files.append(path.relative_to(source_root).as_posix())
        except ValueError:
            tracked_files.append(label)
    if not files:
        raise RuntimeError("installed benchmark package contains no identity-bearing files")
    material: dict[str, Any] = {
        "schema_version": "content-addressed-build/v1",
        "distribution": "inference-endpoint-benchmark",
        "version": _project_version(source_root),
        "files": files,
        "requirements_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    try:
        tracked_files.append(lock.relative_to(source_root).as_posix())
    except ValueError:
        tracked_files.append("requirements.lock")
    return BuildIdentity(
        kind="content-addressed-package",
        revision=digest,
        tree_sha256=digest,
        tree_state="content-addressed",
        source_root=source_root,
        package_root=package_root,
        dependency_lock=lock,
        tracked_files=tuple(sorted(set(tracked_files))),
    )


def resolve_build_identity(
    source_root: Path | None = None,
    *,
    output_dir: Path | None = None,
) -> BuildIdentity:
    """Resolve a clean Git commit or an equivalent content-addressed package build.

    Source archives and installed wheels intentionally have no ``.git`` directory. In that case,
    the exact installed package bytes, distribution version, and packaged dependency lock form the
    revision. A real but unusable Git checkout still fails closed instead of silently downgrading.
    """

    root = (source_root or find_source_root()).resolve()
    git_root = _git(root, "rev-parse", "--show-toplevel")
    use_content_identity = git_root is None or Path(git_root).resolve() != root
    if git_root is None and (root / ".git").exists():
        raise RuntimeError("cannot read benchmark Git source identity")
    if use_content_identity:
        package_root = _package_root(root)
        if output_dir is not None and output_dir.resolve().is_relative_to(package_root):
            raise ValueError("run directory cannot be inside the installed benchmark package")
        return _content_addressed_identity(root, package_root)
    assert git_root is not None

    pathspec = ["--", "."]
    if output_dir is not None:
        try:
            output_relative = output_dir.resolve().relative_to(root).as_posix()
        except ValueError:
            output_relative = None
        if output_relative and output_relative != ".":
            pathspec.append(f":(exclude){output_relative}/**")
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", *pathspec)
    diff = _git(root, "diff", "--binary", "HEAD", *pathspec)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", *pathspec)
    tracked = _git(root, "ls-files")
    if None in {commit, status, diff, untracked, tracked}:
        raise RuntimeError("cannot bind benchmark Git source tree state")
    assert commit is not None and status is not None and diff is not None
    assert untracked is not None and tracked is not None
    if status:
        raise RuntimeError("live runs and reports require clean committed source")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise RuntimeError("benchmark Git source revision is invalid")
    tree_hash = source_tree_state_hash(root, status, diff, untracked)
    if tree_hash is None:
        raise RuntimeError("cannot hash benchmark Git source tree state")
    return BuildIdentity(
        kind="git-commit",
        revision=commit,
        tree_sha256=tree_hash,
        tree_state="clean",
        source_root=root,
        package_root=root,
        dependency_lock=_dependency_lock(root, root),
        tracked_files=tuple(line for line in tracked.splitlines() if line),
    )


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
