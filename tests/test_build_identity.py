from __future__ import annotations

import re
from pathlib import Path

from inference_bench.cli import _runtime_manifest
from inference_bench.environment import find_source_root, resolve_build_identity
from inference_bench.plan import build_plan
from inference_bench.report import _report_source_snapshot


def _source_archive(root: Path) -> Path:
    package = root / "src" / "inference_bench"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[project]
name = "inference-endpoint-benchmark"
version = "9.8.7"
""".lstrip(),
        encoding="utf-8",
    )
    (root / "requirements.lock").write_text("httpx==0.28.1\n", encoding="utf-8")
    (package / "__init__.py").write_text('__version__ = "9.8.7"\n', encoding="utf-8")
    (package / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    return package


def test_source_archive_build_identity_is_stable_and_content_addressed(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    package = _source_archive(archive)

    first = resolve_build_identity(archive)
    second = resolve_build_identity(archive)

    assert first.public_dict() == second.public_dict()
    assert first.kind == "content-addressed-package"
    assert first.tree_state == "content-addressed"
    assert re.fullmatch(r"[0-9a-f]{64}", first.revision)
    assert first.tree_sha256 == first.revision
    assert first.dependency_lock == archive / "requirements.lock"

    (package / "runner.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert resolve_build_identity(archive).revision != first.revision


def test_installed_wheel_does_not_inherit_an_ancestor_checkout_identity(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    installed = checkout / ".venv" / "Lib" / "site-packages" / "inference_bench"
    installed.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\nname='other'\n", encoding="utf-8")
    module = installed / "environment.py"
    module.write_text("# installed wheel\n", encoding="utf-8")

    assert find_source_root(module) == installed.resolve()


def test_archive_plan_run_and_report_share_identity_without_git(
    tmp_path: Path, campaign, monkeypatch
) -> None:
    archive = tmp_path / "archive"
    _source_archive(archive)
    run_dir = tmp_path / "run"
    versions = {
        "httpx": "0.28.1",
        "inference-endpoint-benchmark": "9.8.7",
    }
    monkeypatch.setattr("inference_bench.cli._source_root", lambda: archive)
    monkeypatch.setattr(
        "inference_bench.cli.locked_distribution_versions", lambda lock: versions
    )
    monkeypatch.setattr(
        "inference_bench.report.locked_distribution_versions", lambda lock: versions
    )

    # Config compilation and request planning remain credential-free in an unpacked source build.
    assert build_plan(campaign).campaign_hash == campaign.identity_hash
    runtime = _runtime_manifest(
        campaign,
        ("inference-bench", "run", "campaign.yaml", "--output", str(run_dir)),
        output_dir=run_dir,
    )
    report = _report_source_snapshot(run_dir, source_root=archive)

    assert runtime["source_identity_kind"] == "content-addressed-package"
    assert runtime["source_commit"] == report["source_revision"]
    assert runtime["source_dirty_tree_sha256"] == report["source_dirty_tree_sha256"]
    assert report["source_identity_kind"] == "content-addressed-package"
    assert report["source_tree_state"] == "content-addressed"


def test_wheel_build_includes_the_dependency_lock() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '[tool.hatch.build.targets.wheel.force-include]' in text
    assert '"requirements.lock" = "inference_bench/requirements.lock"' in text
