from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from inference_bench.matrix import load_matrix, run_matrix


def _campaign(path: Path, provider: str, route_id: str) -> None:
    path.write_text(
        f"""
campaign:
  name: {route_id}
  seed: 1
  max_wall_seconds: 60
  max_cost_usd: 2
  launch_reserve_seconds: 1
  launch_reserve_usd: 0.1
  concurrency: 1
  retries: 0
  client_location: test
routes:
  - id: {route_id}
    provider: {provider}
    adapter: openai_compatible
    model: model
    base_url: https://example.invalid/chat/completions
    auth: {{env: TEST_API_KEY}}
    region: test
    api_version: v1
    model_version: v1
    quota_scope: test
    input_usd_per_million: 1
    output_usd_per_million: 1
suites:
  latency: {{enabled: true, repeats: 1, shapes: [short_short]}}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_matrix_runs_different_providers_concurrently(tmp_path: Path) -> None:
    _campaign(tmp_path / "a.yaml", "a-provider", "a")
    _campaign(tmp_path / "b.yaml", "b-provider", "b")
    (tmp_path / "matrix.yaml").write_text(
        """
version: 1
max_parallel_providers: 2
campaigns:
  - {name: a, provider: a-provider, config: a.yaml, output: a}
  - {name: b, provider: b-provider, config: b.yaml, output: b}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    matrix = load_matrix(tmp_path / "matrix.yaml")
    active = 0
    maximum = 0

    async def runner(config, output, invocation):  # type: ignore[no-untyped-def]
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        output.mkdir(parents=True)
        await asyncio.sleep(0.02)
        active -= 1

    asyncio.run(run_matrix(matrix, tmp_path / "runs", runner, invocation=("test",)))
    assert maximum == 2


def test_matrix_rejects_duplicate_provider(tmp_path: Path) -> None:
    _campaign(tmp_path / "a.yaml", "same-provider", "a")
    _campaign(tmp_path / "b.yaml", "same-provider", "b")
    (tmp_path / "matrix.yaml").write_text(
        """
version: 1
campaigns:
  - {name: a, provider: same-provider, config: a.yaml, output: a}
  - {name: b, provider: same-provider, config: b.yaml, output: b}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only once"):
        load_matrix(tmp_path / "matrix.yaml")
