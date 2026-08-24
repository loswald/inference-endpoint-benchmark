from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_bench.config import CampaignConfig
from inference_bench.models import AuthConfig, RouteConfig


@pytest.fixture
def route() -> RouteConfig:
    return RouteConfig(
        id="route-a",
        provider="test",
        adapter="openai_compatible",
        model="model-a",
        base_url="https://example.invalid/v1/chat/completions",
        auth=AuthConfig(env="TEST_API_KEY"),
        quota_scope="test-project",
        region="test-region",
        api_version="test-v1",
        model_version="test-model-v1",
        context_tokens=8_192,
        max_output_tokens=2_048,
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        documentation_source_url="https://example.invalid/documentation",
        pricing_source_url="https://example.invalid/pricing",
        evidence_retrieved_at_utc="2026-08-24T00:00:00Z",
        evidence_bundle_sha256="a" * 64,
        capabilities={"documentation_checked_utc": "2026-08-24T00:00:00Z"},
    )


@pytest.fixture
def campaign(route: RouteConfig) -> CampaignConfig:
    return CampaignConfig(
        name="test",
        seed=7,
        max_wall_seconds=600,
        max_cost_usd=10,
        launch_reserve_seconds=60,
        launch_reserve_usd=1,
        concurrency=4,
        retries=1,
        routes=(route,),
        client_location="test-client",
        suites={"latency": {"enabled": True, "repeats": 1, "shapes": ["short_short"]}},
    )
