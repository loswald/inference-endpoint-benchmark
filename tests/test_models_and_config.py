from __future__ import annotations

import pytest

from inference_bench.models import AuthConfig, RouteConfig


def test_openrouter_requires_exact_upstream() -> None:
    with pytest.raises(ValueError, match="pin upstream_provider"):
        RouteConfig(
            id="or",
            provider="openrouter",
            adapter="openrouter",
            model="vendor/model",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            auth=AuthConfig(env="OPENROUTER_API_KEY"),
            input_usd_per_million=1,
            output_usd_per_million=1,
        )


def test_route_identity_changes_with_api_version(route: RouteConfig) -> None:
    changed = RouteConfig(
        id=route.id,
        provider=route.provider,
        adapter=route.adapter,
        model=route.model,
        base_url=route.base_url,
        auth=route.auth,
        api_version="v2",
        input_usd_per_million=1,
        output_usd_per_million=2,
    )
    assert changed.identity_hash != route.identity_hash


def test_credentials_cannot_be_extra_headers(route: RouteConfig) -> None:
    with pytest.raises(ValueError, match="credentials belong"):
        RouteConfig(
            id="bad",
            provider="test",
            adapter="openai_compatible",
            model="m",
            base_url="https://example.invalid",
            auth=route.auth,
            extra_headers={"api-key": "secret"},
            input_usd_per_million=1,
            output_usd_per_million=1,
        )


def test_cached_input_price_is_stratified() -> None:
    route = RouteConfig(
        id="cache",
        provider="test",
        adapter="openai_compatible",
        model="m",
        base_url="https://example.invalid",
        auth=AuthConfig(env="TEST_API_KEY"),
        input_usd_per_million=2,
        output_usd_per_million=4,
        cached_input_usd_per_million=0.5,
    )
    assert route.actual_cost(1_000, 100, cache_read_input_tokens=800) == pytest.approx(0.0012)
