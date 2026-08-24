from __future__ import annotations

from dataclasses import replace

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


def test_http_adapters_reject_unimplemented_api_families(route: RouteConfig) -> None:
    with pytest.raises(ValueError, match="only api_family=chat_completions"):
        replace(route, api_family="responses")


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


def test_route_identity_binds_safe_request_defaults(route: RouteConfig) -> None:
    changed = RouteConfig(
        id=route.id,
        provider=route.provider,
        adapter=route.adapter,
        model=route.model,
        base_url=route.base_url,
        auth=route.auth,
        input_usd_per_million=1,
        output_usd_per_million=2,
        request_defaults={"user": "benchmark-client"},
    )
    assert changed.identity_hash != route.identity_hash


def test_route_identity_binds_stream_usage_mode(route: RouteConfig) -> None:
    changed = replace(route, stream_usage_mode="required")
    assert changed.identity_hash != route.identity_hash

    with pytest.raises(ValueError, match="stream_usage_mode"):
        replace(route, stream_usage_mode="automatic")


def test_route_identity_binds_timeout_and_documentation_evidence(route: RouteConfig) -> None:
    assert replace(route, request_timeout_seconds=900).identity_hash != route.identity_hash
    assert replace(route, evidence_bundle_sha256="b" * 64).identity_hash != route.identity_hash

    with pytest.raises(ValueError, match="request_timeout_seconds"):
        replace(route, request_timeout_seconds=0)
    with pytest.raises(ValueError, match="evidence_bundle_sha256"):
        replace(route, evidence_bundle_sha256="not-a-digest")
    with pytest.raises(ValueError, match="public absolute HTTPS URL"):
        replace(route, pricing_source_url="https://user@example.invalid/pricing?private=1")


def test_retained_headers_are_restricted_to_fixed_safe_names(route: RouteConfig) -> None:
    with pytest.raises(ValueError, match="fixed safe allowlist"):
        replace(route, retained_header_names=("x-secret-token", "retry-after"))
    with pytest.raises(ValueError, match="accept encoding"):
        replace(route, extra_headers={"Accept-Encoding": "br"})


@pytest.mark.parametrize(
    "key",
    [
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "provider",
        "temperature",
        "tools",
        "response_format",
    ],
)
def test_request_defaults_cannot_override_identity_or_cost_fields(
    route: RouteConfig, key: str
) -> None:
    with pytest.raises(ValueError, match="protected request fields"):
        RouteConfig(
            id=route.id,
            provider=route.provider,
            adapter=route.adapter,
            model=route.model,
            base_url=route.base_url,
            auth=route.auth,
            input_usd_per_million=1,
            output_usd_per_million=2,
            request_defaults={key: "override"},
        )


@pytest.mark.parametrize(
    "key",
    [
        "modalities",
        "audio",
        "service_tier",
        "web_search_options",
        "candidate_count",
        "num_images",
        "vendor_billable_feature",
    ],
)
def test_request_defaults_reject_features_outside_token_cost_model(
    route: RouteConfig, key: str
) -> None:
    with pytest.raises(ValueError, match="outside the token-cost model allowlist"):
        RouteConfig(
            id=route.id,
            provider=route.provider,
            adapter=route.adapter,
            model=route.model,
            base_url=route.base_url,
            auth=route.auth,
            input_usd_per_million=1,
            output_usd_per_million=2,
            request_defaults={key: "enabled"},
        )


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


def test_extra_headers_reject_case_insensitive_duplicates(route: RouteConfig) -> None:
    with pytest.raises(ValueError, match="case-insensitive duplicates"):
        RouteConfig(
            id="duplicate-header",
            provider="test",
            adapter="openai_compatible",
            model="m",
            base_url="https://example.invalid",
            auth=route.auth,
            extra_headers={"X-Test": "one", "x-test": "two"},
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
