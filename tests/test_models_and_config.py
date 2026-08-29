from __future__ import annotations

from dataclasses import replace

import pytest

from inference_bench.config import CampaignConfig
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


def test_route_identity_binds_billing_channel(route: RouteConfig) -> None:
    assert replace(route, billing_channel="pay_as_you_go").identity_hash != route.identity_hash


def test_public_config_redacts_alibaba_workspace_hostname(campaign: CampaignConfig) -> None:
    private_host = "private-workspace-id.ap-southeast-1.maas.aliyuncs.com"
    alibaba = replace(
        campaign.routes[0],
        provider="alibaba-model-studio",
        adapter="alibaba_model_studio",
        base_url=f"https://{private_host}/compatible-mode/v1/chat/completions",
        region="ap-southeast-1",
        billing_channel="pay_as_you_go",
    )
    public = replace(campaign, routes=(alibaba,)).public_dict()["routes"][0]

    assert private_host not in public["base_url"]
    assert public["base_url"].startswith(
        "https://workspace-redacted.ap-southeast-1.maas.aliyuncs.com/"
    )
    assert "base_url_workspace_identifier" in public["omitted_operational_fields"]


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


def test_output_limit_tolerance_is_nonnegative_identity_bound_and_public(
    route: RouteConfig, campaign: CampaignConfig
) -> None:
    assert route.output_limit_tolerance_tokens == 0
    tolerant = replace(route, output_limit_tolerance_tokens=10)
    assert tolerant.identity_hash != route.identity_hash
    public = replace(campaign, routes=(tolerant,)).public_dict()["routes"][0]
    assert public["output_limit_tolerance_tokens"] == 10

    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            replace(route, output_limit_tolerance_tokens=invalid)


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
        "parallel_tool_calls",
        "reasoning",
        "reasoning_effort",
        "text",
        "verbosity",
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


def test_route_reasoning_control_identity_and_exact_api_family_fields(route: RouteConfig) -> None:
    controlled = replace(
        route,
        reasoning_controls={"fast": {"reasoning_effort": "minimal", "verbosity": "low"}},
    )
    assert controlled.identity_hash != route.identity_hash
    assert controlled.reasoning_control("fast") == {
        "reasoning_effort": "minimal",
        "verbosity": "low",
    }
    assert controlled.reasoning_control("provider_default") == {}
    with pytest.raises(ValueError, match="does not declare reasoning budget"):
        route.reasoning_control("fast")
    with pytest.raises(ValueError, match="unsupported chat_completions wire fields"):
        replace(route, reasoning_controls={"bad": {"reasoning.effort": "low"}})


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
