from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from inference_bench.adapters.base import adapter_for
from inference_bench.adapters.providers import (
    AlibabaModelStudioAdapter,
    AlibabaModelStudioResponsesAdapter,
    AzureModelInferenceAdapter,
    BedrockMantleAdapter,
    OpenRouterAdapter,
)
from inference_bench.models import AuthConfig, RequestSpec, RouteConfig


def _route(*, adapter: str, base_url: str, provider: str = "provider") -> RouteConfig:
    return RouteConfig(
        id="r",
        provider=provider,
        adapter=adapter,
        model="exact/model",
        base_url=base_url,
        auth=AuthConfig(env="TEST_PROVIDER_KEY"),
        upstream_provider="Exact upstream" if adapter == "openrouter" else None,
        input_usd_per_million=1,
        output_usd_per_million=2,
    )


def test_bedrock_mantle_is_live_adapter_with_strict_host(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-not-retained")
    route = _route(
        adapter="bedrock_mantle",
        provider="amazon-bedrock",
        base_url="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
    )
    adapter = adapter_for("bedrock_mantle")
    assert isinstance(adapter, BedrockMantleAdapter)
    adapter.preflight(route)
    with pytest.raises(RuntimeError, match="endpoint host"):
        adapter.preflight(replace(route, base_url="https://lookalike.example/chat/completions"))


def test_alibaba_model_studio_requires_payg_host_path_auth_and_matching_region(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-not-retained")
    route = replace(
        _route(
            adapter="alibaba_model_studio",
            provider="alibaba-model-studio",
            base_url=(
                "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/"
                "compatible-mode/v1/chat/completions"
            ),
        ),
        region="ap-southeast-1",
        billing_channel="pay_as_you_go",
    )
    adapter = adapter_for("alibaba_model_studio")
    assert isinstance(adapter, AlibabaModelStudioAdapter)
    adapter.preflight(route)

    adapter.preflight(
        replace(
            route,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
    )
    with pytest.raises(RuntimeError, match="region-bound"):
        adapter.preflight(replace(route, region="cn-beijing"))
    with pytest.raises(RuntimeError, match="pay-as-you-go"):
        adapter.preflight(
            replace(
                route,
                base_url=(
                    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/"
                    "compatible-mode/v1/chat/completions"
                ),
            )
        )
    with pytest.raises(RuntimeError, match="must use"):
        adapter.preflight(
            replace(
                route,
                base_url=(
                    "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/"
                    "compatible-mode/v1/responses"
                ),
            )
        )
    with pytest.raises(RuntimeError, match="Authorization: Bearer"):
        adapter.preflight(replace(route, auth=replace(route.auth, prefix="")))
    with pytest.raises(RuntimeError, match="provider=alibaba-model-studio"):
        adapter.preflight(replace(route, provider="lookalike-provider"))
    with pytest.raises(RuntimeError, match="canonical HTTPS endpoint"):
        adapter.preflight(replace(route, base_url=route.base_url + "?api-version=private"))
    with pytest.raises(RuntimeError, match="canonical HTTPS endpoint"):
        adapter.preflight(
            replace(
                route,
                base_url=route.base_url.replace(
                    "ap-southeast-1.maas.aliyuncs.com",
                    "ap-southeast-1.maas.aliyuncs.com:443",
                ),
            )
        )


def test_alibaba_model_studio_responses_is_a_separate_api_contract(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-not-retained")
    route = RouteConfig(
        id="r",
        provider="alibaba-model-studio",
        adapter="alibaba_model_studio_responses",
        model="exact/model",
        base_url=(
            "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1/responses"
        ),
        auth=AuthConfig(env="TEST_PROVIDER_KEY"),
        region="ap-southeast-1",
        billing_channel="pay_as_you_go",
        api_family="responses",
        output_limit_field="max_output_tokens",
        input_usd_per_million=1,
        output_usd_per_million=2,
    )
    adapter = adapter_for("alibaba_model_studio_responses")
    assert isinstance(adapter, AlibabaModelStudioResponsesAdapter)
    adapter.preflight(route)


@pytest.mark.parametrize(
    ("code", "expected_status", "expected_error_kind"),
    [
        ("Throttling.RateQuota", "rate_limited", "provider_rate_limit"),
        ("Throttling.AllocationQuota", "rate_limited", "provider_rate_limit"),
        ("Throttling.BurstRate", "rate_limited", "provider_rate_limit"),
        ("CommodityNotPurchased", "client_error", "provider_billing_or_entitlement"),
        ("PrepaidBillOverdue", "client_error", "provider_billing_or_entitlement"),
    ],
)
def test_alibaba_429_classification_separates_load_from_account_state(
    code: str, expected_status: str, expected_error_kind: str
) -> None:
    adapter = AlibabaModelStudioAdapter()
    response = httpx.Response(429)

    status, error_kind = adapter._classify_http_error(
        response, json.dumps({"code": code, "message": "not retained"}).encode()
    )

    assert status == expected_status
    assert error_kind == expected_error_kind


def test_alibaba_unknown_429_remains_a_conservative_rate_observation() -> None:
    adapter = AlibabaModelStudioAdapter()
    status, error_kind = adapter._classify_http_error(
        httpx.Response(429), b'{"code":"FutureThrottleClass"}'
    )

    assert status == "rate_limited"
    assert error_kind == "http_429"


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_alibaba_permanent_route_errors_are_not_congestion(status_code: int) -> None:
    adapter = AlibabaModelStudioAdapter()
    status, error_kind = adapter._classify_http_error(
        httpx.Response(status_code), b'{"code":"InvalidParameter"}'
    )

    assert status == "client_error"
    assert error_kind == "provider_route_fatal"


def test_azure_chat_adapter_accepts_both_official_host_families(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-not-retained")
    for host in ("resource.services.ai.azure.com", "resource.openai.azure.com"):
        route = _route(
            adapter="azure_model_inference",
            provider="azure-ai-foundry",
            base_url=f"https://{host}/openai/v1/chat/completions",
        )
        adapter = adapter_for("azure_model_inference")
        assert isinstance(adapter, AzureModelInferenceAdapter)
        adapter.preflight(route)


def test_openrouter_requires_exact_api_path_and_payload_pin(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-not-retained")
    route = _route(
        adapter="openrouter",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1/chat/completions",
    )
    adapter = adapter_for("openrouter")
    assert isinstance(adapter, OpenRouterAdapter)
    adapter.preflight(route)
    with pytest.raises(RuntimeError, match="must use"):
        adapter.preflight(replace(route, base_url="https://openrouter.ai/api/v1/responses"))


def test_openrouter_stream_requires_and_validates_selected_upstream(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-not-retained")
    route = _route(
        adapter="openrouter",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1/chat/completions",
    )
    spec = RequestSpec(
        logical_id="one",
        route_id="r",
        suite="test",
        cell_id="test",
        messages=({"role": "user", "content": "hello"},),
        planned_input_tokens=2,
        max_output_tokens=8,
        stream=True,
    )

    async def run(provider: str | None):
        metadata = (
            {}
            if provider is None
            else {
                "openrouter_metadata": {
                    "endpoints": {"available": [{"provider": provider, "selected": True}]},
                    "attempts": [{"provider": provider}],
                }
            }
        )
        chunks = [
            {
                "id": "generation-id",
                **metadata,
                "choices": [{"index": 0, "delta": {"content": "ok"}}],
            },
            {
                "id": "generation-id",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["X-OpenRouter-Metadata"] == "enabled"
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = OpenRouterAdapter(client)
        result = await adapter.infer(route, spec)
        await client.aclose()
        return result

    accepted = asyncio.run(run("Exact upstream"))
    assert accepted.status == "success"
    assert accepted.provider_request_id == "generation-id"
    missing = asyncio.run(run(None))
    assert missing.status == "server_error"
    assert "provider_attestation_missing" in str(missing.error_kind)
    mismatched = asyncio.run(run("Different provider"))
    assert mismatched.status == "server_error"
    assert "provider_attestation_failed" in str(mismatched.error_kind)
