from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from inference_bench.adapters.base import adapter_for
from inference_bench.adapters.providers import (
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
                    "endpoints": {
                        "available": [{"provider": provider, "selected": True}]
                    },
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
