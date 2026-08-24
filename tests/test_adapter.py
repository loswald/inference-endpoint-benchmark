from __future__ import annotations

import asyncio
import json

import httpx

from inference_bench.adapters.openai_compatible import OpenAICompatibleAdapter, build_payload
from inference_bench.models import AuthConfig, RequestSpec, RouteConfig


def _spec(stream: bool = True) -> RequestSpec:
    return RequestSpec(
        logical_id="logical-1",
        route_id="route-a",
        suite="smoke",
        cell_id="cell",
        messages=({"role": "user", "content": "hello"},),
        planned_input_tokens=3,
        max_output_tokens=8,
        stream=stream,
    )


def test_openrouter_payload_is_hard_pinned() -> None:
    route = RouteConfig(
        id="or",
        provider="openrouter",
        adapter="openrouter",
        model="vendor/model",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        auth=AuthConfig(env="OPENROUTER_API_KEY"),
        upstream_provider="ExactHost",
        input_usd_per_million=1,
        output_usd_per_million=1,
    )
    provider = build_payload(route, _spec())["provider"]
    assert provider == {
        "only": ["ExactHost"],
        "order": ["ExactHost"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_stream_preserves_usage_cache_and_event_count(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")
    chunks = [
        {"choices": [{"delta": {"content": "hello world"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 7},
            },
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer not-written"
        return httpx.Response(200, text=body, headers={"x-request-id": "safe-id"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter(client)

    async def run():
        result = await adapter.infer(route, _spec())
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "success"
    assert result.input_tokens == 10
    assert result.output_tokens == 2
    assert result.cache_read_input_tokens == 7
    assert len(result.output_event_offsets_seconds) == 1
    assert result.output_text == "hello world"


def test_error_body_is_hashed_not_retained(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="private provider response")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "client_error"
    assert result.error_body_sha256
    assert "private provider response" not in json.dumps(result.without_content())
