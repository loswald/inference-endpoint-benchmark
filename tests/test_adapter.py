from __future__ import annotations

import asyncio
import json
from dataclasses import replace

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


def test_stream_requests_ask_for_usage_but_nonstream_requests_do_not(route) -> None:
    assert build_payload(route, _spec(stream=True))["stream_options"] == {
        "include_usage": True
    }
    assert "stream_options" not in build_payload(route, _spec(stream=False))


def test_payload_build_does_not_mutate_route_defaults(route) -> None:
    configured = RouteConfig(
        id=route.id,
        provider=route.provider,
        adapter=route.adapter,
        model=route.model,
        base_url=route.base_url,
        auth=route.auth,
        input_usd_per_million=1,
        output_usd_per_million=2,
        request_defaults={"stream_options": {"vendor_extension": "kept"}},
    )
    before = configured.identity_hash
    payload = build_payload(configured, _spec(stream=True))
    assert payload["stream_options"] == {
        "vendor_extension": "kept",
        "include_usage": True,
    }
    assert configured.request_defaults == {
        "stream_options": {"vendor_extension": "kept"}
    }
    assert configured.identity_hash == before


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


def test_explicit_zero_usage_and_cache_miss_are_not_treated_as_unknown(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")
    body = json.dumps(
        {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cache_read_input_tokens": 99,
            },
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cache_read_input_tokens == 0
    assert result.ttft_seconds is None
    assert result.time_to_headers_seconds is not None


def test_unreported_cache_usage_remains_unknown(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")
    body = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    assert asyncio.run(run()).cache_read_input_tokens is None


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


def test_full_stream_has_one_hard_wall_clock_deadline(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            await asyncio.sleep(0.2)
            yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SlowStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        spec = replace(_spec(), timeout_seconds=0.03)
        result = await OpenAICompatibleAdapter(client).infer(route, spec)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "timeout"
    assert result.total_seconds < 0.15


def test_malformed_success_payload_is_a_protocol_error_not_success(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "server_error"
    assert result.http_status == 200
    assert result.error_kind == "invalid_json_success_body"
    assert result.error_body_sha256
