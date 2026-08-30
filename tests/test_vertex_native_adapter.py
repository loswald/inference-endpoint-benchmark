from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace

import httpx
import pytest

from inference_bench.adapters import adapter_for, adapter_plugin
from inference_bench.adapters.vertex_native import (
    VERTEX_NATIVE_PAYLOAD_GENERATOR_VERSION,
    VertexNativeAdapter,
    _parse_usage,
)
from inference_bench.config import NATIVE_PLACEHOLDER_ADAPTERS
from inference_bench.models import (
    DEFAULT_RETAINED_HEADER_NAMES,
    AuthConfig,
    RequestSpec,
    RouteConfig,
    canonical_json,
)


class FakeCredentials:
    def __init__(self) -> None:
        self.valid = False
        self.token: str | None = None
        self.refresh_calls = 0
        self.refresh_requests: list[object] = []

    def refresh(self, request: object) -> None:
        self.refresh_calls += 1
        self.refresh_requests.append(request)
        self.token = "oauth-fixture"
        self.valid = True


def _route(**changes: object) -> RouteConfig:
    route = RouteConfig(
        id="vertex-gemini-3",
        provider="google-vertex-ai",
        adapter="vertex_native",
        model="gemini-3.6-flash",
        base_url=(
            "https://aiplatform.googleapis.com/v1/projects/benchmark-project/locations/"
            "global/publishers/google/models/gemini-3.6-flash:generateContent"
        ),
        auth=AuthConfig(env="GOOGLE_APPLICATION_CREDENTIALS"),
        region="global",
        api_version="v1",
        output_limit_field="max_output_tokens",
        reasoning_controls={
            "low": {"reasoning_effort": "LOW"},
            "provider_default": {},
        },
        retained_header_names=(*DEFAULT_RETAINED_HEADER_NAMES, "x-goog-request-id"),
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
    )
    return replace(route, **changes)


def _request(*, stream: bool = False, **changes: object) -> RequestSpec:
    request = RequestSpec(
        logical_id="vertex-request-1",
        route_id="vertex-gemini-3",
        suite="fixture",
        cell_id="short_short",
        messages=({"role": "user", "content": "hello"},),
        planned_input_tokens=8,
        max_output_tokens=32,
        stream=stream,
        timeout_seconds=1.0,
    )
    return replace(request, **changes)


def _adapter(
    client: httpx.AsyncClient | None = None,
    credentials: FakeCredentials | None = None,
) -> tuple[VertexNativeAdapter, FakeCredentials]:
    credentials = credentials or FakeCredentials()
    return (
        VertexNativeAdapter(
            client,
            credentials=credentials,
            auth_request_factory=lambda: "refresh-transport",
        ),
        credentials,
    )


def test_vertex_native_absent_thought_count_means_zero_not_unobservable() -> None:
    usage = _parse_usage(
        {
            "promptTokenCount": 8,
            "candidatesTokenCount": 4,
            "totalTokenCount": 12,
        }
    )

    assert usage.errors == ()
    assert usage.input_tokens == 8
    assert usage.output_tokens == 4
    assert usage.reasoning_tokens == 0
    assert usage.total_tokens == 12


def test_vertex_native_materializes_canonical_multimodal_tool_schema_and_thinking() -> None:
    adapter, credentials = _adapter()
    route = _route()
    request = _request(
        reasoning_budget="low",
        messages=(
            {"role": "system", "content": "Answer tersely."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this pixel."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up one item",
                    "parameters": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                        "required": ["item"],
                    },
                },
            },
        ),
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        parallel_tool_calls=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
        temperature=0.2,
        top_p=0.9,
        seed=17,
        stop=("END",),
        logprobs=True,
    )

    prepared = adapter.prepare(route, request)
    value = prepared.payload.value

    assert credentials.refresh_calls == 1
    assert credentials.refresh_requests == ["refresh-transport"]
    assert prepared.headers["Authorization"] == "Bearer oauth-fixture"
    assert prepared.headers["Accept"] == "application/json"
    assert value["systemInstruction"] == {"parts": [{"text": "Answer tersely."}]}
    assert value["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "Describe this pixel."},
                {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
            ],
        }
    ]
    assert value["generationConfig"] == {
        "maxOutputTokens": 32,
        "temperature": 0.2,
        "topP": 0.9,
        "seed": 17,
        "responseLogprobs": True,
        "stopSequences": ["END"],
        "responseMimeType": "application/json",
        "responseJsonSchema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        "thinkingConfig": {"thinkingLevel": "LOW"},
    }
    assert value["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert value["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["lookup"]}
    }
    assert prepared.payload.body == canonical_json(value).encode()
    assert prepared.payload.generator_version == VERTEX_NATIVE_PAYLOAD_GENERATOR_VERSION
    assert prepared.payload.wire_body_sha256 == hashlib.sha256(prepared.payload.body).hexdigest()
    # A caller can mutate the convenience mapping without changing the claimed wire bytes.
    value["generationConfig"]["maxOutputTokens"] = 999
    assert b'"maxOutputTokens":32' in prepared.payload.body


def test_vertex_native_preflight_refreshes_once_and_requires_exact_action_identity() -> None:
    adapter, credentials = _adapter()
    route = _route()

    adapter.preflight(route)
    adapter.prepare(route, _request())

    assert credentials.refresh_calls == 1
    with pytest.raises(RuntimeError, match="explicit v1"):
        adapter.preflight(replace(route, base_url=route.base_url.replace(":generateContent", "")))
    with pytest.raises(RuntimeError, match="location and model"):
        adapter.preflight(
            replace(route, base_url=route.base_url.replace("gemini-3.6-flash", "gemini-other"))
        )
    with pytest.raises(RuntimeError, match="host must exactly match"):
        adapter.preflight(
            replace(
                route,
                base_url=route.base_url.replace(
                    "aiplatform.googleapis.com", "global-aiplatform.googleapis.com"
                ),
            )
        )


def test_vertex_native_materializes_openai_tool_history_as_native_parts() -> None:
    adapter, _ = _adapter()
    prepared = adapter.prepare(
        _route(),
        _request(
            messages=(
                {"role": "user", "content": "Find x"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"item":"x"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": '{"value":7}'},
            )
        ),
    )

    assert prepared.payload.value["contents"] == [
        {"role": "user", "parts": [{"text": "Find x"}]},
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "lookup", "args": {"item": "x"}}}],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "lookup", "response": {"value": 7}}}
            ],
        },
    ]


def test_vertex_native_nonstream_parses_usage_cache_thoughts_and_tools() -> None:
    body = {
        "responseId": "response-1",
        "candidates": [
            {
                "index": 0,
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"thought": True, "text": "private chain of thought"},
                        {"text": "done"},
                        {
                            "functionCall": {"name": "lookup", "args": {"item": "x"}},
                            "thoughtSignature": "opaque-signature",
                        },
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 3,
            "totalTokenCount": 17,
            "toolUsePromptTokenCount": 2,
            "thoughtsTokenCount": 2,
            "cachedContentTokenCount": 4,
        },
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith(":generateContent")
        assert request.headers["accept"] == "application/json"
        return httpx.Response(
            200,
            json=body,
            headers={"x-goog-request-id": "header-request-id"},
        )

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, _ = _adapter(client)
        result = await adapter.infer(_route(), _request())
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "success"
    assert result.output_text == "done"
    assert "private chain of thought" not in result.output_text
    assert result.input_tokens == 12
    assert result.output_tokens == 5
    assert result.reasoning_tokens == 2
    assert result.cache_read_input_tokens == 4
    assert result.finish_reason == "stop"
    assert result.provider_request_id == "response-1"
    assert result.tool_calls[0]["function"] == {
        "name": "lookup",
        "arguments": '{"item":"x"}',
    }
    assert "output_text" not in result.without_content()
    assert "opaque-signature" not in json.dumps(result.without_content())


def test_vertex_native_stream_uses_sse_action_and_records_ttft_offsets() -> None:
    events = [
        {
            "responseId": "stream-response",
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": "Hel"}]}}
            ],
        },
        {
            "responseId": "stream-response",
            "candidates": [
                {
                    "index": 0,
                    "finishReason": "STOP",
                    "content": {"role": "model", "parts": [{"text": "lo"}]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
                "thoughtsTokenCount": 1,
                "cachedContentTokenCount": 2,
            },
        },
    ]
    wire = "".join(f"data: {json.dumps(event)}\r\n\r\n" for event in events).encode()

    observed_bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith(":streamGenerateContent?alt=sse")
        assert request.headers["accept"] == "text/event-stream"
        observed_bodies.append(request.content)
        return httpx.Response(200, content=wire, headers={"x-goog-request-id": "stream-header"})

    async def run() -> tuple[object, bytes]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, _ = _adapter(client)
        route = _route()
        request = _request(stream=True)
        prepared = adapter.prepare(route, request)
        result = await adapter.send_prepared(route, request, prepared)
        await client.aclose()
        return result, prepared.payload.body

    result, prepared_body = asyncio.run(run())
    assert observed_bodies == [prepared_body]
    assert result.status == "success"
    assert result.output_text == "Hello"
    assert result.input_tokens == 4
    assert result.output_tokens == 3
    assert result.reasoning_tokens == 1
    assert result.cache_read_input_tokens == 2
    assert result.ttft_seconds is not None
    assert len(result.output_event_offsets_seconds) == 2
    assert tuple(sorted(result.output_event_offsets_seconds)) == result.output_event_offsets_seconds
    assert result.provider_request_id == "stream-response"


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b'{"candidates":{}}',
        b'{"candidates":[{"index":1,"content":{"parts":[]}}]}',
        b'{"candidates":[{"index":0,"content":{"parts":[{"text":3}]}}]}',
    ],
)
def test_vertex_native_malformed_success_is_fixed_protocol_error(body: bytes) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=body)

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, _ = _adapter(client)
        result = await adapter.infer(_route(), _request())
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "server_error"
    assert result.error_kind == "protocol_error"
    assert result.error_body_sha256 == hashlib.sha256(body).hexdigest()


def test_vertex_native_malformed_sse_is_fixed_protocol_error() -> None:
    wire = b'data: {"candidates":{}}\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=wire)

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, _ = _adapter(client)
        result = await adapter.infer(_route(), _request(stream=True))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "server_error"
    assert result.error_kind == "protocol_error"
    assert result.error_body_sha256 == hashlib.sha256(wire).hexdigest()


def test_vertex_native_429_classifies_error_envelope_without_message_retention() -> None:
    raw = canonical_json(
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "sensitive provider-controlled diagnostic",
            }
        }
    ).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(429, content=raw, headers={"retry-after": "2"})

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, _ = _adapter(client)
        result = await adapter.infer(_route(), _request())
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "rate_limited"
    assert result.error_kind == "provider_rate_limit"
    assert result.error_body_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.retained_headers == {"retry-after": "2"}
    assert "sensitive" not in json.dumps(result.without_content())


def test_vertex_native_timeout_is_measured_and_cancellation_propagates() -> None:
    entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        entered.set()
        await asyncio.sleep(60)
        return httpx.Response(200, json={})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter, _ = _adapter(client)
        timeout_result = await adapter.infer(
            _route(), _request(timeout_seconds=0.001)
        )
        assert timeout_result.status == "timeout"
        assert timeout_result.error_kind == "timeout"

        entered.clear()
        task = asyncio.create_task(adapter.infer(_route(), _request(timeout_seconds=5.0)))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.aclose()

    asyncio.run(run())


def test_vertex_native_rejects_unexpressible_controls_before_send() -> None:
    adapter, _ = _adapter()
    route = _route()
    with pytest.raises(ValueError, match="parallel_tool_calls"):
        adapter.prepare(route, _request(parallel_tool_calls=False))
    with pytest.raises(ValueError, match="data URLs or typed gs"):
        adapter.prepare(
            route,
            _request(
                messages=(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                            }
                        ],
                    },
                )
            ),
        )


def test_vertex_native_connection_pool_identity_is_fail_closed() -> None:
    adapter, _ = _adapter()
    with pytest.raises(RuntimeError, match="connection pool"):
        adapter.preflight(_route(transport_max_connections=17))
    asyncio.run(adapter.close())


def test_vertex_native_is_registered_as_a_live_native_transport() -> None:
    assert isinstance(adapter_for("vertex_native"), VertexNativeAdapter)
    assert adapter_plugin("vertex_native").api_families == (
        "chat_completions",
        "generate_content",
    )
    assert "vertex_native" not in NATIVE_PLACEHOLDER_ADAPTERS
