from __future__ import annotations

import asyncio
import json
import struct
import zlib
from dataclasses import replace

import httpx
import pytest

from inference_bench.adapters import adapter_for, adapter_plugin
from inference_bench.adapters.bedrock_converse import BedrockConverseAdapter
from inference_bench.config import NATIVE_PLACEHOLDER_ADAPTERS, CampaignConfig
from inference_bench.models import (
    DEFAULT_RETAINED_HEADER_NAMES,
    AuthConfig,
    RequestSpec,
    RouteConfig,
)
from inference_bench.payload import (
    BEDROCK_CONVERSE_PAYLOAD_GENERATOR_VERSION,
    materialize_bedrock_converse,
    reserved_input_tokens,
)
from inference_bench.plan import build_plan


def _route(**changes: object) -> RouteConfig:
    route = RouteConfig(
        id="bedrock-sonnet",
        provider="amazon-bedrock",
        adapter="bedrock_converse",
        model="us.anthropic.claude-sonnet-4-6-v1:0",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        auth=AuthConfig(env="AWS_BEARER_TOKEN_BEDROCK"),
        region="us-east-1",
        api_family="converse",
        billing_channel="bedrock_api_key",
        api_version="bedrock-runtime-converse",
        model_version="claude-sonnet-4-6",
        quota_scope="test-account",
        context_tokens=200_000,
        max_output_tokens=64_000,
        input_usd_per_million=3,
        output_usd_per_million=15,
        documentation_source_url="https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html",
        pricing_source_url="https://aws.amazon.com/bedrock/pricing/",
        evidence_retrieved_at_utc="2026-08-30T00:00:00Z",
        evidence_bundle_sha256="a" * 64,
        capabilities={"documentation_checked_utc": "2026-08-30T00:00:00Z"},
        retained_header_names=(*DEFAULT_RETAINED_HEADER_NAMES, "x-amzn-requestid"),
        reasoning_controls={"low": {"outputConfig.effort": "low"}},
    )
    return replace(route, **changes)


def _request(*, stream: bool, **changes: object) -> RequestSpec:
    request = RequestSpec(
        logical_id="bedrock-request",
        route_id="bedrock-sonnet",
        suite="fixture",
        cell_id="text",
        messages=({"role": "user", "content": "hello"},),
        planned_input_tokens=2,
        max_output_tokens=8,
        stream=stream,
        timeout_seconds=2,
    )
    return replace(request, **changes)


def _event_header(name: str, value: str) -> bytes:
    encoded_name = name.encode("utf-8")
    encoded_value = value.encode("utf-8")
    return (
        bytes([len(encoded_name)])
        + encoded_name
        + b"\x07"
        + struct.pack(">H", len(encoded_value))
        + encoded_value
    )


def _event(event_type: str, payload: dict[str, object]) -> bytes:
    headers = b"".join(
        (
            _event_header(":message-type", "event"),
            _event_header(":event-type", event_type),
            _event_header(":content-type", "application/json"),
        )
    )
    body = json.dumps(payload, separators=(",", ":")).encode()
    total_length = 16 + len(headers) + len(body)
    prelude = struct.pack(">II", total_length, len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message_without_crc = prelude + headers + body
    return message_without_crc + struct.pack(
        ">I", zlib.crc32(message_without_crc) & 0xFFFFFFFF
    )


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, delay: float = 0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def test_exact_request_fixture_uses_generated_serializer_and_bearer_path(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fixture")
    route = _route()
    request = _request(stream=False)
    adapter = BedrockConverseAdapter(client=httpx.AsyncClient())

    prepared = adapter.prepare(route, request)

    assert prepared.payload.body == (
        b'{"messages": [{"role": "user", "content": [{"text": "hello"}]}], '
        b'"inferenceConfig": {"maxTokens": 8}}'
    )
    assert prepared.payload.value["operation"] == "Converse"
    assert prepared.payload.value["url_path"] == (
        "/model/us.anthropic.claude-sonnet-4-6-v1%3A0/converse"
    )
    assert prepared.payload.generator_version == BEDROCK_CONVERSE_PAYLOAD_GENERATOR_VERSION
    assert prepared.headers["Authorization"] == "Bearer fixture"
    assert b"fixture" not in prepared.payload.body
    asyncio.run(adapter.close())


def test_complex_request_fixture_translates_system_vision_tools_schema_and_effort() -> None:
    route = _route()
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        },
    }
    request = _request(
        stream=True,
        messages=(
            {"role": "system", "content": "Be concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                        },
                    },
                ],
            },
        ),
        temperature=0.2,
        top_p=0.9,
        stop=("END",),
        tools=(tool,),
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object", "required": ["city"]},
            },
        },
        reasoning_budget="low",
    )

    body = json.loads(materialize_bedrock_converse(route, request).body)

    assert body["system"] == [{"text": "Be concise."}]
    assert body["messages"][0]["content"][1]["image"]["format"] == "png"
    assert body["messages"][0]["content"][1]["image"]["source"]["bytes"] == (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    )
    assert body["inferenceConfig"] == {
        "maxTokens": 8,
        "temperature": 0.2,
        "topP": 0.9,
        "stopSequences": ["END"],
    }
    assert body["toolConfig"]["toolChoice"] == {"tool": {"name": "lookup_weather"}}
    assert body["toolConfig"]["tools"][0]["toolSpec"]["strict"] is True
    assert body["outputConfig"]["effort"] == "low"
    assert body["outputConfig"]["textFormat"]["type"] == "json_schema"
    schema = body["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
    assert json.loads(schema) == {"required": ["city"], "type": "object"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 7, "no generic seed"),
        ("parallel_tool_calls", True, "no generic parallel_tool_calls"),
        ("logprobs", True, "no generic logprobs"),
    ],
)
def test_unmapped_generic_controls_fail_before_claim(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_bedrock_converse(_route(), replace(_request(stream=True), **{field: value}))


def test_nonstream_fixture_sends_bound_bytes_and_parses_usage_tools_without_reasoning_claim(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fixture-token")
    route = _route()
    request = _request(stream=False)
    expected_body = materialize_bedrock_converse(route, request).body

    async def handler(incoming: httpx.Request) -> httpx.Response:
        assert incoming.url.raw_path == (
            b"/model/us.anthropic.claude-sonnet-4-6-v1%3A0/converse"
        )
        assert incoming.content == expected_body
        assert incoming.headers["Authorization"] == "Bearer fixture-token"
        return httpx.Response(
            200,
            headers={"x-amzn-requestid": "request-id-not-persisted-raw"},
            json={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"reasoningContent": {"reasoningText": {"text": "private"}}},
                            {"text": "Calling tool."},
                            {
                                "toolUse": {
                                    "toolUseId": "tool-1",
                                    "name": "lookup_weather",
                                    "input": {"city": "Rome"},
                                }
                            },
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 5,
                    "totalTokens": 17,
                    "cacheReadInputTokens": 7,
                },
            },
        )

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = BedrockConverseAdapter(client=client)
        result = await adapter.infer(route, request)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "success"
    assert result.output_text == "Calling tool."
    assert result.tool_calls == (
        {
            "id": "tool-1",
            "type": "function",
            "function": {"name": "lookup_weather", "arguments": '{"city":"Rome"}'},
        },
    )
    assert result.input_tokens == 12
    assert result.output_tokens == 5
    assert result.cache_read_input_tokens == 7
    assert result.reasoning_tokens is None
    assert result.finish_reason == "tool_calls"
    assert "private" not in result.output_text
    assert result.retained_headers == {"x-amzn-requestid": "request-id-not-persisted-raw"}
    public = result.without_content()
    assert "output_text" not in public
    assert "tool_calls" not in public
    assert "provider_request_id" not in public


def test_nonstream_duplicate_json_is_a_fixed_protocol_error(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fixture-token")
    route = _route()
    request = _request(stream=False)

    async def handler(incoming: httpx.Request) -> httpx.Response:
        del incoming
        return httpx.Response(200, content=b'{"output":{},"output":{},"stopReason":"end_turn"}')

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = BedrockConverseAdapter(client=client)
        result = await adapter.infer(route, request)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "server_error"
    assert str(result.error_kind).startswith("protocol_error:")
    assert result.error_body_sha256 is not None


def test_stream_fixture_uses_aws_frame_parser_and_records_visible_event_times(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fixture-token")
    route = replace(_route(), stream_usage_mode="required")
    request = _request(stream=True)
    wire = b"".join(
        (
            _event("messageStart", {"role": "assistant"}),
            _event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hidden"}}},
            ),
            _event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": "Hello"}},
            ),
            _event("contentBlockStop", {"contentBlockIndex": 0}),
            _event(
                "contentBlockStart",
                {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "tool-1", "name": "lookup"}},
                },
            ),
            _event(
                "contentBlockDelta",
                {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"city"'}}},
            ),
            _event(
                "contentBlockDelta",
                {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ':"Rome"}'}}},
            ),
            _event("contentBlockStop", {"contentBlockIndex": 1}),
            _event("messageStop", {"stopReason": "tool_use"}),
            _event(
                "metadata",
                {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 4,
                        "totalTokens": 14,
                        "cacheReadInputTokens": 6,
                    },
                    "metrics": {"latencyMs": 25},
                },
            ),
        )
    )
    chunks = [wire[:13], wire[13:211], wire[211:477], wire[477:]]

    async def handler(incoming: httpx.Request) -> httpx.Response:
        assert incoming.url.path.endswith("/converse-stream")
        assert incoming.headers["Accept"] == "application/vnd.amazon.eventstream"
        return httpx.Response(
            200,
            headers={
                "content-type": "application/vnd.amazon.eventstream",
                "x-amzn-requestid": "stream-id",
            },
            stream=_Chunks(chunks),
        )

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = BedrockConverseAdapter(client=client)
        result = await adapter.infer(route, request)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "success"
    assert result.output_text == "Hello"
    assert result.tool_calls[0]["function"] == {
        "name": "lookup",
        "arguments": '{"city":"Rome"}',
    }
    assert result.input_tokens == 10
    assert result.output_tokens == 4
    assert result.cache_read_input_tokens == 6
    assert result.reasoning_tokens is None
    assert result.finish_reason == "tool_calls"
    assert result.ttft_seconds is not None
    assert len(result.output_event_offsets_seconds) == 4
    assert result.output_event_offsets_seconds == tuple(sorted(result.output_event_offsets_seconds))


def test_stream_crc_failure_is_protocol_error(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fixture-token")
    route = _route()
    request = _request(stream=True)
    corrupt = bytearray(_event("messageStart", {"role": "assistant"}))
    corrupt[-1] ^= 0xFF

    async def handler(incoming: httpx.Request) -> httpx.Response:
        del incoming
        return httpx.Response(
            200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            stream=_Chunks([bytes(corrupt)]),
        )

    async def run() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = BedrockConverseAdapter(client=client)
        result = await adapter.infer(route, request)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "server_error"
    assert "invalid_aws_event_stream" in str(result.error_kind)


def test_stream_timeout_closes_transport_and_cancellation_propagates(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fixture-token")
    route = _route()
    slow_streams: list[_Chunks] = []

    async def handler(incoming: httpx.Request) -> httpx.Response:
        del incoming
        stream = _Chunks([_event("messageStart", {"role": "assistant"})], delay=0.2)
        slow_streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            stream=stream,
        )

    async def run_timeout() -> object:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = BedrockConverseAdapter(client=client)
        result = await adapter.infer(route, _request(stream=True, timeout_seconds=0.01))
        await client.aclose()
        return result

    timeout_result = asyncio.run(run_timeout())
    assert timeout_result.status == "timeout"
    assert slow_streams[0].closed is True

    async def run_cancel() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = BedrockConverseAdapter(client=client)
        task = asyncio.create_task(adapter.infer(route, _request(stream=True)))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.aclose()

    asyncio.run(run_cancel())
    assert slow_streams[1].closed is True


def test_registry_and_plan_treat_converse_as_live_and_bind_exact_materializer(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fixture-token")
    route = _route()
    campaign = CampaignConfig(
        name="bedrock-plan",
        seed=7,
        max_wall_seconds=600,
        max_cost_usd=100,
        launch_reserve_seconds=30,
        launch_reserve_usd=1,
        concurrency=2,
        retries=0,
        routes=(route,),
        client_location="test-client",
        suites={"latency": {"enabled": True, "repeats": 1, "shapes": ["short_short"]}},
    )

    plan = build_plan(campaign)

    assert isinstance(adapter_for("bedrock_converse"), BedrockConverseAdapter)
    assert isinstance(adapter_for("bedrock_native"), BedrockConverseAdapter)
    assert adapter_plugin("bedrock_converse").api_families == ("converse",)
    assert "bedrock_converse" not in NATIVE_PLACEHOLDER_ADAPTERS
    assert "bedrock_native" not in NATIVE_PLACEHOLDER_ADAPTERS
    assert plan.native_placeholder_routes == ()
    spec = _request(stream=True)
    exact = materialize_bedrock_converse(route, spec)
    assert reserved_input_tokens(route, spec, 1.0) == max(
        spec.planned_input_tokens, exact.input_token_upper_bound
    )
