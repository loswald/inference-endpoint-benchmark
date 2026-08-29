from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx

from inference_bench.adapters.providers import AlibabaModelStudioResponsesAdapter
from inference_bench.adapters.responses import ResponsesAdapter
from inference_bench.models import AuthConfig, RequestSpec, RouteConfig
from inference_bench.payload import (
    build_responses_payload,
    materialize_responses,
    reserved_input_tokens,
)


def _route() -> RouteConfig:
    return RouteConfig(
        id="responses-route",
        provider="azure-ai-foundry",
        adapter="azure_responses",
        model="deployment-name",
        base_url="https://resource.openai.azure.com/openai/v1/responses",
        auth=AuthConfig(env="TEST_RESPONSES_KEY", header="api-key", prefix=""),
        api_family="responses",
        output_limit_field="max_output_tokens",
        input_usd_per_million=1,
        output_usd_per_million=2,
    )


def _spec(stream: bool) -> RequestSpec:
    return RequestSpec(
        logical_id="responses-one",
        route_id="responses-route",
        suite="test",
        cell_id="test",
        messages=({"role": "user", "content": "hello"},),
        planned_input_tokens=2,
        max_output_tokens=16,
        stream=stream,
    )


def _alibaba_route() -> RouteConfig:
    return RouteConfig(
        id="alibaba-responses-route",
        provider="alibaba-model-studio",
        adapter="alibaba_model_studio_responses",
        model="qwen3.8-flash",
        base_url=(
            "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1/responses"
        ),
        auth=AuthConfig(env="TEST_RESPONSES_KEY"),
        region="ap-southeast-1",
        billing_channel="pay_as_you_go",
        api_family="responses",
        output_limit_field="max_output_tokens",
        stream_usage_mode="required",
        input_usd_per_million=1,
        output_usd_per_million=2,
    )


def _fixture(name: str):  # type: ignore[no-untyped-def]
    return json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8"))


def test_responses_payload_flattens_chat_function_tools() -> None:
    spec = _spec(False)
    spec = replace(
        spec,
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            },
        ),
    )
    payload = build_responses_payload(_route(), spec)
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
        }
    ]
    assert payload["max_output_tokens"] == 16


def test_gpt56_responses_reasoning_and_verbosity_are_exact_route_declared_controls() -> None:
    route = replace(
        _route(),
        provider="amazon-bedrock",
        adapter="bedrock_mantle_responses",
        model="openai.gpt-5.6-sol",
        base_url="https://bedrock-mantle.us-east-1.api.aws/v1/responses",
        reasoning_controls={
            "fast": {"reasoning.effort": "minimal", "text.verbosity": "low"},
            "deep": {"reasoning.effort": "high", "text.verbosity": "high"},
        },
    )
    default = _spec(False)
    fast = replace(default, reasoning_budget="fast")
    default_payload = build_responses_payload(route, default)
    fast_payload = build_responses_payload(route, fast)

    assert "reasoning" not in default_payload
    assert "text" not in default_payload
    assert fast_payload["reasoning"] == {"effort": "minimal"}
    assert fast_payload["text"] == {"verbosity": "low"}
    assert fast.payload_hash != default.payload_hash
    assert materialize_responses(route, fast).bound_payload_sha256 != materialize_responses(
        route, default
    ).bound_payload_sha256
    assert reserved_input_tokens(route, fast, 1.0) >= len(
        materialize_responses(route, fast).body
    )


def test_responses_merges_text_format_with_declared_verbosity() -> None:
    route = replace(
        _route(),
        reasoning_controls={"concise": {"text.verbosity": "low"}},
    )
    spec = replace(
        _spec(False),
        reasoning_budget="concise",
        response_format={"type": "json_object"},
    )
    assert build_responses_payload(route, spec)["text"] == {
        "format": {"type": "json_object"},
        "verbosity": "low",
    }


def test_alibaba_responses_provider_default_omits_reasoning_controls() -> None:
    payload = build_responses_payload(_alibaba_route(), _spec(False))
    assert "reasoning" not in payload
    assert "text" not in payload


def test_responses_payload_flattens_json_schema_and_parallel_tool_control() -> None:
    schema = {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }
    spec = replace(
        _spec(False),
        response_format={"type": "json_schema", "json_schema": schema},
        parallel_tool_calls=True,
    )
    payload = build_responses_payload(_route(), spec)
    assert payload["text"]["format"] == {"type": "json_schema", **schema}
    assert payload["parallel_tool_calls"] is True


def test_responses_nonstream_parses_text_tools_and_usage(monkeypatch) -> None:
    monkeypatch.setenv("TEST_RESPONSES_KEY", "not-retained")
    body = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{}",
            },
        ],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }

    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["api-key"] == "not-retained"
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await ResponsesAdapter(client).infer(_route(), _spec(False))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "success"
    assert result.output_text == "ok"
    assert result.input_tokens == 4
    assert result.output_tokens == 2
    assert result.tool_calls[0]["function"]["name"] == "weather"


def test_responses_stream_requires_terminal_event_and_times_visible_deltas(monkeypatch) -> None:
    monkeypatch.setenv("TEST_RESPONSES_KEY", "not-retained")

    async def infer(events):  # type: ignore[no-untyped-def]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await ResponsesAdapter(client).infer(_route(), _spec(True))
        await client.aclose()
        return result

    final = {
        "id": "resp_2",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }
    result = asyncio.run(
        infer(
            [
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 0,
                    "response_id": "resp_2",
                    "delta": "ok",
                },
                {"type": "response.completed", "sequence_number": 1, "response": final},
            ]
        )
    )
    assert result.status == "success"
    assert result.ttft_seconds is not None
    assert result.output_text == "ok"
    missing = asyncio.run(
        infer(
            [
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 0,
                    "delta": "partial",
                }
            ]
        )
    )
    assert missing.status == "server_error"
    assert "missing_terminal_event" in str(missing.error_kind)


def test_alibaba_responses_official_usage_detail_keys_are_accounted(monkeypatch) -> None:
    monkeypatch.setenv("TEST_RESPONSES_KEY", "not-retained")
    nonstream_fixture = _fixture("alibaba_responses_completed.json")
    stream_fixture = _fixture("alibaba_responses_stream.json")

    async def infer(body, *, stream: bool):  # type: ignore[no-untyped-def]
        async def handler(request: httpx.Request) -> httpx.Response:
            if stream:
                wire = "".join(f"data: {json.dumps(event)}\n\n" for event in body)
                return httpx.Response(200, text=wire)
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        spec = replace(
            _spec(stream),
            route_id="alibaba-responses-route",
        )
        result = await AlibabaModelStudioResponsesAdapter(client).infer(
            _alibaba_route(), spec
        )
        await client.aclose()
        return result

    for result in (
        asyncio.run(infer(nonstream_fixture, stream=False)),
        asyncio.run(infer(stream_fixture, stream=True)),
    ):
        assert result.status == "success"
        assert result.input_tokens == 4096
        assert result.output_tokens == 128
        assert result.cache_read_input_tokens == 3072
        assert result.reasoning_tokens == 64
        assert result.usage_parse_errors == ()


def test_alibaba_responses_rejects_missing_usage_nonterminal_status_and_bad_sequence(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEST_RESPONSES_KEY", "not-retained")

    async def infer(body, *, stream: bool):  # type: ignore[no-untyped-def]
        async def handler(request: httpx.Request) -> httpx.Response:
            if stream:
                wire = "".join(f"data: {json.dumps(event)}\n\n" for event in body)
                return httpx.Response(200, text=wire)
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        spec = replace(_spec(stream), route_id="alibaba-responses-route")
        result = await AlibabaModelStudioResponsesAdapter(client).infer(
            _alibaba_route(), spec
        )
        await client.aclose()
        return result

    missing_usage = _fixture("alibaba_responses_completed.json")
    missing_usage.pop("usage")
    result = asyncio.run(infer(missing_usage, stream=False))
    assert result.status == "server_error"
    assert "required_response_usage" in str(result.error_kind)

    nonterminal = _fixture("alibaba_responses_completed.json")
    nonterminal["status"] = "queued"
    result = asyncio.run(infer(nonterminal, stream=False))
    assert result.status == "server_error"
    assert "nonterminal_response_status" in str(result.error_kind)

    bad_sequence = _fixture("alibaba_responses_stream.json")
    bad_sequence[1]["sequence_number"] = 7
    result = asyncio.run(infer(bad_sequence, stream=True))
    assert result.status == "server_error"
    assert "sequence_number" in str(result.error_kind)
