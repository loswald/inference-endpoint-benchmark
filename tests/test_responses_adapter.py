from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx

from inference_bench.adapters.responses import ResponsesAdapter
from inference_bench.models import AuthConfig, RequestSpec, RouteConfig
from inference_bench.payload import build_responses_payload


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
                    "response_id": "resp_2",
                    "delta": "ok",
                },
                {"type": "response.completed", "response": final},
            ]
        )
    )
    assert result.status == "success"
    assert result.ttft_seconds is not None
    assert result.output_text == "ok"
    missing = asyncio.run(infer([{"type": "response.output_text.delta", "delta": "partial"}]))
    assert missing.status == "server_error"
    assert "missing_terminal_event" in str(missing.error_kind)
