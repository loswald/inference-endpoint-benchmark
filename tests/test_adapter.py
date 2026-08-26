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


def test_stream_usage_extension_is_route_identity_bound_and_omittable(route) -> None:
    assert "stream_options" not in build_payload(route, _spec(stream=True))
    assert "stream_options" not in build_payload(route, _spec(stream=False))
    for mode in ("try", "required"):
        configured = replace(route, stream_usage_mode=mode)
        assert build_payload(configured, _spec(stream=True))["stream_options"] == {
            "include_usage": True
        }


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
        stream_usage_mode="try",
        request_defaults={"user": "benchmark-fixture"},
    )
    before = configured.identity_hash
    payload = build_payload(configured, _spec(stream=True))
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["user"] == "benchmark-fixture"
    assert configured.request_defaults == {"user": "benchmark-fixture"}
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
        assert request.headers["Accept-Encoding"] == "identity"
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


def test_finish_only_stream_is_valid_empty_model_outcome_but_done_only_is_protocol_error(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def infer(body: str):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=True))
        await client.aclose()
        return result

    finish_only = (
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":0}}\n\n'
        "data: [DONE]\n\n"
    )
    valid = asyncio.run(infer(finish_only))
    assert valid.status == "success"
    assert valid.output_text == ""
    assert valid.finish_reason == "stop"
    assert valid.ttft_seconds is None
    assert valid.content_event_count == 0

    invalid = asyncio.run(infer("data: [DONE]\n\n"))
    assert invalid.status == "server_error"
    assert "empty_or_invalid_sse_stream" in str(invalid.error_kind)


def test_stream_requires_explicit_terminal_signal_and_rejects_conflicting_finishes(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def infer(body: str):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=True))
        await client.aclose()
        return result

    abrupt_content = asyncio.run(
        infer('data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n')
    )
    assert abrupt_content.status == "server_error"
    assert "sse_stream_ended_without_terminal_signal" in str(abrupt_content.error_kind)

    abrupt_reasoning = asyncio.run(
        infer(
            'data: {"choices":[{"index":0,'
            '"delta":{"reasoning_content":"partial hidden work"}}]}\n\n'
        )
    )
    assert abrupt_reasoning.status == "server_error"
    assert "sse_stream_ended_without_terminal_signal" in str(abrupt_reasoning.error_kind)

    done_terminated = asyncio.run(
        infer('data: {"choices":[{"index":0,"delta":{"content":"complete"}}]}\n\ndata: [DONE]\n\n')
    )
    assert done_terminated.status == "success"
    assert done_terminated.output_text == "complete"
    assert done_terminated.finish_reason is None

    finish_terminated = asyncio.run(
        infer(
            'data: {"choices":[{"index":0,"delta":{"content":"complete"},'
            '"finish_reason":"stop"}]}\n\n'
        )
    )
    assert finish_terminated.status == "success"
    assert finish_terminated.finish_reason == "stop"

    conflicting = asyncio.run(
        infer(
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
        )
    )
    assert conflicting.status == "server_error"
    assert "conflicting_terminal_finish_reasons" in str(conflicting.error_kind)

    repeated = asyncio.run(
        infer(
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
    )
    assert repeated.status == "success"
    assert repeated.finish_reason == "stop"

    repeated_with_content = asyncio.run(
        infer(
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"late"},'
            '"finish_reason":"stop"}]}\n\n'
        )
    )
    assert repeated_with_content.status == "server_error"
    assert "conflicting_terminal_finish_reasons" in str(repeated_with_content.error_kind)


def test_stream_refusal_and_reasoning_only_are_not_provider_failures(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def run(chunks):
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body + "data: [DONE]\n\n")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec())
        await client.aclose()
        return result

    refusal = asyncio.run(
        run(
            [
                {"choices": [{"index": 0, "delta": {"refusal": "cannot comply"}}]},
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            ]
        )
    )
    assert refusal.status == "success"
    assert refusal.output_text == "cannot comply"
    assert refusal.ttft_seconds is not None
    assert refusal.content_event_count == 1

    reasoning = asyncio.run(
        run(
            [
                {"choices": [{"index": 0, "delta": {"reasoning_content": "hidden"}}]},
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            ]
        )
    )
    assert reasoning.status == "success"
    assert reasoning.output_text == ""
    assert reasoning.ttft_seconds is None
    assert reasoning.content_event_count == 0
    assert reasoning.reasoning_tokens == 2


def test_unexpected_multiple_choices_fail_consistently(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def run(body: str, *, stream: bool):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=stream))
        await client.aclose()
        return result

    nonstream = asyncio.run(
        run(
            json.dumps(
                {
                    "choices": [
                        {"message": {"content": "a"}},
                        {"message": {"content": "b"}},
                    ]
                }
            ),
            stream=False,
        )
    )
    assert nonstream.status == "server_error"
    assert nonstream.error_kind == "unexpected_multiple_choices"

    stream_body = (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "a"}},
                    {"index": 1, "delta": {"content": "b"}},
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )
        + "\n\ndata: [DONE]\n\n"
    )
    streamed = asyncio.run(run(stream_body, stream=True))
    assert streamed.status == "server_error"
    assert "unexpected_multiple_choices" in str(streamed.error_kind)
    assert streamed.input_tokens == 3
    assert streamed.output_tokens == 2


def test_nonstream_final_choice_requires_valid_index_semantic_output_or_finish(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def infer(choice: dict[str, object]):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [choice]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    empty = asyncio.run(infer({"message": {}, "finish_reason": None}))
    assert empty.status == "server_error"
    assert empty.error_kind == "empty_choice_without_terminal_finish"

    bad_index = asyncio.run(
        infer({"index": "evil", "message": {"content": "text"}, "finish_reason": "stop"})
    )
    assert bad_index.status == "server_error"
    assert bad_index.error_kind == "unexpected_choice_index"

    empty_finish = asyncio.run(infer({"message": {"content": "text"}, "finish_reason": ""}))
    assert empty_finish.status == "server_error"
    assert empty_finish.error_kind == "invalid_finish_reason"

    finish_only = asyncio.run(infer({"index": 0, "message": {}, "finish_reason": "stop"}))
    assert finish_only.status == "success"
    assert finish_only.output_text == ""
    assert finish_only.finish_reason == "stop"

    content_only = asyncio.run(
        infer({"index": 0, "message": {"content": "text"}, "finish_reason": None})
    )
    assert content_only.status == "success"
    assert content_only.output_text == "text"

    tool_only = asyncio.run(
        infer(
            {
                "index": 0,
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id":1}'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        )
    )
    assert tool_only.status == "success"
    assert len(tool_only.tool_calls) == 1


def test_stream_protocol_error_preserves_parseable_usage(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")
    event = {
        "choices": "wrong-type",
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await OpenAICompatibleAdapter(client).infer(route, _spec())
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "server_error"
    assert result.input_tokens == 11
    assert result.output_tokens == 4
    assert result.cache_read_input_tokens == 0


def test_explicit_zero_usage_and_cache_miss_are_not_treated_as_unknown(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")
    body = json.dumps(
        {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cache_read_input_tokens": 0,
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


def test_usage_aliases_must_agree_and_repeated_stream_usage_cannot_decrease(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def infer(body: str, *, stream: bool):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=stream))
        await client.aclose()
        return result

    equal = asyncio.run(
        infer(
            json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 3,
                        "input_tokens": 3,
                        "completion_tokens": 2,
                        "output_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                        "cache_read_input_tokens": 0,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                        "reasoning_tokens": 0,
                    },
                }
            ),
            stream=False,
        )
    )
    assert equal.usage_parse_errors == ()
    assert (equal.input_tokens, equal.output_tokens, equal.cache_read_input_tokens) == (3, 2, 0)

    conflicting = asyncio.run(
        infer(
            json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 3,
                        "input_tokens": 2,
                        "completion_tokens": 2,
                        "output_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 1},
                        "cache_read_input_tokens": 0,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                        "reasoning_tokens": 1,
                    },
                }
            ),
            stream=False,
        )
    )
    assert conflicting.input_tokens is None
    assert conflicting.cache_read_input_tokens is None
    assert conflicting.reasoning_tokens is None
    assert {
        "input_tokens_alias_conflict",
        "cache_read_input_tokens_alias_conflict",
        "reasoning_tokens_alias_conflict",
    }.issubset(conflicting.usage_parse_errors)

    stream_body = (
        "".join(
            f"data: {json.dumps(event)}\n\n"
            for event in (
                {
                    "choices": [{"index": 0, "delta": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                },
            )
        )
        + "data: [DONE]\n\n"
    )
    streamed = asyncio.run(infer(stream_body, stream=True))
    assert streamed.input_tokens is None
    assert streamed.output_tokens == 2
    assert "stream_input_tokens_decreased" in streamed.usage_parse_errors


def test_total_tokens_must_match_components_for_nonstream_and_stream(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def infer(body: str, *, stream: bool):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=stream))
        await client.aclose()
        return result

    response = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 999},
    }
    nonstream = asyncio.run(infer(json.dumps(response), stream=False))
    assert "total_tokens_mismatch_input_plus_output" in nonstream.usage_parse_errors

    stream_event = {
        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 999},
    }
    stream_body = f"data: {json.dumps(stream_event)}\n\ndata: [DONE]\n\n"
    streamed = asyncio.run(infer(stream_body, stream=True))
    assert "total_tokens_mismatch_input_plus_output" in streamed.usage_parse_errors

    split_usage_body = (
        'data: {"choices":[{"index":0,"delta":{"content":"ok"}}],'
        '"usage":{"prompt_tokens":3,"total_tokens":999}}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n'
    )
    split = asyncio.run(infer(split_usage_body, stream=True))
    assert "total_tokens_mismatch_input_plus_output" in split.usage_parse_errors


def test_duplicate_keys_are_protocol_errors_in_body_sse_and_tool_arguments(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def infer(body: str, *, stream: bool):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=stream))
        await client.aclose()
        return result

    duplicate_body = (
        '{"choices":[{"message":{"content":"first","content":"second"},'
        '"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
    )
    nonstream = asyncio.run(infer(duplicate_body, stream=False))
    assert nonstream.status == "server_error"
    assert nonstream.error_kind == "invalid_json_success_body"

    duplicate_sse = (
        'data: {"choices":[{"index":0,"delta":{"content":"first","content":"second"},'
        '"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    streamed = asyncio.run(infer(duplicate_sse, stream=True))
    assert streamed.status == "server_error"
    assert "invalid_json_event" in str(streamed.error_kind)

    duplicate_tool_arguments = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": '{"city":"wrong","city":"Reykjavík"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    tools = asyncio.run(infer(duplicate_tool_arguments, stream=False))
    assert tools.status == "server_error"
    assert tools.error_kind == "tool_call_arguments_invalid_json"


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


def test_malformed_nonstream_tool_calls_and_nonfinite_json_are_protocol_errors(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")
    bodies = [
        json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": None, "tool_calls": ["not-an-object"]},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":NaN,"completion_tokens":1}}',
    ]

    async def run(body: str):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    malformed_tools, nonfinite = (asyncio.run(run(body)) for body in bodies)
    assert malformed_tools.status == "server_error"
    assert malformed_tools.error_kind == "tool_call_not_object"
    assert nonfinite.status == "server_error"
    assert nonfinite.error_kind == "invalid_json_success_body"


def test_null_tool_calls_are_absent_but_other_falsy_shapes_are_rejected(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def run(value):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "ok", "tool_calls": value}, "finish_reason": "stop"}
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    null_result = asyncio.run(run(None))
    assert null_result.status == "success"

    for value in ({}, "", 0):
        result = asyncio.run(run(value))
        assert result.status == "server_error"
        assert result.error_kind == "invalid_tool_calls"


def test_stream_allows_explicit_null_tool_calls(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")
    body = (
        'data: {"choices":[{"index":0,"delta":{"content":"ok",'
        '"tool_calls":null},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"tool_calls":null},'
        '"finish_reason":"stop"}],"usage":{"prompt_tokens":2,'
        '"completion_tokens":1}}\n\n'
        'data: [DONE]\n\n'
    )

    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=True))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "success"
    assert result.finish_reason == "stop"


def test_stream_tool_delta_rejects_explicit_bad_index_but_allows_absent_index(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def infer(tool_call: dict[str, object]):
        event = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [tool_call]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=True))
        await client.aclose()
        return result

    base_tool: dict[str, object] = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"id":1}'},
    }
    absent = asyncio.run(infer(base_tool))
    assert absent.status == "success"
    assert len(absent.tool_calls) == 1
    assert absent.tool_calls[0]["index"] == 0

    for malformed in ("evil", True, -1, 1.5, None):
        explicit = asyncio.run(infer({**base_tool, "index": malformed}))
        assert explicit.status == "server_error"
        assert "tool_delta_index_invalid" in str(explicit.error_kind)
        assert explicit.tool_calls == ()


def test_http_408_is_a_retryable_timeout_status(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "not-written")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(408, text="timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await OpenAICompatibleAdapter(client).infer(route, _spec(stream=False))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "timeout"
    assert result.http_status == 408
