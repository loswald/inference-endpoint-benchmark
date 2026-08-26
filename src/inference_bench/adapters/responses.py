from __future__ import annotations

import hashlib
import time
from typing import Any

import httpx

from ..json_contract import StrictJSONError, strict_json_loads
from ..models import InferenceResult, RequestSpec, RouteConfig, normalize_finish_reason
from ..payload import materialize_responses
from .openai_compatible import (
    OpenAICompatibleAdapter,
    PreparedTransport,
    _parse_usage,
    _retained,
    _utc_now,
)


def _response_content(data: dict[str, Any]) -> tuple[str, tuple[dict[str, Any], ...]]:
    text: list[str] = []
    tools: list[dict[str, Any]] = []
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        text.append(output_text)
    output = data.get("output", [])
    if not isinstance(output, list):
        raise ValueError("Responses output must be an array")
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Responses output item must be an object")
        if item.get("type") == "function_call":
            tools.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments", ""),
                    },
                }
            )
        for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                value = part.get("text")
                if isinstance(value, str) and value not in text:
                    text.append(value)
    return "".join(text), tuple(tools)


def _finish(data: dict[str, Any]) -> str | None:
    status = data.get("status")
    if status == "completed":
        return "stop"
    if status == "incomplete":
        details = data.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        return "length" if reason == "max_output_tokens" else "other"
    return None


class ResponsesAdapter(OpenAICompatibleAdapter):
    """OpenAI Responses API transport with token-level SSE timing."""

    def prepare(self, route: RouteConfig, request: RequestSpec) -> PreparedTransport:
        self.preflight(route)
        return PreparedTransport(
            payload=materialize_responses(route, request), headers=self.headers(route)
        )

    async def _single(
        self,
        client: httpx.AsyncClient,
        route: RouteConfig,
        request: RequestSpec,
        prepared: PreparedTransport,
        started_utc: str,
        started: float,
    ) -> InferenceResult:
        async with client.stream(
            "POST",
            route.base_url,
            headers=prepared.headers,
            content=prepared.payload.body,
            timeout=request.timeout_seconds,
        ) as response:
            headers_at = time.perf_counter()
            raw = await response.aread()
            ended = time.perf_counter()
            if response.status_code >= 300:
                return self._error(
                    route, request, response, raw, started_utc, started, headers_at, ended
                )
            try:
                data = strict_json_loads(raw)
                if not isinstance(data, dict):
                    raise StrictJSONError("Responses success body must be an object")
                text, tools = _response_content(data)
            except (StrictJSONError, ValueError):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "invalid_responses_success_body",
                )
            usage = _parse_usage(data.get("usage"))
            retained = _retained(response.headers, route)
            return InferenceResult(
                logical_id=request.logical_id,
                status="success",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=headers_at - started,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                usage_parse_errors=usage.errors,
                cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
                finish_reason=normalize_finish_reason(_finish(data)),
                output_text=text,
                tool_calls=tools,
                provider_request_id=(data.get("id") if isinstance(data.get("id"), str) else None),
                retained_headers=retained,
            )

    async def _stream(
        self,
        client: httpx.AsyncClient,
        route: RouteConfig,
        request: RequestSpec,
        prepared: PreparedTransport,
        started_utc: str,
        started: float,
    ) -> InferenceResult:
        pieces: list[str] = []
        event_offsets: list[float] = []
        final_response: dict[str, Any] | None = None
        response_id: str | None = None
        first_visible_at: float | None = None
        headers_at: float | None = None
        wire_digest = hashlib.sha256()
        protocol_errors: list[str] = []
        async with client.stream(
            "POST",
            route.base_url,
            headers=prepared.headers,
            content=prepared.payload.body,
            timeout=request.timeout_seconds,
        ) as response:
            headers_at = time.perf_counter()
            if response.status_code >= 300:
                raw = await response.aread()
                ended = time.perf_counter()
                return self._error(
                    route, request, response, raw, started_utc, started, headers_at, ended
                )
            retained = _retained(response.headers, route)
            async for line in response.aiter_lines():
                wire_digest.update(line.encode("utf-8", errors="replace") + b"\n")
                if not line.startswith("data:"):
                    continue
                raw_event = line[5:].strip()
                if not raw_event or raw_event == "[DONE]":
                    continue
                try:
                    event = strict_json_loads(raw_event)
                except StrictJSONError:
                    protocol_errors.append("invalid_json_event")
                    continue
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    protocol_errors.append("invalid_responses_event")
                    continue
                event_type = event["type"]
                event_response_id = event.get("response_id")
                if isinstance(event_response_id, str):
                    response_id = response_id or event_response_id
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        protocol_errors.append("invalid_output_text_delta")
                        continue
                    now = time.perf_counter()
                    first_visible_at = first_visible_at or now
                    event_offsets.append(now - started)
                    pieces.append(delta)
                elif event_type in {"response.completed", "response.incomplete"}:
                    value = event.get("response")
                    if not isinstance(value, dict):
                        protocol_errors.append("terminal_event_missing_response")
                    else:
                        final_response = value
                        if isinstance(value.get("id"), str):
                            response_id = value["id"]
                elif event_type in {"error", "response.failed"}:
                    protocol_errors.append("provider_reported_failed_response")
        ended = time.perf_counter()
        if final_response is None:
            protocol_errors.append("responses_stream_missing_terminal_event")
        tools: tuple[dict[str, Any], ...] = ()
        final_text = "".join(pieces)
        if final_response is not None:
            try:
                reconstructed_text, tools = _response_content(final_response)
                if reconstructed_text and reconstructed_text != final_text:
                    final_text = reconstructed_text
            except ValueError:
                protocol_errors.append("invalid_terminal_response_output")
        if protocol_errors:
            return InferenceResult(
                logical_id=request.logical_id,
                status="server_error",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=None if headers_at is None else headers_at - started,
                ttft_seconds=None if first_visible_at is None else first_visible_at - started,
                output_event_offsets_seconds=tuple(event_offsets),
                provider_request_id=response_id,
                retained_headers=retained,
                error_kind="protocol_error:" + ",".join(sorted(set(protocol_errors))),
                error_body_sha256=wire_digest.hexdigest(),
            )
        assert final_response is not None
        usage = _parse_usage(final_response.get("usage"))
        ttft = None if first_visible_at is None else first_visible_at - started
        return InferenceResult(
            logical_id=request.logical_id,
            status="success",
            http_status=response.status_code,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=None if headers_at is None else headers_at - started,
            ttft_seconds=ttft,
            decode_seconds=None if ttft is None else max(0.0, ended - started - ttft),
            output_event_offsets_seconds=tuple(event_offsets),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            usage_parse_errors=usage.errors,
            cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
            finish_reason=normalize_finish_reason(_finish(final_response)),
            output_text=final_text,
            tool_calls=tools,
            provider_request_id=response_id,
            retained_headers=retained,
        )
