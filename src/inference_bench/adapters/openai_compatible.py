from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from ..models import InferenceResult, RequestSpec, RouteConfig
from ..workloads import materialize_messages

RETAINED_HEADER_NAMES = {
    "x-request-id",
    "request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _headers(route: RouteConfig) -> dict[str, str]:
    token = os.environ.get(route.auth.env)
    if not token:
        raise RuntimeError(f"required credential environment variable is unset: {route.auth.env}")
    return {
        route.auth.header: f"{route.auth.prefix}{token}",
        "Content-Type": "application/json",
        **route.extra_headers,
    }


def _retained(headers: httpx.Headers) -> dict[str, str]:
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower() in RETAINED_HEADER_NAMES
    }


def _status(http_status: int) -> str:
    if 200 <= http_status < 300:
        return "success"
    if http_status == 429:
        return "rate_limited"
    if 400 <= http_status < 500:
        return "client_error"
    return "server_error"


def build_payload(route: RouteConfig, request: RequestSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **copy.deepcopy(route.request_defaults),
        "model": route.model,
        "messages": materialize_messages(request),
        "stream": request.stream,
        "max_tokens": request.max_output_tokens,
    }
    for key, value in (
        ("temperature", request.temperature),
        ("top_p", request.top_p),
        ("seed", request.seed),
        ("tool_choice", request.tool_choice),
        ("response_format", request.response_format),
        ("logprobs", request.logprobs),
    ):
        if value is not None:
            payload[key] = value
    if request.stop:
        payload["stop"] = list(request.stop)
    if request.tools:
        payload["tools"] = list(request.tools)
    if request.stream:
        # OpenAI-compatible providers commonly require this flag to return the final usage chunk.
        # A route can add other stream options, but cannot replace the measured stream setting.
        options = payload.setdefault("stream_options", {"include_usage": True})
        if isinstance(options, dict):
            options.setdefault("include_usage", True)
    else:
        payload.pop("stream_options", None)
    if route.adapter == "openrouter":
        # Exact serving identity is mandatory: one upstream, no fallback.
        payload["provider"] = {
            "only": [route.upstream_provider],
            "order": [route.upstream_provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    return payload


class OpenAICompatibleAdapter:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            http2=True, limits=httpx.Limits(max_connections=256)
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
        payload = build_payload(route, request)
        started_utc = _utc_now()
        started = time.perf_counter()
        try:
            # httpx read timeouts reset after each received chunk.  The outer deadline instead
            # bounds the full response stream, including a provider that trickles indefinitely.
            async with asyncio.timeout(request.timeout_seconds):
                if request.stream:
                    return await self._stream(route, request, payload, started_utc, started)
                return await self._single(route, request, payload, started_utc, started)
        except (TimeoutError, httpx.TimeoutException):
            ended = time.perf_counter()
            return InferenceResult(
                logical_id=request.logical_id,
                status="timeout",
                http_status=None,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                error_kind="timeout",
            )
        except httpx.TransportError as exc:
            ended = time.perf_counter()
            return InferenceResult(
                logical_id=request.logical_id,
                status="transport_error",
                http_status=None,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                error_kind=type(exc).__name__,
            )

    async def _single(
        self,
        route: RouteConfig,
        request: RequestSpec,
        payload: dict[str, Any],
        started_utc: str,
        started: float,
    ) -> InferenceResult:
        # Use streaming transport even for a non-streaming JSON body so header arrival is measured
        # before body drain. `client.post()` eagerly reads the body and would falsely label
        # time-to-complete as time-to-headers.
        async with self.client.stream(
            "POST",
            route.base_url,
            headers=_headers(route),
            json=payload,
            timeout=request.timeout_seconds,
        ) as response:
            headers_at = time.perf_counter()
            raw = await response.aread()
            ended = time.perf_counter()
            retained = _retained(response.headers)
            if response.status_code >= 300:
                return self._error(
                    request, response, raw, started_utc, started, headers_at, ended
                )
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return self._protocol_error(
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "invalid_json_success_body",
                )
            if not isinstance(data, dict):
                return self._protocol_error(
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "non_object_json_success_body",
                )
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                return self._protocol_error(
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "missing_or_invalid_choice",
                )
            choice = choices[0]
            message = choice.get("message") or {}
            if not isinstance(message, dict):
                return self._protocol_error(
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "invalid_choice_message",
                )
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            return InferenceResult(
                logical_id=request.logical_id,
                status="success",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=headers_at - started,
                # Non-streaming has no first-token observation.
                ttft_seconds=None,
                decode_seconds=None,
                input_tokens=_usage_count(usage, "prompt_tokens", "input_tokens"),
                output_tokens=_usage_count(usage, "completion_tokens", "output_tokens"),
                cache_read_input_tokens=_cache_read_count(usage),
                cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
                finish_reason=choice.get("finish_reason"),
                output_text=str(message.get("content") or ""),
                tool_calls=tuple(message.get("tool_calls") or ()),
                provider_request_id=retained.get("x-request-id")
                or retained.get("request-id"),
                retained_headers=retained,
            )

    async def _stream(
        self,
        route: RouteConfig,
        request: RequestSpec,
        payload: dict[str, Any],
        started_utc: str,
        started: float,
    ) -> InferenceResult:
        content: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        event_offsets: list[float] = []
        usage: dict[str, Any] = {}
        malformed_events: list[str] = []
        finish_reason: str | None = None
        first_content_at: float | None = None
        headers_at: float | None = None
        async with self.client.stream(
            "POST",
            route.base_url,
            headers=_headers(route),
            json=payload,
            timeout=request.timeout_seconds,
        ) as response:
            headers_at = time.perf_counter()
            retained = _retained(response.headers)
            if response.status_code >= 300:
                raw = await response.aread()
                ended = time.perf_counter()
                return self._error(request, response, raw, started_utc, started, headers_at, ended)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if not data_text or data_text == "[DONE]":
                    continue
                try:
                    event = json.loads(data_text)
                except json.JSONDecodeError:
                    malformed_events.append(data_text)
                    continue
                if not isinstance(event, dict):
                    malformed_events.append(data_text)
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    malformed_events.append(data_text)
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                new_tools = delta.get("tool_calls") or []
                # A content event is a provider SSE event carrying non-empty text or tool data,
                # not an assumed token. Event timestamps are retained for diagnostics only.
                if piece or new_tools:
                    now = time.perf_counter()
                    first_content_at = first_content_at or now
                    event_offsets.append(now - started)
                if piece:
                    content.append(str(piece))
                tool_calls.extend(new_tools)
        ended = time.perf_counter()
        ttft = None if first_content_at is None else first_content_at - started
        # Deliberately do not call last-SSE minus first-SSE "decode duration". Providers batch
        # arbitrary token counts per event; that produced fake six-figure token/s measurements.
        # The validity layer derives the comparable request proxy from total_seconds - TTFT.
        decode_proxy = None if ttft is None else max(0.0, (ended - started) - ttft)
        if malformed_events:
            malformed_digest = hashlib.sha256(
                "\n".join(malformed_events).encode("utf-8", errors="replace")
            ).hexdigest()
            return InferenceResult(
                logical_id=request.logical_id,
                status="server_error",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=(
                    None if headers_at is None else headers_at - started
                ),
                ttft_seconds=ttft,
                decode_seconds=decode_proxy,
                output_event_offsets_seconds=tuple(event_offsets),
                input_tokens=_usage_count(usage, "prompt_tokens", "input_tokens"),
                output_tokens=_usage_count(usage, "completion_tokens", "output_tokens"),
                cache_read_input_tokens=_cache_read_count(usage),
                finish_reason=finish_reason,
                provider_request_id=retained.get("x-request-id")
                or retained.get("request-id"),
                retained_headers=retained,
                error_kind="malformed_sse_json_event",
                error_body_sha256=malformed_digest,
            )
        return InferenceResult(
            logical_id=request.logical_id,
            status="success",
            http_status=response.status_code,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=None if headers_at is None else headers_at - started,
            ttft_seconds=ttft,
            decode_seconds=decode_proxy,
            output_event_offsets_seconds=tuple(event_offsets),
            input_tokens=_usage_count(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_count(usage, "completion_tokens", "output_tokens"),
            cache_read_input_tokens=_cache_read_count(usage),
            cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
            finish_reason=finish_reason,
            output_text="".join(content),
            tool_calls=tuple(tool_calls),
            provider_request_id=retained.get("x-request-id") or retained.get("request-id"),
            retained_headers=retained,
        )

    @staticmethod
    def _error(
        request: RequestSpec,
        response: httpx.Response,
        raw: bytes,
        started_utc: str,
        started: float,
        headers_at: float,
        ended: float,
    ) -> InferenceResult:
        retained = _retained(response.headers)
        return InferenceResult(
            logical_id=request.logical_id,
            status=_status(response.status_code),  # type: ignore[arg-type]
            http_status=response.status_code,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=headers_at - started,
            provider_request_id=retained.get("x-request-id") or retained.get("request-id"),
            retained_headers=retained,
            error_kind=f"http_{response.status_code}",
            error_body_sha256=hashlib.sha256(raw).hexdigest(),
        )

    @staticmethod
    def _protocol_error(
        request: RequestSpec,
        response: httpx.Response,
        raw: bytes,
        started_utc: str,
        started: float,
        headers_at: float,
        ended: float,
        error_kind: str,
    ) -> InferenceResult:
        retained = _retained(response.headers)
        return InferenceResult(
            logical_id=request.logical_id,
            status="server_error",
            http_status=response.status_code,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=headers_at - started,
            provider_request_id=retained.get("x-request-id") or retained.get("request-id"),
            retained_headers=retained,
            error_kind=error_kind,
            error_body_sha256=hashlib.sha256(raw).hexdigest(),
        )


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_count(usage: dict[str, Any], *keys: str) -> int | None:
    """Return the first explicitly reported non-null count, preserving a reported zero."""
    for key in keys:
        if key in usage and usage[key] is not None:
            return _int_or_none(usage[key])
    return None


def _cache_read_count(usage: dict[str, Any]) -> int | None:
    """Keep an explicit zero cache read distinct from an unreported/unknown cache state."""
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return _int_or_none(details["cached_tokens"])
    return _usage_count(usage, "cache_read_input_tokens")
