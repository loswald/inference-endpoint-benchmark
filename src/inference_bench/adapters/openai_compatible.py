from __future__ import annotations

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
        "model": route.model,
        "messages": materialize_messages(request),
        "stream": request.stream,
        "max_tokens": request.max_output_tokens,
        **route.request_defaults,
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
            if request.stream:
                return await self._stream(route, request, payload, started_utc, started)
            return await self._single(route, request, payload, started_utc, started)
        except httpx.TimeoutException:
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
        response = await self.client.post(
            route.base_url,
            headers=_headers(route),
            json=payload,
            timeout=request.timeout_seconds,
        )
        headers_at = time.perf_counter()
        raw = await response.aread()
        ended = time.perf_counter()
        retained = _retained(response.headers)
        if response.status_code >= 300:
            return self._error(request, response, raw, started_utc, started, headers_at, ended)
        data = json.loads(raw)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        return InferenceResult(
            logical_id=request.logical_id,
            status="success",
            http_status=response.status_code,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=headers_at - started,
            ttft_seconds=ended - started,
            decode_seconds=None,
            input_tokens=_int_or_none(usage.get("prompt_tokens") or usage.get("input_tokens")),
            output_tokens=_int_or_none(
                usage.get("completion_tokens") or usage.get("output_tokens")
            ),
            cache_read_input_tokens=_int_or_none(
                prompt_details.get("cached_tokens") or usage.get("cache_read_input_tokens")
            ),
            cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
            finish_reason=choice.get("finish_reason"),
            output_text=str(message.get("content") or ""),
            tool_calls=tuple(message.get("tool_calls") or ()),
            provider_request_id=retained.get("x-request-id") or retained.get("request-id"),
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
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
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
            input_tokens=_int_or_none(usage.get("prompt_tokens") or usage.get("input_tokens")),
            output_tokens=_int_or_none(
                usage.get("completion_tokens") or usage.get("output_tokens")
            ),
            cache_read_input_tokens=_int_or_none(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                or usage.get("cache_read_input_tokens")
            ),
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


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
