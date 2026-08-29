from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from ..json_contract import StrictJSONError, strict_json_loads
from ..models import (
    TRANSPORT_HEADER_PROFILE,
    InferenceResult,
    RequestSpec,
    RouteConfig,
    canonical_json,
    normalize_finish_reason,
)
from ..payload import build_openai_compatible_payload, materialize_openai_compatible
from .base import PreparedRequest


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def static_api_key_headers(route: RouteConfig) -> dict[str, str]:
    """Build deterministic headers for routes authenticated by a static API key.

    OAuth-backed adapters override :meth:`OpenAICompatibleAdapter.headers`. Keeping this helper
    public avoids duplicating the control-character and HTTP-header validation contract.
    """
    token = os.environ.get(route.auth.env)
    if not token:
        raise RuntimeError(f"required credential environment variable is unset: {route.auth.env}")
    headers = {
        route.auth.header: f"{route.auth.prefix}{token}",
        "Content-Type": "application/json",
        # httpx otherwise changes this header when optional Brotli/Zstandard packages happen to
        # be installed. Identity encoding makes response-byte behavior deterministic across the
        # exact transport profile committed by RouteConfig and the run manifest.
        "Accept-Encoding": "identity",
        **route.extra_headers,
    }
    if any(
        any(character in name or character in value for character in "\r\n\0")
        for name, value in headers.items()
    ):
        raise RuntimeError("constructed request headers contain prohibited control characters")
    try:
        httpx.Headers(headers)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("constructed request headers are invalid") from exc
    return headers


def _retained(headers: httpx.Headers, route: RouteConfig) -> dict[str, str]:
    allowed = {name.casefold() for name in route.retained_header_names}
    return {name.casefold(): value for name, value in headers.items() if name.casefold() in allowed}


def _status(http_status: int) -> str:
    if 200 <= http_status < 300:
        return "success"
    if http_status == 408:
        return "timeout"
    if http_status == 429:
        return "rate_limited"
    if 400 <= http_status < 500:
        return "client_error"
    return "server_error"


def build_payload(route: RouteConfig, request: RequestSpec) -> dict[str, Any]:
    """Compatibility wrapper around the single canonical payload materializer."""

    return build_openai_compatible_payload(route, request)


@dataclass(frozen=True, slots=True)
class UsageMetrics:
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    errors: tuple[str, ...]


class OpenAICompatibleAdapter:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        http2: bool = False,
        connection_reuse: bool = True,
        transport_max_connections: int = 256,
    ) -> None:
        self.http2 = http2
        self.connection_reuse = connection_reuse
        self.transport_max_connections = transport_max_connections
        self._provided_client = client is not None
        self.client = client
        if self.client is None and self.connection_reuse:
            # HTTP/2 is opt-in and identity-bound. Missing h2 therefore fails during adapter
            # construction/preflight, before a spend-bearing request can be claimed.
            self.client = self._new_client()

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            http2=self.http2,
            # Never inherit an ambient proxy, CA bundle, or netrc credential into a measured
            # route. The explicit pool size is bound into RouteConfig.identity_hash.
            trust_env=False,
            limits=httpx.Limits(
                max_connections=self.transport_max_connections,
                max_keepalive_connections=self.transport_max_connections,
            ),
        )

    async def close(self) -> None:
        if self.client is not None and not self._provided_client:
            await self.client.aclose()

    def headers(self, route: RouteConfig) -> dict[str, str]:
        """Return the exact headers for the next request.

        Subclasses may refresh short-lived credentials here. ``prepare`` is called before the
        durable spend claim, so an authentication refresh failure cannot create an ambiguous
        inference outcome.
        """

        return static_api_key_headers(route)

    def observe_provider_metadata(self, route: RouteConfig, data: dict[str, Any]) -> bool:
        """Validate provider-routing metadata when a specialized adapter supplies it."""

        return False

    def requires_provider_metadata(self, route: RouteConfig) -> bool:
        return False

    def preflight(self, route: RouteConfig) -> None:
        if TRANSPORT_HEADER_PROFILE != "openai-json-accept-encoding-identity/v1":
            raise RuntimeError("unknown transport header profile")
        if not self._provided_client and (
            route.http2 != self.http2 or route.connection_reuse != self.connection_reuse
        ):
            raise RuntimeError("adapter transport does not match route identity")
        if not self._provided_client and (
            route.transport_max_connections != self.transport_max_connections
        ):
            raise RuntimeError("adapter connection pool does not match route identity")
        self.headers(route)

    def prepare(self, route: RouteConfig, request: RequestSpec) -> PreparedRequest:
        self.preflight(route)
        return PreparedRequest(
            payload=materialize_openai_compatible(route, request), headers=self.headers(route)
        )

    @asynccontextmanager
    async def _request_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.client is not None:
            yield self.client
            return
        async with self._new_client() as transient:
            yield transient

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
        """Convenience API for direct callers; the engine always uses the prepared boundary."""

        return await self.send_prepared(route, request, self.prepare(route, request))

    async def send_prepared(
        self, route: RouteConfig, request: RequestSpec, prepared: PreparedRequest
    ) -> InferenceResult:
        started_utc = _utc_now()
        started = time.perf_counter()
        try:
            async with asyncio.timeout(request.timeout_seconds):
                async with self._request_client() as client:
                    if request.stream:
                        return await self._stream(
                            client, route, request, prepared, started_utc, started
                        )
                    return await self._single(
                        client, route, request, prepared, started_utc, started
                    )
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
        client: httpx.AsyncClient,
        route: RouteConfig,
        request: RequestSpec,
        prepared: PreparedRequest,
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
            retained = _retained(response.headers, route)
            if response.status_code >= 300:
                return self._error(
                    route, request, response, raw, started_utc, started, headers_at, ended
                )
            try:
                data = strict_json_loads(raw)
            except StrictJSONError:
                return self._protocol_error(
                    route,
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
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "non_object_json_success_body",
                )
            try:
                canonical_json(data)
            except (TypeError, ValueError):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "nonfinite_or_noncanonical_json_success_body",
                )
            try:
                provider_metadata_seen = self.observe_provider_metadata(route, data)
            except ValueError:
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "provider_attestation_failed",
                )
            if self.requires_provider_metadata(route) and not provider_metadata_seen:
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "provider_attestation_missing",
                )
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "missing_or_invalid_choice",
                )
            if len(choices) != 1:
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "unexpected_multiple_choices",
                )
            choice = choices[0]
            raw_choice_index = choice.get("index", 0)
            if (
                isinstance(raw_choice_index, bool)
                or not isinstance(raw_choice_index, int)
                or raw_choice_index != 0
            ):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "unexpected_choice_index",
                )
            message = choice.get("message")
            if not isinstance(message, dict):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "invalid_choice_message",
                )
            raw_tool_calls: Any = message.get("tool_calls", ())
            # Azure and several compatible gateways serialize this unused
            # optional field as JSON null. Null is equivalent to omission;
            # every other non-array shape remains a protocol error.
            if raw_tool_calls is None:
                raw_tool_calls = ()
            tool_calls, tool_error = _normalize_tool_calls(raw_tool_calls)
            if tool_error:
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    tool_error,
                )
            usage = _parse_usage(data.get("usage"))
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "invalid_message_content",
                )
            refusal = message.get("refusal")
            if refusal is not None and not isinstance(refusal, str):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "invalid_message_refusal",
                )
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and (
                not isinstance(finish_reason, str) or not finish_reason.strip()
            ):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "invalid_finish_reason",
                )
            if not (content or refusal or tool_calls or finish_reason is not None):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "empty_choice_without_terminal_finish",
                )
            if request.logprobs is True and choice.get("logprobs") is None:
                return self._protocol_error(
                    route,
                    request,
                    response,
                    raw,
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    "requested_logprobs_missing",
                )
            return InferenceResult(
                logical_id=request.logical_id,
                status="success",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=headers_at - started,
                ttft_seconds=None,
                decode_seconds=None,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                usage_parse_errors=usage.errors,
                cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
                finish_reason=normalize_finish_reason(finish_reason),
                output_text=content if content is not None else refusal or "",
                tool_calls=tuple(tool_calls),
                provider_request_id=(
                    data.get("id")
                    if isinstance(data.get("id"), str) and data.get("id")
                    else retained.get("x-request-id") or retained.get("request-id")
                ),
                retained_headers=retained,
            )

    async def _stream(
        self,
        client: httpx.AsyncClient,
        route: RouteConfig,
        request: RequestSpec,
        prepared: PreparedRequest,
        started_utc: str,
        started: float,
    ) -> InferenceResult:
        content_by_choice: dict[int, list[str]] = {}
        tool_fragments: dict[tuple[int, int], dict[str, Any]] = {}
        event_offsets: list[float] = []
        usage_values: list[Any] = []
        malformed_reasons: list[str] = []
        finish_reason: str | None = None
        first_visible_at: float | None = None
        headers_at: float | None = None
        valid_event_count = 0
        semantic_event_count = 0
        terminal_choice_count = 0
        terminal_finish_reason_raw: str | None = None
        done_seen = False
        provider_metadata_seen = False
        requested_logprobs_seen = False
        provider_response_id: str | None = None
        wire_digest = hashlib.sha256()
        async with client.stream(
            "POST",
            route.base_url,
            headers=prepared.headers,
            content=prepared.payload.body,
            timeout=request.timeout_seconds,
        ) as response:
            headers_at = time.perf_counter()
            retained = _retained(response.headers, route)
            if response.status_code >= 300:
                raw = await response.aread()
                ended = time.perf_counter()
                return self._error(
                    route, request, response, raw, started_utc, started, headers_at, ended
                )
            async for line in response.aiter_lines():
                wire_digest.update(line.encode("utf-8", errors="replace") + b"\n")
                if not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if not data_text:
                    continue
                if data_text == "[DONE]":
                    if done_seen:
                        malformed_reasons.append("duplicate_done_sentinel")
                    done_seen = True
                    continue
                if done_seen:
                    malformed_reasons.append("data_after_done_sentinel")
                    continue
                try:
                    event = strict_json_loads(data_text)
                except StrictJSONError:
                    malformed_reasons.append("invalid_json_event")
                    continue
                if not isinstance(event, dict):
                    malformed_reasons.append("non_object_event")
                    continue
                try:
                    canonical_json(event)
                except (TypeError, ValueError):
                    malformed_reasons.append("nonfinite_or_noncanonical_json_event")
                    continue
                response_id = event.get("id")
                if isinstance(response_id, str) and response_id:
                    if provider_response_id is not None and provider_response_id != response_id:
                        malformed_reasons.append("conflicting_provider_response_id")
                        continue
                    provider_response_id = response_id
                try:
                    provider_metadata_seen = (
                        self.observe_provider_metadata(route, event) or provider_metadata_seen
                    )
                except ValueError:
                    malformed_reasons.append("provider_attestation_failed")
                    continue
                valid_event_count += 1
                if "usage" in event:
                    usage_values.append(event["usage"])
                choices = event.get("choices", [])
                if not isinstance(choices, list):
                    malformed_reasons.append("choices_not_array")
                    continue
                if len(choices) > 1:
                    malformed_reasons.append("unexpected_multiple_choices")
                    continue
                for position, choice in enumerate(choices):
                    if not isinstance(choice, dict):
                        malformed_reasons.append("choice_not_object")
                        continue
                    raw_choice_index = choice.get("index", position)
                    if (
                        isinstance(raw_choice_index, bool)
                        or not isinstance(raw_choice_index, int)
                        or raw_choice_index != 0
                    ):
                        malformed_reasons.append("unexpected_choice_index")
                        continue
                    choice_index = 0
                    if choice.get("logprobs") is not None:
                        requested_logprobs_seen = True
                    choice_finish_reason = choice.get("finish_reason")
                    if choice_finish_reason is not None and not isinstance(
                        choice_finish_reason, str
                    ):
                        malformed_reasons.append("finish_reason_not_string")
                        continue
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        malformed_reasons.append("delta_not_object")
                        continue
                    piece = delta.get("content")
                    if piece is not None and not isinstance(piece, str):
                        malformed_reasons.append("content_not_string")
                        continue
                    refusal = delta.get("refusal")
                    if refusal is not None and not isinstance(refusal, str):
                        malformed_reasons.append("refusal_not_string")
                        continue
                    reasoning_pieces: list[str] = []
                    reasoning_invalid = False
                    for field in ("reasoning", "reasoning_content"):
                        reasoning_piece = delta.get(field)
                        if reasoning_piece is None:
                            continue
                        if not isinstance(reasoning_piece, str):
                            malformed_reasons.append(f"{field}_not_string")
                            reasoning_invalid = True
                            break
                        reasoning_pieces.append(reasoning_piece)
                    if reasoning_invalid:
                        continue
                    new_tools = delta.get("tool_calls")
                    if new_tools is None:
                        new_tools = []
                    if not isinstance(new_tools, list):
                        malformed_reasons.append("tool_calls_not_array")
                        continue
                    if terminal_choice_count:
                        if (
                            choice_finish_reason == terminal_finish_reason_raw
                            and not piece
                            and not refusal
                            and not reasoning_pieces
                            and not new_tools
                        ):
                            # OpenRouter and some direct providers repeat the same empty terminal
                            # marker before the final usage/DONE event. It carries no additional
                            # model output and is therefore an idempotent transport marker.
                            continue
                        if choice_finish_reason is None:
                            malformed_reasons.append("choice_after_terminal_finish")
                        else:
                            malformed_reasons.append("conflicting_terminal_finish_reasons")
                        continue
                    if choice_finish_reason is not None:
                        if not choice_finish_reason.strip():
                            malformed_reasons.append("empty_terminal_finish_reason")
                            continue
                        # An explicit terminal choice is a valid model outcome even when the
                        # visible completion is empty (for example EOS or a stop-boundary test).
                        # It does not create TTFT/decode evidence, but it prevents a valid empty
                        # model result from being mislabeled as an infrastructure failure.
                        terminal_choice_count += 1
                        terminal_finish_reason_raw = choice_finish_reason
                        finish_reason = normalize_finish_reason(choice_finish_reason)
                    visible = bool(piece or refusal or new_tools)
                    semantic = bool(visible or reasoning_pieces)
                    if semantic:
                        semantic_event_count += 1
                    if visible:
                        now = time.perf_counter()
                        first_visible_at = first_visible_at or now
                        event_offsets.append(now - started)
                    if piece:
                        content_by_choice.setdefault(choice_index, []).append(piece)
                    if refusal:
                        # Refusal text is kept only in memory for deterministic task-quality
                        # scoring and represented by a digest in durable evidence.
                        content_by_choice.setdefault(choice_index, []).append(refusal)
                    for tool_position, fragment in enumerate(new_tools):
                        reason = _merge_tool_delta(
                            tool_fragments, choice_index, tool_position, fragment
                        )
                        if reason:
                            malformed_reasons.append(reason)
        ended = time.perf_counter()
        ttft = None if first_visible_at is None else first_visible_at - started
        decode_proxy = None if ttft is None else max(0.0, (ended - started) - ttft)
        usage = _parse_stream_usage(usage_values)
        if self.requires_provider_metadata(route) and not provider_metadata_seen:
            malformed_reasons.append("provider_attestation_missing")
        if request.logprobs is True and not requested_logprobs_seen:
            malformed_reasons.append("requested_logprobs_missing")
        if route.stream_usage_mode == "required" and not (
            usage.input_tokens is not None and usage.output_tokens is not None
        ):
            usage = UsageMetrics(
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_input_tokens,
                usage.reasoning_tokens,
                usage.total_tokens,
                (*usage.errors, "required_stream_usage_missing"),
            )
        if semantic_event_count > 0 and terminal_choice_count == 0 and not done_seen:
            malformed_reasons.append("sse_stream_ended_without_terminal_signal")
        if (
            malformed_reasons
            or valid_event_count == 0
            or (semantic_event_count == 0 and terminal_choice_count == 0)
        ):
            if valid_event_count == 0:
                malformed_reasons.append("empty_or_invalid_sse_stream")
            elif semantic_event_count == 0:
                malformed_reasons.append("sse_stream_without_content_or_tool_delta")
            return InferenceResult(
                logical_id=request.logical_id,
                status="server_error",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=None if headers_at is None else headers_at - started,
                ttft_seconds=ttft,
                decode_seconds=decode_proxy,
                output_event_offsets_seconds=tuple(event_offsets),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                usage_parse_errors=usage.errors,
                cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
                finish_reason=finish_reason,
                provider_request_id=retained.get("x-request-id") or retained.get("request-id"),
                retained_headers=retained,
                error_kind="protocol_error:" + ",".join(sorted(set(malformed_reasons))),
                error_body_sha256=wire_digest.hexdigest(),
            )
        reconstructed_tools = tuple(
            _finalize_tool_call(key, value) for key, value in sorted(tool_fragments.items())
        )
        reconstructed_tools, tool_error = _normalize_tool_calls(reconstructed_tools)
        if tool_error:
            return InferenceResult(
                logical_id=request.logical_id,
                status="server_error",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=None if headers_at is None else headers_at - started,
                ttft_seconds=ttft,
                decode_seconds=decode_proxy,
                output_event_offsets_seconds=tuple(event_offsets),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                usage_parse_errors=usage.errors,
                cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
                finish_reason=finish_reason,
                provider_request_id=retained.get("x-request-id") or retained.get("request-id"),
                retained_headers=retained,
                error_kind=f"protocol_error:{tool_error}",
                error_body_sha256=wire_digest.hexdigest(),
            )
        output_text = "".join(
            piece
            for choice_index in sorted(content_by_choice)
            for piece in content_by_choice[choice_index]
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
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            usage_parse_errors=usage.errors,
            cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
            finish_reason=finish_reason,
            output_text=output_text,
            tool_calls=reconstructed_tools,
            provider_request_id=(
                provider_response_id or retained.get("x-request-id") or retained.get("request-id")
            ),
            retained_headers=retained,
        )

    def _classify_http_error(
        self, response: httpx.Response, raw: bytes
    ) -> tuple[str, str]:
        """Return a retry/controller status and a sanitized diagnostic category.

        Provider adapters may override this hook when one HTTP status has multiple documented
        meanings. The raw body is available only for local classification and remains represented
        in the ledger by its SHA-256 digest.
        """

        del raw
        return _status(response.status_code), f"http_{response.status_code}"

    def _error(
        self,
        route: RouteConfig,
        request: RequestSpec,
        response: httpx.Response,
        raw: bytes,
        started_utc: str,
        started: float,
        headers_at: float,
        ended: float,
    ) -> InferenceResult:
        retained = _retained(response.headers, route)
        status, error_kind = self._classify_http_error(response, raw)
        return InferenceResult(
            logical_id=request.logical_id,
            status=status,  # type: ignore[arg-type]
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

    @staticmethod
    def _protocol_error(
        route: RouteConfig,
        request: RequestSpec,
        response: httpx.Response,
        raw: bytes,
        started_utc: str,
        started: float,
        headers_at: float,
        ended: float,
        error_kind: str,
    ) -> InferenceResult:
        retained = _retained(response.headers, route)
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


def _merge_tool_delta(
    buffers: dict[tuple[int, int], dict[str, Any]],
    choice_index: int,
    fallback_index: int,
    fragment: Any,
) -> str | None:
    if not isinstance(fragment, dict):
        return "tool_delta_not_object"
    # Some OpenAI-compatible servers omit the tool index when there is only one call. Positional
    # fallback is allowed only for an actually absent field. A present malformed value must fail
    # closed; coercing it could turn a malformed response into a correctly scored tool call.
    if "index" in fragment:
        raw_tool_index = fragment["index"]
        if (
            isinstance(raw_tool_index, bool)
            or not isinstance(raw_tool_index, int)
            or raw_tool_index < 0
        ):
            return "tool_delta_index_invalid"
        tool_index = raw_tool_index
    else:
        tool_index = fallback_index
    key = (choice_index, tool_index)
    target = buffers.setdefault(
        key,
        {
            "index": tool_index,
            "id": "",
            "type": "",
            "function": {"name": "", "arguments": ""},
        },
    )
    identifier = fragment.get("id")
    if identifier is not None:
        if not isinstance(identifier, str):
            return "tool_delta_id_not_string"
        target["id"] += identifier
    tool_type = fragment.get("type")
    if tool_type is not None:
        if not isinstance(tool_type, str):
            return "tool_delta_type_not_string"
        if target["type"] and target["type"] != tool_type:
            return "tool_delta_type_changed"
        target["type"] = tool_type
    function = fragment.get("function")
    if function is not None:
        if not isinstance(function, dict):
            return "tool_delta_function_not_object"
        for field in ("name", "arguments"):
            value = function.get(field)
            if value is not None:
                if not isinstance(value, str):
                    return f"tool_delta_function_{field}_not_string"
                target["function"][field] += value
    return None


def _finalize_tool_call(key: tuple[int, int], value: dict[str, Any]) -> dict[str, Any]:
    return {"choice_index": key[0], **value, "type": value.get("type") or "function"}


def _normalize_tool_calls(value: Any) -> tuple[tuple[dict[str, Any], ...], str | None]:
    if not isinstance(value, (list, tuple)):
        return (), "invalid_tool_calls"
    normalized: list[dict[str, Any]] = []
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            return (), "tool_call_not_object"
        identifier = item.get("id")
        tool_type = item.get("type", "function")
        function = item.get("function")
        if not isinstance(identifier, str) or not identifier:
            return (), "tool_call_id_missing_or_not_string"
        if tool_type != "function" or not isinstance(function, dict):
            return (), "tool_call_type_or_function_invalid"
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, str):
            return (), "tool_call_name_or_arguments_invalid"
        try:
            parsed_arguments = strict_json_loads(arguments)
            canonical_json(parsed_arguments)
        except (StrictJSONError, TypeError, ValueError):
            return (), "tool_call_arguments_invalid_json"
        choice_index = item.get("choice_index", 0)
        tool_index = item.get("index", position)
        if (
            isinstance(choice_index, bool)
            or not isinstance(choice_index, int)
            or choice_index < 0
            or isinstance(tool_index, bool)
            or not isinstance(tool_index, int)
            or tool_index < 0
        ):
            return (), "tool_call_index_invalid"
        normalized.append(
            {
                "choice_index": choice_index,
                "index": tool_index,
                "id": identifier,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return tuple(normalized), None


def _integral_count(value: Any, field: str, errors: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field}_wrong_json_type")
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        errors.append(f"{field}_nonintegral_or_negative")
        return None
    return int(number)


def _consistent_alias_count(
    metric: str, candidates: list[tuple[str, Any]], errors: list[str]
) -> int | None:
    if not candidates:
        return None
    parsed: list[int] = []
    all_valid = True
    for field, raw_value in candidates:
        value = _integral_count(raw_value, field, errors)
        if value is None:
            all_valid = False
        else:
            parsed.append(value)
    if not all_valid:
        return None
    if len(set(parsed)) > 1:
        errors.append(f"{metric}_alias_conflict")
        return None
    return parsed[0]


def _parse_usage(value: Any) -> UsageMetrics:
    if value is None:
        return UsageMetrics(None, None, None, None, None, ())
    if not isinstance(value, dict):
        return UsageMetrics(None, None, None, None, None, ("usage_wrong_json_type",))
    errors: list[str] = []

    input_tokens = _consistent_alias_count(
        "input_tokens",
        [
            (key, value[key])
            for key in ("prompt_tokens", "input_tokens")
            if value.get(key) is not None
        ],
        errors,
    )
    output_tokens = _consistent_alias_count(
        "output_tokens",
        [
            (key, value[key])
            for key in ("completion_tokens", "output_tokens")
            if value.get(key) is not None
        ],
        errors,
    )
    total_tokens = _consistent_alias_count(
        "total_tokens",
        [("total_tokens", value["total_tokens"])] if value.get("total_tokens") is not None else [],
        errors,
    )
    cache_candidates: list[tuple[str, Any]] = []
    for details_field in ("prompt_tokens_details", "input_tokens_details"):
        details = value.get(details_field)
        if details is not None and not isinstance(details, dict):
            errors.append(f"{details_field}_wrong_json_type")
        elif isinstance(details, dict) and details.get("cached_tokens") is not None:
            cache_candidates.append(
                (f"{details_field}.cached_tokens", details["cached_tokens"])
            )
    if "cache_read_input_tokens" in value and value["cache_read_input_tokens"] is not None:
        cache_candidates.append(("cache_read_input_tokens", value["cache_read_input_tokens"]))
    cache_read = _consistent_alias_count("cache_read_input_tokens", cache_candidates, errors)
    reasoning_candidates: list[tuple[str, Any]] = []
    for details_field in ("completion_tokens_details", "output_tokens_details"):
        details = value.get(details_field)
        if details is not None and not isinstance(details, dict):
            errors.append(f"{details_field}_wrong_json_type")
        elif isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            reasoning_candidates.append(
                (f"{details_field}.reasoning_tokens", details["reasoning_tokens"])
            )
    if "reasoning_tokens" in value and value["reasoning_tokens"] is not None:
        reasoning_candidates.append(("reasoning_tokens", value["reasoning_tokens"]))
    reasoning = _consistent_alias_count("reasoning_tokens", reasoning_candidates, errors)
    if (
        total_tokens is not None
        and input_tokens is not None
        and output_tokens is not None
        and total_tokens != input_tokens + output_tokens
        and (
            reasoning is None
            or total_tokens != input_tokens + output_tokens + reasoning
        )
    ):
        errors.append("total_tokens_mismatch_input_plus_output")
    return UsageMetrics(
        input_tokens, output_tokens, cache_read, reasoning, total_tokens, tuple(errors)
    )


def _parse_stream_usage(values: list[Any]) -> UsageMetrics:
    """Merge repeated stream usage snapshots only when counts are monotonic or equal."""

    aggregate: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_input_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }
    errors: list[str] = []
    conflicted: set[str] = set()
    for raw_value in values:
        parsed = _parse_usage(raw_value)
        errors.extend(parsed.errors)
        for field in aggregate:
            current = getattr(parsed, field)
            if current is None or field in conflicted:
                continue
            previous = aggregate[field]
            if previous is not None and current < previous:
                errors.append(f"stream_{field}_decreased")
                aggregate[field] = None
                conflicted.add(field)
            else:
                aggregate[field] = current
    if (
        aggregate["total_tokens"] is not None
        and aggregate["input_tokens"] is not None
        and aggregate["output_tokens"] is not None
        and aggregate["total_tokens"] != aggregate["input_tokens"] + aggregate["output_tokens"]
        and (
            aggregate["reasoning_tokens"] is None
            or aggregate["total_tokens"]
            != aggregate["input_tokens"]
            + aggregate["output_tokens"]
            + aggregate["reasoning_tokens"]
        )
    ):
        errors.append("total_tokens_mismatch_input_plus_output")
    return UsageMetrics(
        aggregate["input_tokens"],
        aggregate["output_tokens"],
        aggregate["cache_read_input_tokens"],
        aggregate["reasoning_tokens"],
        aggregate["total_tokens"],
        tuple(dict.fromkeys(errors)),
    )


def _usage_count(usage: dict[str, Any], *keys: str) -> int | None:
    """Compatibility helper: reject nonintegral/wrong-type counts instead of coercing them."""

    errors: list[str] = []
    for key in keys:
        if key in usage and usage[key] is not None:
            return _integral_count(usage[key], key, errors)
    return None


def _cache_read_count(usage: dict[str, Any]) -> int | None:
    return _parse_usage(usage).cache_read_input_tokens
