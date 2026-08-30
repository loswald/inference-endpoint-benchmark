from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from botocore.eventstream import EventStreamBuffer
from botocore.parsers import ResponseParserFactory
from botocore.session import Session

from ..json_contract import StrictJSONError, strict_json_loads
from ..models import InferenceResult, RequestSpec, RouteConfig, canonical_json
from ..payload import materialize_bedrock_converse
from .base import PreparedRequest


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _retained(headers: httpx.Headers, route: RouteConfig) -> dict[str, str]:
    allowed = {name.casefold() for name in route.retained_header_names}
    return {name.casefold(): value for name, value in headers.items() if name.casefold() in allowed}


def _cache_state(request: RequestSpec) -> str:
    return str(request.metadata.get("cache_state", "uncontrolled"))


def _finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return "other"
    normalized = value.strip().casefold().replace("-", "_")
    return {
        "end_turn": "end_turn",
        "tool_use": "tool_calls",
        "max_tokens": "max_tokens",
        "max_total_tokens": "max_tokens",
        "model_context_window_exceeded": "max_tokens",
        "stop_sequence": "stop",
        "guardrail_intervened": "content_filter",
        "content_filtered": "content_filter",
    }.get(normalized, "other")


@dataclass(frozen=True, slots=True)
class _Usage:
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    errors: tuple[str, ...]


def _usage_count(
    value: dict[str, Any], wire_name: str, public_name: str
) -> tuple[int | None, str | None]:
    if wire_name not in value:
        return None, None
    count = value[wire_name]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None, f"{public_name}_nonintegral_or_negative"
    return count, None


def _parse_usage(value: Any) -> _Usage:
    if not isinstance(value, dict):
        return _Usage(None, None, None, ("usage_wrong_json_type",))
    input_tokens, input_error = _usage_count(value, "inputTokens", "input_tokens")
    output_tokens, output_error = _usage_count(value, "outputTokens", "output_tokens")
    cached_tokens, cached_error = _usage_count(
        value, "cacheReadInputTokens", "cache_read_input_tokens"
    )
    total_tokens, total_error = _usage_count(value, "totalTokens", "total_tokens")
    errors = [
        error for error in (input_error, output_error, cached_error, total_error) if error
    ]
    if (
        total_tokens is not None
        and input_tokens is not None
        and output_tokens is not None
        and total_tokens != input_tokens + output_tokens
    ):
        errors.append("total_tokens_mismatch_input_plus_output")
    return _Usage(input_tokens, output_tokens, cached_tokens, tuple(errors))


def _tool_call(block: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(block, dict):
        return None, "tool_use_not_object"
    tool_id = block.get("toolUseId")
    name = block.get("name")
    arguments = block.get("input")
    if not isinstance(tool_id, str) or not tool_id:
        return None, "tool_use_id_missing"
    if not isinstance(name, str) or not name:
        return None, "tool_use_name_missing"
    try:
        arguments_json = canonical_json(arguments)
    except (TypeError, ValueError):
        return None, "tool_use_input_not_json"
    return (
        {
            "id": tool_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments_json},
        },
        None,
    )


def _parse_output_content(value: Any) -> tuple[str, tuple[dict[str, Any], ...], tuple[str, ...]]:
    if not isinstance(value, list):
        return "", (), ("message_content_not_array",)
    text: list[str] = []
    tools: list[dict[str, Any]] = []
    errors: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            errors.append("content_block_not_object")
            continue
        present = [
            name
            for name in ("text", "toolUse", "reasoningContent")
            if name in block
        ]
        if len(present) != 1:
            errors.append("unsupported_or_ambiguous_content_block")
            continue
        if "text" in block:
            if not isinstance(block["text"], str):
                errors.append("text_content_not_string")
            else:
                text.append(block["text"])
        elif "toolUse" in block:
            tool, error = _tool_call(block["toolUse"])
            if error:
                errors.append(error)
            elif tool is not None:
                tools.append(tool)
        else:
            # Bedrock may return signed or redacted reasoning blocks. They are deliberately not
            # copied into output_text or durable evidence, and TokenUsage exposes no generic
            # reasoning-token count that this adapter could claim.
            reasoning = block["reasoningContent"]
            if not isinstance(reasoning, dict) or not (
                isinstance(reasoning.get("reasoningText"), dict)
                or isinstance(reasoning.get("redactedContent"), str)
            ):
                errors.append("reasoning_content_invalid")
    return "".join(text), tuple(tools), tuple(errors)


@lru_cache(maxsize=1)
def _stream_parser_contract() -> tuple[Any, Any]:
    service_model = Session().get_service_model("bedrock-runtime")
    operation = service_model.operation_model("ConverseStream")
    stream_shape = operation.output_shape.members["stream"]
    parser = ResponseParserFactory().create_parser(service_model.metadata["protocol"])
    parser = parser._event_stream_parser
    return parser, stream_shape


def _parse_event(message: Any) -> dict[str, Any]:
    # The generated parser supplies the current event union while EventStreamBuffer validates
    # both frame CRCs and length bounds. Strict JSON is checked first so duplicate/nonfinite
    # fixture payloads cannot be silently accepted by a permissive decoder.
    if message.payload:
        decoded = strict_json_loads(message.payload)
        if not isinstance(decoded, dict):
            raise ValueError("Bedrock event payload must be a JSON object")
        canonical_json(decoded)
    parser, stream_shape = _stream_parser_contract()
    parsed = parser.parse(message.to_response_dict(), stream_shape)
    if not isinstance(parsed, dict):
        raise ValueError("Bedrock event parser returned a non-object")
    return parsed


class BedrockConverseAdapter:
    """Native Converse/ConverseStream over AWS's generated REST-JSON contract.

    The experimental pre-1.0 AWS async SDK currently requires Python 3.12. This library still
    supports Python 3.11, so it uses botocore's stable generated service model, serializer, and
    CRC-validating event-stream parser with the existing async HTTP pool. No framing or model
    shape is hand-maintained here, and the exact serialized body is fixed before the spend claim.
    """

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
            self.client = self._new_client()

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            http2=self.http2,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=self.transport_max_connections,
                max_keepalive_connections=self.transport_max_connections,
            ),
        )

    async def close(self) -> None:
        if self.client is not None and not self._provided_client:
            await self.client.aclose()

    @asynccontextmanager
    async def _request_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.client is not None:
            yield self.client
            return
        async with self._new_client() as transient:
            yield transient

    def _headers(self, route: RouteConfig, *, stream: bool) -> dict[str, str]:
        token = os.environ.get(route.auth.env)
        if not token:
            raise RuntimeError(
                f"required credential environment variable is unset: {route.auth.env}"
            )
        headers = {
            route.auth.header: f"{route.auth.prefix}{token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.amazon.eventstream" if stream else "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "inference-endpoint-benchmark/0.1 bedrock-converse",
            **route.extra_headers,
        }
        if any(
            any(character in name or character in value for character in "\r\n\0")
            for name, value in headers.items()
        ):
            raise RuntimeError("constructed Bedrock headers contain prohibited control characters")
        try:
            httpx.Headers(headers)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("constructed Bedrock request headers are invalid") from exc
        return headers

    def preflight(self, route: RouteConfig) -> None:
        if route.provider != "amazon-bedrock":
            raise RuntimeError("Bedrock Converse adapter requires provider=amazon-bedrock")
        if route.adapter not in {"bedrock_converse", "bedrock_native"}:
            raise RuntimeError("Bedrock Converse adapter route name is invalid")
        if route.api_family != "converse":
            raise RuntimeError("Bedrock Converse adapter requires api_family=converse")
        if (
            route.auth.env != "AWS_BEARER_TOKEN_BEDROCK"
            or route.auth.header.casefold() != "authorization"
            or route.auth.prefix != "Bearer "
        ):
            raise RuntimeError(
                "Bedrock Converse requires AWS_BEARER_TOKEN_BEDROCK via Authorization: Bearer"
            )
        parsed = urlsplit(route.base_url)
        suffix = "amazonaws.com.cn" if route.region.startswith("cn-") else "amazonaws.com"
        expected_host = f"bedrock-runtime.{route.region}.{suffix}"
        if (
            parsed.scheme != "https"
            or parsed.hostname != expected_host
            or parsed.netloc != expected_host
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError(
                "Bedrock Converse base_url must be the canonical regional Runtime endpoint"
            )
        if route.request_defaults:
            raise RuntimeError("Bedrock Converse does not accept generic route request_defaults")
        if any(name.casefold() in {"accept", "user-agent"} for name in route.extra_headers):
            raise RuntimeError("Bedrock route extra_headers cannot override transport identity")
        if not self._provided_client and (
            route.http2 != self.http2
            or route.connection_reuse != self.connection_reuse
            or route.transport_max_connections != self.transport_max_connections
        ):
            raise RuntimeError("adapter connection pool does not match route identity")
        self._headers(route, stream=False)

    def prepare(self, route: RouteConfig, request: RequestSpec) -> PreparedRequest:
        self.preflight(route)
        return PreparedRequest(
            payload=materialize_bedrock_converse(route, request),
            headers=self._headers(route, stream=request.stream),
        )

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
        return await self.send_prepared(route, request, self.prepare(route, request))

    def _url(self, route: RouteConfig, request: RequestSpec, prepared: PreparedRequest) -> str:
        expected_operation = "ConverseStream" if request.stream else "Converse"
        expected_suffix = "/converse-stream" if request.stream else "/converse"
        expected_path = f"/model/{quote(route.model, safe='-._~')}{expected_suffix}"
        value = prepared.payload.value
        if value.get("operation") != expected_operation or value.get("url_path") != expected_path:
            raise RuntimeError("prepared Bedrock operation/path does not match route and request")
        try:
            decoded = strict_json_loads(prepared.payload.body)
        except StrictJSONError as exc:
            raise RuntimeError("prepared Bedrock body is not strict JSON") from exc
        if value.get("body") != decoded:
            raise RuntimeError("prepared Bedrock body metadata does not match exact wire bytes")
        return route.base_url.rstrip("/") + expected_path

    async def send_prepared(
        self, route: RouteConfig, request: RequestSpec, prepared: PreparedRequest
    ) -> InferenceResult:
        started_utc = _utc_now()
        started = time.perf_counter()
        try:
            url = self._url(route, request, prepared)
            async with asyncio.timeout(request.timeout_seconds):
                async with self._request_client() as client:
                    if request.stream:
                        return await self._stream(
                            client, url, route, request, prepared, started_utc, started
                        )
                    return await self._single(
                        client, url, route, request, prepared, started_utc, started
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

    def _http_error(
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
        status_code = response.status_code
        if status_code in {408, 504}:
            status, error_kind = "timeout", "timeout"
        elif status_code == 429:
            status, error_kind = "rate_limited", "provider_rate_limit"
        elif status_code in {401, 403}:
            status, error_kind = "client_error", "provider_billing_or_entitlement"
        elif status_code == 404:
            status, error_kind = "client_error", "provider_route_fatal"
        elif 400 <= status_code < 500:
            status, error_kind = "client_error", f"http_{status_code}"
        else:
            status, error_kind = "server_error", f"http_{status_code}"
        return InferenceResult(
            logical_id=request.logical_id,
            status=status,  # type: ignore[arg-type]
            http_status=status_code,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=headers_at - started,
            cache_state=_cache_state(request),  # type: ignore[arg-type]
            provider_request_id=_retained(response.headers, route).get("x-amzn-requestid"),
            retained_headers=_retained(response.headers, route),
            error_kind=error_kind,
            error_body_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def _protocol_error(
        self,
        route: RouteConfig,
        request: RequestSpec,
        response: httpx.Response,
        raw_sha256: str,
        started_utc: str,
        started: float,
        headers_at: float,
        ended: float,
        reasons: list[str] | tuple[str, ...],
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
            cache_state=_cache_state(request),  # type: ignore[arg-type]
            provider_request_id=retained.get("x-amzn-requestid"),
            retained_headers=retained,
            error_kind="protocol_error:" + ",".join(sorted(set(reasons))),
            error_body_sha256=raw_sha256,
        )

    async def _single(
        self,
        client: httpx.AsyncClient,
        url: str,
        route: RouteConfig,
        request: RequestSpec,
        prepared: PreparedRequest,
        started_utc: str,
        started: float,
    ) -> InferenceResult:
        async with client.stream(
            "POST",
            url,
            headers=prepared.headers,
            content=prepared.payload.body,
            timeout=request.timeout_seconds,
        ) as response:
            headers_at = time.perf_counter()
            raw = await response.aread()
            ended = time.perf_counter()
            if response.status_code >= 300:
                return self._http_error(
                    route, request, response, raw, started_utc, started, headers_at, ended
                )
            try:
                data = strict_json_loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("non-object")
                canonical_json(data)
            except (StrictJSONError, TypeError, ValueError):
                return self._protocol_error(
                    route,
                    request,
                    response,
                    hashlib.sha256(raw).hexdigest(),
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    ["invalid_json_success_body"],
                )
            output = data.get("output")
            message = output.get("message") if isinstance(output, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                return self._protocol_error(
                    route,
                    request,
                    response,
                    hashlib.sha256(raw).hexdigest(),
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    ["missing_assistant_message"],
                )
            text, tools, content_errors = _parse_output_content(message.get("content"))
            stop_reason = data.get("stopReason")
            if not isinstance(stop_reason, str) or not stop_reason:
                content_errors = (*content_errors, "missing_stop_reason")
            if content_errors:
                return self._protocol_error(
                    route,
                    request,
                    response,
                    hashlib.sha256(raw).hexdigest(),
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    content_errors,
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
                # Converse TokenUsage currently has no generic reasoning-token member.
                reasoning_tokens=None,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                usage_parse_errors=usage.errors,
                cache_state=_cache_state(request),  # type: ignore[arg-type]
                finish_reason=_finish_reason(stop_reason),
                output_text=text,
                tool_calls=tools,
                provider_request_id=retained.get("x-amzn-requestid"),
                retained_headers=retained,
            )

    async def _stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        route: RouteConfig,
        request: RequestSpec,
        prepared: PreparedRequest,
        started_utc: str,
        started: float,
    ) -> InferenceResult:
        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, Any]] = {}
        stopped_blocks: set[int] = set()
        event_offsets: list[float] = []
        first_visible_at: float | None = None
        message_started = False
        stop_reason: str | None = None
        usage: _Usage | None = None
        errors: list[str] = []
        wire_digest = hashlib.sha256()
        buffer = EventStreamBuffer()

        async with client.stream(
            "POST",
            url,
            headers=prepared.headers,
            content=prepared.payload.body,
            timeout=request.timeout_seconds,
        ) as response:
            headers_at = time.perf_counter()
            if response.status_code >= 300:
                raw = await response.aread()
                ended = time.perf_counter()
                return self._http_error(
                    route, request, response, raw, started_utc, started, headers_at, ended
                )
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
            if content_type != "application/vnd.amazon.eventstream":
                raw = await response.aread()
                ended = time.perf_counter()
                return self._protocol_error(
                    route,
                    request,
                    response,
                    hashlib.sha256(raw).hexdigest(),
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    ["unexpected_stream_content_type"],
                )
            try:
                async for chunk in response.aiter_raw():
                    wire_digest.update(chunk)
                    buffer.add_data(chunk)
                    for message in buffer:
                        parsed = _parse_event(message)
                        keys = [
                            key
                            for key in (
                                "messageStart",
                                "contentBlockStart",
                                "contentBlockDelta",
                                "contentBlockStop",
                                "messageStop",
                                "metadata",
                                "internalServerException",
                                "modelStreamErrorException",
                                "validationException",
                                "throttlingException",
                                "serviceUnavailableException",
                            )
                            if key in parsed
                        ]
                        if len(keys) != 1:
                            errors.append("unknown_or_ambiguous_stream_event")
                            continue
                        kind = keys[0]
                        event = parsed[kind]
                        if kind.endswith("Exception"):
                            ended = time.perf_counter()
                            retained = _retained(response.headers, route)
                            if kind == "throttlingException":
                                status, error_kind = "rate_limited", "provider_rate_limit"
                            elif kind == "validationException":
                                status, error_kind = "client_error", "provider_route_fatal"
                            elif kind == "internalServerException":
                                status, error_kind = "server_error", "provider_internal_error"
                            elif kind == "modelStreamErrorException":
                                status, error_kind = (
                                    "server_error",
                                    "provider_model_stream_error",
                                )
                            elif kind == "serviceUnavailableException":
                                status, error_kind = (
                                    "server_error",
                                    "provider_service_unavailable",
                                )
                            else:
                                status, error_kind = "server_error", "provider_stream_error"
                            return InferenceResult(
                                logical_id=request.logical_id,
                                status=status,  # type: ignore[arg-type]
                                http_status=response.status_code,
                                started_at_utc=started_utc,
                                ended_at_utc=_utc_now(),
                                total_seconds=ended - started,
                                time_to_headers_seconds=headers_at - started,
                                cache_state=_cache_state(request),  # type: ignore[arg-type]
                                provider_request_id=retained.get("x-amzn-requestid"),
                                retained_headers=retained,
                                error_kind=error_kind,
                                error_body_sha256=wire_digest.hexdigest(),
                            )
                        if not isinstance(event, dict):
                            errors.append(f"{kind}_not_object")
                            continue
                        if kind == "messageStart":
                            if message_started or event.get("role") != "assistant":
                                errors.append("invalid_message_start")
                            message_started = True
                        elif kind == "contentBlockStart":
                            index = event.get("contentBlockIndex")
                            start = event.get("start")
                            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                                errors.append("invalid_content_block_index")
                                continue
                            if not isinstance(start, dict):
                                errors.append("content_block_start_not_object")
                                continue
                            tool = start.get("toolUse")
                            if tool is not None:
                                if not isinstance(tool, dict):
                                    errors.append("tool_start_not_object")
                                    continue
                                tool_id, name = tool.get("toolUseId"), tool.get("name")
                                if (
                                    not isinstance(tool_id, str)
                                    or not tool_id
                                    or not isinstance(name, str)
                                    or not name
                                    or index in tool_parts
                                ):
                                    errors.append("invalid_tool_start")
                                    continue
                                tool_parts[index] = {"id": tool_id, "name": name, "input": []}
                                now = time.perf_counter()
                                first_visible_at = first_visible_at or now
                                event_offsets.append(now - started)
                        elif kind == "contentBlockDelta":
                            index = event.get("contentBlockIndex")
                            delta = event.get("delta")
                            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                                errors.append("invalid_content_block_index")
                                continue
                            if not isinstance(delta, dict):
                                errors.append("content_block_delta_not_object")
                                continue
                            present = [
                                name
                                for name in (
                                    "text",
                                    "toolUse",
                                    "reasoningContent",
                                    "citation",
                                    "image",
                                    "toolResult",
                                )
                                if name in delta
                            ]
                            if len(present) != 1:
                                errors.append("ambiguous_content_block_delta")
                                continue
                            visible = False
                            if "text" in delta:
                                piece = delta["text"]
                                if not isinstance(piece, str):
                                    errors.append("text_delta_not_string")
                                    continue
                                text_parts.append(piece)
                                visible = bool(piece)
                            elif "toolUse" in delta:
                                tool_delta = delta["toolUse"]
                                if not isinstance(tool_delta, dict) or not isinstance(
                                    tool_delta.get("input"), str
                                ):
                                    errors.append("invalid_tool_delta")
                                    continue
                                if index not in tool_parts:
                                    errors.append("tool_delta_without_start")
                                    continue
                                piece = tool_delta["input"]
                                tool_parts[index]["input"].append(piece)
                                visible = bool(piece)
                            elif "reasoningContent" in delta:
                                reasoning = delta["reasoningContent"]
                                if not isinstance(reasoning, dict) or any(
                                    key in reasoning and not isinstance(reasoning[key], str)
                                    for key in ("text", "redactedContent", "signature")
                                ):
                                    errors.append("invalid_reasoning_delta")
                            elif "citation" in delta:
                                if not isinstance(delta["citation"], dict):
                                    errors.append("invalid_citation_delta")
                            else:
                                errors.append("unsupported_stream_output_delta")
                            if visible:
                                now = time.perf_counter()
                                first_visible_at = first_visible_at or now
                                event_offsets.append(now - started)
                        elif kind == "contentBlockStop":
                            index = event.get("contentBlockIndex")
                            if (
                                isinstance(index, bool)
                                or not isinstance(index, int)
                                or index < 0
                                or index in stopped_blocks
                            ):
                                errors.append("invalid_content_block_stop")
                            else:
                                stopped_blocks.add(index)
                        elif kind == "messageStop":
                            candidate = event.get("stopReason")
                            if (
                                stop_reason is not None
                                or not isinstance(candidate, str)
                                or not candidate
                            ):
                                errors.append("invalid_message_stop")
                            else:
                                stop_reason = candidate
                        elif kind == "metadata":
                            if usage is not None:
                                errors.append("duplicate_stream_metadata")
                            else:
                                usage = _parse_usage(event.get("usage"))
            except (StrictJSONError, TypeError, ValueError) as exc:
                del exc
                errors.append("invalid_aws_event_stream")
            except Exception as exc:
                # Botocore's frame parser raises typed CRC/length/parser exceptions. Their class
                # names are intentionally not persisted; the fixed category plus wire digest is
                # enough to diagnose a fixture without retaining provider content.
                del exc
                errors.append("invalid_aws_event_stream")

            ended = time.perf_counter()
            if getattr(buffer, "_data", b""):
                errors.append("truncated_aws_event_stream")
            if not message_started:
                errors.append("message_start_missing")
            if stop_reason is None:
                errors.append("message_stop_missing")
            if route.stream_usage_mode == "required" and (
                usage is None or usage.input_tokens is None or usage.output_tokens is None
            ):
                if usage is None:
                    usage = _Usage(None, None, None, ("required_stream_usage_missing",))
                else:
                    usage = _Usage(
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.cache_read_input_tokens,
                        (*usage.errors, "required_stream_usage_missing"),
                    )
            if errors:
                result = self._protocol_error(
                    route,
                    request,
                    response,
                    wire_digest.hexdigest(),
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    errors,
                )
                result.ttft_seconds = (
                    None if first_visible_at is None else first_visible_at - started
                )
                result.decode_seconds = (
                    None
                    if first_visible_at is None
                    else max(0.0, (ended - started) - (first_visible_at - started))
                )
                result.output_event_offsets_seconds = tuple(event_offsets)
                return result

            tools: list[dict[str, Any]] = []
            for index, fragments in sorted(tool_parts.items()):
                if index not in stopped_blocks:
                    errors.append("tool_block_stop_missing")
                    continue
                raw_arguments = "".join(fragments["input"])
                try:
                    arguments = strict_json_loads(raw_arguments)
                    arguments_json = canonical_json(arguments)
                except (StrictJSONError, TypeError, ValueError):
                    errors.append("invalid_tool_arguments_json")
                    continue
                tools.append(
                    {
                        "id": fragments["id"],
                        "type": "function",
                        "function": {"name": fragments["name"], "arguments": arguments_json},
                    }
                )
            if errors:
                return self._protocol_error(
                    route,
                    request,
                    response,
                    wire_digest.hexdigest(),
                    started_utc,
                    started,
                    headers_at,
                    ended,
                    errors,
                )
            observed_usage = usage or _parse_usage(None)
            retained = _retained(response.headers, route)
            ttft = None if first_visible_at is None else first_visible_at - started
            return InferenceResult(
                logical_id=request.logical_id,
                status="success",
                http_status=response.status_code,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=headers_at - started,
                ttft_seconds=ttft,
                decode_seconds=None if ttft is None else max(0.0, (ended - started) - ttft),
                output_event_offsets_seconds=tuple(event_offsets),
                input_tokens=observed_usage.input_tokens,
                output_tokens=observed_usage.output_tokens,
                reasoning_tokens=None,
                cache_read_input_tokens=observed_usage.cache_read_input_tokens,
                usage_parse_errors=observed_usage.errors,
                cache_state=_cache_state(request),  # type: ignore[arg-type]
                finish_reason=_finish_reason(stop_reason),
                output_text="".join(text_parts),
                tool_calls=tuple(tools),
                provider_request_id=retained.get("x-amzn-requestid"),
                retained_headers=retained,
            )
