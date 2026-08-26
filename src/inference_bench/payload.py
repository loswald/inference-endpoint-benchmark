from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .models import RequestSpec, RouteConfig, canonical_json
from .workloads import materialize_messages

PAYLOAD_GENERATOR_VERSION = "openai-compatible-synthetic/v3"
RESPONSES_PAYLOAD_GENERATOR_VERSION = "openai-responses-synthetic/v1"


@dataclass(frozen=True, slots=True)
class MaterializedPayload:
    value: dict[str, Any]
    body: bytes
    wire_body_sha256: str
    bound_payload_sha256: str
    input_token_upper_bound: int
    generator_version: str = PAYLOAD_GENERATOR_VERSION


def build_openai_compatible_payload(route: RouteConfig, request: RequestSpec) -> dict[str, Any]:
    """Materialize the exact JSON object that will be encoded and sent.

    Generic route defaults were validated when ``RouteConfig`` was constructed. Measured fields
    are written last and therefore cannot be replaced by a route default.
    """

    import copy

    payload: dict[str, Any] = {
        **copy.deepcopy(route.request_defaults),
        "model": route.model,
        "messages": materialize_messages(request),
        "stream": request.stream,
        route.output_limit_field: request.max_output_tokens,
    }
    for key, value in (
        ("temperature", request.temperature),
        ("top_p", request.top_p),
        ("seed", request.seed),
        ("tool_choice", request.tool_choice),
        ("parallel_tool_calls", request.parallel_tool_calls),
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
        if route.stream_usage_mode in {"required", "try"}:
            payload["stream_options"] = {"include_usage": True}
        else:
            payload.pop("stream_options", None)
    else:
        payload.pop("stream_options", None)
    if route.adapter == "openrouter":
        payload["provider"] = {
            "only": [route.upstream_provider],
            "order": [route.upstream_provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    return payload


def materialize_openai_compatible(route: RouteConfig, request: RequestSpec) -> MaterializedPayload:
    value = build_openai_compatible_payload(route, request)
    # The adapter sends these exact UTF-8 bytes with Content-Type application/json. This avoids a
    # second serializer changing whitespace, escaping, or key order after the claim is written.
    body = canonical_json(value).encode("utf-8")
    wire_hash = hashlib.sha256(body).hexdigest()
    bound_hash = hashlib.sha256(
        b"materialized-payload/v2\0" + PAYLOAD_GENERATOR_VERSION.encode("utf-8") + b"\0" + body
    ).hexdigest()
    # For text JSON transports, UTF-8 bytes are a tokenizer-independent conservative upper bound
    # on subword pieces. The explicit route overhead covers provider framing/tool/image accounting
    # that is not present in the user text. The campaign factor is applied separately at claim.
    upper = len(body) + route.input_token_reservation_overhead
    if upper <= 0 or not math.isfinite(float(upper)):
        raise ValueError("materialized input-token upper bound is invalid")
    # Round-trip once before any claim; NaN/Infinity and non-JSON values fail locally.
    decoded = json.loads(body)
    if decoded != value:
        raise ValueError("materialized payload failed canonical JSON round trip")
    return MaterializedPayload(
        value=value,
        body=body,
        wire_body_sha256=wire_hash,
        bound_payload_sha256=bound_hash,
        input_token_upper_bound=upper,
    )


def _responses_tools(tools: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            converted.append(dict(tool))
            continue
        function = tool["function"]
        converted.append(
            {
                "type": "function",
                **{
                    key: function[key]
                    for key in ("name", "description", "parameters", "strict")
                    if key in function
                },
            }
        )
    return converted


def _responses_input(request: RequestSpec) -> tuple[str | None, list[dict[str, Any]]]:
    instructions: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in materialize_messages(request):
        role = str(message.get("role") or "user")
        content = message.get("content")
        if role == "system" and isinstance(content, str):
            instructions.append(content)
            continue
        if isinstance(content, list):
            converted: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Responses message content parts must be objects")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    converted.append({"type": "input_text", "text": part["text"]})
                elif part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                    url = part["image_url"].get("url")
                    if not isinstance(url, str) or not url:
                        raise ValueError("Responses image_url content requires a URL")
                    converted.append({"type": "input_image", "image_url": url})
                else:
                    converted.append(dict(part))
            content = converted
        messages.append({"role": role, "content": content})
    return "\n\n".join(instructions) or None, messages


def build_responses_payload(route: RouteConfig, request: RequestSpec) -> dict[str, Any]:
    instructions, input_messages = _responses_input(request)
    payload: dict[str, Any] = {
        "model": route.model,
        "input": input_messages,
        "stream": request.stream,
        "max_output_tokens": request.max_output_tokens,
    }
    if instructions:
        payload["instructions"] = instructions
    for key, value in (
        ("temperature", request.temperature),
        ("top_p", request.top_p),
        ("seed", request.seed),
        ("logprobs", request.logprobs),
    ):
        if value is not None:
            payload[key] = value
    if request.stop:
        payload["stop"] = list(request.stop)
    if request.tools:
        payload["tools"] = _responses_tools(request.tools)
    if request.tool_choice is not None:
        choice = request.tool_choice
        if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
            choice = {"type": "function", "name": choice["function"].get("name")}
        payload["tool_choice"] = choice
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.response_format is not None:
        response_format = request.response_format
        if response_format.get("type") == "json_schema" and isinstance(
            response_format.get("json_schema"), dict
        ):
            response_format = {
                "type": "json_schema",
                **response_format["json_schema"],
            }
        payload["text"] = {"format": response_format}
    return payload


def materialize_responses(route: RouteConfig, request: RequestSpec) -> MaterializedPayload:
    value = build_responses_payload(route, request)
    body = canonical_json(value).encode("utf-8")
    wire_hash = hashlib.sha256(body).hexdigest()
    bound_hash = hashlib.sha256(
        b"materialized-payload/v2\0"
        + RESPONSES_PAYLOAD_GENERATOR_VERSION.encode("utf-8")
        + b"\0"
        + body
    ).hexdigest()
    upper = len(body) + route.input_token_reservation_overhead
    if json.loads(body) != value:
        raise ValueError("materialized Responses payload failed canonical JSON round trip")
    return MaterializedPayload(
        value=value,
        body=body,
        wire_body_sha256=wire_hash,
        bound_payload_sha256=bound_hash,
        input_token_upper_bound=upper,
        generator_version=RESPONSES_PAYLOAD_GENERATOR_VERSION,
    )


def reserved_input_tokens(
    route: RouteConfig, request: RequestSpec, reservation_factor: float
) -> int:
    materialized = materialize_openai_compatible(route, request)
    return math.ceil(
        max(request.planned_input_tokens, materialized.input_token_upper_bound) * reservation_factor
    )
