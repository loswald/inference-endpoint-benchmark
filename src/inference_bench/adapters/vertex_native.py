from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import math
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from ..json_contract import StrictJSONError, strict_json_loads
from ..models import InferenceResult, RequestSpec, RouteConfig, canonical_json
from ..payload import MaterializedPayload, payload_binding_sha256
from ..workloads import materialize_messages
from .base import PreparedRequest

VERTEX_NATIVE_PAYLOAD_GENERATOR_VERSION = "vertex-native-generate-content/v1"
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
_ACTION_PATH = re.compile(
    r"^/v1/projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/"
    r"publishers/google/models/(?P<model>[^/:]+):generateContent$"
)
_DATA_URL = re.compile(
    r"^data:(?P<mime>[A-Za-z0-9][A-Za-z0-9.+-]*/[A-Za-z0-9][A-Za-z0-9.+-]*);"
    r"base64,(?P<data>[A-Za-z0-9+/]*={0,2})$"
)
_THINKING_LEVELS = frozenset({"MINIMAL", "LOW", "MEDIUM", "HIGH"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _retained(headers: httpx.Headers, route: RouteConfig) -> dict[str, str]:
    allowed = {name.casefold() for name in route.retained_header_names}
    return {name.casefold(): value for name, value in headers.items() if name.casefold() in allowed}


def _native_part(part: dict[str, Any]) -> dict[str, Any]:
    """Validate a native Gemini part that the generic request surface can already express."""

    if "text" in part:
        if not isinstance(part["text"], str):
            raise ValueError("Vertex text parts require a string")
        result: dict[str, Any] = {"text": part["text"]}
    elif "inlineData" in part:
        value = part["inlineData"]
        if not isinstance(value, dict):
            raise ValueError("Vertex inlineData must be an object")
        mime = value.get("mimeType")
        data = value.get("data")
        if not isinstance(mime, str) or not mime or not isinstance(data, str) or not data:
            raise ValueError("Vertex inlineData requires mimeType and base64 data")
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Vertex inlineData data must be strict base64") from exc
        result = {"inlineData": {"mimeType": mime, "data": data}}
    elif "fileData" in part:
        value = part["fileData"]
        if not isinstance(value, dict):
            raise ValueError("Vertex fileData must be an object")
        mime = value.get("mimeType")
        uri = value.get("fileUri")
        if (
            not isinstance(mime, str)
            or not mime
            or not isinstance(uri, str)
            or not uri.startswith("gs://")
        ):
            raise ValueError("Vertex fileData requires mimeType and a gs:// fileUri")
        result = {"fileData": {"mimeType": mime, "fileUri": uri}}
    elif "functionCall" in part:
        value = part["functionCall"]
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("Vertex functionCall requires a name")
        args = value.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Vertex functionCall args must be an object")
        result = {"functionCall": {"name": value["name"], "args": copy.deepcopy(args)}}
    elif "functionResponse" in part:
        value = part["functionResponse"]
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("Vertex functionResponse requires a name")
        response = value.get("response")
        if not isinstance(response, dict):
            raise ValueError("Vertex functionResponse response must be an object")
        result = {
            "functionResponse": {
                "name": value["name"],
                "response": copy.deepcopy(response),
            }
        }
    else:
        raise ValueError("unsupported Vertex native content part")
    signature = part.get("thoughtSignature")
    if signature is not None:
        if not isinstance(signature, str) or not signature:
            raise ValueError("Vertex thoughtSignature must be a nonempty string")
        result["thoughtSignature"] = signature
    unknown = set(part) - {next(iter(result)), "thoughtSignature"}
    if unknown:
        raise ValueError("unsupported fields in Vertex native content part")
    return result


def _image_url_part(part: dict[str, Any]) -> dict[str, Any]:
    image = part.get("image_url")
    if not isinstance(image, dict):
        raise ValueError("image_url content requires an object")
    url = image.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("image_url content requires a URL")
    match = _DATA_URL.fullmatch(url)
    if match is not None:
        data = match.group("data")
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image data URLs must contain strict base64") from exc
        return {"inlineData": {"mimeType": match.group("mime"), "data": data}}
    mime = image.get("mime_type")
    if url.startswith("gs://") and isinstance(mime, str) and mime:
        return {"fileData": {"mimeType": mime, "fileUri": url}}
    raise ValueError("Vertex image URLs must be data URLs or typed gs:// objects")


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    values = content if isinstance(content, (list, tuple)) else [content]
    parts: list[dict[str, Any]] = []
    for part in values:
        if not isinstance(part, dict):
            raise ValueError("Vertex message content parts must be objects")
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Vertex text parts require a string")
            parts.append({"text": text})
        elif part_type in {"image_url", "input_image"}:
            parts.append(_image_url_part(part))
        elif part_type is None:
            parts.append(_native_part(part))
        else:
            raise ValueError("unsupported Vertex message content part type")
    return parts


def _openai_tool_call_parts(
    message: dict[str, Any], call_names: dict[str, str]
) -> list[dict[str, Any]]:
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, (list, tuple)):
        raise ValueError("tool_calls must be an array")
    parts: list[dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict) or call.get("type", "function") != "function":
            raise ValueError("Vertex supports function tool calls only")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("function tool calls require a name")
        call_id = call.get("id")
        if call_id is not None:
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("function tool-call ids must be nonempty strings")
            call_names[call_id] = function["name"]
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = strict_json_loads(raw_args)
            except StrictJSONError as exc:
                raise ValueError("function tool-call arguments must be strict JSON") from exc
        else:
            args = copy.deepcopy(raw_args)
        if not isinstance(args, dict):
            raise ValueError("function tool-call arguments must be an object")
        parts.append({"functionCall": {"name": function["name"], "args": args}})
    return parts


def _contents(request: RequestSpec) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    for message in materialize_messages(request):
        if not isinstance(message, dict):
            raise ValueError("Vertex messages must be objects")
        raw_role = message.get("role")
        if raw_role in {"system", "developer"}:
            parts = _content_parts(message.get("content"))
            if any(set(part) != {"text"} for part in parts):
                raise ValueError("Vertex system instructions support text parts only")
            system_parts.extend(parts)
            continue
        if raw_role in {"assistant", "model"}:
            role = "model"
        elif raw_role in {"user", "tool"}:
            role = "user"
        else:
            raise ValueError("Vertex message role must be system, developer, user, tool, or model")
        if raw_role == "tool":
            direct_parts = _content_parts(message.get("content"))
            if direct_parts and all("functionResponse" in part for part in direct_parts):
                parts = direct_parts
            else:
                call_id = message.get("tool_call_id")
                name = message.get("name")
                if name is None and isinstance(call_id, str):
                    name = call_names.get(call_id)
                if not isinstance(name, str) or not name:
                    raise ValueError("Vertex tool responses require a function name")
                raw_response = message.get("content")
                if isinstance(raw_response, str):
                    try:
                        decoded = strict_json_loads(raw_response)
                    except StrictJSONError:
                        decoded = raw_response
                else:
                    decoded = copy.deepcopy(raw_response)
                response = decoded if isinstance(decoded, dict) else {"result": decoded}
                parts = [{"functionResponse": {"name": name, "response": response}}]
        else:
            parts = _content_parts(message.get("content"))
            parts.extend(_openai_tool_call_parts(message, call_names))
        if raw_role == "tool" and not parts:
            name = message.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Vertex tool responses require a function name")
            parts.append({"functionResponse": {"name": name, "response": {}}})
        if not parts:
            raise ValueError("Vertex non-system messages require at least one content part")
        contents.append({"role": role, "parts": parts})
    if not contents:
        raise ValueError("Vertex requests require at least one non-system content message")
    return ({"parts": system_parts} if system_parts else None), contents


def _tools(request: RequestSpec) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    native_tools: list[dict[str, Any]] = []
    for tool in request.tools:
        if not isinstance(tool, dict):
            raise ValueError("Vertex tools must be objects")
        if "functionDeclarations" in tool and "type" not in tool:
            declarations_value = tool["functionDeclarations"]
            if not isinstance(declarations_value, list) or not declarations_value:
                raise ValueError("native Vertex functionDeclarations must be a nonempty array")
            native_tools.append(copy.deepcopy(tool))
            continue
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Vertex supports function tools only")
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Vertex function declarations require a name")
        declaration: dict[str, Any] = {"name": name}
        description = function.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("Vertex function descriptions must be strings")
            declaration["description"] = description
        parameters = function.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError("Vertex function parameters must be an object")
            declaration["parameters"] = copy.deepcopy(parameters)
        declarations.append(declaration)
    if declarations:
        native_tools.insert(0, {"functionDeclarations": declarations})
    return native_tools


def _tool_config(request: RequestSpec) -> dict[str, Any] | None:
    choice = request.tool_choice
    if choice is None:
        return None
    config: dict[str, Any]
    if isinstance(choice, str):
        modes = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
        try:
            config = {"mode": modes[choice.casefold()]}
        except KeyError as exc:
            raise ValueError("unsupported Vertex tool_choice") from exc
    elif isinstance(choice, dict):
        function = choice.get("function")
        if choice.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("Vertex named tool_choice must select a function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Vertex named tool_choice requires a name")
        config = {"mode": "ANY", "allowedFunctionNames": [name]}
    else:
        raise ValueError("unsupported Vertex tool_choice")
    return {"functionCallingConfig": config}


def _response_format(config: dict[str, Any], value: dict[str, Any] | None) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("Vertex response_format must be an object")
    format_type = value.get("type")
    if format_type == "json_object":
        config["responseMimeType"] = "application/json"
    elif format_type == "json_schema":
        wrapper = value.get("json_schema")
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("schema"), dict):
            raise ValueError("Vertex json_schema response_format requires schema")
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = copy.deepcopy(wrapper["schema"])
    elif format_type == "text":
        config["responseMimeType"] = "text/plain"
    else:
        raise ValueError("unsupported Vertex response_format type")


def _thinking_config(route: RouteConfig, request: RequestSpec) -> dict[str, str] | None:
    controls = route.reasoning_control(request.reasoning_budget)
    if not controls:
        return None
    # The current shared chat schema calls this exact declared route control
    # ``reasoning_effort``. Native profiles can move to the wire-native name when the shared
    # schema and registry are promoted; accepting both keeps this adapter self-contained.
    allowed = {"reasoning_effort", "thinkingConfig.thinkingLevel"}
    if set(controls) - allowed or len(controls) != 1:
        raise ValueError("Vertex native reasoning controls must declare one thinking level")
    raw = next(iter(controls.values()))
    level = raw.strip().upper()
    if level not in _THINKING_LEVELS:
        raise ValueError("Vertex thinking level must be MINIMAL, LOW, MEDIUM, or HIGH")
    return {"thinkingLevel": level}


def build_vertex_native_payload(route: RouteConfig, request: RequestSpec) -> dict[str, Any]:
    if route.request_defaults:
        raise ValueError("Vertex native routes do not accept generic request_defaults")
    if request.parallel_tool_calls is False:
        raise ValueError("Vertex native cannot enforce parallel_tool_calls=false")
    system_instruction, contents = _contents(request)
    generation_config: dict[str, Any] = {"maxOutputTokens": request.max_output_tokens}
    for field, value in (
        ("temperature", request.temperature),
        ("topP", request.top_p),
        ("seed", request.seed),
        ("responseLogprobs", request.logprobs),
    ):
        if value is not None:
            generation_config[field] = value
    if request.stop:
        generation_config["stopSequences"] = list(request.stop)
    _response_format(generation_config, request.response_format)
    if thinking := _thinking_config(route, request):
        generation_config["thinkingConfig"] = thinking
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_instruction is not None:
        payload["systemInstruction"] = system_instruction
    if tools := _tools(request):
        payload["tools"] = tools
    if tool_config := _tool_config(request):
        if not request.tools:
            raise ValueError("Vertex tool_choice requires tools")
        payload["toolConfig"] = tool_config
    return payload


def materialize_vertex_native(route: RouteConfig, request: RequestSpec) -> MaterializedPayload:
    value = build_vertex_native_payload(route, request)
    body = canonical_json(value).encode("utf-8")
    if strict_json_loads(body) != value:
        raise ValueError("materialized Vertex payload failed canonical JSON round trip")
    upper = len(body) + route.input_token_reservation_overhead
    if upper <= 0 or not math.isfinite(float(upper)):
        raise ValueError("materialized Vertex input-token upper bound is invalid")
    return MaterializedPayload(
        value=value,
        body=body,
        wire_body_sha256=hashlib.sha256(body).hexdigest(),
        bound_payload_sha256=payload_binding_sha256(
            body, VERTEX_NATIVE_PAYLOAD_GENERATOR_VERSION
        ),
        input_token_upper_bound=upper,
        generator_version=VERTEX_NATIVE_PAYLOAD_GENERATOR_VERSION,
    )


@dataclass(frozen=True, slots=True)
class VertexUsage:
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    errors: tuple[str, ...]


def _count(value: Any, field: str, errors: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field}_wrong_json_type")
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        errors.append(f"{field}_nonintegral_or_negative")
        return None
    return int(numeric)


def _parse_usage(value: Any) -> VertexUsage:
    if value is None:
        return VertexUsage(None, None, None, None, None, ())
    if not isinstance(value, dict):
        return VertexUsage(None, None, None, None, None, ("usage_wrong_json_type",))
    errors: list[str] = []
    prompt = _count(value.get("promptTokenCount"), "input_tokens", errors)
    tool_prompt = _count(value.get("toolUsePromptTokenCount"), "input_tokens", errors)
    candidates = _count(value.get("candidatesTokenCount"), "output_tokens", errors)
    thoughts = _count(value.get("thoughtsTokenCount"), "reasoning_tokens", errors)
    cached = _count(value.get("cachedContentTokenCount"), "cache_read_input_tokens", errors)
    total = _count(value.get("totalTokenCount"), "total_tokens", errors)
    tool_prompt_invalid = value.get("toolUsePromptTokenCount") is not None and tool_prompt is None
    thoughts_invalid = value.get("thoughtsTokenCount") is not None and thoughts is None
    input_tokens = (
        None if prompt is None or tool_prompt_invalid else prompt + (tool_prompt or 0)
    )
    output_tokens = (
        None if candidates is None or thoughts_invalid else candidates + (thoughts or 0)
    )
    if (
        total is not None
        and input_tokens is not None
        and output_tokens is not None
        and total != input_tokens + output_tokens
    ):
        errors.append("total_tokens_mismatch_input_plus_output")
    return VertexUsage(
        input_tokens,
        output_tokens,
        cached,
        thoughts,
        total,
        tuple(dict.fromkeys(errors)),
    )


def _merge_usage(values: list[Any]) -> VertexUsage:
    aggregate: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_input_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }
    errors: list[str] = []
    conflicted: set[str] = set()
    for raw in values:
        current = _parse_usage(raw)
        errors.extend(current.errors)
        for field in aggregate:
            observed = getattr(current, field)
            if observed is None or field in conflicted:
                continue
            prior = aggregate[field]
            if prior is not None and observed < prior:
                errors.append(f"stream_{field}_decreased")
                aggregate[field] = None
                conflicted.add(field)
            else:
                aggregate[field] = observed
    if (
        aggregate["total_tokens"] is not None
        and aggregate["input_tokens"] is not None
        and aggregate["output_tokens"] is not None
        and aggregate["total_tokens"]
        != aggregate["input_tokens"] + aggregate["output_tokens"]
    ):
        errors.append("total_tokens_mismatch_input_plus_output")
    return VertexUsage(
        aggregate["input_tokens"],
        aggregate["output_tokens"],
        aggregate["cache_read_input_tokens"],
        aggregate["reasoning_tokens"],
        aggregate["total_tokens"],
        tuple(dict.fromkeys(errors)),
    )


_FILTER_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
    }
)


def _finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("invalid Vertex finishReason")
    normalized = value.strip().upper()
    if normalized == "STOP":
        return "stop"
    if normalized == "MAX_TOKENS":
        return "max_tokens"
    if normalized in _FILTER_FINISH_REASONS:
        return "content_filter"
    return "other"


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    text: tuple[str, ...]
    function_calls: tuple[tuple[str, dict[str, Any]], ...]
    finish_reason: str | None
    usage: Any
    response_id: str | None
    terminal: bool


def _parse_success(value: Any) -> ParsedResponse:
    if not isinstance(value, dict):
        raise ValueError("Vertex success envelope must be an object")
    response_id = value.get("responseId")
    if response_id is not None and (not isinstance(response_id, str) or not response_id):
        raise ValueError("Vertex responseId must be a nonempty string")
    prompt_feedback = value.get("promptFeedback")
    blocked = False
    if prompt_feedback is not None:
        if not isinstance(prompt_feedback, dict):
            raise ValueError("Vertex promptFeedback must be an object")
        block_reason = prompt_feedback.get("blockReason")
        blocked = isinstance(block_reason, str) and bool(block_reason)
    candidates = value.get("candidates")
    if candidates is None:
        candidates = []
    if not isinstance(candidates, list) or len(candidates) > 1:
        raise ValueError("Vertex responses require zero or one candidate")
    text: list[str] = []
    calls: list[tuple[str, dict[str, Any]]] = []
    finish: str | None = "content_filter" if blocked else None
    if candidates:
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ValueError("Vertex candidate must be an object")
        index = candidate.get("index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or index != 0:
            raise ValueError("Vertex candidate index must be zero")
        finish = _finish_reason(candidate.get("finishReason"))
        content = candidate.get("content")
        if content is not None:
            if not isinstance(content, dict):
                raise ValueError("Vertex candidate content must be an object")
            role = content.get("role")
            if role is not None and role != "model":
                raise ValueError("Vertex candidate role must be model")
            parts = content.get("parts", [])
            if not isinstance(parts, list):
                raise ValueError("Vertex candidate parts must be an array")
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError("Vertex response parts must be objects")
                thought = part.get("thought", False)
                if not isinstance(thought, bool):
                    raise ValueError("Vertex part thought marker must be boolean")
                if "text" in part:
                    piece = part["text"]
                    if not isinstance(piece, str):
                        raise ValueError("Vertex response text must be a string")
                    if piece and not thought:
                        text.append(piece)
                if "functionCall" in part:
                    call = part["functionCall"]
                    if not isinstance(call, dict):
                        raise ValueError("Vertex functionCall must be an object")
                    name = call.get("name")
                    args = call.get("args", {})
                    if not isinstance(name, str) or not name or not isinstance(args, dict):
                        raise ValueError("Vertex functionCall requires name and object args")
                    calls.append((name, copy.deepcopy(args)))
                known = {"text", "functionCall", "thought", "thoughtSignature"}
                if set(part) - known:
                    raise ValueError("unsupported Vertex response part")
    terminal = finish is not None or blocked
    if not (text or calls or terminal or value.get("usageMetadata") is not None):
        raise ValueError("empty Vertex response")
    return ParsedResponse(
        tuple(text),
        tuple(calls),
        finish,
        value.get("usageMetadata"),
        response_id,
        terminal,
    )


def _tool_calls(values: list[tuple[str, dict[str, Any]]]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for index, (name, args) in enumerate(values):
        arguments = canonical_json(args)
        digest = hashlib.sha256(f"{name}\0{arguments}\0{index}".encode()).hexdigest()[:16]
        result.append(
            {
                "choice_index": 0,
                "index": index,
                "id": f"vertex-{digest}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return tuple(result)


async def _sse_payloads(
    response: httpx.Response, digest: Any
) -> AsyncIterator[bytes]:
    buffer = bytearray()
    data_lines: list[bytes] = []

    def dispatch() -> bytes | None:
        if not data_lines:
            return None
        payload = b"\n".join(data_lines)
        data_lines.clear()
        return payload

    async for chunk in response.aiter_bytes():
        digest.update(chunk)
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line:
                payload = dispatch()
                if payload is not None:
                    yield payload
            elif line.startswith(b"data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(b" ") else value)
            elif line.startswith(b":") or b":" in line:
                continue
            else:
                raise ValueError("malformed SSE field")
    if buffer:
        line = bytes(buffer[:-1] if buffer.endswith(b"\r") else buffer)
        if line.startswith(b"data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(b" ") else value)
        elif line and not line.startswith(b":") and b":" not in line:
            raise ValueError("malformed SSE tail")
    payload = dispatch()
    if payload is not None:
        yield payload


class VertexNativeAdapter:
    """Strict Vertex Gemini ``generateContent``/``streamGenerateContent`` transport.

    ``RouteConfig.base_url`` is the complete v1 ``:generateContent`` action URL. Streaming swaps
    only that registered action suffix for ``:streamGenerateContent?alt=sse``; model and location
    are never guessed from an alias.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        http2: bool = False,
        connection_reuse: bool = True,
        transport_max_connections: int = 256,
        credentials: Any | None = None,
        credential_loader: Callable[[RouteConfig], Any] | None = None,
        auth_request_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.http2 = http2
        self.connection_reuse = connection_reuse
        self.transport_max_connections = transport_max_connections
        self._provided_client = client is not None
        self.client = client
        self._injected_credentials = credentials
        self._credential_loader = credential_loader or self._load_credentials
        self._auth_request_factory = auth_request_factory
        self._credentials_by_env: dict[str, Any] = {}
        self._credential_lock = threading.Lock()
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

    @staticmethod
    def _load_credentials(route: RouteConfig) -> Any:
        try:
            import google.auth
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - installation preflight
            raise RuntimeError("Vertex native support requires google-auth") from exc
        configured = os.environ.get(route.auth.env)
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    f"{route.auth.env} must name a readable service-account JSON file"
                )
            return service_account.Credentials.from_service_account_file(
                str(path), scopes=_SCOPES
            )
        credentials, _ = google.auth.default(scopes=_SCOPES)
        return credentials

    def _credentials(self, route: RouteConfig) -> Any:
        if self._injected_credentials is not None:
            return self._injected_credentials
        if route.auth.env not in self._credentials_by_env:
            self._credentials_by_env[route.auth.env] = self._credential_loader(route)
        return self._credentials_by_env[route.auth.env]

    def _auth_request(self) -> Any:
        if self._auth_request_factory is not None:
            return self._auth_request_factory()
        try:
            from google.auth.transport.requests import Request
        except ImportError as exc:  # pragma: no cover - installation preflight
            raise RuntimeError("Vertex native support requires google-auth") from exc
        return Request()

    def _headers(self, route: RouteConfig, *, stream: bool) -> dict[str, str]:
        credentials = self._credentials(route)
        token = getattr(credentials, "token", None)
        if not bool(getattr(credentials, "valid", False)) or not token:
            with self._credential_lock:
                token = getattr(credentials, "token", None)
                if not bool(getattr(credentials, "valid", False)) or not token:
                    credentials.refresh(self._auth_request())
                    token = getattr(credentials, "token", None)
        if not isinstance(token, str) or not token:
            raise RuntimeError("Google OAuth refresh did not produce an access token")
        headers = {
            **route.extra_headers,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if any(
            any(character in name or character in value for character in "\r\n\0")
            for name, value in headers.items()
        ):
            raise RuntimeError("constructed Vertex headers contain prohibited control characters")
        try:
            httpx.Headers(headers)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("constructed Vertex headers are invalid") from exc
        return headers

    @staticmethod
    def _validate_action(route: RouteConfig) -> None:
        parsed = urlsplit(route.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError("Vertex base_url must be one canonical HTTPS action URL")
        match = _ACTION_PATH.fullmatch(parsed.path)
        if match is None:
            raise RuntimeError("Vertex base_url must be the explicit v1 :generateContent action")
        location = unquote(match.group("location"))
        model = unquote(match.group("model"))
        if location != route.region or model != route.model:
            raise RuntimeError("Vertex action URL must exactly match the route location and model")
        expected_host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        if parsed.hostname.casefold() != expected_host.casefold():
            raise RuntimeError("Vertex action URL host must exactly match its location")
        if route.provider != "google-vertex-ai" or route.adapter != "vertex_native":
            raise RuntimeError("Vertex native adapter requires google-vertex-ai/vertex_native")
        if route.api_family not in {"chat_completions", "generate_content"}:
            raise RuntimeError("Vertex native adapter requires the generate-content API family")
        if route.api_version != "v1":
            raise RuntimeError("Vertex native adapter requires api_version=v1")
        if route.output_limit_field != "max_output_tokens":
            raise RuntimeError(
                "Vertex native adapter requires output_limit_field=max_output_tokens"
            )
        if route.auth.header.casefold() != "authorization" or route.auth.prefix != "Bearer ":
            raise RuntimeError("Vertex native OAuth requires Authorization: Bearer")
        if any(name.casefold() == "accept" for name in route.extra_headers):
            raise RuntimeError("Vertex native routes cannot override the negotiated Accept header")

    def preflight(self, route: RouteConfig) -> None:
        self._validate_action(route)
        if not self._provided_client and (
            route.http2 != self.http2
            or route.connection_reuse != self.connection_reuse
            or route.transport_max_connections != self.transport_max_connections
        ):
            raise RuntimeError("adapter connection pool does not match route identity")
        # OAuth refresh is deliberately before prepare returns to the durable claim boundary.
        self._headers(route, stream=False)

    def prepare(self, route: RouteConfig, request: RequestSpec) -> PreparedRequest:
        self.preflight(route)
        return PreparedRequest(
            payload=materialize_vertex_native(route, request),
            headers=self._headers(route, stream=request.stream),
        )

    @staticmethod
    def _url(route: RouteConfig, *, stream: bool) -> str:
        if not stream:
            return route.base_url
        return route.base_url[: -len(":generateContent")] + ":streamGenerateContent?alt=sse"

    @asynccontextmanager
    async def _request_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.client is not None:
            yield self.client
            return
        async with self._new_client() as transient:
            yield transient

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
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
        except httpx.TransportError:
            ended = time.perf_counter()
            return InferenceResult(
                logical_id=request.logical_id,
                status="transport_error",
                http_status=None,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                error_kind="transport_error",
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
            self._url(route, stream=False),
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
            parsed = _parse_success(strict_json_loads(raw))
        except (StrictJSONError, ValueError, TypeError):
            return self._protocol_error(
                route, request, response, raw, started_utc, started, headers_at, ended
            )
        usage = _parse_usage(parsed.usage)
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
            finish_reason=parsed.finish_reason,
            output_text="".join(parsed.text),
            tool_calls=_tool_calls(list(parsed.function_calls)),
            provider_request_id=parsed.response_id or retained.get("x-goog-request-id"),
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
        digest = hashlib.sha256()
        parsed_events: list[ParsedResponse] = []
        text: list[str] = []
        calls: list[tuple[str, dict[str, Any]]] = []
        offsets: list[float] = []
        usage_values: list[Any] = []
        first_visible_at: float | None = None
        headers_at: float | None = None
        finish: str | None = None
        response_id: str | None = None
        try:
            async with client.stream(
                "POST",
                self._url(route, stream=True),
                headers=prepared.headers,
                content=prepared.payload.body,
                timeout=request.timeout_seconds,
            ) as response:
                headers_at = time.perf_counter()
                if response.status_code >= 300:
                    raw = await response.aread()
                    ended = time.perf_counter()
                    return self._http_error(
                        route,
                        request,
                        response,
                        raw,
                        started_utc,
                        started,
                        headers_at,
                        ended,
                    )
                async for payload in _sse_payloads(response, digest):
                    if payload == b"[DONE]":
                        continue
                    parsed = _parse_success(strict_json_loads(payload))
                    parsed_events.append(parsed)
                    response_id = parsed.response_id or response_id
                    if parsed.usage is not None:
                        usage_values.append(parsed.usage)
                    if parsed.finish_reason is not None:
                        finish = parsed.finish_reason
                    for piece in parsed.text:
                        now = time.perf_counter()
                        first_visible_at = first_visible_at or now
                        offsets.append(now - started)
                        text.append(piece)
                    for call in parsed.function_calls:
                        now = time.perf_counter()
                        first_visible_at = first_visible_at or now
                        offsets.append(now - started)
                        calls.append(call)
                ended = time.perf_counter()
                retained = _retained(response.headers, route)
        except (StrictJSONError, ValueError, TypeError):
            ended = time.perf_counter()
            return InferenceResult(
                logical_id=request.logical_id,
                status="server_error",
                http_status=200,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=None if headers_at is None else headers_at - started,
                error_kind="protocol_error",
                error_body_sha256=digest.hexdigest(),
            )
        terminal = any(event.terminal for event in parsed_events)
        if not parsed_events or not (text or calls or terminal):
            ended = time.perf_counter()
            return InferenceResult(
                logical_id=request.logical_id,
                status="server_error",
                http_status=200,
                started_at_utc=started_utc,
                ended_at_utc=_utc_now(),
                total_seconds=ended - started,
                time_to_headers_seconds=None if headers_at is None else headers_at - started,
                error_kind="protocol_error",
                error_body_sha256=digest.hexdigest(),
            )
        usage = _merge_usage(usage_values)
        ttft = None if first_visible_at is None else first_visible_at - started
        return InferenceResult(
            logical_id=request.logical_id,
            status="success",
            http_status=200,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=None if headers_at is None else headers_at - started,
            ttft_seconds=ttft,
            decode_seconds=None if ttft is None else max(0.0, ended - started - ttft),
            output_event_offsets_seconds=tuple(offsets),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            usage_parse_errors=usage.errors,
            cache_state=str(request.metadata.get("cache_state", "uncontrolled")),  # type: ignore[arg-type]
            finish_reason=finish,
            output_text="".join(text),
            tool_calls=_tool_calls(calls),
            provider_request_id=response_id or retained.get("x-goog-request-id"),
            retained_headers=retained,
        )

    @staticmethod
    def _error_status(http_status: int, raw: bytes) -> tuple[str, str]:
        envelope_status: str | None = None
        try:
            value = strict_json_loads(raw)
            error = value.get("error") if isinstance(value, dict) else None
            status = error.get("status") if isinstance(error, dict) else None
            if isinstance(status, str):
                envelope_status = status.strip().upper()
        except StrictJSONError:
            pass
        if http_status == 429 or envelope_status == "RESOURCE_EXHAUSTED":
            return "rate_limited", "provider_rate_limit"
        if http_status == 408 or envelope_status == "DEADLINE_EXCEEDED":
            return "timeout", "timeout"
        if http_status in {400, 401, 403, 404} or envelope_status in {
            "INVALID_ARGUMENT",
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
            "NOT_FOUND",
            "FAILED_PRECONDITION",
        }:
            return "client_error", "provider_route_fatal"
        return "server_error", f"http_{http_status}"

    @classmethod
    def _http_error(
        cls,
        route: RouteConfig,
        request: RequestSpec,
        response: httpx.Response,
        raw: bytes,
        started_utc: str,
        started: float,
        headers_at: float,
        ended: float,
    ) -> InferenceResult:
        status, error_kind = cls._error_status(response.status_code, raw)
        retained = _retained(response.headers, route)
        return InferenceResult(
            logical_id=request.logical_id,
            status=status,  # type: ignore[arg-type]
            http_status=response.status_code,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            total_seconds=ended - started,
            time_to_headers_seconds=headers_at - started,
            provider_request_id=retained.get("x-goog-request-id"),
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
            provider_request_id=retained.get("x-goog-request-id"),
            retained_headers=retained,
            error_kind="protocol_error",
            error_body_sha256=hashlib.sha256(raw).hexdigest(),
        )
