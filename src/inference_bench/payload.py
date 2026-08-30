from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from botocore.serialize import create_serializer
from botocore.session import Session

from .models import RequestSpec, RouteConfig, canonical_json
from .workloads import materialize_messages

PAYLOAD_GENERATOR_VERSION = "openai-compatible-synthetic/v4"
RESPONSES_PAYLOAD_GENERATOR_VERSION = "openai-responses-synthetic/v2"
BEDROCK_CONVERSE_PAYLOAD_GENERATOR_VERSION = "aws-bedrock-converse-rest-json/v1"
_PAYLOAD_BINDING_PREFIX = b"materialized-payload/v2\0"


def payload_binding_sha256(body: bytes, generator_version: str) -> str:
    """Bind an adapter's exact wire bytes to its declared materializer version."""

    if not isinstance(body, bytes):
        raise TypeError("materialized request body must be bytes")
    if not isinstance(generator_version, str) or not generator_version:
        raise ValueError("payload generator version must be a nonempty string")
    return hashlib.sha256(
        _PAYLOAD_BINDING_PREFIX + generator_version.encode("utf-8") + b"\0" + body
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class MaterializedPayload:
    value: dict[str, Any]
    body: bytes
    wire_body_sha256: str
    bound_payload_sha256: str
    input_token_upper_bound: int
    generator_version: str = PAYLOAD_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.value, dict):
            raise TypeError("materialized payload value must be a mapping")
        if not isinstance(self.body, bytes):
            raise TypeError("materialized request body must be bytes")
        if hashlib.sha256(self.body).hexdigest() != self.wire_body_sha256:
            raise ValueError("wire_body_sha256 does not match the exact prepared bytes")
        if payload_binding_sha256(self.body, self.generator_version) != self.bound_payload_sha256:
            raise ValueError("bound_payload_sha256 does not match the prepared bytes and version")
        if (
            isinstance(self.input_token_upper_bound, bool)
            or not isinstance(self.input_token_upper_bound, int)
            or self.input_token_upper_bound <= 0
        ):
            raise ValueError("materialized input-token upper bound must be a positive integer")


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
    # Chat controls are emitted only from this route's exact named mapping. The explicit
    # provider_default state resolves to an empty mapping and therefore stays off the wire.
    payload.update(route.reasoning_control(request.reasoning_budget))
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
    bound_hash = payload_binding_sha256(body, PAYLOAD_GENERATOR_VERSION)
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
    controls = route.reasoning_control(request.reasoning_budget)
    if effort := controls.get("reasoning.effort"):
        payload["reasoning"] = {"effort": effort}
    if verbosity := controls.get("text.verbosity"):
        payload.setdefault("text", {})["verbosity"] = verbosity
    return payload


def materialize_responses(route: RouteConfig, request: RequestSpec) -> MaterializedPayload:
    value = build_responses_payload(route, request)
    body = canonical_json(value).encode("utf-8")
    wire_hash = hashlib.sha256(body).hexdigest()
    bound_hash = payload_binding_sha256(body, RESPONSES_PAYLOAD_GENERATOR_VERSION)
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


@lru_cache(maxsize=1)
def _bedrock_runtime_service_model() -> Any:
    # Botocore ships AWS's generated Bedrock Runtime service model. Loading it is local and
    # credential-free; it performs no provider discovery or network request.
    return Session().get_service_model("bedrock-runtime")


def _bedrock_image_block(url: Any) -> dict[str, Any]:
    if not isinstance(url, str) or not url:
        raise ValueError("Bedrock image_url content requires a nonempty URL")
    if url.startswith("s3://"):
        return {"image": {"format": "png", "source": {"s3Location": {"uri": url}}}}
    header, separator, encoded = url.partition(",")
    media_types = {
        "data:image/png;base64": "png",
        "data:image/jpeg;base64": "jpeg",
        "data:image/gif;base64": "gif",
        "data:image/webp;base64": "webp",
    }
    image_format = media_types.get(header.casefold())
    if separator != "," or image_format is None or not encoded:
        raise ValueError(
            "Bedrock vision accepts exact base64 png/jpeg/gif/webp data URLs or s3:// URIs"
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Bedrock image data URL contains invalid base64") from exc
    if not raw:
        raise ValueError("Bedrock image data URL decodes to an empty image")
    return {"image": {"format": image_format, "source": {"bytes": raw}}}


def _bedrock_tool_use(value: dict[str, Any]) -> dict[str, Any]:
    tool_id = value.get("id", value.get("toolUseId"))
    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = value.get("name")
        arguments = value.get("input", {})
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("Bedrock assistant tool use requires a nonempty id")
    if not isinstance(name, str) or not name:
        raise ValueError("Bedrock assistant tool use requires a nonempty function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError) as exc:
            raise ValueError("Bedrock assistant tool arguments must be valid JSON") from exc
    try:
        canonical_json(arguments)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bedrock assistant tool arguments must be finite JSON") from exc
    return {"toolUse": {"toolUseId": tool_id, "name": name, "input": arguments}}


def _bedrock_tool_result_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, dict):
        try:
            canonical_json(content)
        except (TypeError, ValueError) as exc:
            raise ValueError("Bedrock tool result must be finite JSON") from exc
        return [{"json": content}]
    if isinstance(content, list):
        result: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Bedrock tool-result content parts must be objects")
            kind = part.get("type")
            if kind == "text" and isinstance(part.get("text"), str):
                result.append({"text": part["text"]})
            elif kind == "image_url" and isinstance(part.get("image_url"), dict):
                result.append(_bedrock_image_block(part["image_url"].get("url")))
            else:
                raise ValueError("unsupported Bedrock tool-result content block")
        if result:
            return result
    raise ValueError("Bedrock tool result requires text, JSON, or supported content blocks")


def _bedrock_message_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list) or not content:
        raise ValueError("Bedrock message content must be text or a nonempty content-block list")
    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Bedrock message content parts must be objects")
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            converted.append({"text": part["text"]})
        elif kind == "image_url" and isinstance(part.get("image_url"), dict):
            converted.append(_bedrock_image_block(part["image_url"].get("url")))
        elif kind in {"tool_use", "tool_call"}:
            converted.append(_bedrock_tool_use(part))
        elif kind == "tool_result":
            tool_id = part.get("tool_use_id", part.get("toolUseId"))
            if not isinstance(tool_id, str) or not tool_id:
                raise ValueError("Bedrock tool result requires a nonempty tool-use id")
            block: dict[str, Any] = {
                "toolUseId": tool_id,
                "content": _bedrock_tool_result_content(part.get("content")),
            }
            if part.get("is_error") is True:
                block["status"] = "error"
            converted.append({"toolResult": block})
        else:
            raise ValueError("unsupported Bedrock message content block")
    return converted


def _bedrock_messages(request: RequestSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for message in materialize_messages(request):
        if not isinstance(message, dict):
            raise ValueError("Bedrock messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            blocks = _bedrock_message_content(content)
            if any("text" not in block for block in blocks):
                raise ValueError("Bedrock system messages currently support text blocks only")
            system.extend(blocks)
            continue
        if role == "tool":
            tool_id = message.get("tool_call_id")
            if not isinstance(tool_id, str) or not tool_id:
                raise ValueError("Bedrock tool-role messages require tool_call_id")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_id,
                                "content": _bedrock_tool_result_content(content),
                            }
                        }
                    ],
                }
            )
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(
                "Bedrock Converse message role must be user, assistant, system, or tool"
            )
        blocks = _bedrock_message_content(content)
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if role != "assistant" or not isinstance(tool_calls, list):
                raise ValueError("Bedrock tool_calls must be an array on an assistant message")
            blocks.extend(_bedrock_tool_use(call) for call in tool_calls)
        messages.append({"role": role, "content": blocks})
    if not messages:
        raise ValueError("Bedrock Converse requires at least one non-system message")
    return system, messages


def _bedrock_tools(request: RequestSpec) -> dict[str, Any] | None:
    if not request.tools:
        if request.tool_choice is not None:
            raise ValueError("Bedrock tool_choice requires at least one tool")
        return None
    tools: list[dict[str, Any]] = []
    for tool in request.tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Bedrock Converse currently accepts OpenAI-style function tools")
        function = tool["function"]
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError("Bedrock function tool requires a nonempty name")
        if not isinstance(parameters, dict):
            raise ValueError("Bedrock function tool requires a JSON-object parameters schema")
        tool_spec: dict[str, Any] = {"name": name, "inputSchema": {"json": parameters}}
        description = function.get("description")
        if description is not None:
            if not isinstance(description, str) or not description:
                raise ValueError("Bedrock function-tool description must be nonempty when present")
            tool_spec["description"] = description
        strict = function.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise ValueError("Bedrock function-tool strict must be boolean")
            tool_spec["strict"] = strict
        tools.append({"toolSpec": tool_spec})
    result: dict[str, Any] = {"tools": tools}
    choice = request.tool_choice
    if choice is None or choice == "auto":
        if choice == "auto":
            result["toolChoice"] = {"auto": {}}
    elif choice == "required" or choice == "any":
        result["toolChoice"] = {"any": {}}
    elif choice == "none":
        raise ValueError("Bedrock Converse has no generic tool_choice=none wire control")
    elif isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else choice.get("name")
        if choice.get("type") not in {"function", "tool"} or not isinstance(name, str) or not name:
            raise ValueError("Bedrock named tool_choice requires an exact function name")
        result["toolChoice"] = {"tool": {"name": name}}
    else:
        raise ValueError("unsupported Bedrock tool_choice")
    return result


def _bedrock_output_config(route: RouteConfig, request: RequestSpec) -> dict[str, Any] | None:
    output: dict[str, Any] = {}
    if request.response_format is not None:
        response_format = request.response_format
        kind = response_format.get("type")
        if kind == "json_object":
            definition = {
                "name": "json_object",
                "schema": canonical_json({"type": "object"}),
            }
        elif kind == "json_schema" and isinstance(response_format.get("json_schema"), dict):
            source = response_format["json_schema"]
            name = source.get("name")
            schema = source.get("schema")
            if not isinstance(name, str) or not name or not isinstance(schema, dict):
                raise ValueError("Bedrock json_schema requires a nonempty name and object schema")
            definition = {"name": name, "schema": canonical_json(schema)}
            description = source.get("description")
            if description is not None:
                if not isinstance(description, str) or not description:
                    raise ValueError("Bedrock JSON-schema description must be nonempty")
                definition["description"] = description
        else:
            raise ValueError("Bedrock Converse supports json_object or named json_schema formats")
        output["textFormat"] = {
            "type": "json_schema",
            "structure": {"jsonSchema": definition},
        }
    controls = route.reasoning_control(request.reasoning_budget)
    if effort := controls.get("outputConfig.effort"):
        output["effort"] = effort
    return output or None


def build_bedrock_converse_request(
    route: RouteConfig, request: RequestSpec
) -> dict[str, Any]:
    """Translate a provider-neutral request to AWS's generated Converse input shape."""

    if request.seed is not None:
        raise ValueError("Bedrock Converse has no generic seed field")
    if request.parallel_tool_calls is not None:
        raise ValueError("Bedrock Converse has no generic parallel_tool_calls field")
    if request.logprobs is not None:
        raise ValueError("Bedrock Converse has no generic logprobs field")
    system, messages = _bedrock_messages(request)
    inference: dict[str, Any] = {"maxTokens": request.max_output_tokens}
    if request.temperature is not None:
        inference["temperature"] = request.temperature
    if request.top_p is not None:
        inference["topP"] = request.top_p
    if request.stop:
        inference["stopSequences"] = list(request.stop)
    value: dict[str, Any] = {
        "modelId": route.model,
        "messages": messages,
        "inferenceConfig": inference,
    }
    if system:
        value["system"] = system
    if tool_config := _bedrock_tools(request):
        value["toolConfig"] = tool_config
    if output_config := _bedrock_output_config(route, request):
        value["outputConfig"] = output_config
    return value


def materialize_bedrock_converse(
    route: RouteConfig, request: RequestSpec
) -> MaterializedPayload:
    """Use AWS's generated REST-JSON serializer before the durable spend claim."""

    parameters = build_bedrock_converse_request(route, request)
    service_model = _bedrock_runtime_service_model()
    operation_name = "ConverseStream" if request.stream else "Converse"
    operation = service_model.operation_model(operation_name)
    serializer = create_serializer(service_model.metadata["protocol"])
    serialized = serializer.serialize_to_request(parameters, operation)
    if (
        serialized.get("method") != "POST"
        or serialized.get("query_string")
        or not isinstance(serialized.get("url_path"), str)
    ):
        raise ValueError("unexpected Bedrock REST-JSON request shape")
    body = serialized.get("body")
    if not isinstance(body, bytes):
        raise TypeError("Bedrock serializer did not produce exact request bytes")
    try:
        decoded_body = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bedrock serializer produced invalid JSON") from exc
    if not isinstance(decoded_body, dict):
        raise ValueError("Bedrock serializer produced a non-object JSON body")
    wire_hash = hashlib.sha256(body).hexdigest()
    bound_hash = payload_binding_sha256(body, BEDROCK_CONVERSE_PAYLOAD_GENERATOR_VERSION)
    upper = len(body) + route.input_token_reservation_overhead
    return MaterializedPayload(
        value={
            "operation": operation_name,
            "url_path": serialized["url_path"],
            "body": decoded_body,
        },
        body=body,
        wire_body_sha256=wire_hash,
        bound_payload_sha256=bound_hash,
        input_token_upper_bound=upper,
        generator_version=BEDROCK_CONVERSE_PAYLOAD_GENERATOR_VERSION,
    )


def reserved_input_tokens(
    route: RouteConfig, request: RequestSpec, reservation_factor: float
) -> int:
    if route.api_family == "responses":
        materialized = materialize_responses(route, request)
    elif route.api_family == "converse":
        materialized = materialize_bedrock_converse(route, request)
    else:
        materialized = materialize_openai_compatible(route, request)
    return math.ceil(
        max(request.planned_input_tokens, materialized.input_token_upper_bound) * reservation_factor
    )
