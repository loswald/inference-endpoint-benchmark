"""Typed embedding transports with a shared privacy-safe result boundary."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import httpx

from ..embedding_models import (
    EMBEDDING_ADAPTER_ID,
    EMBEDDING_API_FAMILY,
    EMBEDDING_GENERATOR_VERSION,
    EmbeddingRepeatabilityObservation,
    EmbeddingRequestSpec,
    EmbeddingResult,
    EmbeddingRouteConfig,
    EmbeddingVectorObservation,
    canonical_vector_sha256,
)
from ..json_contract import StrictJSONError, strict_json_loads
from ..models import canonical_json, sha256_json
from ..payload import MaterializedPayload, payload_binding_sha256
from .openai_compatible import static_api_key_headers


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class PreparedEmbeddingRequest:
    payload: MaterializedPayload
    headers: dict[str, str] = field(default_factory=dict, repr=False)


class EmbeddingAdapter(Protocol):
    def preflight(self, route: EmbeddingRouteConfig) -> None: ...

    def prepare(
        self, route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
    ) -> PreparedEmbeddingRequest: ...

    async def send_prepared(
        self,
        route: EmbeddingRouteConfig,
        request: EmbeddingRequestSpec,
        prepared: PreparedEmbeddingRequest,
    ) -> EmbeddingResult: ...

    async def close(self) -> None: ...


class EmbeddingAdapterFactory(Protocol):
    def __call__(
        self,
        *,
        http2: bool,
        connection_reuse: bool,
        transport_max_connections: int,
    ) -> EmbeddingAdapter: ...


@dataclass(frozen=True, slots=True)
class EmbeddingAdapterPlugin:
    id: str
    version: str
    api_family: str
    input_cardinality: Literal["one", "one_or_many"]
    factory: EmbeddingAdapterFactory = field(repr=False, compare=False)
    materializer: Callable[
        [EmbeddingRouteConfig, EmbeddingRequestSpec], MaterializedPayload
    ] = field(repr=False, compare=False)
    route_validator: Callable[[EmbeddingRouteConfig], None] = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("embedding adapter id contains invalid characters")
        if not self.version:
            raise ValueError("embedding adapter version is required")
        if self.api_family != EMBEDDING_API_FAMILY:
            raise ValueError("embedding adapter plugins require api_family=embeddings")
        if self.input_cardinality not in {"one", "one_or_many"}:
            raise ValueError("embedding adapter input_cardinality must be one or one_or_many")
        for name in ("factory", "materializer", "route_validator"):
            if not callable(getattr(self, name)):
                raise TypeError(f"embedding adapter {name} must be callable")

    def public_identity(self) -> dict[str, Any]:
        value = {
            "schema": "embedding-adapter-plugin/v1",
            "id": self.id,
            "version": self.version,
            "api_family": self.api_family,
            "input_cardinality": self.input_cardinality,
        }
        return {**value, "identity_sha256": sha256_json(value)}

    def validate_route(self, route: EmbeddingRouteConfig) -> None:
        if route.adapter != self.id:
            raise ValueError("embedding adapter plugin does not match route adapter")
        if route.api_family != self.api_family:
            raise ValueError("embedding adapter API family does not match route")
        self.route_validator(route)


_EMBEDDING_ADAPTERS: dict[str, EmbeddingAdapterPlugin] = {}


def register_embedding_adapter(
    plugin: EmbeddingAdapterPlugin, *, replace: bool = False
) -> None:
    if not isinstance(plugin, EmbeddingAdapterPlugin):
        raise TypeError("embedding adapter registration requires EmbeddingAdapterPlugin")
    if plugin.id in _EMBEDDING_ADAPTERS and not replace:
        raise ValueError(f"embedding adapter already registered: {plugin.id}")
    _EMBEDDING_ADAPTERS[plugin.id] = plugin


def embedding_adapter_plugin(adapter_id: str) -> EmbeddingAdapterPlugin:
    _register_builtin()
    try:
        return _EMBEDDING_ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ValueError(f"unknown embedding adapter: {adapter_id}") from exc


def embedding_adapter_for(route: EmbeddingRouteConfig) -> EmbeddingAdapter:
    plugin = embedding_adapter_plugin(route.adapter)
    plugin.validate_route(route)
    adapter = plugin.factory(
        http2=route.http2,
        connection_reuse=route.connection_reuse,
        transport_max_connections=route.transport_max_connections,
    )
    for method in ("preflight", "prepare", "send_prepared", "close"):
        if not callable(getattr(adapter, method, None)):
            raise TypeError(f"embedding adapter is missing required method {method}")
    return adapter


def materialize_embedding_request_for(
    route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
) -> MaterializedPayload:
    plugin = embedding_adapter_plugin(route.adapter)
    plugin.validate_route(route)
    return plugin.materializer(route, request)


def build_embedding_payload(
    route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
) -> dict[str, Any]:
    input_value: str | list[str]
    if request.input_encoding == "single_string":
        input_value = request.inputs[0]
    else:
        input_value = list(request.inputs)
    value: dict[str, Any] = {
        "model": route.model,
        "input": input_value,
        "encoding_format": "float",
    }
    if request.dimensions is not None:
        value["dimensions"] = request.dimensions
    return value


def materialize_embedding_request(
    route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
) -> MaterializedPayload:
    value = build_embedding_payload(route, request)
    body = canonical_json(value).encode("utf-8")
    if strict_json_loads(body) != value:
        raise ValueError("embedding payload failed canonical JSON round trip")
    wire_hash = hashlib.sha256(body).hexdigest()
    bound_hash = payload_binding_sha256(body, EMBEDDING_GENERATOR_VERSION)
    # The provider usage count is authoritative.  This is only the conservative pre-send
    # reservation and deliberately combines the registered workload target with framing overhead.
    upper = max(1, request.planned_input_tokens + route.input_token_reservation_overhead)
    return MaterializedPayload(
        value=value,
        body=body,
        wire_body_sha256=wire_hash,
        bound_payload_sha256=bound_hash,
        input_token_upper_bound=upper,
        generator_version=EMBEDDING_GENERATOR_VERSION,
    )


def validate_openai_compatible_embedding_route(route: EmbeddingRouteConfig) -> None:
    from urllib.parse import urlsplit

    path = urlsplit(route.base_url).path.rstrip("/").casefold()
    if not path.endswith("/embeddings"):
        raise ValueError(
            "openai_compatible_embeddings requires a canonical /embeddings action URL"
        )


def _count(value: object, field_name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{field_name}_invalid")
        return None
    return value


def _provider_request_id_sha256(headers: httpx.Headers) -> str | None:
    for name in (
        "x-request-id",
        "request-id",
        "x-ms-request-id",
        "openai-request-id",
        "x-goog-request-id",
    ):
        value = headers.get(name)
        if value:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return None


def validate_embedding_vectors(
    route: EmbeddingRouteConfig,
    request: EmbeddingRequestSpec,
    indexed_vectors: list[tuple[object, object]],
) -> tuple[
    tuple[EmbeddingVectorObservation, ...],
    tuple[EmbeddingRepeatabilityObservation, ...],
    tuple[str, ...],
]:
    """Validate vectors without retaining their values beyond the transient call boundary."""

    errors: list[str] = []
    observations: list[EmbeddingVectorObservation] = []
    transient_vectors: dict[int, list[float]] = {}
    seen_indices: set[int] = set()
    expected_dimensions = request.dimensions or route.capabilities.default_dimensions
    for raw_index, raw_vector in indexed_vectors:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            errors.append("embedding_index_invalid")
            continue
        if raw_index in seen_indices:
            errors.append("embedding_index_duplicate")
            continue
        seen_indices.add(raw_index)
        if not isinstance(raw_vector, list) or not raw_vector:
            errors.append("embedding_vector_missing_or_empty")
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_vector
        ):
            errors.append("embedding_vector_nonfinite_or_nonnumeric")
            continue
        vector = [float(value) for value in raw_vector]
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            errors.append("embedding_vector_zero_or_invalid_norm")
        if len(vector) != expected_dimensions:
            errors.append("embedding_dimension_mismatch")
        transient_vectors[raw_index] = vector
        observations.append(
            EmbeddingVectorObservation(
                index=raw_index,
                dimensions=len(vector),
                l2_norm=norm,
                vector_sha256=canonical_vector_sha256(vector),
            )
        )
    if seen_indices != set(range(len(request.inputs))):
        errors.append("embedding_indices_not_contiguous")

    repeatability: list[EmbeddingRepeatabilityObservation] = []
    for left, right in request.repeatability_pairs:
        left_vector = transient_vectors.get(left)
        right_vector = transient_vectors.get(right)
        if left_vector is None or right_vector is None or len(left_vector) != len(right_vector):
            errors.append("repeatability_pair_unavailable")
            continue
        left_norm = math.sqrt(sum(value * value for value in left_vector))
        right_norm = math.sqrt(sum(value * value for value in right_vector))
        if left_norm <= 0 or right_norm <= 0:
            cosine = 0.0
        else:
            cosine = sum(a * b for a, b in zip(left_vector, right_vector, strict=True)) / (
                left_norm * right_norm
            )
            cosine = max(-1.0, min(1.0, cosine))
        max_difference = max(
            abs(a - b) for a, b in zip(left_vector, right_vector, strict=True)
        )
        passed = cosine >= route.capabilities.repeatability_cosine_minimum
        if not passed:
            errors.append("repeatability_cosine_below_contract")
        repeatability.append(
            EmbeddingRepeatabilityObservation(
                left_index=left,
                right_index=right,
                cosine_similarity=cosine,
                max_absolute_difference=max_difference,
                passed=passed,
            )
        )
    return (
        tuple(sorted(observations, key=lambda item: item.index)),
        tuple(repeatability),
        tuple(dict.fromkeys(errors)),
    )


class OpenAICompatibleEmbeddingsAdapter:
    """Exact-byte JSON adapter for the OpenAI-compatible ``/embeddings`` contract."""

    def __init__(
        self,
        *,
        http2: bool = False,
        connection_reuse: bool = True,
        transport_max_connections: int = 64,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.http2 = http2
        self.connection_reuse = connection_reuse
        self.transport_max_connections = transport_max_connections
        self._provided_client = client is not None
        self.client = client or httpx.AsyncClient(
            http2=http2,
            limits=httpx.Limits(
                max_connections=transport_max_connections,
                max_keepalive_connections=(transport_max_connections if connection_reuse else 0),
            ),
        )

    def preflight(self, route: EmbeddingRouteConfig) -> None:
        validate_openai_compatible_embedding_route(route)
        if not self._provided_client and (
            route.http2 != self.http2
            or route.connection_reuse != self.connection_reuse
            or route.transport_max_connections != self.transport_max_connections
        ):
            raise RuntimeError("embedding adapter transport does not match route identity")
        static_api_key_headers(route)  # type: ignore[arg-type]

    def prepare(
        self, route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
    ) -> PreparedEmbeddingRequest:
        self.preflight(route)
        return PreparedEmbeddingRequest(
            payload=materialize_embedding_request(route, request),
            headers=static_api_key_headers(route),  # type: ignore[arg-type]
        )

    async def close(self) -> None:
        if not self._provided_client:
            await self.client.aclose()

    async def send_prepared(
        self,
        route: EmbeddingRouteConfig,
        request: EmbeddingRequestSpec,
        prepared: PreparedEmbeddingRequest,
    ) -> EmbeddingResult:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(route.request_timeout_seconds):
                response = await self.client.post(
                    route.base_url,
                    headers=prepared.headers,
                    content=prepared.payload.body,
                    timeout=route.request_timeout_seconds,
                )
        except (TimeoutError, httpx.TimeoutException):
            return EmbeddingResult(
                logical_id=request.logical_id,
                status="timeout",
                http_status=None,
                total_seconds=time.perf_counter() - started,
            )
        except httpx.TransportError:
            return EmbeddingResult(
                logical_id=request.logical_id,
                status="transport_error",
                http_status=None,
                total_seconds=time.perf_counter() - started,
            )
        elapsed = time.perf_counter() - started
        raw = response.content
        provider_id = _provider_request_id_sha256(response.headers)
        if response.status_code >= 300:
            if response.status_code == 429:
                status = "rate_limited"
            elif response.status_code >= 500:
                status = "server_error"
            else:
                status = "client_error"
            return EmbeddingResult(
                logical_id=request.logical_id,
                status=status,
                http_status=response.status_code,
                total_seconds=elapsed,
                provider_request_id_sha256=provider_id,
                error_body_sha256=hashlib.sha256(raw).hexdigest(),
            )
        try:
            payload = strict_json_loads(raw)
        except StrictJSONError:
            payload = None
        if not isinstance(payload, dict):
            return EmbeddingResult(
                logical_id=request.logical_id,
                status="protocol_error",
                http_status=response.status_code,
                total_seconds=elapsed,
                validation_errors=("success_body_not_json_object",),
                provider_request_id_sha256=provider_id,
            )
        return self._parse_success(
            route, request, payload, response.status_code, elapsed, provider_id
        )

    def _parse_success(
        self,
        route: EmbeddingRouteConfig,
        request: EmbeddingRequestSpec,
        payload: dict[str, Any],
        http_status: int,
        elapsed: float,
        provider_id: str | None,
    ) -> EmbeddingResult:
        errors: list[str] = []
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            errors.append("usage_missing_or_invalid")
            prompt_tokens = total_tokens = None
        else:
            prompt_tokens = _count(usage.get("prompt_tokens"), "prompt_tokens", errors)
            total_tokens = _count(usage.get("total_tokens"), "total_tokens", errors)
            if (
                prompt_tokens is not None
                and total_tokens is not None
                and total_tokens < prompt_tokens
            ):
                errors.append("total_tokens_less_than_prompt_tokens")
            if prompt_tokens == 0 and any(request.inputs):
                errors.append("prompt_tokens_zero_for_nonempty_input")

        raw_data = payload.get("data")
        if not isinstance(raw_data, list):
            raw_data = []
            errors.append("data_missing_or_invalid")
        if len(raw_data) != len(request.inputs):
            errors.append("embedding_count_mismatch")
        indexed_vectors: list[tuple[object, object]] = []
        for position, raw_item in enumerate(raw_data):
            if not isinstance(raw_item, dict):
                errors.append("embedding_item_not_object")
                continue
            indexed_vectors.append(
                (raw_item.get("index", position), raw_item.get("embedding"))
            )
        observations, repeatability, vector_errors = validate_embedding_vectors(
            route, request, indexed_vectors
        )
        errors.extend(vector_errors)

        return EmbeddingResult(
            logical_id=request.logical_id,
            status="success",
            http_status=http_status,
            total_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            vectors=observations,
            repeatability=repeatability,
            validation_errors=tuple(dict.fromkeys(errors)),
            provider_request_id_sha256=provider_id,
        )


def _register_builtin() -> None:
    if EMBEDDING_ADAPTER_ID not in _EMBEDDING_ADAPTERS:
        register_embedding_adapter(
            EmbeddingAdapterPlugin(
                id=EMBEDDING_ADAPTER_ID,
                version="builtin/v1",
                api_family=EMBEDDING_API_FAMILY,
                input_cardinality="one_or_many",
                factory=OpenAICompatibleEmbeddingsAdapter,
                materializer=materialize_embedding_request,
                route_validator=validate_openai_compatible_embedding_route,
            )
        )
    from ..embedding_models import VERTEX_EMBED_CONTENT_ADAPTER_ID
    from .vertex_embeddings import (
        VertexEmbedContentAdapter,
        materialize_vertex_embed_content,
        validate_vertex_embed_content_route,
    )

    if VERTEX_EMBED_CONTENT_ADAPTER_ID not in _EMBEDDING_ADAPTERS:
        register_embedding_adapter(
            EmbeddingAdapterPlugin(
                id=VERTEX_EMBED_CONTENT_ADAPTER_ID,
                version="builtin/v1",
                api_family=EMBEDDING_API_FAMILY,
                input_cardinality="one",
                factory=VertexEmbedContentAdapter,
                materializer=materialize_vertex_embed_content,
                route_validator=validate_vertex_embed_content_route,
            )
        )


__all__ = [
    "EmbeddingAdapter",
    "EmbeddingAdapterPlugin",
    "OpenAICompatibleEmbeddingsAdapter",
    "PreparedEmbeddingRequest",
    "build_embedding_payload",
    "embedding_adapter_for",
    "embedding_adapter_plugin",
    "materialize_embedding_request",
    "materialize_embedding_request_for",
    "register_embedding_adapter",
    "validate_embedding_vectors",
    "validate_openai_compatible_embedding_route",
]
