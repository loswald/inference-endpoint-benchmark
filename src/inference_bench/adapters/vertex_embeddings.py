from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from ..embedding_models import (
    EMBEDDING_API_FAMILY,
    VERTEX_EMBED_CONTENT_ADAPTER_ID,
    EmbeddingRequestSpec,
    EmbeddingResult,
    EmbeddingRouteConfig,
)
from ..json_contract import StrictJSONError, strict_json_loads
from ..models import canonical_json
from ..payload import MaterializedPayload, payload_binding_sha256
from .embeddings import PreparedEmbeddingRequest, validate_embedding_vectors
from .google_oauth import GoogleOAuthBearer

VERTEX_EMBED_CONTENT_GENERATOR_VERSION = "vertex-embed-content/v1"
_ACTION_PATH = re.compile(
    r"^/v1/projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/"
    r"publishers/google/models/(?P<model>[^/:]+):embedContent$"
)


def validate_vertex_embed_content_route(route: EmbeddingRouteConfig) -> None:
    parsed = urlsplit(route.base_url)
    match = _ACTION_PATH.fullmatch(parsed.path)
    if match is None:
        raise ValueError("vertex_embed_content requires the explicit v1 :embedContent action")
    location = unquote(match.group("location"))
    model = unquote(match.group("model"))
    if location != route.region or model != route.model:
        raise ValueError("Vertex embedding action must exactly match route location and model")
    allowed_hosts = {
        f"aiplatform.{location}.rep.googleapis.com",
        f"{location}-aiplatform.googleapis.com",
    }
    if location == "global":
        allowed_hosts.add("aiplatform.googleapis.com")
    if (parsed.hostname or "").casefold() not in {
        host.casefold() for host in allowed_hosts
    }:
        raise ValueError("Vertex embedding action host must exactly match its location")
    if route.provider != "google-vertex-ai":
        raise ValueError("vertex_embed_content requires provider=google-vertex-ai")
    if route.adapter != VERTEX_EMBED_CONTENT_ADAPTER_ID:
        raise ValueError("Vertex embedding route declares the wrong adapter")
    if route.api_family != EMBEDDING_API_FAMILY or route.api_version != "v1":
        raise ValueError("Vertex embedContent routes require embeddings/v1")
    if route.capabilities.max_batch_inputs != 1:
        raise ValueError("Vertex embedContent accepts exactly one Content per HTTP request")
    if route.auth.header.casefold() != "authorization" or route.auth.prefix != "Bearer ":
        raise ValueError("Vertex embedContent requires renewable Authorization: Bearer OAuth")
    if any(name.casefold() == "accept" for name in route.extra_headers):
        raise ValueError("Vertex embedding routes cannot override the negotiated Accept header")


def build_vertex_embed_content_payload(
    route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
) -> dict[str, Any]:
    validate_vertex_embed_content_route(route)
    if request.input_encoding != "single_string" or len(request.inputs) != 1:
        raise ValueError("Vertex embedContent materializes exactly one input per request")
    dimensions = request.dimensions or route.capabilities.default_dimensions
    if dimensions not in route.capabilities.supported_dimensions:
        raise ValueError("requested Vertex embedding dimensions are outside the route contract")
    return {
        "content": {"role": "user", "parts": [{"text": request.inputs[0]}]},
        "embedContentConfig": {
            "outputDimensionality": dimensions,
        },
    }


def materialize_vertex_embed_content(
    route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
) -> MaterializedPayload:
    value = build_vertex_embed_content_payload(route, request)
    body = canonical_json(value).encode("utf-8")
    if strict_json_loads(body) != value:
        raise ValueError("Vertex embedding payload failed canonical JSON round trip")
    upper = max(1, request.planned_input_tokens + route.input_token_reservation_overhead)
    return MaterializedPayload(
        value=value,
        body=body,
        wire_body_sha256=hashlib.sha256(body).hexdigest(),
        bound_payload_sha256=payload_binding_sha256(
            body, VERTEX_EMBED_CONTENT_GENERATOR_VERSION
        ),
        input_token_upper_bound=upper,
        generator_version=VERTEX_EMBED_CONTENT_GENERATOR_VERSION,
    )


def _token_count(value: object, name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{name}_invalid")
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        errors.append(f"{name}_invalid")
        return None
    return int(number)


def _provider_request_id_sha256(headers: httpx.Headers) -> str | None:
    value = headers.get("x-goog-request-id")
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None


class VertexEmbedContentAdapter:
    """Native Gemini ``embedContent`` transport with renewable Google OAuth."""

    def __init__(
        self,
        *,
        http2: bool = False,
        connection_reuse: bool = True,
        transport_max_connections: int = 64,
        client: httpx.AsyncClient | None = None,
        credentials: Any | None = None,
        credential_loader: Callable[[EmbeddingRouteConfig], Any] | None = None,
        auth_request_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.http2 = http2
        self.connection_reuse = connection_reuse
        self.transport_max_connections = transport_max_connections
        self._provided_client = client is not None
        self.client = client or httpx.AsyncClient(
            http2=http2,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=transport_max_connections,
                max_keepalive_connections=(
                    transport_max_connections if connection_reuse else 0
                ),
            ),
        )
        self._oauth = GoogleOAuthBearer(
            credentials=credentials,
            credential_loader=credential_loader,  # type: ignore[arg-type]
            auth_request_factory=auth_request_factory,
        )

    def preflight(self, route: EmbeddingRouteConfig) -> None:
        validate_vertex_embed_content_route(route)
        if not self._provided_client and (
            route.http2 != self.http2
            or route.connection_reuse != self.connection_reuse
            or route.transport_max_connections != self.transport_max_connections
        ):
            raise RuntimeError("Vertex embedding transport does not match route identity")
        self._oauth.headers(route, accept="application/json")

    def prepare(
        self, route: EmbeddingRouteConfig, request: EmbeddingRequestSpec
    ) -> PreparedEmbeddingRequest:
        self.preflight(route)
        return PreparedEmbeddingRequest(
            payload=materialize_vertex_embed_content(route, request),
            headers=self._oauth.headers(route, accept="application/json"),
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
            elif response.status_code == 408:
                status = "timeout"
            elif response.status_code >= 500:
                status = "server_error"
            else:
                status = "client_error"
            return EmbeddingResult(
                logical_id=request.logical_id,
                status=status,  # type: ignore[arg-type]
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
                error_body_sha256=hashlib.sha256(raw).hexdigest(),
            )

        errors: list[str] = []
        usage = payload.get("usageMetadata")
        if not isinstance(usage, dict):
            prompt_tokens = total_tokens = None
            errors.append("usage_missing_or_invalid")
        else:
            prompt_tokens = _token_count(
                usage.get("promptTokenCount"), "prompt_tokens", errors
            )
            total_tokens = _token_count(
                usage.get("totalTokenCount"), "total_tokens", errors
            )
            if (
                prompt_tokens is not None
                and total_tokens is not None
                and total_tokens < prompt_tokens
            ):
                errors.append("total_tokens_less_than_prompt_tokens")
            if prompt_tokens == 0 and request.inputs[0]:
                errors.append("prompt_tokens_zero_for_nonempty_input")
        truncated = payload.get("truncated")
        if truncated not in {None, False, True}:
            errors.append("truncated_flag_invalid")
            truncated = None
        if request.expectation == "success_with_truncation" and truncated is not True:
            errors.append("expected_truncation_not_observed")
        if request.expectation != "success_with_truncation" and truncated is True:
            errors.append("unexpected_provider_truncation")
        embedding = payload.get("embedding")
        raw_vector = embedding.get("values") if isinstance(embedding, dict) else None
        observations, repeatability, vector_errors = validate_embedding_vectors(
            route, request, [(0, raw_vector)]
        )
        errors.extend(vector_errors)
        return EmbeddingResult(
            logical_id=request.logical_id,
            status="success",
            http_status=response.status_code,
            total_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            truncated=truncated,
            vectors=observations,
            repeatability=repeatability,
            validation_errors=tuple(dict.fromkeys(errors)),
            provider_request_id_sha256=provider_id,
        )


__all__ = [
    "VERTEX_EMBED_CONTENT_GENERATOR_VERSION",
    "VertexEmbedContentAdapter",
    "build_vertex_embed_content_payload",
    "materialize_vertex_embed_content",
    "validate_vertex_embed_content_route",
]
