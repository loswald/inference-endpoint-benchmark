"""Typed contracts for provider-neutral text-embedding benchmarks.

Embedding endpoints are not chat endpoints with a different URL.  Their request shape, limits,
usage accounting, output validation, and privacy boundary are different, so this module gives
them a separate typed surface.  Input text and returned vectors exist only in transient request
and result objects.  Public projections contain hashes and scalar diagnostics, never either
payload.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from .models import AuthConfig, canonical_json, sha256_json

EMBEDDING_PROFILE_SCHEMA = "embedding-benchmark/v1"
EMBEDDING_API_FAMILY = "embeddings"
EMBEDDING_ADAPTER_ID = "openai_compatible_embeddings"
EMBEDDING_GENERATOR_VERSION = "openai-compatible-embeddings/v1"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class EmbeddingCapabilityContract:
    """Documented limits that define one embedding-route estimand."""

    max_input_tokens_per_item: int
    max_batch_inputs: int
    default_dimensions: int
    supported_dimensions: tuple[int, ...]
    max_total_tokens_per_request: int | None = None
    empty_input: Literal["documented_invalid", "supported", "unknown"] = "documented_invalid"
    unicode_input: Literal["documented_supported", "unknown"] = "unknown"
    repeatability_cosine_minimum: float = 0.999999
    long_input_fraction: float = 0.90

    def __post_init__(self) -> None:
        _positive_int(self.max_input_tokens_per_item, "max_input_tokens_per_item")
        _positive_int(self.max_batch_inputs, "max_batch_inputs")
        _positive_int(self.default_dimensions, "default_dimensions")
        if (
            not isinstance(self.supported_dimensions, tuple)
            or not self.supported_dimensions
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.supported_dimensions
            )
        ):
            raise ValueError("supported_dimensions must be a nonempty tuple of positive integers")
        normalized = tuple(sorted(set(self.supported_dimensions)))
        object.__setattr__(self, "supported_dimensions", normalized)
        if self.default_dimensions not in normalized:
            raise ValueError("default_dimensions must appear in supported_dimensions")
        if self.max_total_tokens_per_request is not None:
            _positive_int(self.max_total_tokens_per_request, "max_total_tokens_per_request")
            if self.max_total_tokens_per_request < self.max_input_tokens_per_item:
                raise ValueError(
                    "max_total_tokens_per_request cannot be smaller than the per-item limit"
                )
        if self.empty_input not in {"documented_invalid", "supported", "unknown"}:
            raise ValueError("invalid empty_input capability state")
        if self.unicode_input not in {"documented_supported", "unknown"}:
            raise ValueError("invalid unicode_input capability state")
        cosine = _positive_number(
            self.repeatability_cosine_minimum,
            "repeatability_cosine_minimum",
            allow_zero=True,
        )
        if cosine > 1:
            raise ValueError("repeatability_cosine_minimum cannot exceed 1")
        fraction = _positive_number(self.long_input_fraction, "long_input_fraction")
        if not 0.5 <= fraction < 1:
            raise ValueError("long_input_fraction must be in [0.5, 1)")

    def public_dict(self) -> dict[str, Any]:
        return {
            "max_input_tokens_per_item": self.max_input_tokens_per_item,
            "max_batch_inputs": self.max_batch_inputs,
            "max_total_tokens_per_request": self.max_total_tokens_per_request,
            "default_dimensions": self.default_dimensions,
            "supported_dimensions": list(self.supported_dimensions),
            "empty_input": self.empty_input,
            "unicode_input": self.unicode_input,
            "repeatability_cosine_minimum": self.repeatability_cosine_minimum,
            "long_input_fraction": self.long_input_fraction,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingRouteConfig:
    id: str
    provider: str
    adapter: str
    model: str
    base_url: str
    auth: AuthConfig
    region: str
    billing_channel: str
    api_version: str
    model_version: str
    quota_scope: str
    capabilities: EmbeddingCapabilityContract
    input_usd_per_million: float
    documentation_source_url: str
    pricing_source_url: str
    evidence_retrieved_at_utc: str
    evidence_bundle_sha256: str
    api_family: str = EMBEDDING_API_FAMILY
    request_timeout_seconds: float = 180.0
    input_token_reservation_overhead: int = 128
    http2: bool = False
    connection_reuse: bool = True
    transport_max_connections: int = 64
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "id",
            "provider",
            "adapter",
            "model",
            "base_url",
            "region",
            "billing_channel",
            "api_version",
            "model_version",
            "quota_scope",
            "documentation_source_url",
            "pricing_source_url",
            "evidence_retrieved_at_utc",
            "evidence_bundle_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"embedding route {name} is required")
        if self.api_family != EMBEDDING_API_FAMILY:
            raise ValueError("embedding routes require api_family=embeddings")
        if self.adapter != EMBEDDING_ADAPTER_ID:
            raise ValueError(
                "embedding routes require the typed openai_compatible_embeddings adapter"
            )
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.query
            or parsed.path.rstrip("/").casefold().endswith("/embeddings") is False
        ):
            raise ValueError(
                "embedding base_url must be an absolute canonical HTTPS /embeddings endpoint"
            )
        _positive_number(self.request_timeout_seconds, "request_timeout_seconds")
        _positive_int(self.transport_max_connections, "transport_max_connections")
        if (
            isinstance(self.input_token_reservation_overhead, bool)
            or not isinstance(self.input_token_reservation_overhead, int)
            or self.input_token_reservation_overhead < 0
        ):
            raise ValueError("input_token_reservation_overhead must be nonnegative")
        _positive_number(self.input_usd_per_million, "input_usd_per_million", allow_zero=True)
        if not isinstance(self.http2, bool) or not isinstance(self.connection_reuse, bool):
            raise ValueError("embedding transport flags must be booleans")
        if not isinstance(self.extra_headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.extra_headers.items()
        ):
            raise ValueError("extra_headers must map strings to strings")
        blocked = {"authorization", "api-key", "x-api-key", "content-type"}
        if any(key.casefold() in blocked for key in self.extra_headers):
            raise ValueError(
                "credentials and content type cannot be supplied through extra_headers"
            )
        for source_name in ("documentation_source_url", "pricing_source_url"):
            source = urlsplit(getattr(self, source_name))
            if source.scheme != "https" or not source.hostname or source.query or source.fragment:
                raise ValueError(f"{source_name} must be a public absolute HTTPS URL")
        if not (
            len(self.evidence_bundle_sha256) == 64
            and all(character in "0123456789abcdef" for character in self.evidence_bundle_sha256)
        ):
            raise ValueError("evidence_bundle_sha256 must be a lowercase SHA-256")

    @property
    def identity_hash(self) -> str:
        return sha256_json(self.public_dict())

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "adapter": self.adapter,
            "model": self.model,
            "base_url": self.base_url,
            "auth_transport": {
                "env": self.auth.env,
                "header": self.auth.header,
                "prefix": self.auth.prefix,
            },
            "region": self.region,
            "api_family": self.api_family,
            "billing_channel": self.billing_channel,
            "api_version": self.api_version,
            "model_version": self.model_version,
            "quota_scope": self.quota_scope,
            "capabilities": self.capabilities.public_dict(),
            "input_usd_per_million": self.input_usd_per_million,
            "request_timeout_seconds": self.request_timeout_seconds,
            "input_token_reservation_overhead": self.input_token_reservation_overhead,
            "http2": self.http2,
            "connection_reuse": self.connection_reuse,
            "transport_max_connections": self.transport_max_connections,
            "extra_headers": self.extra_headers,
            "documentation_source_url": self.documentation_source_url,
            "pricing_source_url": self.pricing_source_url,
            "evidence_retrieved_at_utc": self.evidence_retrieved_at_utc,
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
        }

    def reserved_cost(self, input_tokens: int) -> float:
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
            raise ValueError("input_tokens must be a nonnegative integer")
        return input_tokens * self.input_usd_per_million / 1_000_000


EmbeddingExpectation = Literal["success", "client_error"]


@dataclass(frozen=True, slots=True)
class EmbeddingRequestSpec:
    logical_id: str
    route_id: str
    cell_id: str
    inputs: tuple[str, ...] = field(repr=False)
    planned_input_tokens: int = 0
    dimensions: int | None = None
    expectation: EmbeddingExpectation = "success"
    input_encoding: Literal["single_string", "string_array"] = "single_string"
    repeatability_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.logical_id or not self.route_id or not self.cell_id:
            raise ValueError("embedding request identity fields are required")
        if not isinstance(self.inputs, tuple) or not self.inputs or any(
            not isinstance(value, str) for value in self.inputs
        ):
            raise ValueError("embedding inputs must be a nonempty tuple of strings")
        if self.input_encoding == "single_string" and len(self.inputs) != 1:
            raise ValueError("single_string requests require exactly one input")
        if self.input_encoding not in {"single_string", "string_array"}:
            raise ValueError("unsupported embedding input encoding")
        if (
            isinstance(self.planned_input_tokens, bool)
            or not isinstance(self.planned_input_tokens, int)
            or self.planned_input_tokens < 0
        ):
            raise ValueError("planned_input_tokens must be nonnegative")
        if self.dimensions is not None:
            _positive_int(self.dimensions, "dimensions")
        if self.expectation not in {"success", "client_error"}:
            raise ValueError("invalid embedding expectation")
        for left, right in self.repeatability_pairs:
            if left == right or min(left, right) < 0 or max(left, right) >= len(self.inputs):
                raise ValueError("repeatability pair indices must name two distinct inputs")

    @property
    def input_sha256(self) -> tuple[str, ...]:
        return tuple(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in self.inputs)

    def public_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "route_id": self.route_id,
            "cell_id": self.cell_id,
            "input_count": len(self.inputs),
            "input_sha256": list(self.input_sha256),
            "planned_input_tokens": self.planned_input_tokens,
            "dimensions": self.dimensions,
            "expectation": self.expectation,
            "input_encoding": self.input_encoding,
            "repeatability_pairs": [list(pair) for pair in self.repeatability_pairs],
        }


EmbeddingStatus = Literal[
    "success",
    "client_error",
    "rate_limited",
    "server_error",
    "timeout",
    "transport_error",
    "protocol_error",
]


@dataclass(frozen=True, slots=True)
class EmbeddingVectorObservation:
    index: int
    dimensions: int
    l2_norm: float
    vector_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "dimensions": self.dimensions,
            "l2_norm": self.l2_norm,
            "vector_sha256": self.vector_sha256,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingRepeatabilityObservation:
    left_index: int
    right_index: int
    cosine_similarity: float
    max_absolute_difference: float
    passed: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "left_index": self.left_index,
            "right_index": self.right_index,
            "cosine_similarity": self.cosine_similarity,
            "max_absolute_difference": self.max_absolute_difference,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    logical_id: str
    status: EmbeddingStatus
    http_status: int | None
    total_seconds: float
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    vectors: tuple[EmbeddingVectorObservation, ...] = ()
    repeatability: tuple[EmbeddingRepeatabilityObservation, ...] = ()
    validation_errors: tuple[str, ...] = ()
    provider_request_id_sha256: str | None = None
    error_body_sha256: str | None = None

    @property
    def validation_passed(self) -> bool:
        return self.status == "success" and not self.validation_errors

    def public_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "status": self.status,
            "http_status": self.http_status,
            "total_seconds": self.total_seconds,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "vectors": [value.public_dict() for value in self.vectors],
            "repeatability": [value.public_dict() for value in self.repeatability],
            "validation_errors": list(self.validation_errors),
            "provider_request_id_sha256": self.provider_request_id_sha256,
            "error_body_sha256": self.error_body_sha256,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingCampaignConfig:
    name: str
    route: EmbeddingRouteConfig
    max_cost_usd: float
    concurrency: int = 8
    retries: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("embedding campaign name is required")
        _positive_number(self.max_cost_usd, "max_cost_usd")
        _positive_int(self.concurrency, "concurrency")
        if isinstance(self.retries, bool) or not isinstance(self.retries, int) or self.retries < 0:
            raise ValueError("retries must be a nonnegative integer")

    @property
    def identity_hash(self) -> str:
        return sha256_json(
            {
                "schema": EMBEDDING_PROFILE_SCHEMA,
                "name": self.name,
                "route_identity_sha256": self.route.identity_hash,
                "max_cost_usd": self.max_cost_usd,
                "concurrency": self.concurrency,
                "retries": self.retries,
            }
        )


def canonical_vector_sha256(values: list[float]) -> str:
    return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
