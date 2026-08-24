from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

# These fields define the measured request itself or can multiply its billed output.  Allowing
# generic route defaults to set them would let configuration silently disagree with the immutable
# route/request identity and the pre-send cost reservation.
PROTECTED_REQUEST_DEFAULT_KEYS = frozenset(
    {
        "model",
        "messages",
        "prompt",
        "input",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "n",
        "best_of",
        "provider",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "logprobs",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthConfig:
    env: str
    header: str = "Authorization"
    prefix: str = "Bearer "

    def __post_init__(self) -> None:
        if not self.env or not self.env.replace("_", "").isalnum():
            raise ValueError("auth.env must be an environment-variable name")
        if any(character in self.header or character in self.prefix for character in "\r\n"):
            raise ValueError("invalid authentication header")


@dataclass(frozen=True, slots=True)
class RouteConfig:
    id: str
    provider: str
    adapter: str
    model: str
    base_url: str
    auth: AuthConfig
    region: str = "not_reported"
    api_family: str = "chat_completions"
    api_version: str = "not_reported"
    model_version: str = "not_reported"
    upstream_provider: str | None = None
    context_tokens: int | None = None
    max_output_tokens: int | None = None
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    cached_input_usd_per_million: float | None = None
    capabilities: dict[str, bool | str] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    request_defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "provider", "adapter", "model", "base_url"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"route {name} is required")
        if self.adapter == "openrouter" and not self.upstream_provider:
            raise ValueError("OpenRouter routes must pin upstream_provider")
        for name in ("context_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        for name in (
            "input_usd_per_million",
            "output_usd_per_million",
            "cached_input_usd_per_million",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be nonnegative")
        blocked = {"authorization", "api-key", "x-api-key"}
        if any(key.lower() in blocked for key in self.extra_headers):
            raise ValueError("credentials belong in auth, not extra_headers")
        if any(
            key.casefold() in {self.auth.header.casefold(), "content-type"}
            for key in self.extra_headers
        ):
            raise ValueError("extra_headers cannot override authentication or content type")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.extra_headers.items()
        ):
            raise ValueError("extra_headers must map strings to strings")
        if any(not isinstance(key, str) for key in self.request_defaults):
            raise ValueError("request_defaults keys must be strings")
        protected = sorted(
            str(key)
            for key in self.request_defaults
            if str(key).lower() in PROTECTED_REQUEST_DEFAULT_KEYS
        )
        if protected:
            raise ValueError(
                "request_defaults cannot override protected request fields: "
                + ", ".join(protected)
            )
        stream_options = self.request_defaults.get("stream_options")
        if stream_options is not None and (
            not isinstance(stream_options, dict)
            or stream_options.get("include_usage", True) is not True
        ):
            raise ValueError(
                "request_defaults.stream_options must be a mapping with include_usage=true"
            )

    @property
    def identity_hash(self) -> str:
        return sha256_json(
            {
                "provider": self.provider,
                "adapter": self.adapter,
                "model": self.model,
                "base_url": self.base_url,
                "auth_transport": {
                    "header": self.auth.header,
                    "prefix": self.auth.prefix,
                },
                "region": self.region,
                "api_family": self.api_family,
                "api_version": self.api_version,
                "model_version": self.model_version,
                "upstream_provider": self.upstream_provider,
                "context_tokens": self.context_tokens,
                "max_output_tokens": self.max_output_tokens,
                "input_usd_per_million": self.input_usd_per_million,
                "output_usd_per_million": self.output_usd_per_million,
                "cached_input_usd_per_million": self.cached_input_usd_per_million,
                "capabilities": self.capabilities,
                "extra_headers": self.extra_headers,
                "request_defaults": self.request_defaults,
            }
        )

    def worst_case_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            raise ValueError(f"route {self.id} has incomplete pricing")
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        return (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000

    def actual_cost(
        self, input_tokens: int, output_tokens: int, cache_read_input_tokens: int = 0
    ) -> float:
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            raise ValueError(f"route {self.id} has incomplete pricing")
        if input_tokens < 0 or output_tokens < 0 or cache_read_input_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        cached = min(max(0, cache_read_input_tokens), input_tokens)
        uncached = input_tokens - cached
        cached_price = (
            self.input_usd_per_million
            if self.cached_input_usd_per_million is None
            else self.cached_input_usd_per_million
        )
        return (
            uncached * self.input_usd_per_million
            + cached * cached_price
            + output_tokens * self.output_usd_per_million
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class RequestSpec:
    logical_id: str
    route_id: str
    suite: str
    cell_id: str
    messages: tuple[dict[str, Any], ...]
    planned_input_tokens: int
    max_output_tokens: int
    stream: bool = True
    timeout_seconds: float = 180.0
    temperature: float | None = 0.0
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.logical_id or not self.route_id or not self.suite or not self.cell_id:
            raise ValueError("request identity fields are required")
        if (
            isinstance(self.planned_input_tokens, bool)
            or not isinstance(self.planned_input_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.planned_input_tokens < 0
            or self.max_output_tokens <= 0
        ):
            raise ValueError("token counts must be nonnegative/positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def payload_material(self) -> dict[str, Any]:
        return {
            "messages": list(self.messages),
            "max_output_tokens": self.max_output_tokens,
            "stream": self.stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": list(self.stop),
            "tools": list(self.tools),
            "tool_choice": self.tool_choice,
            "response_format": self.response_format,
            "logprobs": self.logprobs,
            "synthetic_payload_descriptor": {
                key: value
                for key, value in self.metadata.items()
                if key in {"prompt_kind", "target_tokens", "nonce", "cache_state", "prompt_seed"}
            },
        }

    @property
    def payload_hash(self) -> str:
        return sha256_json(self.payload_material())


ResultStatus = Literal[
    "success",
    "client_error",
    "rate_limited",
    "server_error",
    "timeout",
    "transport_error",
    "adapter_unavailable",
    "unknown",
]


@dataclass(slots=True)
class InferenceResult:
    logical_id: str
    status: ResultStatus
    http_status: int | None
    started_at_utc: str
    ended_at_utc: str
    total_seconds: float
    time_to_headers_seconds: float | None = None
    ttft_seconds: float | None = None
    decode_seconds: float | None = None
    output_event_offsets_seconds: tuple[float, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_state: Literal["cached_trial", "uncached_trial", "uncontrolled"] = "uncontrolled"
    finish_reason: str | None = None
    output_text: str = field(default="", repr=False)
    tool_calls: tuple[dict[str, Any], ...] = field(default=(), repr=False)
    provider_request_id: str | None = None
    retained_headers: dict[str, str] = field(default_factory=dict)
    error_kind: str | None = None
    error_body_sha256: str | None = None
    cost_usd: float | None = None
    cost_basis: Literal[
        "provider_usage",
        "provider_usage_cache_unknown_upper_bound",
        "reserved_upper_bound",
        "unpriced",
    ] = "unpriced"
    queue_delay_seconds: float = 0.0

    @property
    def usage_complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def content_event_count(self) -> int:
        return len(self.output_event_offsets_seconds)

    @property
    def output_sha256(self) -> str:
        return hashlib.sha256(self.output_text.encode()).hexdigest()

    def without_content(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "status": self.status,
            "http_status": self.http_status,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "total_seconds": self.total_seconds,
            "time_to_headers_seconds": self.time_to_headers_seconds,
            "ttft_seconds": self.ttft_seconds,
            "decode_seconds": self.decode_seconds,
            "output_event_offsets_seconds": list(self.output_event_offsets_seconds),
            "content_event_count": self.content_event_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_state": self.cache_state,
            "finish_reason": self.finish_reason,
            "output_sha256": self.output_sha256,
            "provider_request_id": self.provider_request_id,
            "retained_headers": self.retained_headers,
            "error_kind": self.error_kind,
            "error_body_sha256": self.error_body_sha256,
            "cost_usd": self.cost_usd,
            "cost_basis": self.cost_basis,
            "queue_delay_seconds": self.queue_delay_seconds,
        }


@dataclass(frozen=True, slots=True)
class ValidityAssessment:
    classification: Literal["valid", "anomalous", "invalid", "censored"]
    reasons: tuple[str, ...]
    latency_eligible: bool
    usage_eligible: bool
    decode_eligible: bool
    quality_eligible: bool
