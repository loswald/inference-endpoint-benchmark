from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

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
        "parallel_tool_calls",
        "response_format",
        "logprobs",
        "stream_options",
        "reasoning",
        "reasoning_effort",
        "text",
        "verbosity",
    }
)
# The hard cost guard models text/image prompt tokens and billed completion tokens only. Route
# defaults therefore admit only fields known not to multiply outputs or activate separately billed
# provider services. Additions require an explicit cost/reservation contract and tests first.
SAFE_REQUEST_DEFAULT_KEYS = frozenset({"user"})

DEFAULT_RETAINED_HEADER_NAMES = (
    "x-request-id",
    "request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
)
SAFE_RETAINED_HEADER_NAMES = frozenset(
    {
        *DEFAULT_RETAINED_HEADER_NAMES,
        "ratelimit-limit",
        "ratelimit-remaining",
        "ratelimit-reset",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-amzn-requestid",
        "x-ms-request-id",
        "x-goog-request-id",
        "openai-request-id",
    }
)
TRANSPORT_HEADER_PROFILE = "openai-json-accept-encoding-identity/v1"
OUTPUT_LIMIT_FIELDS = frozenset({"max_tokens", "max_completion_tokens", "max_output_tokens"})
PUBLIC_FINISH_REASONS = frozenset(
    {
        "stop",
        "length",
        "tool_calls",
        "function_call",
        "content_filter",
        "end_turn",
        "max_tokens",
    }
)
PUBLIC_RESULT_STATUSES = frozenset(
    {
        "success",
        "client_error",
        "rate_limited",
        "server_error",
        "timeout",
        "transport_error",
        "adapter_unavailable",
        "unknown",
    }
)
PUBLIC_ARRIVAL_LATENCY_CENSOR_REASONS = frozenset(
    {"resumed_retry_arrival_latency_unavailable", "other_arrival_latency_censor_reason"}
)
_USAGE_COUNT_FIELDS = frozenset(
    {
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens_details.cached_tokens",
        "input_tokens_details.cached_tokens",
        "cache_read_input_tokens",
        "completion_tokens_details.reasoning_tokens",
        "output_tokens_details.reasoning_tokens",
        "reasoning_tokens",
    }
)
PUBLIC_USAGE_PARSE_ERRORS = frozenset(
    {
        "usage_wrong_json_type",
        "prompt_tokens_details_wrong_json_type",
        "input_tokens_details_wrong_json_type",
        "completion_tokens_details_wrong_json_type",
        "output_tokens_details_wrong_json_type",
        "input_tokens_alias_conflict",
        "output_tokens_alias_conflict",
        "total_tokens_alias_conflict",
        "cache_read_input_tokens_alias_conflict",
        "reasoning_tokens_alias_conflict",
        "total_tokens_mismatch_input_plus_output",
        "required_stream_usage_missing",
        "provider_input_tokens_zero_for_nonempty_request",
        "provider_output_tokens_zero_for_nonempty_response",
        "provider_input_tokens_exceed_reservation",
        "provider_output_tokens_exceed_request_limit",
        "other_usage_parse_error",
        *(f"{field}_wrong_json_type" for field in _USAGE_COUNT_FIELDS),
        *(f"{field}_nonintegral_or_negative" for field in _USAGE_COUNT_FIELDS),
        *(
            f"stream_{field}_decreased"
            for field in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_read_input_tokens",
                "reasoning_tokens",
            )
        ),
    }
)
HTTP_ADAPTER_NAMES = frozenset(
    {
        "openai_compatible",
        "alibaba_model_studio",
        "alibaba_model_studio_responses",
        "bedrock_mantle",
        "bedrock_mantle_responses",
        "azure_openai",
        "azure_model_inference",
        "azure_responses",
        "vertex_openai",
        "openrouter",
    }
)
HTTP_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
ROUTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REASONING_BUDGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROVIDER_DEFAULT_REASONING_BUDGET = "provider_default"
_REASONING_CONTROL_FIELDS_BY_API_FAMILY = {
    "chat_completions": frozenset({"reasoning_effort", "verbosity"}),
    "responses": frozenset({"reasoning.effort", "text.verbosity"}),
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalize_finish_reason(value: object) -> str | None:
    """Return a fixed public category without retaining provider-controlled text."""

    if value is None:
        return None
    if not isinstance(value, str):
        return "other"
    normalized = value.strip().casefold().replace("-", "_")
    return normalized if normalized in PUBLIC_FINISH_REASONS else "other"


def normalize_result_status(value: object) -> str:
    return value if isinstance(value, str) and value in PUBLIC_RESULT_STATUSES else "server_error"


def normalize_arrival_latency_censor_reason(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value in PUBLIC_ARRIVAL_LATENCY_CENSOR_REASONS:
        return value
    return "other_arrival_latency_censor_reason"


def normalize_usage_parse_errors(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ("other_usage_parse_error",)
    normalized = [
        value
        if isinstance(value, str) and value in PUBLIC_USAGE_PARSE_ERRORS
        else "other_usage_parse_error"
        for value in values
    ]
    return tuple(dict.fromkeys(normalized))


_PUBLIC_VALIDITY_REASONS = frozenset(
    {
        "total_seconds_nonpositive",
        "headers_after_first_token",
        "arrival_to_completion_precedes_component_duration",
        "stream_event_offset_invalid_seconds",
        "stream_event_offsets_nonmonotonic",
        "stream_event_after_request_end",
        "http_status_invalid",
        "adapter_total_seconds_disagrees_with_engine_clock",
        "reasoning_tokens_invalid_count",
        "reasoning_tokens_exceed_output_tokens",
        "cache_read_input_tokens_invalid_count",
        "cache_read_input_tokens_exceeds_input_tokens",
        "expected_probe_observed_validation_http_status",
        "expected_validation_rejection_not_enforced_observed_acceptance",
        "expected_probe_failed_for_nonvalidation_client_reason",
        "parameter_acceptance_probe_observed_client_error",
        "unexpected_client_error",
        "provider_usage_missing",
        "first_output_event_missing",
        "resumed_retry_arrival_latency_unavailable",
        "other_arrival_latency_censor_reason",
        "decode_proxy_missing_ttft",
        "decode_proxy_insufficient_content_events",
        "decode_proxy_near_zero_with_multiple_tokens",
        "decode_proxy_observation_window_below_one_second",
        "decode_proxy_reasoning_token_state_unknown",
        "decode_proxy_hidden_reasoning_tokens_present",
        "decode_proxy_extreme_tokens_per_second",
        "decode_proxy_requires_meaningful_output_tokens",
        "unknown_provider_outcome",
        "other_validity_reason",
        *(
            f"{name}_invalid_seconds"
            for name in (
                "total_seconds",
                "time_to_headers_seconds",
                "ttft_seconds",
                "decode_seconds",
                "queue_delay_seconds",
                "arrival_to_completion_seconds",
            )
        ),
        *(
            f"{name}_exceeds_total"
            for name in ("time_to_headers_seconds", "ttft_seconds", "decode_seconds")
        ),
        *(f"{name}_invalid_count" for name in ("input_tokens", "output_tokens")),
        *(f"request_{status}" for status in PUBLIC_RESULT_STATUSES if status != "success"),
        *(f"usage_parse_error:{reason}" for reason in PUBLIC_USAGE_PARSE_ERRORS),
    }
)


def normalize_validity_reasons(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ("other_validity_reason",)
    normalized = [
        value
        if isinstance(value, str) and value in _PUBLIC_VALIDITY_REASONS
        else "other_validity_reason"
        for value in values
    ]
    return tuple(dict.fromkeys(normalized))


def public_error_category(value: object) -> str | None:
    """Reduce adapter-controlled diagnostic text to a fixed event-projection category."""

    if value is None:
        return None
    normalized = str(value).strip().casefold()
    if normalized.startswith("protocol_error"):
        return "protocol_error"
    if re.fullmatch(r"http_[1-5][0-9]{2}", normalized):
        return "http_error"
    if normalized == "provider_rate_limit":
        return "provider_rate_limit"
    if normalized == "provider_billing_or_entitlement":
        return "provider_billing_or_entitlement"
    if normalized == "provider_route_fatal":
        return "provider_route_fatal"
    if "timeout" in normalized:
        return "timeout"
    if any(token in normalized for token in ("transport", "connect", "network")):
        return "transport_error"
    if normalized.startswith("post_claim_exception"):
        return "post_claim_exception"
    if normalized.startswith("process_interrupted"):
        return "process_interrupted"
    return "other_error"


def json_safe_number(value: int | float | None) -> int | float | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True, slots=True)
class AuthConfig:
    env: str
    header: str = "Authorization"
    prefix: str = "Bearer "

    def __post_init__(self) -> None:
        if not isinstance(self.env, str) or not self.env or not self.env.replace("_", "").isalnum():
            raise ValueError("auth.env must be an environment-variable name")
        if not isinstance(self.header, str) or not isinstance(self.prefix, str):
            raise ValueError("authentication header and prefix must be strings")
        if not HTTP_HEADER_NAME_RE.fullmatch(self.header) or any(
            character in self.prefix for character in "\r\n\0"
        ):
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
    billing_channel: str = "not_reported"
    api_version: str = "not_reported"
    model_version: str = "not_reported"
    upstream_provider: str | None = None
    quota_scope: str = "not_reported"
    context_tokens: int | None = None
    max_output_tokens: int | None = None
    output_limit_field: str = "max_tokens"
    output_limit_tolerance_tokens: int = 0
    stream_usage_mode: Literal["required", "try", "omit"] = "omit"
    request_timeout_seconds: float = 180.0
    http2: bool = False
    connection_reuse: bool = True
    transport_max_connections: int = 256
    input_token_reservation_overhead: int = 1_024
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    cached_input_usd_per_million: float | None = None
    documentation_source_url: str = "not_reported"
    pricing_source_url: str = "not_reported"
    evidence_retrieved_at_utc: str = "not_reported"
    evidence_bundle_sha256: str = "not_reported"
    capabilities: dict[str, bool | str] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    retained_header_names: tuple[str, ...] = DEFAULT_RETAINED_HEADER_NAMES
    request_defaults: dict[str, Any] = field(default_factory=dict)
    # Named budgets are an exact route-specific wire contract. An empty mapping means that this
    # route declares no controllable reasoning budget. provider_default always omits controls.
    reasoning_controls: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "provider", "adapter", "model", "base_url"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"route {name} is required")
        if not ROUTE_ID_RE.fullmatch(self.id):
            raise ValueError(
                "route id must start with an ASCII alphanumeric and contain only "
                "ASCII alphanumerics, '.', '_', or '-' (maximum 128 characters)"
            )
        for name in (
            "region",
            "api_family",
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
                raise ValueError(f"route {name} must be a nonempty string")
        evidence_placeholders = {"not_reported", "unknown"}
        for name in ("documentation_source_url", "pricing_source_url"):
            value = getattr(self, name).strip()
            if value.casefold() in evidence_placeholders or value.casefold().startswith(
                "replace-with-"
            ):
                continue
            try:
                parsed_evidence_url = urlsplit(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a public absolute HTTPS URL") from exc
            if (
                parsed_evidence_url.scheme != "https"
                or not parsed_evidence_url.netloc
                or not parsed_evidence_url.hostname
                or parsed_evidence_url.username is not None
                or parsed_evidence_url.password is not None
                or parsed_evidence_url.query
                or parsed_evidence_url.fragment
            ):
                raise ValueError(
                    f"{name} must be a public absolute HTTPS URL without credentials, query, "
                    "or fragment"
                )
        evidence_sha = self.evidence_bundle_sha256.strip()
        if (
            evidence_sha.casefold() not in evidence_placeholders
            and not evidence_sha.casefold().startswith("replace-with-")
            and not re.fullmatch(r"[0-9a-f]{64}", evidence_sha)
        ):
            raise ValueError("evidence_bundle_sha256 must be a lowercase SHA-256 digest")
        if self.adapter in HTTP_ADAPTER_NAMES:
            try:
                parsed_url = urlsplit(self.base_url)
                parsed_port = parsed_url.port
            except ValueError as exc:
                raise ValueError("base_url must be a valid absolute HTTPS URL") from exc
            if (
                parsed_url.scheme != "https"
                or not parsed_url.netloc
                or not parsed_url.hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
                or parsed_url.fragment
                or parsed_port is not None
                and not 0 < parsed_port < 65_536
            ):
                raise ValueError(
                    "base_url must be an absolute HTTPS URL without credentials or a fragment"
                )
            expected_api_family = (
                "responses"
                if self.adapter
                in {
                    "alibaba_model_studio_responses",
                    "bedrock_mantle_responses",
                    "azure_responses",
                }
                else "chat_completions"
            )
            if self.api_family != expected_api_family:
                message = (
                    f"adapter {self.adapter} currently supports only api_family=chat_completions"
                    if expected_api_family == "chat_completions"
                    else f"adapter {self.adapter} requires api_family=responses"
                )
                raise ValueError(message)
            if (
                expected_api_family == "responses"
                and self.output_limit_field != "max_output_tokens"
            ):
                raise ValueError("Responses adapters require output_limit_field=max_output_tokens")
        if self.adapter == "openrouter" and not self.upstream_provider:
            raise ValueError("OpenRouter routes must pin upstream_provider")
        if self.upstream_provider is not None and not isinstance(self.upstream_provider, str):
            raise ValueError("upstream_provider must be a string or null")
        if self.output_limit_field not in OUTPUT_LIMIT_FIELDS:
            raise ValueError(
                "output_limit_field must be max_tokens, max_completion_tokens, or max_output_tokens"
            )
        if (
            isinstance(self.output_limit_tolerance_tokens, bool)
            or not isinstance(self.output_limit_tolerance_tokens, int)
            or self.output_limit_tolerance_tokens < 0
        ):
            raise ValueError("output_limit_tolerance_tokens must be a nonnegative integer")
        if self.stream_usage_mode not in {"required", "try", "omit"}:
            raise ValueError("stream_usage_mode must be required, try, or omit")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be finite and positive")
        if not isinstance(self.http2, bool) or not isinstance(self.connection_reuse, bool):
            raise ValueError("http2 and connection_reuse must be booleans")
        if (
            isinstance(self.transport_max_connections, bool)
            or not isinstance(self.transport_max_connections, int)
            or self.transport_max_connections <= 0
        ):
            raise ValueError("transport_max_connections must be a positive integer")
        if (
            isinstance(self.input_token_reservation_overhead, bool)
            or not isinstance(self.input_token_reservation_overhead, int)
            or self.input_token_reservation_overhead < 0
        ):
            raise ValueError("input_token_reservation_overhead must be a nonnegative integer")
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
        if not isinstance(self.capabilities, dict) or any(
            not isinstance(key, str) or not isinstance(value, (bool, str))
            for key, value in self.capabilities.items()
        ):
            raise ValueError("capabilities must map strings to booleans or documented states")
        if not isinstance(self.extra_headers, dict):
            raise ValueError("extra_headers must be a mapping")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.extra_headers.items()
        ):
            raise ValueError("extra_headers must map strings to strings")
        lowered_extra_headers = [key.casefold() for key in self.extra_headers]
        if len(set(lowered_extra_headers)) != len(lowered_extra_headers):
            raise ValueError("extra_headers cannot contain case-insensitive duplicates")
        blocked = {"authorization", "api-key", "x-api-key"}
        if any(key.casefold() in blocked for key in self.extra_headers):
            raise ValueError("credentials belong in auth, not extra_headers")
        if any(
            key.casefold() in {self.auth.header.casefold(), "content-type", "accept-encoding"}
            for key in self.extra_headers
        ):
            raise ValueError(
                "extra_headers cannot override authentication, content type, or accept encoding"
            )
        if any(
            not HTTP_HEADER_NAME_RE.fullmatch(key)
            or any(character in value for character in "\r\n\0")
            for key, value in self.extra_headers.items()
        ):
            raise ValueError("extra_headers contains an invalid HTTP header name or value")
        if (
            not isinstance(self.retained_header_names, tuple)
            or not self.retained_header_names
            or any(
                not isinstance(name, str)
                or not name.strip()
                or not HTTP_HEADER_NAME_RE.fullmatch(name)
                for name in self.retained_header_names
            )
        ):
            raise ValueError("retained_header_names must be a nonempty tuple of HTTP header names")
        lowered_retained = [name.casefold() for name in self.retained_header_names]
        if len(set(lowered_retained)) != len(lowered_retained):
            raise ValueError("retained_header_names cannot contain case-insensitive duplicates")
        unsupported_retained = sorted(set(lowered_retained) - SAFE_RETAINED_HEADER_NAMES)
        if unsupported_retained:
            raise ValueError(
                "retained_header_names contains a header outside the fixed safe allowlist: "
                + ", ".join(unsupported_retained)
            )
        if "retry-after" not in lowered_retained:
            raise ValueError(
                "retained_header_names must include retry-after so provider-directed backoff "
                "cannot be disabled by reporting configuration"
            )
        if self.auth.header.casefold() in lowered_retained or any(
            name in {"authorization", "api-key", "x-api-key", "cookie", "set-cookie"}
            for name in lowered_retained
        ):
            raise ValueError("credential-bearing headers cannot be retained")
        if not isinstance(self.request_defaults, dict):
            raise ValueError("request_defaults must be a mapping")
        if any(not isinstance(key, str) for key in self.request_defaults):
            raise ValueError("request_defaults keys must be strings")
        protected = sorted(
            str(key)
            for key in self.request_defaults
            if str(key).lower() in PROTECTED_REQUEST_DEFAULT_KEYS
        )
        if protected:
            raise ValueError(
                "request_defaults cannot override protected request fields: " + ", ".join(protected)
            )
        unsupported_defaults = sorted(
            key for key in self.request_defaults if key not in SAFE_REQUEST_DEFAULT_KEYS
        )
        if unsupported_defaults:
            raise ValueError(
                "request_defaults contains fields outside the token-cost model allowlist: "
                + ", ".join(unsupported_defaults)
            )
        if not isinstance(self.reasoning_controls, dict):
            raise ValueError("reasoning_controls must be a mapping")
        allowed_control_fields = _REASONING_CONTROL_FIELDS_BY_API_FAMILY.get(self.api_family)
        if self.reasoning_controls and allowed_control_fields is None:
            raise ValueError(
                f"reasoning_controls are not implemented for api_family={self.api_family}"
            )
        for budget, controls in self.reasoning_controls.items():
            if not isinstance(budget, str) or not REASONING_BUDGET_RE.fullmatch(budget):
                raise ValueError(
                    "reasoning control budget names must contain only ASCII letters, digits, "
                    "'.', '_', or '-' (maximum 64 characters)"
                )
            if budget == PROVIDER_DEFAULT_REASONING_BUDGET:
                if controls != {}:
                    raise ValueError(
                        "provider_default reasoning budget must omit all wire controls"
                    )
                continue
            if not isinstance(controls, dict) or not controls:
                raise ValueError(
                    f"reasoning_controls.{budget} must be a nonempty mapping of wire fields"
                )
            unsupported_fields = sorted(set(controls) - set(allowed_control_fields or ()))
            if unsupported_fields:
                raise ValueError(
                    f"reasoning_controls.{budget} contains unsupported {self.api_family} "
                    "wire fields: " + ", ".join(unsupported_fields)
                )
            if any(
                not isinstance(field_name, str)
                or not isinstance(value, str)
                or not value.strip()
                for field_name, value in controls.items()
            ):
                raise ValueError(
                    f"reasoning_controls.{budget} must map wire fields to nonempty strings"
                )

    def reasoning_control(self, budget: str) -> dict[str, str]:
        """Resolve one exact named control without silently applying provider guesses."""

        if budget == PROVIDER_DEFAULT_REASONING_BUDGET:
            return {}
        try:
            return dict(self.reasoning_controls[budget])
        except KeyError as exc:
            raise ValueError(
                f"route {self.id} does not declare reasoning budget {budget!r}; "
                "use provider_default to omit reasoning controls"
            ) from exc

    @property
    def identity_hash(self) -> str:
        return sha256_json(
            {
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
                "upstream_provider": self.upstream_provider,
                "quota_scope": self.quota_scope,
                "context_tokens": self.context_tokens,
                "max_output_tokens": self.max_output_tokens,
                "output_limit_field": self.output_limit_field,
                "output_limit_tolerance_tokens": self.output_limit_tolerance_tokens,
                "stream_usage_mode": self.stream_usage_mode,
                "request_timeout_seconds": self.request_timeout_seconds,
                "http2": self.http2,
                "connection_reuse": self.connection_reuse,
                "transport_max_connections": self.transport_max_connections,
                "transport_header_profile": TRANSPORT_HEADER_PROFILE,
                "input_token_reservation_overhead": self.input_token_reservation_overhead,
                "input_usd_per_million": self.input_usd_per_million,
                "output_usd_per_million": self.output_usd_per_million,
                "cached_input_usd_per_million": self.cached_input_usd_per_million,
                "documentation_source_url": self.documentation_source_url,
                "pricing_source_url": self.pricing_source_url,
                "evidence_retrieved_at_utc": self.evidence_retrieved_at_utc,
                "evidence_bundle_sha256": self.evidence_bundle_sha256,
                "capabilities": self.capabilities,
                "extra_headers": self.extra_headers,
                "retained_header_names": self.retained_header_names,
                "request_defaults": self.request_defaults,
                "reasoning_controls": self.reasoning_controls,
            }
        )

    def worst_case_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            raise ValueError(f"route {self.id} has incomplete pricing")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (input_tokens, output_tokens)
        ):
            raise ValueError("token counts must be nonnegative integers")
        conservative_input_price = max(
            self.input_usd_per_million,
            self.cached_input_usd_per_million
            if self.cached_input_usd_per_million is not None
            else self.input_usd_per_million,
        )
        return (
            input_tokens * conservative_input_price + output_tokens * self.output_usd_per_million
        ) / 1_000_000

    def usage_cost_with_unknown_cache(self, input_tokens: int, output_tokens: int) -> float:
        """Conservative provider-usage cost when cached-token count was not reported."""

        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            raise ValueError(f"route {self.id} has incomplete pricing")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (input_tokens, output_tokens)
        ):
            raise ValueError("token counts must be nonnegative integers")
        input_price = max(
            self.input_usd_per_million,
            self.cached_input_usd_per_million
            if self.cached_input_usd_per_million is not None
            else self.input_usd_per_million,
        )
        return (
            input_tokens * input_price + output_tokens * self.output_usd_per_million
        ) / 1_000_000

    def actual_cost(
        self, input_tokens: int, output_tokens: int, cache_read_input_tokens: int = 0
    ) -> float:
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            raise ValueError(f"route {self.id} has incomplete pricing")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (input_tokens, output_tokens, cache_read_input_tokens)
        ):
            raise ValueError("token counts must be nonnegative integers")
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
    # Optional sampling controls are omitted from common baselines. Capability
    # and interaction suites add them explicitly, so a model that does not
    # implement temperature is not excluded from otherwise comparable tests.
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None
    reasoning_budget: str = PROVIDER_DEFAULT_REASONING_BUDGET
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
        if not isinstance(self.reasoning_budget, str) or not REASONING_BUDGET_RE.fullmatch(
            self.reasoning_budget
        ):
            raise ValueError(
                "reasoning_budget must contain only ASCII letters, digits, '.', '_', or '-' "
                "(maximum 64 characters)"
            )

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
            "parallel_tool_calls": self.parallel_tool_calls,
            "response_format": self.response_format,
            "logprobs": self.logprobs,
            # Explicitly bind intentional provider-default omission into request identity.
            "reasoning_budget": self.reasoning_budget,
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
    reasoning_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    usage_parse_errors: tuple[str, ...] = ()
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
    arrival_to_completion_seconds: float | None = None
    arrival_latency_censor_reason: str | None = None
    # Fixed, engine-authored reasons for custom-adapter values that were quarantined at the
    # trust boundary. Adapters cannot author these: the engine always replaces the tuple.
    adapter_contract_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.status = normalize_result_status(self.status)  # type: ignore[assignment]
        self.finish_reason = normalize_finish_reason(self.finish_reason)
        self.usage_parse_errors = normalize_usage_parse_errors(self.usage_parse_errors)
        self.arrival_latency_censor_reason = normalize_arrival_latency_censor_reason(
            self.arrival_latency_censor_reason
        )

    @property
    def usage_complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def content_event_count(self) -> int:
        return len(self.output_event_offsets_seconds)

    @property
    def output_sha256(self) -> str:
        # Tool-only responses are semantically nonempty. Bind reconstructed tool calls into the
        # response digest so two different tool argument payloads cannot share a text-only hash.
        return sha256_json({"output_text": self.output_text, "tool_calls": list(self.tool_calls)})

    def without_content(self) -> dict[str, Any]:
        provider_request_id_sha256 = (
            hashlib.sha256(self.provider_request_id.encode("utf-8")).hexdigest()
            if isinstance(self.provider_request_id, str) and self.provider_request_id
            else None
        )
        return {
            "logical_id": self.logical_id,
            "status": normalize_result_status(self.status),
            "http_status": self.http_status,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "total_seconds": json_safe_number(self.total_seconds),
            "time_to_headers_seconds": json_safe_number(self.time_to_headers_seconds),
            "ttft_seconds": json_safe_number(self.ttft_seconds),
            "decode_seconds": json_safe_number(self.decode_seconds),
            "output_event_offsets_seconds": [
                json_safe_number(value) for value in self.output_event_offsets_seconds
            ],
            "content_event_count": self.content_event_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "usage_parse_error_count": len(self.usage_parse_errors),
            "cache_state": self.cache_state,
            "finish_reason": normalize_finish_reason(self.finish_reason),
            "output_sha256": self.output_sha256,
            "provider_request_id_sha256": provider_request_id_sha256,
            "retained_header_count": len(self.retained_headers),
            "retained_headers_sha256": sha256_json(self.retained_headers),
            "error_kind": public_error_category(self.error_kind),
            "error_body_sha256": self.error_body_sha256,
            "cost_usd": json_safe_number(self.cost_usd),
            "cost_basis": self.cost_basis,
            "queue_delay_seconds": json_safe_number(self.queue_delay_seconds),
            "arrival_to_completion_seconds": json_safe_number(self.arrival_to_completion_seconds),
            "arrival_latency_censor_reason": normalize_arrival_latency_censor_reason(
                self.arrival_latency_censor_reason
            ),
            "adapter_contract_error_count": len(self.adapter_contract_errors),
        }


@dataclass(frozen=True, slots=True)
class ValidityAssessment:
    classification: Literal["valid", "anomalous", "invalid", "censored"]
    reasons: tuple[str, ...]
    latency_eligible: bool
    usage_eligible: bool
    decode_eligible: bool
    quality_eligible: bool
