from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from .models import TRANSPORT_HEADER_PROFILE, AuthConfig, RouteConfig, sha256_json

_PUBLIC_CAPABILITY_KEYS = {
    "batch_inference",
    "caching",
    "documentation_checked_utc",
    "json_schema",
    "logprobs",
    "parallel_tool_calls",
    "seed",
    "stop",
    "streaming",
    "structured_output",
    "tool_calling",
    "tools",
    "vision",
}
_PUBLIC_SUITE_KEYS = {
    "static": {"enabled", "offered_rps", "route_ids"},
    "warmup": {
        "enabled",
        "repeats",
        "shapes",
        "long_input_tokens",
        "long_input_tokens_by_route",
        "long_input_overflow",
        "long_output_tokens",
        "long_output_tokens_by_route",
        "long_output_overflow",
        "route_ids",
    },
    "latency": {
        "enabled",
        "repeats",
        "shapes",
        "long_input_tokens",
        "long_input_tokens_by_route",
        "long_input_overflow",
        "long_output_tokens",
        "long_output_tokens_by_route",
        "long_output_overflow",
    },
    "capability": {
        "enabled",
        "temperatures",
        "top_ps",
        "tool_counts",
        "route_ids",
        "probe_groups",
        "probe_groups_by_route",
    },
    "interactions": {
        "enabled",
        "temperatures",
        "top_ps",
        "stream",
        "output_tokens",
    },
    "context": {"enabled", "percentages", "fixed_tokens", "route_ids"},
    "output": {
        "enabled",
        "fallback_max_output_tokens",
        "realized_generation_ceiling",
        "route_ids",
    },
    "quality": {"enabled", "repeats", "route_ids"},
    "cache": {"enabled", "repeats", "prefix_tokens", "route_ids"},
    "time_variation": {
        "enabled",
        "panels",
        "interval_minutes",
        "samples_per_route_shape",
        "stable_exact_prompt_repeats",
        "panel_unique_cache_cold_repeats",
        "shapes",
        "offered_rps",
        "concurrency",
        "long_input_tokens",
        "long_input_tokens_by_route",
        "long_input_overflow",
        "long_output_tokens",
        "long_output_tokens_by_route",
        "long_output_overflow",
        "route_ids",
        "interleave_gap_work",
        "panel_guard_seconds",
        "panel_deadline_seconds",
        "send_cutoff_seconds",
    },
    "aimd": {
        "enabled",
        "shapes",
        "initial_rps",
        "additive_rps",
        "multiplicative_decrease",
        "bracket_epochs",
        "bracket_multiplier",
        "max_rps",
        "max_rps_by_shape",
        "epochs",
        "epoch_seconds",
        "concurrency",
        "baseline_rps",
        "baseline_samples",
        "baseline_attempts",
        "baseline_multiplicative_decrease",
        "confirmation_max_stages",
        "confirmation_multiplicative_decrease",
        "confirmation_separator_samples",
        "minimum_rps",
        "long_input_tokens",
        "long_input_tokens_by_route",
        "long_input_overflow",
        "long_output_tokens",
        "long_output_tokens_by_route",
        "long_output_overflow",
        "route_ids",
        "cells",
    },
    "soak": {
        "enabled",
        "shapes",
        "rate_rps",
        "rate_rps_by_route",
        "rate_rps_by_route_shape",
        "blocks",
        "block_seconds",
        "concurrency",
        "baseline_rps",
        "baseline_samples",
        "baseline_attempts",
        "baseline_multiplicative_decrease",
        "max_rate_stages",
        "rate_multiplicative_decrease",
        "minimum_rps",
        "long_input_tokens",
        "long_input_tokens_by_route",
        "long_input_overflow",
        "long_output_tokens",
        "long_output_tokens_by_route",
        "long_output_overflow",
        "route_ids",
        "cells",
    },
}

_TOP_LEVEL_KEYS = {"campaign", "route_defaults", "routes", "suites"}
_CAMPAIGN_KEYS = {
    "name",
    "seed",
    "max_wall_seconds",
    "max_cost_usd",
    "launch_reserve_seconds",
    "launch_reserve_usd",
    "concurrency",
    "retries",
    "input_token_reservation_factor",
    "client_location",
}
_ROUTE_KEYS = {
    "id",
    "provider",
    "adapter",
    "model",
    "base_url",
    "auth",
    "region",
    "api_family",
    "api_version",
    "model_version",
    "upstream_provider",
    "quota_scope",
    "context_tokens",
    "max_output_tokens",
    "output_limit_field",
    "stream_usage_mode",
    "request_timeout_seconds",
    "http2",
    "connection_reuse",
    "transport_max_connections",
    "input_token_reservation_overhead",
    "input_usd_per_million",
    "output_usd_per_million",
    "cached_input_usd_per_million",
    "documentation_source_url",
    "pricing_source_url",
    "evidence_retrieved_at_utc",
    "evidence_bundle_sha256",
    "capabilities",
    "extra_headers",
    "retained_header_names",
    "request_defaults",
}
_AUTH_KEYS = {"env", "header", "prefix"}
_SHAPES = {"short_short", "long_short", "short_long", "mixed"}
_CAPABILITY_PROBE_GROUPS = {
    "transport_baseline",
    "structured_output",
    "tool_calling",
    "vision",
    "parameter_validation",
}


def suite_applies_to_route(values: dict[str, Any], route_id: str) -> bool:
    """Return whether a suite explicitly includes an endpoint.

    ``route_ids`` is a plan-level selector. It never changes route identity and therefore keeps
    one immutable campaign manifest while avoiding duplicate requests for already-settled cells.
    """

    selected = values.get("route_ids")
    return selected is None or route_id in selected


def selected_capacity_cells(
    config: CampaignConfig, suite_name: str
) -> list[tuple[RouteConfig, str]]:
    """Resolve the exact endpoint/workload cells for one capacity suite."""

    if suite_name not in {"aimd", "soak"}:
        raise ValueError("capacity selection supports only aimd or soak")
    suite = config.suites.get(suite_name)
    if not suite or not suite.get("enabled", True):
        return []
    shapes = tuple(suite.get("shapes", ("short_short", "long_short", "short_long", "mixed")))
    declared = suite.get("cells")
    selected = (
        {tuple(str(value).rsplit(":", 1)) for value in declared}
        if declared is not None
        else None
    )
    return [
        (route, shape)
        for route in config.routes
        if suite_applies_to_route(suite, route.id)
        for shape in shapes
        if selected is None or (route.id, shape) in selected
    ]
NATIVE_PLACEHOLDER_ADAPTERS = frozenset(
    {
        "bedrock_native",
        "vertex_native",
        "azure_model_inference_native",
    }
)


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every nesting level."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _safe_public_url(value: str) -> str:
    """Remove credentials, queries, and fragments from an endpoint descriptor."""
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname or ""
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    port = f":{parsed_port}" if parsed_port is not None else ""
    return urlunsplit((parsed.scheme, hostname + port, parsed.path, "", ""))


def _safe_capabilities(values: dict[str, bool | str]) -> dict[str, bool | str]:
    public: dict[str, bool | str] = {}
    for key in sorted(_PUBLIC_CAPABILITY_KEYS & values.keys()):
        value = values[key]
        if isinstance(value, bool):
            public[key] = value
        elif key == "documentation_checked_utc":
            public[key] = str(value)
        elif str(value).lower() in {"supported", "unsupported", "unknown", "partial"}:
            public[key] = str(value).lower()
    return public


def _safe_suites(values: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    public: dict[str, dict[str, Any]] = {}
    for suite, allowed in _PUBLIC_SUITE_KEYS.items():
        config = values.get(suite)
        if not isinstance(config, dict):
            continue
        public[suite] = {
            key: copy.deepcopy(config[key])
            for key in sorted(allowed & config.keys())
            if isinstance(config[key], (bool, int, float, str, list, tuple, dict))
        }
    return public


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    name: str
    seed: int
    max_wall_seconds: float
    max_cost_usd: float
    launch_reserve_seconds: float
    launch_reserve_usd: float
    concurrency: int
    retries: int
    routes: tuple[RouteConfig, ...]
    client_location: str = "not_reported"
    input_token_reservation_factor: float = 1.5
    suites: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("campaign.name is required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("campaign.seed must be an integer")
        numeric = (
            self.max_wall_seconds,
            self.max_cost_usd,
            self.launch_reserve_seconds,
            self.launch_reserve_usd,
            self.input_token_reservation_factor,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("campaign time, cost, reserve, and factor values must be finite")
        if self.max_wall_seconds <= 0 or self.max_cost_usd <= 0:
            raise ValueError("positive time and cost caps are mandatory")
        if self.launch_reserve_seconds < 0 or self.launch_reserve_usd < 0:
            raise ValueError("reserves must be nonnegative")
        if self.launch_reserve_seconds >= self.max_wall_seconds:
            raise ValueError("time reserve must be smaller than wall cap")
        if self.launch_reserve_usd >= self.max_cost_usd:
            raise ValueError("cost reserve must be smaller than cost cap")
        if (
            isinstance(self.concurrency, bool)
            or not isinstance(self.concurrency, int)
            or isinstance(self.retries, bool)
            or not isinstance(self.retries, int)
            or self.concurrency <= 0
            or self.retries < 0
        ):
            raise ValueError("concurrency/retries are invalid")
        if self.input_token_reservation_factor < 1:
            raise ValueError("input_token_reservation_factor must be at least 1")
        if not isinstance(self.client_location, str) or not self.client_location.strip():
            raise ValueError("client_location must be a nonempty region/location label")
        if not self.routes or any(not isinstance(route, RouteConfig) for route in self.routes):
            raise ValueError("at least one valid route is required")
        if len({route.id for route in self.routes}) != len(self.routes):
            raise ValueError("route IDs must be unique")
        if any(
            not isinstance(name, str) or not isinstance(value, dict)
            for name, value in self.suites.items()
        ):
            raise ValueError("suites must map names to configuration mappings")
        _validate_suites(
            self.suites,
            self.concurrency,
            {route.id for route in self.routes},
            retries=self.retries,
        )

    @property
    def identity_hash(self) -> str:
        # Public serialization intentionally omits operational fields that can contain secrets or
        # private account identifiers. The identity still binds their effects through each route
        # identity hash and binds the complete suite configuration without publishing it.
        return sha256_json(
            {
                "campaign": {
                    "name": self.name,
                    "seed": self.seed,
                    "max_wall_seconds": self.max_wall_seconds,
                    "max_cost_usd": self.max_cost_usd,
                    "launch_reserve_seconds": self.launch_reserve_seconds,
                    "launch_reserve_usd": self.launch_reserve_usd,
                    "concurrency": self.concurrency,
                    "retries": self.retries,
                    "input_token_reservation_factor": self.input_token_reservation_factor,
                    "client_location": self.client_location,
                },
                "routes": [
                    {"id": route.id, "identity_hash": route.identity_hash} for route in self.routes
                ],
                "suites": self.suites,
            }
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "campaign": {
                "name": self.name,
                "seed": self.seed,
                "max_wall_seconds": self.max_wall_seconds,
                "max_cost_usd": self.max_cost_usd,
                "launch_reserve_seconds": self.launch_reserve_seconds,
                "launch_reserve_usd": self.launch_reserve_usd,
                "concurrency": self.concurrency,
                "retries": self.retries,
                "input_token_reservation_factor": self.input_token_reservation_factor,
                "client_location": self.client_location,
            },
            "routes": [
                {
                    "id": route.id,
                    "provider": route.provider,
                    "adapter": route.adapter,
                    "model": route.model,
                    "base_url": _safe_public_url(route.base_url),
                    "auth": {
                        "env": route.auth.env,
                    },
                    "region": route.region,
                    "api_family": route.api_family,
                    "api_version": route.api_version,
                    "model_version": route.model_version,
                    "upstream_provider": route.upstream_provider,
                    "quota_scope_hash": sha256_json(route.quota_scope),
                    "context_tokens": route.context_tokens,
                    "max_output_tokens": route.max_output_tokens,
                    "output_limit_field": route.output_limit_field,
                    "stream_usage_mode": route.stream_usage_mode,
                    "request_timeout_seconds": route.request_timeout_seconds,
                    "http2": route.http2,
                    "connection_reuse": route.connection_reuse,
                    "transport_max_connections": route.transport_max_connections,
                    "transport_header_profile": TRANSPORT_HEADER_PROFILE,
                    "input_token_reservation_overhead": route.input_token_reservation_overhead,
                    "input_usd_per_million": route.input_usd_per_million,
                    "output_usd_per_million": route.output_usd_per_million,
                    "cached_input_usd_per_million": route.cached_input_usd_per_million,
                    "documentation_source_url": route.documentation_source_url,
                    "pricing_source_url": route.pricing_source_url,
                    "evidence_retrieved_at_utc": route.evidence_retrieved_at_utc,
                    "evidence_bundle_sha256": route.evidence_bundle_sha256,
                    "capabilities": _safe_capabilities(route.capabilities),
                    "retained_header_names": list(route.retained_header_names),
                    "omitted_operational_fields": sorted(
                        name
                        for name, present in {
                            "auth_transport": bool(route.auth.header or route.auth.prefix),
                            "extra_headers": bool(route.extra_headers),
                            "request_defaults": bool(route.request_defaults),
                            "base_url_query_or_credentials": (
                                _safe_public_url(route.base_url) != route.base_url
                            ),
                        }.items()
                        if present
                    ),
                    "identity_hash": route.identity_hash,
                }
                for route in self.routes
            ],
            "suites": _safe_suites(self.suites),
            "public_serialization": {
                "schema_version": "campaign-public/v1",
                "policy": "explicit-allowlist",
                "note": (
                    "Operational auth transport, arbitrary headers, request defaults, URL query "
                    "parameters, and unknown extension fields are intentionally omitted."
                ),
                "hash_commitment_warning": (
                    "Route identity hashes still commit to omitted operational fields. Never put "
                    "credentials or low-entropy private values in those fields."
                ),
            },
        }


def validate_route_evidence_identity(config: CampaignConfig, route: RouteConfig) -> None:
    """Validate non-credential identity fields used by plans and live evidence."""

    placeholders = {"", "unknown", "not_reported"}
    client_location = config.client_location.strip()
    if client_location.casefold() in placeholders or client_location.casefold().startswith(
        "replace-with-"
    ):
        raise ValueError("benchmark plans require an exact campaign.client_location")
    quota_scope = route.quota_scope.strip()
    if quota_scope.casefold() in placeholders or quota_scope.casefold().startswith("replace-with-"):
        raise ValueError(f"benchmark route {route.id} requires an exact opaque quota_scope")
    for field_name in ("model", "region", "api_version", "model_version"):
        value = str(getattr(route, field_name)).strip()
        if value.casefold() in placeholders or value.casefold().startswith("replace-with-"):
            raise ValueError(
                f"benchmark route {route.id} requires an exact {field_name}; use a deliberate "
                "provider_does_not_expose_* sentinel when applicable"
            )
    documentation_checked = route.capabilities.get("documentation_checked_utc")
    if documentation_checked is None:
        raise ValueError(
            f"benchmark route {route.id} requires capabilities.documentation_checked_utc"
        )
    documentation_value = str(documentation_checked).strip()
    if documentation_value.casefold().startswith("replace-with-"):
        raise ValueError(f"benchmark route {route.id} requires an exact documentation_checked_utc")
    try:
        checked_at = datetime.fromisoformat(documentation_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"benchmark route {route.id} has invalid documentation_checked_utc"
        ) from exc
    if checked_at.tzinfo is None or checked_at.utcoffset() != timedelta(0):
        raise ValueError(
            f"benchmark route {route.id} documentation_checked_utc must include a UTC offset"
        )
    for field_name in ("documentation_source_url", "pricing_source_url"):
        value = str(getattr(route, field_name)).strip()
        if value.casefold() in placeholders or value.casefold().startswith("replace-with-"):
            raise ValueError(f"benchmark route {route.id} requires an exact {field_name}")
    evidence_sha = route.evidence_bundle_sha256.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", evidence_sha):
        raise ValueError(
            f"benchmark route {route.id} requires an exact lowercase evidence_bundle_sha256"
        )
    retrieved_value = route.evidence_retrieved_at_utc.strip()
    if retrieved_value.casefold() in placeholders or retrieved_value.casefold().startswith(
        "replace-with-"
    ):
        raise ValueError(f"benchmark route {route.id} requires an exact evidence_retrieved_at_utc")
    try:
        retrieved_at = datetime.fromisoformat(retrieved_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"benchmark route {route.id} has invalid evidence_retrieved_at_utc"
        ) from exc
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() != timedelta(0):
        raise ValueError(
            f"benchmark route {route.id} evidence_retrieved_at_utc must include a UTC offset"
        )
    required_connections = config.concurrency
    for suite_name in ("aimd", "soak"):
        suite = config.suites.get(suite_name) or {}
        if suite.get("enabled", True):
            required_connections = max(
                required_connections, int(suite.get("concurrency", config.concurrency))
            )
    if route.transport_max_connections < required_connections:
        raise ValueError(
            f"route {route.id} transport_max_connections is below measured concurrency "
            f"({route.transport_max_connections} < {required_connections})"
        )


def _route(raw: dict[str, Any]) -> RouteConfig:
    _reject_unknown("route", raw, _ROUTE_KEYS)
    auth = raw.get("auth") or {}
    if not isinstance(auth, dict):
        raise ValueError("route.auth must be a mapping")
    _reject_unknown("route.auth", auth, _AUTH_KEYS)
    capabilities = raw.get("capabilities") or {}
    extra_headers = raw.get("extra_headers") or {}
    request_defaults = raw.get("request_defaults") or {}
    retained_header_names = raw.get("retained_header_names")
    for field_name, value in (
        ("route.capabilities", capabilities),
        ("route.extra_headers", extra_headers),
        ("route.request_defaults", request_defaults),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a mapping")
    if retained_header_names is not None and (
        not isinstance(retained_header_names, list) or not retained_header_names
    ):
        raise ValueError("route.retained_header_names must be a nonempty list")
    return RouteConfig(
        id=_string(raw.get("id"), "route.id"),
        provider=_string(raw.get("provider"), "route.provider"),
        adapter=_string(raw.get("adapter", "openai_compatible"), "route.adapter"),
        model=_string(raw.get("model"), "route.model"),
        base_url=_string(raw.get("base_url"), "route.base_url"),
        auth=AuthConfig(
            env=_string(auth.get("env"), "route.auth.env"),
            header=_string(auth.get("header", "Authorization"), "route.auth.header"),
            prefix=_string(auth.get("prefix", "Bearer "), "route.auth.prefix"),
        ),
        region=_string(raw.get("region", "not_reported"), "route.region"),
        api_family=_string(raw.get("api_family", "chat_completions"), "route.api_family"),
        api_version=_string(raw.get("api_version", "not_reported"), "route.api_version"),
        model_version=_string(raw.get("model_version", "not_reported"), "route.model_version"),
        upstream_provider=raw.get("upstream_provider"),
        quota_scope=_string(raw.get("quota_scope", "not_reported"), "route.quota_scope"),
        context_tokens=raw.get("context_tokens"),
        max_output_tokens=raw.get("max_output_tokens"),
        output_limit_field=_string(
            raw.get("output_limit_field", "max_tokens"), "route.output_limit_field"
        ),
        stream_usage_mode=_string(raw.get("stream_usage_mode", "omit"), "route.stream_usage_mode"),  # type: ignore[arg-type]
        request_timeout_seconds=_number(
            raw.get("request_timeout_seconds", 180), "route.request_timeout_seconds"
        ),
        http2=_strict_bool(raw.get("http2", False), "route.http2"),
        connection_reuse=_strict_bool(raw.get("connection_reuse", True), "route.connection_reuse"),
        transport_max_connections=_positive_integer(
            raw.get("transport_max_connections", 256), "route.transport_max_connections"
        ),
        input_token_reservation_overhead=raw.get("input_token_reservation_overhead", 1_024),
        input_usd_per_million=raw.get("input_usd_per_million"),
        output_usd_per_million=raw.get("output_usd_per_million"),
        cached_input_usd_per_million=raw.get("cached_input_usd_per_million"),
        documentation_source_url=_string(
            raw.get("documentation_source_url", "not_reported"),
            "route.documentation_source_url",
        ),
        pricing_source_url=_string(
            raw.get("pricing_source_url", "not_reported"), "route.pricing_source_url"
        ),
        evidence_retrieved_at_utc=_string(
            raw.get("evidence_retrieved_at_utc", "not_reported"),
            "route.evidence_retrieved_at_utc",
        ),
        evidence_bundle_sha256=_string(
            raw.get("evidence_bundle_sha256", "not_reported"),
            "route.evidence_bundle_sha256",
        ),
        capabilities=dict(capabilities),
        extra_headers=dict(extra_headers),
        retained_header_names=tuple(
            retained_header_names
            or (
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
        ),
        request_defaults=dict(request_defaults),
    )


def load_config(path: str | Path) -> CampaignConfig:
    try:
        raw = yaml.load(
            Path(path).read_text(encoding="utf-8"),
            Loader=_StrictSafeLoader,
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid campaign YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("campaign file must contain a mapping")
    _reject_unknown("top level", raw, _TOP_LEVEL_KEYS)
    campaign = raw.get("campaign") or {}
    if not isinstance(campaign, dict):
        raise ValueError("campaign must be a mapping")
    _reject_unknown("campaign", campaign, _CAMPAIGN_KEYS)
    route_defaults = raw.get("route_defaults") or {}
    if not isinstance(route_defaults, dict):
        raise ValueError("route_defaults must be a mapping")
    _reject_unknown("route_defaults", route_defaults, _ROUTE_KEYS)
    routes = raw.get("routes") or []
    if not isinstance(routes, list) or not routes:
        raise ValueError("at least one route is required")
    merged_routes: list[dict[str, Any]] = []
    default_capabilities = route_defaults.get("capabilities") or {}
    for index, item in enumerate(routes):
        if not isinstance(item, dict):
            raise ValueError(f"routes[{index}] must be a mapping")
        merged = {**route_defaults, **item}
        item_capabilities = item.get("capabilities") or {}
        if default_capabilities or item_capabilities:
            if not isinstance(default_capabilities, dict) or not isinstance(
                item_capabilities, dict
            ):
                raise ValueError("route capabilities must be mappings")
            merged["capabilities"] = {**default_capabilities, **item_capabilities}
        merged_routes.append(merged)
    suites = raw.get("suites") or {}
    if not isinstance(suites, dict):
        raise ValueError("suites must be a mapping")
    return CampaignConfig(
        name=_string(campaign.get("name", ""), "campaign.name"),
        seed=_integer(campaign.get("seed", 1), "campaign.seed"),
        max_wall_seconds=_number(campaign.get("max_wall_seconds", 0), "campaign.max_wall_seconds"),
        max_cost_usd=_number(campaign.get("max_cost_usd", 0), "campaign.max_cost_usd"),
        launch_reserve_seconds=_number(
            campaign.get("launch_reserve_seconds", 180), "campaign.launch_reserve_seconds"
        ),
        launch_reserve_usd=_number(
            campaign.get("launch_reserve_usd", 5), "campaign.launch_reserve_usd"
        ),
        concurrency=_integer(campaign.get("concurrency", 16), "campaign.concurrency"),
        retries=_integer(campaign.get("retries", 2), "campaign.retries"),
        client_location=_string(
            campaign.get("client_location", "not_reported"), "campaign.client_location"
        ),
        input_token_reservation_factor=_number(
            campaign.get("input_token_reservation_factor", 1.5),
            "campaign.input_token_reservation_factor",
        ),
        routes=tuple(_route(item) for item in merged_routes),
        suites=dict(suites),
    )


def _reject_unknown(scope: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {scope} field(s): {', '.join(unknown)}")


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _positive_number(value: Any, field_name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        comparator = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be finite and {comparator}")
    return number


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_unique(values: list[Any], field_name: str) -> None:
    """Reject duplicate design levels before they can collapse deterministic cell IDs."""

    fingerprints = [(type(value).__name__, repr(value)) for value in values]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(f"{field_name} must not contain duplicates")


def _validate_suites(
    suites: dict[str, dict[str, Any]],
    default_concurrency: int,
    route_ids: set[str],
    *,
    retries: int,
) -> None:
    unknown_suites = sorted(set(suites) - set(_PUBLIC_SUITE_KEYS))
    if unknown_suites:
        raise ValueError(f"unknown suite(s): {', '.join(unknown_suites)}")
    if not suites or not any(values.get("enabled", True) for values in suites.values()):
        raise ValueError("at least one benchmark suite must be enabled")
    time_variation = suites.get("time_variation")
    if time_variation and time_variation.get("enabled", True):
        other_enabled = [
            name
            for name, values in suites.items()
            if name != "time_variation" and values.get("enabled", True)
        ]
        if other_enabled and not time_variation.get("interleave_gap_work", False):
            raise ValueError(
                "time_variation must run as a dedicated low-load campaign; overlapping it with "
                "capacity or capability suites requires explicit interleave_gap_work scheduling"
            )
    for name, values in suites.items():
        _reject_unknown(f"suites.{name}", values, _PUBLIC_SUITE_KEYS[name])
        if "enabled" in values:
            _strict_bool(values["enabled"], f"suites.{name}.enabled")
        if not values.get("enabled", True):
            continue
        if "route_ids" in values:
            selected_routes = values["route_ids"]
            if (
                not isinstance(selected_routes, list)
                or not selected_routes
                or any(not isinstance(route_id, str) for route_id in selected_routes)
            ):
                raise ValueError(f"suites.{name}.route_ids must be a nonempty list")
            _require_unique(selected_routes, f"suites.{name}.route_ids")
            unknown_routes = sorted(set(selected_routes) - route_ids)
            if unknown_routes:
                raise ValueError(
                    f"suites.{name}.route_ids has unknown routes: {unknown_routes}"
                )
        if "repeats" in values:
            _positive_integer(values["repeats"], f"suites.{name}.repeats")
        if "shapes" in values:
            shapes = values["shapes"]
            if (
                not isinstance(shapes, list)
                or not shapes
                or any(not isinstance(shape, str) for shape in shapes)
            ):
                raise ValueError(f"suites.{name}.shapes must be a nonempty list")
            _require_unique(shapes, f"suites.{name}.shapes")
            invalid = sorted(set(shapes) - _SHAPES)
            if invalid:
                raise ValueError(f"suites.{name}.shapes has unknown shapes: {invalid}")
        if name in {"warmup", "latency", "aimd", "soak", "time_variation"}:
            for axis in ("input", "output"):
                target_key = f"long_{axis}_tokens"
                by_route_key = f"{target_key}_by_route"
                overflow_key = f"long_{axis}_overflow"
                if target_key in values:
                    _positive_integer(values[target_key], f"suites.{name}.{target_key}")
                if by_route_key in values:
                    by_route = values[by_route_key]
                    if not isinstance(by_route, dict) or not by_route:
                        raise ValueError(f"suites.{name}.{by_route_key} must be a nonempty mapping")
                    for route_id, target in by_route.items():
                        if route_id not in route_ids:
                            raise ValueError(
                                f"suites.{name}.{by_route_key} has unknown route ID: {route_id}"
                            )
                        _positive_integer(
                            target,
                            f"suites.{name}.{by_route_key}.{route_id}",
                        )
                if overflow_key in values:
                    if target_key not in values and by_route_key not in values:
                        raise ValueError(
                            f"suites.{name}.{overflow_key} requires {target_key} or {by_route_key}"
                        )
                    if values[overflow_key] not in {"fail", "clip"}:
                        raise ValueError(f"suites.{name}.{overflow_key} must be fail or clip")
        if name == "static":
            _positive_number(values.get("offered_rps", 1.0), "suites.static.offered_rps")
        if name == "time_variation":
            _positive_integer(values.get("panels", 12), "suites.time_variation.panels")
            _positive_number(
                values.get("interval_minutes", 120),
                "suites.time_variation.interval_minutes",
            )
            _positive_integer(
                values.get("samples_per_route_shape", 3),
                "suites.time_variation.samples_per_route_shape",
            )
            stable_repeats = values.get("stable_exact_prompt_repeats")
            cold_repeats = values.get("panel_unique_cache_cold_repeats")
            if (stable_repeats is None) != (cold_repeats is None):
                raise ValueError(
                    "time_variation prompt-repeat design requires both "
                    "stable_exact_prompt_repeats and panel_unique_cache_cold_repeats"
                )
            if stable_repeats is not None and cold_repeats is not None:
                _positive_integer(
                    stable_repeats,
                    "suites.time_variation.stable_exact_prompt_repeats",
                )
                _positive_integer(
                    cold_repeats,
                    "suites.time_variation.panel_unique_cache_cold_repeats",
                )
                if stable_repeats + cold_repeats != int(
                    values.get("samples_per_route_shape", 3)
                ):
                    raise ValueError(
                        "time_variation stable and cache-cold repeats must sum to "
                        "samples_per_route_shape"
                    )
            _positive_number(values.get("offered_rps", 0.2), "suites.time_variation.offered_rps")
            _positive_integer(
                values.get("concurrency", default_concurrency),
                "suites.time_variation.concurrency",
            )
            if "interleave_gap_work" in values:
                _strict_bool(
                    values["interleave_gap_work"],
                    "suites.time_variation.interleave_gap_work",
                )
            guard = _positive_number(
                values.get("panel_guard_seconds", 300),
                "suites.time_variation.panel_guard_seconds",
                allow_zero=True,
            )
            cutoff = _positive_number(
                values.get("send_cutoff_seconds", 0),
                "suites.time_variation.send_cutoff_seconds",
                allow_zero=True,
            )
            deadline = _positive_number(
                values.get("panel_deadline_seconds", 600),
                "suites.time_variation.panel_deadline_seconds",
            )
            if values.get("interleave_gap_work", False):
                if retries:
                    raise ValueError(
                        "interleaved time_variation requires campaign.retries=0 so the "
                        "panel deadline has one hard request-timeout bound"
                    )
                selected_route_count = len(values.get("route_ids", route_ids))
                panel_arrivals = (
                    selected_route_count
                    * len(values.get("shapes", ["short_short", "long_short"]))
                    * int(values.get("samples_per_route_shape", 3))
                )
                if int(values.get("concurrency", default_concurrency)) < panel_arrivals:
                    raise ValueError(
                        "interleaved time_variation concurrency must admit every registered "
                        "panel arrival without client-side queueing"
                    )
                if cutoff <= 0:
                    raise ValueError(
                        "interleaved time_variation requires a positive send_cutoff_seconds"
                    )
                last_panel = (
                    (int(values.get("panels", 12)) - 1)
                    * float(values.get("interval_minutes", 120))
                    * 60
                )
                if last_panel >= cutoff:
                    raise ValueError(
                        "the last time_variation panel must begin before send_cutoff_seconds"
                    )
                if guard >= float(values.get("interval_minutes", 120)) * 60:
                    raise ValueError(
                        "time_variation.panel_guard_seconds must be shorter than the panel interval"
                    )
                if deadline >= float(values.get("interval_minutes", 120)) * 60:
                    raise ValueError(
                        "time_variation.panel_deadline_seconds must be shorter than "
                        "the panel interval"
                    )
        if name == "context":
            percentages = values.get("percentages", [1, 10, 25, 50, 75, 90, 95, 99])
            fixed_tokens = values.get("fixed_tokens", [])
            if not isinstance(percentages, list):
                raise ValueError("suites.context.percentages must be a list")
            if not isinstance(fixed_tokens, list):
                raise ValueError("suites.context.fixed_tokens must be a list")
            if not percentages and not fixed_tokens:
                raise ValueError(
                    "suites.context requires at least one percentage or fixed token anchor"
                )
            _require_unique(percentages, "suites.context.percentages")
            for index, percentage in enumerate(percentages):
                number = _positive_number(percentage, f"suites.context.percentages[{index}]")
                if number > 100:
                    raise ValueError("suites.context.percentages must not exceed 100")
            _require_unique(fixed_tokens, "suites.context.fixed_tokens")
            for index, tokens in enumerate(fixed_tokens):
                _positive_integer(tokens, f"suites.context.fixed_tokens[{index}]")
        if name in {"capability", "interactions"}:
            for key in ("temperatures", "top_ps"):
                if key in values and (not isinstance(values[key], list) or not values[key]):
                    raise ValueError(f"suites.{name}.{key} must be a nonempty list")
                if key in values:
                    _require_unique(values[key], f"suites.{name}.{key}")
                for index, value in enumerate(values.get(key, [])):
                    _number(value, f"suites.{name}.{key}[{index}]")
                    if not math.isfinite(float(value)):
                        raise ValueError(f"suites.{name}.{key}[{index}] must be finite")
        if name == "capability":
            groups = values.get("probe_groups")
            if groups is not None:
                if not isinstance(groups, list) or not groups:
                    raise ValueError("suites.capability.probe_groups must be a nonempty list")
                _require_unique(groups, "suites.capability.probe_groups")
                unknown_groups = sorted(set(groups) - _CAPABILITY_PROBE_GROUPS)
                if unknown_groups:
                    raise ValueError(
                        "suites.capability.probe_groups has unknown groups: "
                        + ", ".join(unknown_groups)
                    )
            groups_by_route = values.get("probe_groups_by_route")
            if groups_by_route is not None:
                if not isinstance(groups_by_route, dict) or not groups_by_route:
                    raise ValueError(
                        "suites.capability.probe_groups_by_route must be a nonempty mapping"
                    )
                for route_id, route_groups in groups_by_route.items():
                    if route_id not in route_ids:
                        raise ValueError(f"unknown capability probe route ID: {route_id}")
                    if not isinstance(route_groups, list) or not route_groups:
                        raise ValueError(
                            f"capability probe groups for {route_id} must be a nonempty list"
                        )
                    _require_unique(
                        route_groups,
                        f"suites.capability.probe_groups_by_route.{route_id}",
                    )
                    unknown_groups = sorted(set(route_groups) - _CAPABILITY_PROBE_GROUPS)
                    if unknown_groups:
                        raise ValueError(
                            f"unknown capability probe groups for {route_id}: {unknown_groups}"
                        )
        if name == "capability" and "tool_counts" in values:
            tool_counts = values["tool_counts"]
            if not isinstance(tool_counts, list) or not tool_counts:
                raise ValueError("suites.capability.tool_counts must be a nonempty list")
            _require_unique(tool_counts, "suites.capability.tool_counts")
            for index, value in enumerate(tool_counts):
                _positive_integer(value, f"suites.capability.tool_counts[{index}]")
        if name == "interactions" and "stream" in values:
            stream = values["stream"]
            if (
                not isinstance(stream, list)
                or not stream
                or any(not isinstance(value, bool) for value in stream)
            ):
                raise ValueError("suites.interactions.stream must be a nonempty boolean list")
            _require_unique(stream, "suites.interactions.stream")
        if name == "interactions" and "output_tokens" in values:
            tokens = values["output_tokens"]
            if not isinstance(tokens, list) or not tokens:
                raise ValueError("suites.interactions.output_tokens must be a nonempty list")
            _require_unique(tokens, "suites.interactions.output_tokens")
            for index, value in enumerate(tokens):
                _positive_integer(value, f"suites.interactions.output_tokens[{index}]")
        if name == "cache":
            _positive_integer(values.get("prefix_tokens", 4_096), "suites.cache.prefix_tokens")
        if name == "output":
            _positive_integer(
                values.get("fallback_max_output_tokens", 4_096),
                "suites.output.fallback_max_output_tokens",
            )
            _positive_integer(
                values.get("realized_generation_ceiling", 16_384),
                "suites.output.realized_generation_ceiling",
            )
        if name == "aimd":
            from .load import validate_aimd_config

            validate_aimd_config(values, default_concurrency)
        if name in {"aimd", "soak"} and "cells" in values:
            cells = values["cells"]
            if not isinstance(cells, list) or not cells:
                raise ValueError(f"suites.{name}.cells must be a nonempty list")
            _require_unique(cells, f"suites.{name}.cells")
            selected_routes = set(values.get("route_ids") or route_ids)
            selected_shapes = set(values.get("shapes") or _SHAPES)
            for index, cell in enumerate(cells):
                if not isinstance(cell, str) or ":" not in cell:
                    raise ValueError(
                        f"suites.{name}.cells[{index}] must use '<route_id>:<shape>'"
                    )
                route_id, shape = cell.rsplit(":", 1)
                if route_id not in selected_routes:
                    raise ValueError(f"{name} cell uses an unselected route: {route_id}")
                if shape not in selected_shapes:
                    raise ValueError(f"{name} cell uses an unselected shape: {shape}")
        if name == "soak":
            from .load import validate_soak_config

            validate_soak_config(values, default_concurrency)
            by_route = values.get("rate_rps_by_route") or {}
            if not isinstance(by_route, dict):
                raise ValueError("suites.soak.rate_rps_by_route must be a mapping")
            for route_id, rate in by_route.items():
                if not isinstance(route_id, str) or not route_id:
                    raise ValueError("suites.soak.rate_rps_by_route keys must be route IDs")
                if route_id not in route_ids:
                    raise ValueError(f"unknown soak route ID: {route_id}")
                _positive_number(rate, f"suites.soak.rate_rps_by_route.{route_id}")
            by_cell = values.get("rate_rps_by_route_shape") or {}
            if not isinstance(by_cell, dict):
                raise ValueError("suites.soak.rate_rps_by_route_shape must be a mapping")
            configured_cells: set[tuple[str, str]] = set()
            for key, rate_or_map in by_cell.items():
                if isinstance(rate_or_map, dict):
                    if key not in route_ids:
                        raise ValueError(f"unknown soak route ID: {key}")
                    if not rate_or_map:
                        raise ValueError(f"empty route-shape rates for {key}")
                    for shape, rate in rate_or_map.items():
                        if shape not in _SHAPES:
                            raise ValueError(f"unknown soak shape {shape}")
                        if (key, shape) in configured_cells:
                            raise ValueError(f"duplicate soak route-shape override: {key}:{shape}")
                        configured_cells.add((key, shape))
                        _positive_number(rate, f"suites.soak.rate_rps_by_route_shape.{key}.{shape}")
                else:
                    if not isinstance(key, str) or ":" not in key:
                        raise ValueError("flat soak route-shape keys must use '<route_id>:<shape>'")
                    route_id, shape = key.rsplit(":", 1)
                    if route_id not in route_ids:
                        raise ValueError(f"unknown soak route ID: {route_id}")
                    if shape not in _SHAPES:
                        raise ValueError(f"unknown soak shape {shape}")
                    if (route_id, shape) in configured_cells:
                        raise ValueError(f"duplicate soak route-shape override: {key}")
                    configured_cells.add((route_id, shape))
                    _positive_number(rate_or_map, f"suites.soak.rate_rps_by_route_shape.{key}")
            # Resolve the fallback too, even when route-specific rates are supplied.
            _positive_number(values.get("rate_rps", 0.25), "suites.soak.rate_rps")
