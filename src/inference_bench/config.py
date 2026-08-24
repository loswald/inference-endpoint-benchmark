from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from .models import AuthConfig, RouteConfig, sha256_json

_PUBLIC_CAPABILITY_KEYS = {
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
    "static": {"enabled", "offered_rps"},
    "latency": {"enabled", "repeats", "shapes"},
    "capability": {
        "enabled",
        "temperatures",
        "top_ps",
    },
    "interactions": {
        "enabled",
        "temperatures",
        "top_ps",
        "stream",
        "output_tokens",
    },
    "context": {"enabled", "percentages"},
    "output": {"enabled", "fallback_max_output_tokens"},
    "quality": {"enabled", "repeats"},
    "cache": {"enabled", "repeats", "prefix_tokens"},
    "aimd": {
        "enabled",
        "shapes",
        "initial_rps",
        "additive_rps",
        "multiplicative_decrease",
        "epochs",
        "epoch_seconds",
        "concurrency",
        "baseline_rps",
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
    },
}


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
    input_token_reservation_factor: float = 1.5
    suites: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign.name is required")
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
        if self.concurrency <= 0 or self.retries < 0:
            raise ValueError("concurrency/retries are invalid")
        if self.input_token_reservation_factor < 1:
            raise ValueError("input_token_reservation_factor must be at least 1")
        if len({route.id for route in self.routes}) != len(self.routes):
            raise ValueError("route IDs must be unique")
        if any(
            not isinstance(name, str) or not isinstance(value, dict)
            for name, value in self.suites.items()
        ):
            raise ValueError("suites must map names to configuration mappings")

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
                },
                "routes": [
                    {"id": route.id, "identity_hash": route.identity_hash}
                    for route in self.routes
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
                    "context_tokens": route.context_tokens,
                    "max_output_tokens": route.max_output_tokens,
                    "input_usd_per_million": route.input_usd_per_million,
                    "output_usd_per_million": route.output_usd_per_million,
                    "cached_input_usd_per_million": route.cached_input_usd_per_million,
                    "capabilities": _safe_capabilities(route.capabilities),
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


def _route(raw: dict[str, Any]) -> RouteConfig:
    auth = raw.get("auth") or {}
    return RouteConfig(
        id=str(raw["id"]),
        provider=str(raw["provider"]),
        adapter=str(raw.get("adapter", "openai_compatible")),
        model=str(raw["model"]),
        base_url=str(raw["base_url"]),
        auth=AuthConfig(
            env=str(auth["env"]),
            header=str(auth.get("header", "Authorization")),
            prefix=str(auth.get("prefix", "Bearer ")),
        ),
        region=str(raw.get("region", "not_reported")),
        api_family=str(raw.get("api_family", "chat_completions")),
        api_version=str(raw.get("api_version", "not_reported")),
        model_version=str(raw.get("model_version", "not_reported")),
        upstream_provider=raw.get("upstream_provider"),
        context_tokens=raw.get("context_tokens"),
        max_output_tokens=raw.get("max_output_tokens"),
        input_usd_per_million=raw.get("input_usd_per_million"),
        output_usd_per_million=raw.get("output_usd_per_million"),
        cached_input_usd_per_million=raw.get("cached_input_usd_per_million"),
        capabilities=dict(raw.get("capabilities") or {}),
        extra_headers=dict(raw.get("extra_headers") or {}),
        request_defaults=dict(raw.get("request_defaults") or {}),
    )


def load_config(path: str | Path) -> CampaignConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("campaign file must contain a mapping")
    campaign = raw.get("campaign") or {}
    routes = raw.get("routes") or []
    if not isinstance(routes, list) or not routes:
        raise ValueError("at least one route is required")
    return CampaignConfig(
        name=str(campaign.get("name", "")),
        seed=int(campaign.get("seed", 1)),
        max_wall_seconds=float(campaign.get("max_wall_seconds", 0)),
        max_cost_usd=float(campaign.get("max_cost_usd", 0)),
        launch_reserve_seconds=float(campaign.get("launch_reserve_seconds", 180)),
        launch_reserve_usd=float(campaign.get("launch_reserve_usd", 5)),
        concurrency=int(campaign.get("concurrency", 16)),
        retries=int(campaign.get("retries", 2)),
        input_token_reservation_factor=float(campaign.get("input_token_reservation_factor", 1.5)),
        routes=tuple(_route(item) for item in routes),
        suites=dict(raw.get("suites") or {}),
    )
