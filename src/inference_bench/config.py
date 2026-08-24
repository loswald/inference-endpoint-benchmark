from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import AuthConfig, RouteConfig, sha256_json


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

    @property
    def identity_hash(self) -> str:
        return sha256_json(self.public_dict())

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
                    "base_url": route.base_url,
                    "auth": {
                        "env": route.auth.env,
                        "header": route.auth.header,
                        "prefix": route.auth.prefix,
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
                    "capabilities": route.capabilities,
                    "extra_headers": route.extra_headers,
                    "request_defaults": route.request_defaults,
                    "identity_hash": route.identity_hash,
                }
                for route in self.routes
            ],
            "suites": copy.deepcopy(self.suites),
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
