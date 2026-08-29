"""Provider-neutral composition of reusable provider catalogs and experiments.

The layer deliberately has two small, strict YAML contracts:

``provider-profile/v1``
    ``schema`` (exact string), ``provider`` (provider identity), ``route_defaults``
    (transport/auth/evidence defaults accepted by :class:`RouteConfig`), and ``routes``
    (the provider's route catalog).  Catalog rows require ``id`` and may override defaults.
    ``provider`` is injected uniformly and therefore cannot be repeated in route mappings.

``benchmark-experiment/v1``
    ``schema`` (exact string), ``campaign`` (the normal campaign mapping), ``suites`` (the
    normal suite mapping), optional ``route_selection`` with ``include``/``exclude`` ID lists,
    and optional ``route_overrides`` keyed by catalog route ID.  Overrides cannot change route
    identity fields ``id`` or ``provider``.

Composition uses the precedence ``profile defaults < catalog row < experiment override``.
The mapping-valued route fields ``auth``, ``capabilities``, ``extra_headers``, and
``request_defaults`` merge one level deep; every other field replaces the earlier value.
Selected routes are emitted in route-ID order.  The result then crosses the same strict
``CampaignConfig`` validation boundary as a handwritten campaign file.  No environment variable
is resolved and no credential value is accepted by this layer.
"""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import (
    ROUTE_CONFIG_KEYS,
    CampaignConfig,
    campaign_config_from_mapping,
    load_yaml_mapping,
    route_config_from_mapping,
)
from .models import sha256_json

PROVIDER_PROFILE_SCHEMA = "provider-profile/v1"
EXPERIMENT_PROFILE_SCHEMA = "benchmark-experiment/v1"

_PROVIDER_PROFILE_KEYS = {"schema", "provider", "route_defaults", "routes"}
_EXPERIMENT_PROFILE_KEYS = {
    "schema",
    "campaign",
    "route_selection",
    "route_overrides",
    "suites",
}
_ROUTE_SELECTION_KEYS = {"include", "exclude"}
_NESTED_ROUTE_FIELDS = {
    "auth",
    "capabilities",
    "extra_headers",
    "request_defaults",
    "reasoning_controls",
}
_PROFILE_CONTROLLED_ROUTE_FIELDS = {"id", "provider"}


@dataclass(frozen=True, slots=True)
class ProfileCompilation:
    """Validated output plus stable identities for its two declarative inputs."""

    config: CampaignConfig
    mapping: dict[str, Any]
    provider_profile_sha256: str
    experiment_profile_sha256: str

    @property
    def yaml_bytes(self) -> bytes:
        return yaml.safe_dump(
            self.mapping,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        ).encode("utf-8")

    @property
    def compiled_sha256(self) -> str:
        return hashlib.sha256(self.yaml_bytes).hexdigest()


def compose_profile_config(
    provider_profile: dict[str, Any], experiment_profile: dict[str, Any]
) -> ProfileCompilation:
    """Compile two parsed profile mappings into one validated campaign configuration."""

    _reject_unknown("provider profile", provider_profile, _PROVIDER_PROFILE_KEYS)
    _require_schema(provider_profile, PROVIDER_PROFILE_SCHEMA, "provider profile")
    provider = _nonempty_string(provider_profile.get("provider"), "provider profile.provider")

    route_defaults = _mapping(
        provider_profile.get("route_defaults", {}), "provider profile.route_defaults"
    )
    _validate_route_fields(
        route_defaults,
        "provider profile.route_defaults",
        forbidden=_PROFILE_CONTROLLED_ROUTE_FIELDS,
    )
    raw_routes = provider_profile.get("routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise ValueError("provider profile.routes must be a nonempty list")

    catalog: dict[str, dict[str, Any]] = {}
    for index, raw_route in enumerate(raw_routes):
        route = _mapping(raw_route, f"provider profile.routes[{index}]")
        _validate_route_fields(
            route,
            f"provider profile.routes[{index}]",
            forbidden={"provider"},
        )
        route_id = _nonempty_string(route.get("id"), f"provider profile.routes[{index}].id")
        if route_id in catalog:
            raise ValueError(f"duplicate provider profile route ID: {route_id}")
        catalog[route_id] = route

    # A provider profile is an admitted catalog, not a bag of fragments.  Validate every route,
    # including routes that a particular experiment does not select, so stale or malformed catalog
    # entries cannot remain hidden until a later campaign.
    for route_id, route in catalog.items():
        resolved = _merge_route_layers(route_defaults, route)
        resolved["id"] = route_id
        resolved["provider"] = provider
        route_config_from_mapping(resolved)

    _reject_unknown("experiment profile", experiment_profile, _EXPERIMENT_PROFILE_KEYS)
    _require_schema(experiment_profile, EXPERIMENT_PROFILE_SCHEMA, "experiment profile")
    campaign = _mapping(experiment_profile.get("campaign"), "experiment profile.campaign")
    suites = _mapping(experiment_profile.get("suites"), "experiment profile.suites")
    if not suites:
        raise ValueError("experiment profile.suites must not be empty")

    selection = _mapping(
        experiment_profile.get("route_selection", {}),
        "experiment profile.route_selection",
    )
    _reject_unknown("experiment profile.route_selection", selection, _ROUTE_SELECTION_KEYS)
    include = _route_id_list(selection.get("include"), "route_selection.include")
    exclude = _route_id_list(selection.get("exclude", []), "route_selection.exclude")
    if include is not None and not include:
        raise ValueError("route_selection.include must not be empty")
    include_set = set(catalog) if include is None else set(include)
    exclude_set = set(exclude or [])
    unknown_selected = sorted((include_set | exclude_set) - set(catalog))
    if unknown_selected:
        raise ValueError(
            "route selection references unknown route(s): " + ", ".join(unknown_selected)
        )
    overlap = sorted(include_set & exclude_set)
    if overlap:
        raise ValueError("routes cannot be both included and excluded: " + ", ".join(overlap))
    selected_ids = sorted(include_set - exclude_set)
    if not selected_ids:
        raise ValueError("route selection produced no routes")

    overrides = _mapping(
        experiment_profile.get("route_overrides", {}),
        "experiment profile.route_overrides",
    )
    unknown_overrides = sorted(set(overrides) - set(catalog))
    if unknown_overrides:
        raise ValueError(
            "route overrides reference unknown route(s): " + ", ".join(unknown_overrides)
        )
    unselected_overrides = sorted(set(overrides) - set(selected_ids))
    if unselected_overrides:
        raise ValueError(
            "route overrides reference unselected route(s): " + ", ".join(unselected_overrides)
        )

    compiled_routes: list[dict[str, Any]] = []
    for route_id in selected_ids:
        override = _mapping(
            overrides.get(route_id, {}), f"experiment profile.route_overrides.{route_id}"
        )
        _validate_route_fields(
            override,
            f"experiment profile.route_overrides.{route_id}",
            forbidden=_PROFILE_CONTROLLED_ROUTE_FIELDS,
        )
        compiled = _merge_route_layers(route_defaults, catalog[route_id], override)
        compiled["id"] = route_id
        compiled["provider"] = provider
        compiled_routes.append(compiled)

    mapping = {
        "campaign": copy.deepcopy(campaign),
        "routes": compiled_routes,
        "suites": copy.deepcopy(suites),
    }
    config = campaign_config_from_mapping(mapping)
    return ProfileCompilation(
        config=config,
        mapping=mapping,
        provider_profile_sha256=sha256_json(provider_profile),
        experiment_profile_sha256=sha256_json(experiment_profile),
    )


def load_profile_config(
    provider_profile_path: str | Path, experiment_profile_path: str | Path
) -> ProfileCompilation:
    """Load and compose a provider profile and experiment profile without resolving secrets."""

    return compose_profile_config(
        load_yaml_mapping(provider_profile_path, document_name="provider profile"),
        load_yaml_mapping(experiment_profile_path, document_name="experiment profile"),
    )


def compile_profile_files(
    provider_profile_path: str | Path,
    experiment_profile_path: str | Path,
    output_path: str | Path,
) -> ProfileCompilation:
    """Write a byte-stable canonical campaign YAML assembled from the two profile files."""

    provider_path = Path(provider_profile_path).resolve()
    experiment_path = Path(experiment_profile_path).resolve()
    output = Path(output_path).resolve()
    if output in {provider_path, experiment_path}:
        raise ValueError("compiled output cannot overwrite either input profile")
    compilation = load_profile_config(provider_path, experiment_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(compilation.yaml_bytes)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    reloaded = campaign_config_from_mapping(
        load_yaml_mapping(output, document_name="compiled campaign")
    )
    if reloaded.identity_hash != compilation.config.identity_hash:
        raise RuntimeError("compiled campaign did not round-trip to the same identity")
    return compilation


def _merge_route_layers(*layers: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if key in _NESTED_ROUTE_FIELDS:
                nested = _mapping(value, f"route.{key}")
                prior = merged.get(key, {})
                if not isinstance(prior, dict):
                    raise ValueError(f"route.{key} must be a mapping")
                merged[key] = {**copy.deepcopy(prior), **copy.deepcopy(nested)}
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def _validate_route_fields(
    route: dict[str, Any], scope: str, *, forbidden: set[str] | frozenset[str]
) -> None:
    unknown = sorted(set(route) - set(ROUTE_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown {scope} field(s): {', '.join(unknown)}")
    controlled = sorted(set(route) & set(forbidden))
    if controlled:
        raise ValueError(f"{scope} cannot set profile-controlled field(s): {', '.join(controlled)}")


def _route_id_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_name} must be a list of nonempty route IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} contains duplicate route IDs")
    return list(value)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a string-keyed mapping")
    return value


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _require_schema(values: dict[str, Any], expected: str, scope: str) -> None:
    if values.get("schema") != expected:
        raise ValueError(f"{scope}.schema must be exactly {expected!r}")


def _reject_unknown(scope: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {scope} field(s): {', '.join(unknown)}")
