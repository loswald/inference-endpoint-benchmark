"""Provider-neutral composition of reusable provider catalogs and experiments.

The layer deliberately has two small, strict YAML contracts:

``provider-profile/v1``
    ``schema`` (exact string), ``provider`` (provider identity), ``route_defaults``
    (transport/auth/evidence defaults accepted by :class:`RouteConfig`), and ``routes``
    (the provider's route catalog).  Catalog rows require ``id`` and may override defaults.
    ``provider`` is injected uniformly and therefore cannot be repeated in route mappings.

``provider-profile/v2``
    Retains the v1 route configuration and adds a strict catalog freshness declaration plus
    per-route lifecycle, benchmark-role, and live-admission metadata.  A selected v2 route must
    be current, released, not retired or superseded, explicitly admitted by live evidence, and
    drawn from a catalog that is fresh at the experiment's deterministic ``as_of_utc``.

``benchmark-experiment/v1``
    ``schema`` (exact string), ``campaign`` (the normal campaign mapping), ``suites`` (the
    normal suite mapping), optional ``route_selection`` with ``include``/``exclude`` ID lists,
    optional ``provider_route_overrides`` keyed by provider identity, optional
    ``route_overrides`` keyed by catalog route ID, and optional UTC-aware ``as_of_utc`` (required
    for provider-profile/v2).  Overrides cannot change route identity fields ``id`` or
    ``provider``.  Provider-scoped overrides let one portable experiment tighten a transport
    timeout without mutating the reusable provider catalog.

Composition uses the precedence ``profile defaults < catalog row < provider experiment override
< exact-route experiment override``.
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
from datetime import UTC, datetime, timedelta
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
PROVIDER_PROFILE_SCHEMA_V2 = "provider-profile/v2"
EXPERIMENT_PROFILE_SCHEMA = "benchmark-experiment/v1"

_PROVIDER_PROFILE_V1_KEYS = {"schema", "provider", "route_defaults", "routes"}
_PROVIDER_PROFILE_V2_KEYS = _PROVIDER_PROFILE_V1_KEYS | {"catalog"}
_EXPERIMENT_PROFILE_KEYS = {
    "schema",
    "as_of_utc",
    "campaign",
    "route_selection",
    "provider_route_overrides",
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
_V2_ROUTE_METADATA_FIELDS = {"lifecycle", "role", "live_admission"}
_CATALOG_KEYS = {
    "documentation_checked_at_utc",
    "revalidated_at_utc",
    "freshness_window_days",
}
_LIFECYCLE_KEYS = {
    "stage",
    "status",
    "released_at_utc",
    "retirement_at_utc",
    "superseded_by",
}
_LIVE_ADMISSION_KEYS = {
    "status",
    "verified_at_utc",
    "evidence_sha256",
    "reason",
}
_LIFECYCLE_STAGES = frozenset({"ga", "preview", "private_preview", "experimental"})
_LIFECYCLE_STATUSES = frozenset({"current", "superseded", "retired"})
_ROUTE_ROLES = frozenset({"primary", "preview", "control"})
_LIVE_ADMISSION_STATUSES = frozenset(
    {"live_proved", "unverified", "live_failed", "excluded"}
)


@dataclass(frozen=True, slots=True)
class _CatalogFreshness:
    documentation_checked_at_utc: datetime
    revalidated_at_utc: datetime
    freshness_window_days: int


@dataclass(frozen=True, slots=True)
class _RouteAdmission:
    stage: str
    lifecycle_status: str
    released_at_utc: datetime
    retirement_at_utc: datetime | None
    superseded_by: str | None
    role: str
    live_status: str
    live_verified_at_utc: datetime | None


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

    provider_schema = provider_profile.get("schema")
    if provider_schema == PROVIDER_PROFILE_SCHEMA:
        _reject_unknown("provider profile", provider_profile, _PROVIDER_PROFILE_V1_KEYS)
        catalog_freshness = None
    elif provider_schema == PROVIDER_PROFILE_SCHEMA_V2:
        _reject_unknown("provider profile", provider_profile, _PROVIDER_PROFILE_V2_KEYS)
        catalog_freshness = _parse_catalog_freshness(provider_profile.get("catalog"))
    else:
        raise ValueError(
            "provider profile.schema must be exactly "
            f"{PROVIDER_PROFILE_SCHEMA!r} or {PROVIDER_PROFILE_SCHEMA_V2!r}"
        )
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
    route_admissions: dict[str, _RouteAdmission] = {}
    for index, raw_route in enumerate(raw_routes):
        route_scope = f"provider profile.routes[{index}]"
        route = _mapping(raw_route, route_scope)
        if provider_schema == PROVIDER_PROFILE_SCHEMA_V2:
            _reject_unknown(
                route_scope,
                route,
                set(ROUTE_CONFIG_KEYS) | _V2_ROUTE_METADATA_FIELDS,
            )
            route_config = {
                key: value for key, value in route.items() if key not in _V2_ROUTE_METADATA_FIELDS
            }
            admission = _parse_route_admission(route, route_scope)
        else:
            route_config = route
            admission = None
        _validate_route_fields(
            route_config,
            route_scope,
            forbidden={"provider"},
        )
        route_id = _nonempty_string(route_config.get("id"), f"{route_scope}.id")
        if route_id in catalog:
            raise ValueError(f"duplicate provider profile route ID: {route_id}")
        catalog[route_id] = route_config
        if admission is not None:
            route_admissions[route_id] = admission

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
    raw_as_of_utc = experiment_profile.get("as_of_utc")
    as_of_utc = (
        None
        if raw_as_of_utc is None
        else _utc_datetime(raw_as_of_utc, "experiment profile.as_of_utc")
    )
    if provider_schema == PROVIDER_PROFILE_SCHEMA_V2:
        if as_of_utc is None:
            raise ValueError("experiment profile.as_of_utc is required for provider-profile/v2")
        assert catalog_freshness is not None
        _require_fresh_catalog(catalog_freshness, as_of_utc)
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
    if provider_schema == PROVIDER_PROFILE_SCHEMA_V2:
        assert as_of_utc is not None
        for route_id in selected_ids:
            _require_route_admitted(route_id, route_admissions[route_id], as_of_utc)

    overrides = _mapping(
        experiment_profile.get("route_overrides", {}),
        "experiment profile.route_overrides",
    )
    provider_overrides = _mapping(
        experiment_profile.get("provider_route_overrides", {}),
        "experiment profile.provider_route_overrides",
    )
    for provider_id, raw_override in provider_overrides.items():
        _nonempty_string(provider_id, "experiment profile.provider_route_overrides key")
        provider_override = _mapping(
            raw_override,
            f"experiment profile.provider_route_overrides.{provider_id}",
        )
        _validate_route_fields(
            provider_override,
            f"experiment profile.provider_route_overrides.{provider_id}",
            forbidden=_PROFILE_CONTROLLED_ROUTE_FIELDS,
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
    selected_provider_override = _mapping(
        provider_overrides.get(provider, {}),
        f"experiment profile.provider_route_overrides.{provider}",
    )
    for route_id in selected_ids:
        override = _mapping(
            overrides.get(route_id, {}), f"experiment profile.route_overrides.{route_id}"
        )
        _validate_route_fields(
            override,
            f"experiment profile.route_overrides.{route_id}",
            forbidden=_PROFILE_CONTROLLED_ROUTE_FIELDS,
        )
        compiled = _merge_route_layers(
            route_defaults,
            catalog[route_id],
            selected_provider_override,
            override,
        )
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


def _parse_catalog_freshness(value: Any) -> _CatalogFreshness:
    scope = "provider profile.catalog"
    catalog = _mapping(value, scope)
    _reject_unknown(scope, catalog, _CATALOG_KEYS)
    documentation_checked_at_utc = _utc_datetime(
        catalog.get("documentation_checked_at_utc"),
        f"{scope}.documentation_checked_at_utc",
    )
    revalidated_at_utc = _utc_datetime(
        catalog.get("revalidated_at_utc"),
        f"{scope}.revalidated_at_utc",
    )
    freshness_window_days = catalog.get("freshness_window_days")
    if (
        isinstance(freshness_window_days, bool)
        or not isinstance(freshness_window_days, int)
        or freshness_window_days <= 0
    ):
        raise ValueError(f"{scope}.freshness_window_days must be a positive integer")
    if revalidated_at_utc < documentation_checked_at_utc:
        raise ValueError(
            f"{scope}.revalidated_at_utc cannot precede documentation_checked_at_utc"
        )
    return _CatalogFreshness(
        documentation_checked_at_utc=documentation_checked_at_utc,
        revalidated_at_utc=revalidated_at_utc,
        freshness_window_days=freshness_window_days,
    )


def _parse_route_admission(route: dict[str, Any], scope: str) -> _RouteAdmission:
    lifecycle_scope = f"{scope}.lifecycle"
    lifecycle = _mapping(route.get("lifecycle"), lifecycle_scope)
    _reject_unknown(lifecycle_scope, lifecycle, _LIFECYCLE_KEYS)
    stage = _enum_string(
        lifecycle.get("stage"), f"{lifecycle_scope}.stage", _LIFECYCLE_STAGES
    )
    lifecycle_status = _enum_string(
        lifecycle.get("status"),
        f"{lifecycle_scope}.status",
        _LIFECYCLE_STATUSES,
    )
    released_at_utc = _utc_datetime(
        lifecycle.get("released_at_utc"), f"{lifecycle_scope}.released_at_utc"
    )
    retirement_value = lifecycle.get("retirement_at_utc")
    retirement_at_utc = (
        None
        if retirement_value is None
        else _utc_datetime(retirement_value, f"{lifecycle_scope}.retirement_at_utc")
    )
    if retirement_at_utc is not None and retirement_at_utc <= released_at_utc:
        raise ValueError(f"{lifecycle_scope}.retirement_at_utc must follow released_at_utc")

    superseded_value = lifecycle.get("superseded_by")
    superseded_by = (
        None
        if superseded_value is None
        else _nonempty_string(superseded_value, f"{lifecycle_scope}.superseded_by")
    )
    if lifecycle_status == "superseded" and superseded_by is None:
        raise ValueError(f"{lifecycle_scope}.superseded_by is required when status is superseded")
    if lifecycle_status != "superseded" and superseded_by is not None:
        raise ValueError(
            f"{lifecycle_scope}.superseded_by is only valid when status is superseded"
        )
    if lifecycle_status == "retired" and retirement_at_utc is None:
        raise ValueError(f"{lifecycle_scope}.retirement_at_utc is required when status is retired")

    role = _enum_string(route.get("role"), f"{scope}.role", _ROUTE_ROLES)
    live_scope = f"{scope}.live_admission"
    live = _mapping(route.get("live_admission"), live_scope)
    _reject_unknown(live_scope, live, _LIVE_ADMISSION_KEYS)
    live_status = _enum_string(
        live.get("status"), f"{live_scope}.status", _LIVE_ADMISSION_STATUSES
    )
    verified_value = live.get("verified_at_utc")
    live_verified_at_utc = (
        None
        if verified_value is None
        else _utc_datetime(verified_value, f"{live_scope}.verified_at_utc")
    )
    evidence_value = live.get("evidence_sha256")
    reason_value = live.get("reason")
    if live_status == "live_proved":
        if live_verified_at_utc is None:
            raise ValueError(
                f"{live_scope}.verified_at_utc is required when status is live_proved"
            )
        _sha256(evidence_value, f"{live_scope}.evidence_sha256")
        if reason_value is not None:
            _nonempty_string(reason_value, f"{live_scope}.reason")
    else:
        if live_verified_at_utc is not None or evidence_value is not None:
            raise ValueError(
                f"{live_scope} cannot claim verification evidence when status is {live_status}"
            )
        _nonempty_string(reason_value, f"{live_scope}.reason")
    if live_verified_at_utc is not None and live_verified_at_utc < released_at_utc:
        raise ValueError(f"{live_scope}.verified_at_utc cannot precede route release")

    return _RouteAdmission(
        stage=stage,
        lifecycle_status=lifecycle_status,
        released_at_utc=released_at_utc,
        retirement_at_utc=retirement_at_utc,
        superseded_by=superseded_by,
        role=role,
        live_status=live_status,
        live_verified_at_utc=live_verified_at_utc,
    )


def _require_fresh_catalog(catalog: _CatalogFreshness, as_of_utc: datetime) -> None:
    if catalog.documentation_checked_at_utc > as_of_utc:
        raise ValueError("provider profile catalog documentation timestamp is after as_of_utc")
    if catalog.revalidated_at_utc > as_of_utc:
        raise ValueError("provider profile catalog revalidation timestamp is after as_of_utc")
    maximum_age = timedelta(days=catalog.freshness_window_days)
    if as_of_utc - catalog.revalidated_at_utc > maximum_age:
        raise ValueError(
            "stale catalog: provider profile revalidation exceeds its freshness window at "
            "experiment profile.as_of_utc"
        )


def _require_route_admitted(
    route_id: str, admission: _RouteAdmission, as_of_utc: datetime
) -> None:
    if admission.released_at_utc > as_of_utc:
        raise ValueError(f"selected route {route_id!r} is not released as of as_of_utc")
    if admission.lifecycle_status == "retired":
        raise ValueError(f"selected route {route_id!r} is retired")
    if admission.lifecycle_status == "superseded":
        raise ValueError(
            f"selected route {route_id!r} is superseded by {admission.superseded_by!r}"
        )
    if admission.retirement_at_utc is not None and admission.retirement_at_utc <= as_of_utc:
        raise ValueError(f"selected route {route_id!r} is retired as of as_of_utc")
    if admission.live_status == "excluded":
        raise ValueError(f"selected route {route_id!r} is explicitly excluded")
    if admission.live_status != "live_proved":
        raise ValueError(f"selected route {route_id!r} is not live-proved")
    assert admission.live_verified_at_utc is not None
    if admission.live_verified_at_utc > as_of_utc:
        raise ValueError(f"selected route {route_id!r} has live proof after as_of_utc")


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


def _enum_string(value: Any, field_name: str, allowed: frozenset[str]) -> str:
    parsed = _nonempty_string(value, field_name)
    if parsed not in allowed:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(allowed))}")
    return parsed


def _utc_datetime(value: Any, field_name: str) -> datetime:
    raw = _nonempty_string(value, field_name)
    candidate = f"{raw[:-1]}+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return parsed.astimezone(UTC)


def _sha256(value: Any, field_name: str) -> str:
    digest = _nonempty_string(value, field_name)
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return digest.lower()


def _require_schema(values: dict[str, Any], expected: str, scope: str) -> None:
    if values.get("schema") != expected:
        raise ValueError(f"{scope}.schema must be exactly {expected!r}")


def _reject_unknown(scope: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {scope} field(s): {', '.join(unknown)}")
