from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from .config import CampaignConfig, load_config, selected_capacity_cells
from .models import RouteConfig, canonical_json, sha256_json
from .plan import build_plan
from .workloads import shape_spec

PROFILE_SCHEMA = "capacity-closure-profile/v1"
MANIFEST_SCHEMA = "capacity-closure-plan/v1"
CONTROLLER_EVIDENCE_MANIFEST_SCHEMA = "capacity-controller-evidence/v1"

_CAPACITY_SUITES = frozenset({"aimd", "soak"})
_IDENTITY_COLUMN_KEYS = frozenset(
    {
        "route_identity_sha256",
        "source_campaign_identity_sha256",
        "workload_recipe_sha256",
        "input_target",
        "output_target",
    }
)
_PREDICATE_OPERATORS = frozenset(
    {"equals", "not_equals", "in", "not_in", "empty", "not_empty", "matches", "not_matches"}
)
_ROUTE_PREDICATE_FIELDS = frozenset(
    {
        "id",
        "provider",
        "adapter",
        "model",
        "region",
        "api_family",
        "api_version",
        "model_version",
        "upstream_provider",
    }
)
_CAMPAIGN_FIELDS = (
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
)
_CONTROLLED_SUITE_FIELDS = frozenset({"enabled", "route_ids", "shapes", "cells"})


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
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
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _filename(value: object, name: str, suffix: str) -> str:
    filename = _string(value, name)
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"{name} must be a filename, not a path")
    if Path(filename).suffix.casefold() != suffix:
        raise ValueError(f"{name} must end in {suffix}")
    return filename


def load_capacity_closure_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    try:
        raw = yaml.load(profile_path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid capacity-closure profile YAML: {exc}") from exc
    return validate_capacity_closure_profile(raw)


def _validate_predicates(
    value: object,
    name: str,
    *,
    allowed_fields: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    predicates: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        predicate = _mapping(item, f"{name}[{index}]")
        unknown = sorted(set(predicate) - {"field", "op", "value"})
        if unknown:
            raise ValueError(f"{name}[{index}] has unknown fields: {', '.join(unknown)}")
        field = _string(predicate.get("field"), f"{name}[{index}].field")
        if allowed_fields is not None and field not in allowed_fields:
            raise ValueError(f"{name}[{index}].field is not allowed: {field}")
        operator = _string(predicate.get("op"), f"{name}[{index}].op")
        if operator not in _PREDICATE_OPERATORS:
            raise ValueError(f"{name}[{index}].op is not supported: {operator}")
        has_value = "value" in predicate
        if operator in {"empty", "not_empty"} and has_value:
            raise ValueError(f"{name}[{index}] must omit value for the {operator} operator")
        if operator not in {"empty", "not_empty"} and not has_value:
            raise ValueError(f"{name}[{index}] requires value for the {operator} operator")
        if operator in {"in", "not_in"} and (
            not isinstance(predicate.get("value"), list) or not predicate["value"]
        ):
            raise ValueError(f"{name}[{index}].value must be a nonempty list for {operator}")
        if operator in {"matches", "not_matches"}:
            expression = _string(predicate.get("value"), f"{name}[{index}].value")
            try:
                re.compile(expression)
            except re.error as exc:
                raise ValueError(
                    f"{name}[{index}].value is not a valid regular expression"
                ) from exc
        predicates.append(copy.deepcopy(predicate))
    return predicates


def validate_capacity_closure_profile(value: object) -> dict[str, Any]:
    profile = _mapping(value, "capacity-closure profile")
    unknown = sorted(
        set(profile)
        - {"schema", "suite", "selection", "mapping", "route_predicates", "overrides", "output"}
    )
    if unknown:
        raise ValueError("capacity-closure profile has unknown fields: " + ", ".join(unknown))
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"capacity-closure profile schema must be {PROFILE_SCHEMA}")
    suite_name = _string(profile.get("suite"), "capacity-closure profile.suite")
    if suite_name not in _CAPACITY_SUITES:
        raise ValueError("capacity-closure profile.suite must be aimd or soak")

    selection = _mapping(profile.get("selection"), "capacity-closure profile.selection")
    unknown_selection = sorted(set(selection) - {"columns", "where", "expected_cells"})
    if unknown_selection:
        raise ValueError("selection has unknown fields: " + ", ".join(unknown_selection))
    columns = _mapping(selection.get("columns"), "selection.columns")
    allowed_columns = {"route_id", "workload", "prior_state", *_IDENTITY_COLUMN_KEYS}
    unknown_columns = sorted(set(columns) - allowed_columns)
    if unknown_columns:
        raise ValueError("selection.columns has unknown fields: " + ", ".join(unknown_columns))
    for required in ("route_id", "workload", *_IDENTITY_COLUMN_KEYS):
        _string(columns.get(required), f"selection.columns.{required}")
    if "prior_state" in columns:
        _string(columns["prior_state"], "selection.columns.prior_state")
    _validate_predicates(selection.get("where"), "selection.where")
    expected = selection.get("expected_cells")
    if expected is not None and (
        isinstance(expected, bool) or not isinstance(expected, int) or expected < 1
    ):
        raise ValueError("selection.expected_cells must be a positive integer when provided")

    mapping = _mapping(profile.get("mapping"), "capacity-closure profile.mapping")
    if set(mapping) != {"workload_to_shape"}:
        raise ValueError("mapping must contain only workload_to_shape")
    workload_to_shape = _mapping(mapping["workload_to_shape"], "mapping.workload_to_shape")
    if not workload_to_shape:
        raise ValueError("mapping.workload_to_shape must not be empty")
    for workload, shape in workload_to_shape.items():
        _string(workload, "mapping.workload_to_shape key")
        _string(shape, f"mapping.workload_to_shape.{workload}")

    route_predicates = _validate_predicates(
        profile.get("route_predicates"),
        "route_predicates",
        allowed_fields=_ROUTE_PREDICATE_FIELDS,
    )
    if not any(
        predicate["field"] == "provider" and predicate["op"] in {"equals", "in"}
        for predicate in route_predicates
    ):
        raise ValueError("route_predicates must explicitly select one or more providers")
    overrides = _mapping(profile.get("overrides"), "capacity-closure profile.overrides")
    if set(overrides) != {"campaign", "suite"}:
        raise ValueError("overrides must contain exactly campaign and suite mappings")
    campaign_overrides = _mapping(overrides["campaign"], "overrides.campaign")
    unknown_campaign = sorted(set(campaign_overrides) - set(_CAMPAIGN_FIELDS))
    if unknown_campaign:
        raise ValueError("overrides.campaign has unknown fields: " + ", ".join(unknown_campaign))
    suite_overrides = _mapping(overrides["suite"], "overrides.suite")
    controlled = sorted(set(suite_overrides) & _CONTROLLED_SUITE_FIELDS)
    if controlled:
        raise ValueError(
            "overrides.suite cannot replace planner-controlled fields: " + ", ".join(controlled)
        )

    output = _mapping(profile.get("output"), "capacity-closure profile.output")
    if set(output) != {"config_filename", "manifest_filename"}:
        raise ValueError("output must contain exactly config_filename and manifest_filename")
    _filename(output["config_filename"], "output.config_filename", ".yaml")
    _filename(output["manifest_filename"], "output.manifest_filename", ".json")

    # Reject YAML-only scalar types and other values that would make profile identity ambiguous.
    canonical_json(profile)
    return copy.deepcopy(profile)


def _predicate_matches(actual: object, predicate: Mapping[str, Any]) -> bool:
    operator = predicate["op"]
    expected = predicate.get("value")
    empty = actual is None or actual == ""
    if operator == "empty":
        return empty
    if operator == "not_empty":
        return not empty
    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "in":
        return actual in expected
    if operator == "not_in":
        return actual not in expected
    if operator == "matches":
        return re.search(str(expected), "" if actual is None else str(actual)) is not None
    if operator == "not_matches":
        return re.search(str(expected), "" if actual is None else str(actual)) is None
    raise AssertionError(f"unhandled predicate operator: {operator}")


def _select_evidence_rows(
    capacity_csv: Path, profile: Mapping[str, Any]
) -> list[dict[str, str]]:
    with capacity_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames:
        raise ValueError("capacity evidence CSV must contain a header")

    selection = profile["selection"]
    columns = selection["columns"]
    required_columns = {
        columns["route_id"],
        columns["workload"],
        *(columns[key] for key in _IDENTITY_COLUMN_KEYS),
    }
    if "prior_state" in columns:
        required_columns.add(columns["prior_state"])
    required_columns.update(predicate["field"] for predicate in selection["where"])
    missing = sorted(required_columns - fieldnames)
    if missing:
        raise ValueError("capacity evidence CSV is missing required columns: " + ", ".join(missing))

    selected = [
        row
        for row in rows
        if all(
            _predicate_matches(row[predicate["field"]], predicate)
            for predicate in selection["where"]
        )
    ]
    expected = selection.get("expected_cells")
    if expected is not None and len(selected) != expected:
        raise ValueError(f"selection expected {expected} capacity cells, found {len(selected)}")
    if not selected:
        raise ValueError("capacity evidence selection matched no cells")

    identities: list[tuple[str, str]] = []
    for row in selected:
        route_id = row[columns["route_id"]]
        workload = row[columns["workload"]]
        if not route_id or not workload:
            raise ValueError("selected capacity cells require nonempty route and workload values")
        identities.append((route_id, workload))
    if len(set(identities)) != len(identities):
        raise ValueError("capacity evidence selection contains duplicate route/workload cells")
    return sorted(
        selected,
        key=lambda row: (row[columns["route_id"]], row[columns["workload"]]),
    )


def _capacity_target_descriptors(
    route: RouteConfig,
    suite_name: str,
    suite_config: Mapping[str, Any],
    shape: str,
) -> tuple[str, str]:
    """Return exact, stable target descriptors for one capacity estimand.

    Ordinary shapes have one input/output target. ``mixed`` deliberately samples four
    deterministic subtypes, so its exact target is the sorted set of all possible realized targets
    under the bound route and suite configuration. Strings let the CSV contract represent both
    scalar and set-valued targets without lossy coercion.
    """

    def targets(workload_key: str) -> tuple[int, int]:
        spec = shape_spec(
            route,
            shape,
            f"capacity-recipe:{route.id}:{suite_name}:{shape}:{workload_key}",
            suite=suite_name,
            seed=1,
            workload_key=workload_key,
            shape_config=dict(suite_config),
        )
        return spec.planned_input_tokens, spec.max_output_tokens

    if shape != "mixed":
        input_tokens, output_tokens = targets(
            f"capacity-recipe:{{route}}:{suite_name}:{shape}"
        )
        return str(input_tokens), str(output_tokens)

    realized: set[tuple[int, int]] = set()
    for index in range(256):
        realized.add(targets(f"capacity-recipe:{{route}}:{suite_name}:mixed:{index}"))
        if len(realized) == 4:
            break
    if len(realized) != 4:
        raise AssertionError("mixed capacity recipe did not realize all four registered subtypes")
    return (
        canonical_json(sorted({input_tokens for input_tokens, _ in realized})),
        canonical_json(sorted({output_tokens for _, output_tokens in realized})),
    )


def capacity_workload_identity(
    route: RouteConfig,
    suite_name: str,
    suite_config: Mapping[str, Any],
    shape: str,
) -> dict[str, str]:
    """Build identity columns that reports must carry into closure evidence."""

    input_target, output_target = _capacity_target_descriptors(
        route, suite_name, suite_config, shape
    )
    recipe = {
        "schema": "capacity-workload-recipe/v1",
        "suite": suite_name,
        "shape": shape,
        "suite_config": copy.deepcopy(dict(suite_config)),
        "route_identity_sha256": route.identity_hash,
        "input_target": input_target,
        "output_target": output_target,
    }
    return {
        "route_identity_sha256": route.identity_hash,
        "workload_recipe_sha256": sha256_json(recipe),
        "input_target": input_target,
        "output_target": output_target,
    }


def export_controller_capacity_evidence(
    source_config_path: str | Path,
    controller_summary_path: str | Path,
    report_manifest_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Bind a terminal controller summary to its exact campaign and workload identities.

    The report deliberately contains no endpoint configuration.  This export joins it to the
    immutable source configuration without copying endpoint URLs, credentials, or headers into
    the closure evidence.  The adjacent manifest proves which terminal report and source config
    were used, so a later closure planner can reject stale or cross-campaign rows.
    """

    source_path = Path(source_config_path)
    summary_path = Path(controller_summary_path)
    report_path = Path(report_manifest_path)
    target = Path(output_path)
    source = load_config(source_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("campaign"), dict):
        raise ValueError("report manifest must contain a campaign mapping")
    campaign = report["campaign"]
    if campaign.get("identity_hash") != source.identity_hash:
        raise ValueError("report manifest campaign identity does not match source configuration")
    if not campaign.get("terminal_event") or not campaign.get("ended_at_utc"):
        raise ValueError("controller evidence requires a terminal report manifest")

    with summary_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    required = {"suite", "route_id", "shape", "controller_completion_state"}
    missing_columns = sorted(required - set(fieldnames))
    if missing_columns:
        raise ValueError(
            "controller summary is missing required columns: " + ", ".join(missing_columns)
        )
    if any(field in fieldnames for field in _IDENTITY_COLUMN_KEYS):
        raise ValueError("controller summary already contains reserved identity columns")

    routes = {route.id: route for route in source.routes}
    seen: set[tuple[str, str, str]] = set()
    bound_rows: list[dict[str, str]] = []
    for row in rows:
        suite = str(row.get("suite") or "")
        route_id = str(row.get("route_id") or "")
        shape = str(row.get("shape") or "")
        key = (suite, route_id, shape)
        if key in seen:
            raise ValueError(f"duplicate controller summary cell: {suite}:{route_id}:{shape}")
        seen.add(key)
        if suite not in _CAPACITY_SUITES or suite not in source.suites:
            raise ValueError(f"controller summary names an unknown capacity suite: {suite}")
        route = routes.get(route_id)
        if route is None:
            raise ValueError(f"controller summary names an unknown route: {route_id}")
        try:
            shape_spec(
                route,
                shape,
                f"controller-evidence:{route_id}:{suite}:{shape}",
                suite=suite,
                shape_config=dict(source.suites[suite]),
            )
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"controller summary names an invalid workload shape: {shape}"
            ) from exc
        identity = capacity_workload_identity(route, suite, source.suites[suite], shape)
        bound_rows.append(
            {
                **{field: str(row.get(field) or "") for field in fieldnames},
                **identity,
                "source_campaign_identity_sha256": source.identity_hash,
            }
        )
    if not bound_rows:
        raise ValueError("controller summary contains no capacity cells")
    bound_rows.sort(key=lambda row: (row["suite"], row["route_id"], row["shape"]))

    output_fields = [*fieldnames, *sorted(_IDENTITY_COLUMN_KEYS)]
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=output_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(bound_rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(target, buffer.getvalue())

    manifest_path = target.with_suffix(".manifest.json")
    manifest = {
        "schema": CONTROLLER_EVIDENCE_MANIFEST_SCHEMA,
        "source_config_sha256": _canonical_text_sha256(source_path),
        "source_campaign_identity_sha256": source.identity_hash,
        "controller_summary_sha256": _canonical_text_sha256(summary_path),
        "report_manifest_sha256": _canonical_text_sha256(report_path),
        "output_sha256": _canonical_text_sha256(target),
        "cell_count": len(bound_rows),
        "live_traffic_sent": False,
    }
    _write_text_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target, manifest_path


def _route_values(route: RouteConfig) -> dict[str, Any]:
    return {
        field: getattr(route, field)
        for field in _ROUTE_PREDICATE_FIELDS
    }


def _plain(value: object) -> object:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _route_document(route: RouteConfig) -> dict[str, Any]:
    return _plain(asdict(route))  # type: ignore[return-value]


def _campaign_document(config: CampaignConfig) -> dict[str, Any]:
    return {field: getattr(config, field) for field in _CAMPAIGN_FIELDS}


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_capacity_closure_package(
    base_config: CampaignConfig,
    capacity_csv: str | Path,
    profile: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Compile a deterministic, credential-free capacity-closure package.

    The planner has no provider catalog, report-vintage, workload-alias, or expected-cardinality
    knowledge. Those choices are explicit in ``profile``. It performs no network access, never
    reads a credential environment variable, and emits only the exact selected routes and one
    capacity suite. Live execution remains a separate explicit command.
    """

    validated_profile = validate_capacity_closure_profile(profile)
    capacity_path = Path(capacity_csv).resolve()
    destination = Path(output_dir).resolve()
    if not capacity_path.is_file():
        raise FileNotFoundError(capacity_path)

    selected = _select_evidence_rows(capacity_path, validated_profile)
    selection = validated_profile["selection"]
    columns = selection["columns"]
    workload_to_shape = validated_profile["mapping"]["workload_to_shape"]
    workloads = {row[columns["workload"]] for row in selected}
    unmapped = sorted(workloads - set(workload_to_shape))
    if unmapped:
        raise ValueError(
            "selected capacity workloads have no shape mapping: " + ", ".join(unmapped)
        )

    route_by_id = {route.id: route for route in base_config.routes}
    selected_route_ids = {row[columns["route_id"]] for row in selected}
    missing_routes = sorted(selected_route_ids - set(route_by_id))
    if missing_routes:
        raise ValueError(
            "capacity evidence references routes absent from the base campaign: "
            + ", ".join(missing_routes)
        )
    for route_id in sorted(selected_route_ids):
        route_values = _route_values(route_by_id[route_id])
        failed = [
            predicate
            for predicate in validated_profile["route_predicates"]
            if not _predicate_matches(route_values[predicate["field"]], predicate)
        ]
        if failed:
            raise ValueError(
                f"selected route {route_id} violates route_predicates: "
                + ", ".join(f"{item['field']} {item['op']}" for item in failed)
            )

    source_suite = base_config.suites.get(validated_profile["suite"])
    if source_suite is None:
        raise ValueError(
            f"base campaign does not contain source suite {validated_profile['suite']}"
        )
    identity_mismatches: list[str] = []
    for row in selected:
        route_id = row[columns["route_id"]]
        workload = row[columns["workload"]]
        shape = workload_to_shape[workload]
        expected_identity = capacity_workload_identity(
            route_by_id[route_id], validated_profile["suite"], source_suite, shape
        )
        expected_identity["source_campaign_identity_sha256"] = base_config.identity_hash
        for key in sorted(_IDENTITY_COLUMN_KEYS):
            observed = row[columns[key]].strip()
            expected_value = expected_identity[key]
            if observed != expected_value:
                identity_mismatches.append(
                    f"{route_id}/{workload}:{key} expected {expected_value!r}, "
                    f"observed {observed!r}"
                )
    if identity_mismatches:
        raise ValueError(
            "capacity evidence identity does not match the source campaign: "
            + "; ".join(identity_mismatches)
        )

    normalized_cells = [
        (
            row[columns["route_id"]],
            row[columns["workload"]],
            workload_to_shape[row[columns["workload"]]],
        )
        for row in selected
    ]
    runner_identities = [(route_id, shape) for route_id, _workload, shape in normalized_cells]
    if len(set(runner_identities)) != len(runner_identities):
        raise ValueError("workload mapping collapses distinct evidence cells onto one runner cell")

    suite_name = validated_profile["suite"]
    suite = copy.deepcopy(base_config.suites.get(suite_name, {}))
    for controlled in _CONTROLLED_SUITE_FIELDS:
        suite.pop(controlled, None)
    suite.update(copy.deepcopy(validated_profile["overrides"]["suite"]))
    suite.update(
        {
            "enabled": True,
            "shapes": sorted({shape for _route_id, _workload, shape in normalized_cells}),
            "cells": sorted(f"{route_id}:{shape}" for route_id, shape in runner_identities),
        }
    )

    campaign = _campaign_document(base_config)
    campaign.update(copy.deepcopy(validated_profile["overrides"]["campaign"]))
    routes = [
        _route_document(route)
        for route in base_config.routes
        if route.id in selected_route_ids
    ]
    document = {
        "campaign": campaign,
        "routes": routes,
        "suites": {suite_name: suite},
    }
    config_text = yaml.safe_dump(document, sort_keys=False, width=100)

    output = validated_profile["output"]
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / output["config_filename"]
    temporary_config = config_path.with_name(f".{config_path.name}.validation.tmp")
    try:
        temporary_config.write_text(config_text, encoding="utf-8", newline="\n")
        compiled = load_config(temporary_config)
    finally:
        temporary_config.unlink(missing_ok=True)

    compiled_cells = {
        (route.id, shape) for route, shape in selected_capacity_cells(compiled, suite_name)
    }
    if compiled_cells != set(runner_identities):
        raise AssertionError("compiled capacity closure does not match the selected evidence cells")
    if set(compiled.suites) != {suite_name}:
        raise AssertionError("compiled capacity closure contains an unexpected suite")
    compiled_route_by_id = {route.id: route for route in compiled.routes}
    for route_id in selected_route_ids:
        if asdict(compiled_route_by_id[route_id]) != asdict(route_by_id[route_id]):
            raise AssertionError(f"compiled capacity closure changed route identity: {route_id}")
    plan = build_plan(compiled).to_dict()
    if plan["static_requests"] != 0:
        raise AssertionError("capacity closure unexpectedly planned static requests")
    _write_text_atomic(config_path, config_text)

    prior_state_column = columns.get("prior_state")
    cells = [
        {
            "route_id": route_id,
            "source_workload": workload,
            "runner_shape": shape,
            "source_identity": {
                key: row[columns[key]] for key in sorted(_IDENTITY_COLUMN_KEYS)
            },
            **(
                {"prior_state": row[prior_state_column]}
                if prior_state_column is not None
                else {}
            ),
        }
        for row, (route_id, workload, shape) in zip(selected, normalized_cells, strict=True)
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "profile_schema": PROFILE_SCHEMA,
        "live_traffic_sent": False,
        "claim_boundary": (
            "evidence-driven capacity closure plan only; selected prior states are not endpoint "
            "failures and this package makes no new capacity claim before execution"
        ),
        "source": {
            "base_campaign_identity_sha256": base_config.identity_hash,
            "capacity_evidence_filename": capacity_path.name,
            "capacity_evidence_sha256": _canonical_text_sha256(capacity_path),
            "profile_sha256": hashlib.sha256(
                canonical_json(validated_profile).encode("utf-8")
            ).hexdigest(),
            "compiled_config_sha256": _canonical_text_sha256(config_path),
        },
        "selection": {
            "cell_count": len(cells),
            "route_count": len(selected_route_ids),
            "predicates": validated_profile["selection"]["where"],
            "route_predicates": validated_profile["route_predicates"],
            "workload_to_shape": workload_to_shape,
            "cells": cells,
        },
        "execution_contract": {
            "suite_name": suite_name,
            "retries": compiled.retries,
            "exact_evidence_cell_selection": True,
            "excluded_base_suites": sorted(set(base_config.suites) - {suite_name}),
            "suite_config": compiled.public_dict()["suites"][suite_name],
        },
        "campaign_public": compiled.public_dict(),
        "plan": plan,
    }
    manifest_path = destination / output["manifest_filename"]
    _write_text_atomic(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return config_path, manifest_path


def build_capacity_closure_package_from_files(
    base_config_path: str | Path,
    capacity_csv: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """File-oriented wrapper used by the generic command-line interface."""

    return build_capacity_closure_package(
        load_config(base_config_path),
        capacity_csv,
        load_capacity_closure_profile(profile_path),
        output_dir,
    )
