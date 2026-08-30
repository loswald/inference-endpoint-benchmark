"""Plan, run, resume, and report text-embedding endpoint benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .adapters.embeddings import (
    EmbeddingAdapter,
    embedding_adapter_for,
    embedding_adapter_plugin,
    materialize_embedding_request_for,
)
from .config import load_yaml_mapping
from .embedding_models import (
    EMBEDDING_PROFILE_SCHEMA,
    EmbeddingCampaignConfig,
    EmbeddingCapabilityContract,
    EmbeddingRequestSpec,
    EmbeddingRouteConfig,
)
from .models import AuthConfig, canonical_json, sha256_json

_PROFILE_KEYS = {"schema", "campaign", "route"}
_CAMPAIGN_KEYS = {"name", "max_cost_usd", "concurrency", "retries"}
_ROUTE_KEYS = {
    "id",
    "provider",
    "adapter",
    "model",
    "base_url",
    "auth",
    "region",
    "api_family",
    "billing_channel",
    "api_version",
    "model_version",
    "quota_scope",
    "capabilities",
    "input_usd_per_million",
    "documentation_source_url",
    "pricing_source_url",
    "evidence_retrieved_at_utc",
    "evidence_bundle_sha256",
    "request_timeout_seconds",
    "input_token_reservation_overhead",
    "http2",
    "connection_reuse",
    "transport_max_connections",
    "extra_headers",
}
_AUTH_KEYS = {"env", "header", "prefix"}
_CAPABILITY_KEYS = {
    "max_input_tokens_per_item",
    "max_batch_inputs",
    "max_total_tokens_per_request",
    "default_dimensions",
    "supported_dimensions",
    "empty_input",
    "unicode_input",
    "over_limit_input",
    "repeatability_cosine_minimum",
    "long_input_fraction",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _reject_unknown(scope: str, value: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {scope} field(s): {', '.join(unknown)}")


def embedding_config_from_mapping(raw: dict[str, Any]) -> EmbeddingCampaignConfig:
    _reject_unknown("embedding profile", raw, _PROFILE_KEYS)
    if raw.get("schema") != EMBEDDING_PROFILE_SCHEMA:
        raise ValueError(f"schema must be exactly {EMBEDDING_PROFILE_SCHEMA!r}")
    campaign = _mapping(raw.get("campaign"), "campaign")
    route = _mapping(raw.get("route"), "route")
    _reject_unknown("campaign", campaign, _CAMPAIGN_KEYS)
    _reject_unknown("route", route, _ROUTE_KEYS)
    auth = _mapping(route.get("auth"), "route.auth")
    capabilities = _mapping(route.get("capabilities"), "route.capabilities")
    _reject_unknown("route.auth", auth, _AUTH_KEYS)
    _reject_unknown("route.capabilities", capabilities, _CAPABILITY_KEYS)
    dimensions = capabilities.get("supported_dimensions")
    if not isinstance(dimensions, list):
        raise ValueError("supported_dimensions must be a list")
    route_config = EmbeddingRouteConfig(
        id=route.get("id"),
        provider=route.get("provider"),
        adapter=route.get("adapter"),
        model=route.get("model"),
        base_url=route.get("base_url"),
        auth=AuthConfig(
            env=auth.get("env"),
            header=auth.get("header", "Authorization"),
            prefix=auth.get("prefix", "Bearer "),
        ),
        region=route.get("region"),
        api_family=route.get("api_family", "embeddings"),
        billing_channel=route.get("billing_channel"),
        api_version=route.get("api_version"),
        model_version=route.get("model_version"),
        quota_scope=route.get("quota_scope"),
        capabilities=EmbeddingCapabilityContract(
            max_input_tokens_per_item=capabilities.get("max_input_tokens_per_item"),
            max_batch_inputs=capabilities.get("max_batch_inputs"),
            max_total_tokens_per_request=capabilities.get("max_total_tokens_per_request"),
            default_dimensions=capabilities.get("default_dimensions"),
            supported_dimensions=tuple(dimensions),
            empty_input=capabilities.get("empty_input", "documented_invalid"),
            unicode_input=capabilities.get("unicode_input", "unknown"),
            over_limit_input=capabilities.get(
                "over_limit_input", "documented_client_error"
            ),
            repeatability_cosine_minimum=capabilities.get(
                "repeatability_cosine_minimum", 0.999999
            ),
            long_input_fraction=capabilities.get("long_input_fraction", 0.90),
        ),
        input_usd_per_million=route.get("input_usd_per_million"),
        documentation_source_url=route.get("documentation_source_url"),
        pricing_source_url=route.get("pricing_source_url"),
        evidence_retrieved_at_utc=route.get("evidence_retrieved_at_utc"),
        evidence_bundle_sha256=route.get("evidence_bundle_sha256"),
        request_timeout_seconds=route.get("request_timeout_seconds", 180),
        input_token_reservation_overhead=route.get("input_token_reservation_overhead", 128),
        http2=route.get("http2", False),
        connection_reuse=route.get("connection_reuse", True),
        transport_max_connections=route.get("transport_max_connections", 64),
        extra_headers=dict(_mapping(route.get("extra_headers", {}), "route.extra_headers")),
    )
    # Resolve the typed plugin while still credential-free.  Unknown adapters and API-family
    # mismatches therefore fail at plan time, not after an output directory or spend claim exists.
    plugin = embedding_adapter_plugin(route_config.adapter)
    plugin.validate_route(route_config)
    return EmbeddingCampaignConfig(
        name=campaign.get("name"),
        route=route_config,
        max_cost_usd=campaign.get("max_cost_usd"),
        concurrency=campaign.get("concurrency", 8),
        retries=campaign.get("retries", 0),
    )


def load_embedding_config(path: str | Path) -> EmbeddingCampaignConfig:
    return embedding_config_from_mapping(
        load_yaml_mapping(path, document_name="embedding benchmark profile")
    )


def _token_like_text(target_tokens: int, nonce: str) -> str:
    if target_tokens <= 0:
        return ""
    marker = hashlib.sha256(nonce.encode()).hexdigest()[:12]
    return f"benchmark-{marker}" + " token" * max(0, target_tokens - 1)


def _spec(
    route: EmbeddingRouteConfig,
    cell: str,
    inputs: tuple[str, ...],
    planned_tokens: int,
    *,
    dimensions: int | None = None,
    expectation: str = "success",
    encoding: str | None = None,
    repeatability_pairs: tuple[tuple[int, int], ...] = (),
    repeatability_group: str | None = None,
    repeatability_ordinal: int | None = None,
) -> EmbeddingRequestSpec:
    material = {
        "route_identity_sha256": route.identity_hash,
        "cell": cell,
        "input_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in inputs],
        "planned_tokens": planned_tokens,
        "dimensions": dimensions,
        "expectation": expectation,
        "repeatability_group": repeatability_group,
        "repeatability_ordinal": repeatability_ordinal,
    }
    logical_id = f"embedding-{cell}-{sha256_json(material)[:16]}"
    return EmbeddingRequestSpec(
        logical_id=logical_id,
        route_id=route.id,
        cell_id=cell,
        inputs=inputs,
        planned_input_tokens=planned_tokens,
        dimensions=dimensions,
        expectation=expectation,  # type: ignore[arg-type]
        input_encoding=encoding or ("single_string" if len(inputs) == 1 else "string_array"),  # type: ignore[arg-type]
        repeatability_pairs=repeatability_pairs,
        repeatability_group=repeatability_group,
        repeatability_ordinal=repeatability_ordinal,
    )


def plan_embedding_requests(route: EmbeddingRouteConfig) -> tuple[EmbeddingRequestSpec, ...]:
    capability = route.capabilities
    adapter = embedding_adapter_plugin(route.adapter)
    short = "Portable embedding benchmark: semantic retrieval systems need stable vectors."
    specs: list[EmbeddingRequestSpec] = [
        _spec(route, "single-short", (short,), 16, dimensions=capability.default_dimensions),
        _spec(
            route,
            "unicode-multilingual",
            ("多语言检索 — café — مرحبا — नमस्ते — 🧭",),
            24,
            dimensions=capability.default_dimensions,
        ),
    ]
    if adapter.input_cardinality == "one_or_many":
        specs.append(
            _spec(
                route,
                "repeatability-pair",
                (short, short),
                32,
                dimensions=capability.default_dimensions,
                encoding="string_array",
                repeatability_pairs=((0, 1),),
            )
        )
    else:
        for ordinal in (1, 2):
            specs.append(
                _spec(
                    route,
                    f"repeatability-call-{ordinal}",
                    (short,),
                    16,
                    dimensions=capability.default_dimensions,
                    repeatability_group="repeatability-exact-across-requests",
                    repeatability_ordinal=ordinal,
                )
            )
    long_target = math.floor(
        capability.max_input_tokens_per_item * capability.long_input_fraction
    )
    specs.append(
        _spec(
            route,
            "single-near-documented-limit",
            (_token_like_text(long_target, f"{route.identity_hash}:near-limit"),),
            long_target,
            dimensions=capability.default_dimensions,
        )
    )
    if adapter.input_cardinality == "one_or_many":
        batch_sizes = sorted(
            {
                size
                for size in (
                    2,
                    max(2, capability.max_batch_inputs // 2),
                    capability.max_batch_inputs,
                )
                if size <= capability.max_batch_inputs
            }
        )
        for size in batch_sizes:
            inputs = tuple(f"batch item {index}" for index in range(size))
            specs.append(
                _spec(
                    route,
                    f"batch-{size}",
                    inputs,
                    4 * size,
                    dimensions=capability.default_dimensions,
                    encoding="string_array",
                )
            )
    minimum_dimensions = min(capability.supported_dimensions)
    if minimum_dimensions != capability.default_dimensions:
        specs.append(
            _spec(
                route,
                f"dimensions-{minimum_dimensions}",
                (short,),
                16,
                dimensions=minimum_dimensions,
            )
        )
    if capability.empty_input == "documented_invalid":
        specs.append(
            _spec(
                route,
                "invalid-empty-input",
                ("",),
                0,
                dimensions=capability.default_dimensions,
                expectation="client_error",
            )
        )
    if adapter.input_cardinality == "one_or_many":
        over_batch = tuple(
            f"over batch item {index}" for index in range(capability.max_batch_inputs + 1)
        )
        specs.append(
            _spec(
                route,
                "invalid-batch-over-documented-max",
                over_batch,
                4 * len(over_batch),
                dimensions=capability.default_dimensions,
                expectation="client_error",
                encoding="string_array",
            )
        )
    if capability.over_limit_input != "unknown":
        specs.append(
            _spec(
                route,
                "item-over-documented-max",
                (
                    _token_like_text(
                        capability.max_input_tokens_per_item + 1,
                        f"{route.identity_hash}:over-limit",
                    ),
                ),
                capability.max_input_tokens_per_item + 1,
                dimensions=capability.default_dimensions,
                expectation=(
                    "client_error"
                    if capability.over_limit_input == "documented_client_error"
                    else "success_with_truncation"
                ),
            )
        )
    logical_ids = [spec.logical_id for spec in specs]
    if len(set(logical_ids)) != len(logical_ids):
        raise RuntimeError("embedding planner produced duplicate logical IDs")
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class EmbeddingPlan:
    campaign_identity_sha256: str
    route_identity_sha256: str
    adapter_identity: dict[str, Any]
    requests: tuple[dict[str, Any], ...]
    derived_cells: tuple[dict[str, Any], ...]
    physical_attempts_upper_bound: int
    worst_case_cost_usd: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema": "embedding-plan/v1",
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "route_identity_sha256": self.route_identity_sha256,
            "adapter_identity": self.adapter_identity,
            "request_count": len(self.requests),
            "derived_cell_count": len(self.derived_cells),
            "physical_attempts_upper_bound": self.physical_attempts_upper_bound,
            "worst_case_cost_usd": self.worst_case_cost_usd,
            "requests": list(self.requests),
            "derived_cells": list(self.derived_cells),
            "privacy": {
                "input_text_retained": False,
                "embedding_vectors_retained": False,
                "input_and_vector_sha256_only": True,
            },
        }


def build_embedding_plan(config: EmbeddingCampaignConfig) -> EmbeddingPlan:
    specs = plan_embedding_requests(config.route)
    rows: list[dict[str, Any]] = []
    cost = 0.0
    for spec in specs:
        payload = materialize_embedding_request_for(config.route, spec)
        reserved_tokens = payload.input_token_upper_bound
        reserved_cost = config.route.reserved_cost(reserved_tokens)
        cost += reserved_cost * (config.retries + 1)
        rows.append(
            {
                **spec.public_dict(),
                "wire_body_sha256": payload.wire_body_sha256,
                "bound_payload_sha256": payload.bound_payload_sha256,
                "payload_generator_version": payload.generator_version,
                "reserved_input_tokens": reserved_tokens,
                "reserved_cost_usd_per_attempt": reserved_cost,
            }
        )
    repeatability_groups: dict[str, list[EmbeddingRequestSpec]] = {}
    for spec in specs:
        if spec.repeatability_group is not None:
            repeatability_groups.setdefault(spec.repeatability_group, []).append(spec)
    derived_cells: list[dict[str, Any]] = []
    for group, members in sorted(repeatability_groups.items()):
        ordered = sorted(members, key=lambda item: item.repeatability_ordinal or 0)
        if len(ordered) != 2 or [item.repeatability_ordinal for item in ordered] != [1, 2]:
            raise ValueError("cross-request repeatability groups require exactly ordinals 1 and 2")
        derived_cells.append(
            {
                "cell_id": group,
                "kind": "cross_request_exact_vector_equality",
                "member_logical_ids": [item.logical_id for item in ordered],
            }
        )
    if cost > config.max_cost_usd:
        raise ValueError(
            f"embedding plan worst-case cost ${cost:.6f} exceeds cap ${config.max_cost_usd:.6f}"
        )
    return EmbeddingPlan(
        campaign_identity_sha256=config.identity_hash,
        route_identity_sha256=config.route.identity_hash,
        adapter_identity=embedding_adapter_plugin(config.route.adapter).public_identity(),
        requests=tuple(rows),
        derived_cells=tuple(derived_cells),
        physical_attempts_upper_bound=len(rows) * (config.retries + 1),
        worst_case_cost_usd=cost,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class EmbeddingJournal:
    """Append-only idempotency journal; a claimed request is never replayed after ambiguity."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid embedding journal JSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"embedding journal line {line_number} is not an object")
            rows.append(value)
        return rows

    async def append(self, value: dict[str, Any]) -> None:
        serialized = canonical_json(value) + "\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())


def embedding_request_id(
    config: EmbeddingCampaignConfig,
    spec: EmbeddingRequestSpec,
    bound_payload_sha256: str,
    attempt: int,
) -> str:
    return sha256_json(
        {
            "schema": "embedding-request-id/v1",
            "campaign_identity_sha256": config.identity_hash,
            "route_identity_sha256": config.route.identity_hash,
            "logical_id": spec.logical_id,
            "bound_payload_sha256": bound_payload_sha256,
            "attempt": attempt,
        }
    )


async def run_embedding_campaign(
    config: EmbeddingCampaignConfig,
    output_dir: str | Path,
    *,
    adapter: EmbeddingAdapter | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    plan = build_embedding_plan(config)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "embedding-plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan.public_dict():
            raise ValueError("output directory is bound to a different embedding plan")
    else:
        _atomic_json(plan_path, plan.public_dict())
    journal = EmbeddingJournal(output / "embedding-events.jsonl")
    existing_rows = journal.rows()
    settled = {
        str(row["request_id"])
        for row in existing_rows
        if row.get("kind") == "request_settled"
    }
    claimed = {
        str(row["request_id"])
        for row in existing_rows
        if row.get("kind") == "request_claimed"
    }
    implementation = adapter or embedding_adapter_for(config.route)
    implementation.preflight(config.route)
    semaphore = asyncio.Semaphore(config.concurrency)
    planned_by_logical = {row["logical_id"]: row for row in plan.requests}

    async def execute(spec: EmbeddingRequestSpec) -> None:
        prepared = implementation.prepare(config.route, spec)
        plan_row = planned_by_logical[spec.logical_id]
        if prepared.payload.bound_payload_sha256 != plan_row["bound_payload_sha256"]:
            raise RuntimeError("prepared embedding payload differs from the registered plan")
        for attempt in range(1, config.retries + 2):
            request_id = embedding_request_id(
                config, spec, prepared.payload.bound_payload_sha256, attempt
            )
            if request_id in settled:
                return
            if request_id in claimed:
                # The prior process may have sent it.  Fail closed rather than double-billing or
                # silently manufacturing repeatability evidence from a replay.
                return
            reserved_tokens = prepared.payload.input_token_upper_bound
            reserved_cost = config.route.reserved_cost(reserved_tokens)
            await journal.append(
                {
                    "schema": "embedding-event/v1",
                    "kind": "request_claimed",
                    "recorded_at_utc": _utc_now(),
                    "request_id": request_id,
                    "attempt": attempt,
                    "logical_id": spec.logical_id,
                    "route_identity_sha256": config.route.identity_hash,
                    "wire_body_sha256": prepared.payload.wire_body_sha256,
                    "bound_payload_sha256": prepared.payload.bound_payload_sha256,
                    "payload_generator_version": prepared.payload.generator_version,
                    "reserved_input_tokens": reserved_tokens,
                    "reserved_cost_usd": reserved_cost,
                    "request": spec.public_dict(),
                }
            )
            claimed.add(request_id)
            async with semaphore:
                result = await implementation.send_prepared(config.route, spec, prepared)
            if result.prompt_tokens is not None:
                cost = config.route.reserved_cost(result.prompt_tokens)
                cost_basis = "provider_usage"
            else:
                cost = reserved_cost
                cost_basis = "reserved_upper_bound"
            await journal.append(
                {
                    "schema": "embedding-event/v1",
                    "kind": "request_settled",
                    "recorded_at_utc": _utc_now(),
                    "request_id": request_id,
                    "attempt": attempt,
                    "logical_id": spec.logical_id,
                    "result": result.public_dict(),
                    "cost_usd": cost,
                    "cost_basis": cost_basis,
                }
            )
            settled.add(request_id)
            explicitly_retryable = result.status == "rate_limited" or (
                result.status == "server_error"
                and result.http_status is not None
                and result.http_status >= 500
            )
            if not explicitly_retryable:
                return

    await asyncio.gather(*(execute(spec) for spec in plan_embedding_requests(config.route)))
    await implementation.close()
    report = build_embedding_report(plan.public_dict(), journal.rows())
    _atomic_json(output / "embedding-report.json", report)
    (output / "embedding-report.md").write_text(
        render_embedding_report_markdown(report), encoding="utf-8"
    )
    return report


def build_embedding_report(
    plan: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    claims_by_id = {
        str(row["request_id"]): row
        for row in events
        if row.get("kind") == "request_claimed"
    }
    settled_by_id = {
        str(row["request_id"]): row
        for row in events
        if row.get("kind") == "request_settled"
    }
    settled_by_logical: dict[str, dict[str, Any]] = {}
    for row in settled_by_id.values():
        settled_by_logical[str(row["logical_id"])] = row
    ambiguous_logical = {
        str(row["logical_id"])
        for request_id, row in claims_by_id.items()
        if request_id not in settled_by_id
    }
    coverage: list[dict[str, Any]] = []
    for planned in plan.get("requests", []):
        logical_id = str(planned["logical_id"])
        settled = settled_by_logical.get(logical_id)
        expectation = planned["expectation"]
        if settled is None:
            state = "ambiguous_unsettled_claim" if logical_id in ambiguous_logical else "not_run"
            coverage.append(
                {
                    "logical_id": logical_id,
                    "cell_id": planned["cell_id"],
                    "expectation": expectation,
                    "state": state,
                    "result": None,
                    "cost_usd": None,
                    "cost_basis": None,
                }
            )
            continue
        result = settled["result"]
        if expectation == "success":
            passed = result["status"] == "success" and not result["validation_errors"]
        elif expectation == "client_error":
            passed = result["status"] == "client_error"
        else:
            passed = (
                result["status"] == "success"
                and not result["validation_errors"]
                and result.get("truncated") is True
            )
        coverage.append(
            {
                "logical_id": logical_id,
                "cell_id": planned["cell_id"],
                "expectation": expectation,
                "state": "passed" if passed else "measured_nonpass",
                "result": result,
                "cost_usd": settled["cost_usd"],
                "cost_basis": settled["cost_basis"],
            }
        )
    for derived in plan.get("derived_cells", []):
        member_ids = [str(value) for value in derived.get("member_logical_ids", [])]
        members = [settled_by_logical.get(logical_id) for logical_id in member_ids]
        if any(member is None for member in members):
            state = (
                "ambiguous_unsettled_claim"
                if any(logical_id in ambiguous_logical for logical_id in member_ids)
                else "not_run"
            )
            result = None
        else:
            member_results = [member["result"] for member in members if member is not None]
            vector_hashes = [
                result_value["vectors"][0]["vector_sha256"]
                for result_value in member_results
                if result_value.get("status") == "success"
                and not result_value.get("validation_errors")
                and len(result_value.get("vectors", [])) == 1
            ]
            exact_match = len(vector_hashes) == 2 and len(set(vector_hashes)) == 1
            state = "passed" if exact_match else "measured_nonpass"
            result = {
                "comparison_kind": derived.get("kind"),
                "member_count": len(member_ids),
                "vector_hashes_equal": exact_match,
            }
        coverage.append(
            {
                "logical_id": f"derived:{derived['cell_id']}",
                "cell_id": derived["cell_id"],
                "expectation": "exact_vector_equality",
                "state": state,
                "result": result,
                "cost_usd": None,
                "cost_basis": "derived_from_member_requests",
            }
        )
    state_counts: dict[str, int] = {}
    for row in coverage:
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
    return {
        "schema": "embedding-report/v1",
        "campaign_identity_sha256": plan.get("campaign_identity_sha256"),
        "route_identity_sha256": plan.get("route_identity_sha256"),
        "generated_at_utc": _utc_now(),
        "planned_cells": len(coverage),
        "state_counts": state_counts,
        "settled_cost_usd": sum(
            float(row["cost_usd"]) for row in coverage if row["cost_usd"] is not None
        ),
        "privacy": plan.get("privacy"),
        "coverage": coverage,
    }


def render_embedding_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text-embedding endpoint benchmark",
        "",
        "This report tests the embedding API itself: accepted input shapes, documented limits, "
        "returned vector dimensions and norms, usage accounting, and same-request repeatability. "
        "It stores neither input text nor embedding vectors.",
        "",
        f"Planned cells: {report['planned_cells']}. Conservative settled exposure: "
        f"${report['settled_cost_usd']:.6f}.",
        "",
        "| Workload | Expected | Result | HTTP | Dimensions | Input tokens | Cost basis |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["coverage"]:
        result = row.get("result") or {}
        dimensions = sorted(
            {value["dimensions"] for value in result.get("vectors", [])}
        )
        dimension_text = ", ".join(str(value) for value in dimensions) or "—"
        lines.append(
            (
                "| {cell} | {expectation} | {state} | {http} | {dimensions} | "
                "{tokens} | {basis} |"
            ).format(
                cell=row["cell_id"],
                expectation=row["expectation"],
                state=row["state"],
                http=result.get("http_status", "—"),
                dimensions=dimension_text,
                tokens=result.get("prompt_tokens", "—"),
                basis=row.get("cost_basis") or "—",
            )
        )
    lines.extend(
        [
            "",
            "`measured_nonpass` is a result, not missing data. `not_run` means no request was "
            "claimed. `ambiguous_unsettled_claim` means a prior process may have sent the exact "
            "request, so the resumable runner correctly refused to replay it.",
            "",
        ]
    )
    return "\n".join(lines)


def write_embedding_plan(config: EmbeddingCampaignConfig, path: str | Path) -> EmbeddingPlan:
    plan = build_embedding_plan(config)
    _atomic_json(Path(path), plan.public_dict())
    return plan


def load_embedding_events(path: str | Path) -> list[dict[str, Any]]:
    return EmbeddingJournal(Path(path)).rows()


def write_embedding_report(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    plan = json.loads((output / "embedding-plan.json").read_text(encoding="utf-8"))
    events = load_embedding_events(output / "embedding-events.jsonl")
    report = build_embedding_report(plan, events)
    _atomic_json(output / "embedding-report.json", report)
    (output / "embedding-report.md").write_text(
        render_embedding_report_markdown(report), encoding="utf-8"
    )
    return report


def profile_sha256(path: str | Path) -> str:
    raw = load_yaml_mapping(path, document_name="embedding benchmark profile")
    return hashlib.sha256(yaml.safe_dump(raw, sort_keys=True).encode()).hexdigest()


__all__ = [
    "EmbeddingJournal",
    "EmbeddingPlan",
    "build_embedding_plan",
    "build_embedding_report",
    "embedding_config_from_mapping",
    "embedding_request_id",
    "load_embedding_config",
    "plan_embedding_requests",
    "render_embedding_report_markdown",
    "run_embedding_campaign",
    "write_embedding_plan",
    "write_embedding_report",
]
