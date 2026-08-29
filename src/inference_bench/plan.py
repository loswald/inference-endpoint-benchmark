from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapters import validate_adapter_route
from .config import (
    NATIVE_PLACEHOLDER_ADAPTERS,
    CampaignConfig,
    selected_capacity_cells,
    validate_route_evidence_identity,
)
from .load import (
    adaptive_baseline_epoch_id,
    aimd_confirmation_epoch_id,
    aimd_max_rps,
    aimd_separator_epoch_id,
    baseline_attempt_count,
    baseline_design,
    next_healthy_aimd_rate,
    scheduled_offsets,
    soak_block_epoch_id,
    soak_rate_rps,
    validate_aimd_config,
    validate_soak_config,
)
from .models import RouteConfig
from .payload import reserved_input_tokens
from .workloads import plan_static_suites, shape_spec


@dataclass(frozen=True, slots=True)
class PlanSummary:
    campaign_hash: str
    static_requests: int
    load_requests_upper_path: int
    load_arrival_window_seconds_sequential_upper_path: float
    request_timeout_seconds_by_route: dict[str, float]
    max_single_request_timeout_seconds: float
    total_requests_upper_path: int
    max_attempts_per_logical_request: int
    physical_attempts_upper_bound: int
    static_worst_case_usd: float
    load_worst_case_usd_upper_path: float
    total_worst_case_usd_upper_path: float
    launch_budget_usd: float
    native_placeholder_routes: tuple[str, ...]
    coverage_cells: tuple[dict[str, Any], ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_hash": self.campaign_hash,
            "static_requests": self.static_requests,
            "load_requests_upper_path": self.load_requests_upper_path,
            "load_arrival_window_seconds_sequential_upper_path": (
                self.load_arrival_window_seconds_sequential_upper_path
            ),
            "request_timeout_seconds_by_route": self.request_timeout_seconds_by_route,
            "max_single_request_timeout_seconds": self.max_single_request_timeout_seconds,
            "total_requests_upper_path": self.total_requests_upper_path,
            "max_attempts_per_logical_request": self.max_attempts_per_logical_request,
            "physical_attempts_upper_bound": self.physical_attempts_upper_bound,
            "static_worst_case_usd": self.static_worst_case_usd,
            "load_worst_case_usd_upper_path": self.load_worst_case_usd_upper_path,
            "total_worst_case_usd_upper_path": self.total_worst_case_usd_upper_path,
            "launch_budget_usd": self.launch_budget_usd,
            "native_placeholder_routes": list(self.native_placeholder_routes),
            "coverage_cells": list(self.coverage_cells),
            "notes": list(self.notes),
        }


def build_plan(config: CampaignConfig) -> PlanSummary:
    # Adapter discovery and API-family compatibility are credential-free structural checks. They
    # must fail before suite planning or any live command creates an output directory.
    for route in config.routes:
        validate_adapter_route(route)
    static_specs = plan_static_suites(config.routes, config.suites, seed=config.seed)
    route_by_id = {route.id: route for route in config.routes}
    attempts_per_logical = config.retries + 1
    coverage_cells: list[dict[str, Any]] = [
        {
            "plan_cell_id": f"request:{spec.logical_id}",
            "logical_id": spec.logical_id,
            "route_id": spec.route_id,
            "suite": spec.suite,
            "cell_id": spec.cell_id,
            "planned_disposition": "required",
        }
        for spec in static_specs
    ]
    context_suite = config.suites.get("context")
    if context_suite and context_suite.get("enabled", True):
        for route in config.routes:
            if route.context_tokens is None:
                coverage_cells.append(
                    {
                        "plan_cell_id": f"context-config-missing:{route.id}",
                        "logical_id": None,
                        "route_id": route.id,
                        "suite": "context",
                        "cell_id": "documented_context_limit_missing",
                        "planned_disposition": "configuration_required",
                        "initial_state": "inconclusive",
                        "reason": "route_context_tokens_not_configured",
                    }
                )
    output_suite = config.suites.get("output")
    if output_suite and output_suite.get("enabled", True):
        for route in config.routes:
            if route.max_output_tokens is None:
                coverage_cells.append(
                    {
                        "plan_cell_id": f"output-config-missing:{route.id}",
                        "logical_id": None,
                        "route_id": route.id,
                        "suite": "output",
                        "cell_id": "documented_output_limit_missing",
                        "planned_disposition": "configuration_required",
                        "initial_state": "inconclusive",
                        "reason": "route_max_output_tokens_not_configured",
                    }
                )
    static_cost = attempts_per_logical * sum(
        route_by_id[spec.route_id].worst_case_cost(
            reserved_input_tokens(
                route_by_id[spec.route_id], spec, config.input_token_reservation_factor
            ),
            spec.max_output_tokens,
        )
        for spec in static_specs
    )
    for route in config.routes:
        if route.adapter not in NATIVE_PLACEHOLDER_ADAPTERS:
            validate_route_evidence_identity(config, route)
    load_count = 0
    load_cost = 0.0
    load_arrival_window_seconds = 0.0
    shapes = ["short_short", "long_short", "short_long", "mixed"]
    aimd = config.suites.get("aimd")
    soak = config.suites.get("soak")
    aimd_cells = {(route.id, shape) for route, shape in selected_capacity_cells(config, "aimd")}
    soak_cells = {(route.id, shape) for route, shape in selected_capacity_cells(config, "soak")}
    if aimd and aimd.get("enabled", True):
        validate_aimd_config(aimd, config.concurrency)
    if soak and soak.get("enabled", True):
        validate_soak_config(soak, config.concurrency)
    for route in config.routes:
        if aimd and aimd.get("enabled", True):
            epochs = int(aimd.get("epochs", 12))
            seconds = float(aimd.get("epoch_seconds", 20))
            additive = float(aimd.get("additive_rps", 0.25))
            bracket_epochs = int(aimd.get("bracket_epochs", min(6, epochs)))
            bracket_multiplier = float(aimd.get("bracket_multiplier", 2.0))
            for shape in aimd.get("shapes", shapes):
                if (route.id, shape) not in aimd_cells:
                    continue
                max_rps = aimd_max_rps(aimd, shape)
                rate = float(aimd.get("initial_rps", 0.25))
                shape_cost = _shape_cost(
                    route,
                    shape,
                    config.input_token_reservation_factor,
                    shape_config=aimd,
                )
                baseline_samples, baseline_seconds, _baseline_rate = baseline_design(
                    aimd, seconds, default_rps=min(rate, 0.1)
                )
                baseline_attempts = baseline_attempt_count(
                    aimd, _baseline_rate, field_prefix="aimd"
                )
                baseline_decrease = float(aimd.get("baseline_multiplicative_decrease", 0.5))
                minimum_rps = float(aimd.get("minimum_rps", 0.01))
                for attempt in range(baseline_attempts):
                    attempt_rate = max(
                        minimum_rps, _baseline_rate * baseline_decrease**attempt
                    )
                    attempt_seconds = max(baseline_seconds, baseline_samples / attempt_rate)
                    load_arrival_window_seconds += attempt_seconds
                    baseline_id = adaptive_baseline_epoch_id("aimd", route.id, shape, attempt)
                    coverage_cells.append(
                        _load_plan_cell(
                            route,
                            shape,
                            baseline_id,
                            "baseline",
                            planned_disposition=(
                                "required"
                                if attempt == 0
                                else "conditional_on_prior_baseline_failure"
                            ),
                            shape_config=aimd,
                        )
                    )
                    n = baseline_samples
                    load_count += n
                    load_cost += n * shape_cost * attempts_per_logical
                best = 0.0
                for index in range(epochs):
                    epoch_id = f"aimd-{route.id}-{shape}-{index:03d}"
                    coverage_cells.append(
                        _load_plan_cell(route, shape, epoch_id, "aimd", shape_config=aimd)
                    )
                    n = len(
                        scheduled_offsets(
                            rate,
                            seconds,
                            seed=config.seed,
                            epoch_id=epoch_id,
                        )
                    )
                    load_count += n
                    load_cost += n * shape_cost * attempts_per_logical
                    load_arrival_window_seconds += seconds
                    best = max(best, rate)
                    rate = next_healthy_aimd_rate(
                        rate,
                        healthy_increases=index,
                        overload_observed=False,
                        additive_rps=additive,
                        bracket_epochs=bracket_epochs,
                        bracket_multiplier=bracket_multiplier,
                        max_rps=max_rps,
                    )
                confirmation_stages = int(aimd.get("confirmation_max_stages", 4))
                separator_samples = int(
                    aimd.get("confirmation_separator_samples", baseline_samples)
                )
                separator_seconds = max(
                    seconds, separator_samples / _baseline_rate
                )
                confirmation_decrease = float(
                    aimd.get(
                        "confirmation_multiplicative_decrease",
                        aimd.get("multiplicative_decrease", 0.5),
                    )
                )
                confirmation_rate = max(minimum_rps, best or minimum_rps)
                for stage in range(confirmation_stages):
                    for confirmation in range(3):
                        confirmation_id = aimd_confirmation_epoch_id(
                            route.id, shape, stage, confirmation
                        )
                        coverage_cells.append(
                            _load_plan_cell(
                                route,
                                shape,
                                confirmation_id,
                                "confirmation",
                                planned_disposition=(
                                    "required"
                                    if stage == 0
                                    else "conditional_on_prior_confirmation_failure"
                                ),
                                shape_config=aimd,
                            )
                        )
                        n = len(
                            scheduled_offsets(
                                confirmation_rate,
                                seconds,
                                seed=config.seed + stage * 1000 + confirmation + 1,
                                epoch_id=confirmation_id,
                            )
                        )
                        load_count += n
                        load_cost += n * shape_cost * attempts_per_logical
                        load_arrival_window_seconds += seconds
                        if confirmation < 2:
                            separator_id = aimd_separator_epoch_id(
                                route.id, shape, stage, confirmation
                            )
                            coverage_cells.append(
                                _load_plan_cell(
                                    route,
                                    shape,
                                    separator_id,
                                    "confirmation_separator",
                                    planned_disposition=(
                                        "required"
                                        if stage == 0
                                        else "conditional_on_prior_confirmation_failure"
                                    ),
                                    shape_config=aimd,
                                )
                            )
                            n = separator_samples
                            load_count += n
                            load_cost += n * shape_cost * attempts_per_logical
                            load_arrival_window_seconds += separator_seconds
                    confirmation_rate = max(minimum_rps, confirmation_rate * confirmation_decrease)
                # Recovery is conditional on an observed two-epoch overload at runtime. Include
                # its maximal possible schedule in the upper bound even though the all-healthy
                # trajectory itself would omit it.
                recovery_id = f"aimd-{route.id}-{shape}-recovery"
                coverage_cells.append(
                    _load_plan_cell(
                        route,
                        shape,
                        recovery_id,
                        "recovery_after_observed_overload",
                        planned_disposition="conditional_on_overload",
                        shape_config=aimd,
                    )
                )
                n = len(
                    scheduled_offsets(
                        best * 0.5,
                        seconds,
                        seed=config.seed + 100,
                        epoch_id=recovery_id,
                    )
                )
                load_count += n
                load_cost += n * shape_cost * attempts_per_logical
                load_arrival_window_seconds += seconds
        if soak and soak.get("enabled", True):
            blocks = int(soak.get("blocks", 4))
            seconds = float(soak.get("block_seconds", 30))
            for shape in soak.get("shapes", shapes):
                if (route.id, shape) not in soak_cells:
                    continue
                rate = soak_rate_rps(soak, route.id, shape)
                shape_cost = _shape_cost(
                    route,
                    shape,
                    config.input_token_reservation_factor,
                    shape_config=soak,
                )
                baseline_samples, baseline_seconds, _baseline_rate = baseline_design(
                    soak, seconds, default_rps=min(rate, 0.1)
                )
                baseline_attempts = baseline_attempt_count(
                    soak, _baseline_rate, field_prefix="soak"
                )
                baseline_decrease = float(soak.get("baseline_multiplicative_decrease", 0.5))
                minimum_rps = float(soak.get("minimum_rps", 0.01))
                for attempt in range(baseline_attempts):
                    attempt_rate = max(
                        minimum_rps, _baseline_rate * baseline_decrease**attempt
                    )
                    attempt_seconds = max(baseline_seconds, baseline_samples / attempt_rate)
                    load_arrival_window_seconds += attempt_seconds
                    baseline_id = adaptive_baseline_epoch_id("soak", route.id, shape, attempt)
                    coverage_cells.append(
                        _load_plan_cell(
                            route,
                            shape,
                            baseline_id,
                            "soak_baseline",
                            planned_disposition=(
                                "required"
                                if attempt == 0
                                else "conditional_on_prior_baseline_failure"
                            ),
                            shape_config=soak,
                        )
                    )
                    n = baseline_samples
                    load_count += n
                    load_cost += n * shape_cost * attempts_per_logical
                rate_stages = int(soak.get("max_rate_stages", 4))
                rate_decrease = float(soak.get("rate_multiplicative_decrease", 0.5))
                stage_rate = max(minimum_rps, rate)
                for stage in range(rate_stages):
                    for block in range(blocks):
                        epoch_id = soak_block_epoch_id(route.id, shape, stage, block)
                        coverage_cells.append(
                            _load_plan_cell(
                                route,
                                shape,
                                epoch_id,
                                "soak_block",
                                planned_disposition=(
                                    "required"
                                    if stage == 0
                                    else "conditional_on_prior_soak_failure"
                                ),
                                shape_config=soak,
                            )
                        )
                        n = len(
                            scheduled_offsets(
                                stage_rate,
                                seconds,
                                seed=config.seed + stage * 1000 + block,
                                epoch_id=epoch_id,
                            )
                        )
                        load_count += n
                        load_cost += n * shape_cost * attempts_per_logical
                        load_arrival_window_seconds += seconds
                    stage_rate = max(minimum_rps, stage_rate * rate_decrease)
    placeholders = tuple(
        route.id for route in config.routes if route.adapter in NATIVE_PLACEHOLDER_ADAPTERS
    )
    total = static_cost + load_cost
    notes = [
        "Costs include every allowed physical retry attempt at the full prompt/output ceiling.",
        "AIMD upper path assumes every geometric-bracket/additive epoch is healthy; actual "
        "offered rate adapts downward.",
        "Mixed workload cost uses its most expensive possible subtype, not one sampled subtype.",
        "Runtime reserves each deterministic send atomically and stops before the launch guard.",
        "Each provider send and complete response stream has the route-specific full-stream "
        "timeout reported in request_timeout_seconds_by_route; a final request may drain for up "
        "to that interval after its arrival.",
    ]
    if total > config.max_cost_usd - config.launch_reserve_usd:
        notes.append(
            "Upper-path plan exceeds launch budget; later cells can be honestly budget-censored."
        )
    if load_arrival_window_seconds > config.max_wall_seconds - config.launch_reserve_seconds:
        notes.append(
            "Sequential load arrival windows alone exceed the launchable wall-time budget; later "
            "load cells will be time-censored unless the configuration is changed."
        )
    max_request_timeout = max(route.request_timeout_seconds for route in config.routes)
    if max_request_timeout > config.max_wall_seconds - config.launch_reserve_seconds:
        notes.append(
            "At least one route timeout exceeds the launchable campaign wall interval; that route "
            "cannot launch unless the wall cap or route timeout is changed."
        )
    variation = config.suites.get("time_variation")
    if variation and variation.get("interleave_gap_work", False):
        for cell in coverage_cells:
            if cell.get("suite") != "time_variation":
                cell["planned_disposition"] = "optional_gap_closure_within_six_hour_window"
        notes.append(
            "Matched time-variation panels are the guaranteed six-hour core; all non-panel "
            "cells are optional gap closure and may remain honestly untested at the send cutoff."
        )
    if not coverage_cells:
        raise ValueError(
            "enabled suites produced zero planned cells; add an applicable suite or route limits"
        )
    plan_ids = [str(cell["plan_cell_id"]) for cell in coverage_cells]
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("benchmark plan contains duplicate plan_cell_id values")
    logical_ids = [str(cell["logical_id"]) for cell in coverage_cells if cell["logical_id"]]
    if len(set(logical_ids)) != len(logical_ids):
        raise ValueError("benchmark plan contains duplicate logical request IDs")
    return PlanSummary(
        campaign_hash=config.identity_hash,
        static_requests=len(static_specs),
        load_requests_upper_path=load_count,
        load_arrival_window_seconds_sequential_upper_path=load_arrival_window_seconds,
        request_timeout_seconds_by_route={
            route.id: float(route.request_timeout_seconds) for route in config.routes
        },
        max_single_request_timeout_seconds=max_request_timeout,
        total_requests_upper_path=len(static_specs) + load_count,
        max_attempts_per_logical_request=attempts_per_logical,
        physical_attempts_upper_bound=(len(static_specs) + load_count) * attempts_per_logical,
        static_worst_case_usd=static_cost,
        load_worst_case_usd_upper_path=load_cost,
        total_worst_case_usd_upper_path=total,
        launch_budget_usd=config.max_cost_usd - config.launch_reserve_usd,
        native_placeholder_routes=placeholders,
        coverage_cells=tuple(coverage_cells),
        notes=tuple(notes),
    )


def _shape_cost(
    route: RouteConfig,
    shape: str,
    input_factor: float,
    *,
    shape_config: dict[str, Any] | None = None,
) -> float:
    if shape == "mixed":
        # shape_spec samples one of three named shapes or a 1,024-in/512-out structured task.
        # Planning must reserve the worst subtype rather than whichever one a plan-only hash picks.
        named = [
            _shape_cost(route, item, input_factor, shape_config=shape_config)
            for item in (
                "short_short",
                "long_short",
                "short_long",
            )
        ]
        structured_spec = next(
            candidate
            for index in range(64)
            if (
                candidate := shape_spec(
                    route,
                    "mixed",
                    f"plan:{route.id}:mixed:structured:{index}",
                    suite="plan",
                    workload_key=f"plan:{{route}}:mixed:structured:{index}",
                    shape_config=shape_config,
                )
            ).metadata.get("mixed_subtype")
            == "structured"
        )
        structured = route.worst_case_cost(
            reserved_input_tokens(route, structured_spec, input_factor),
            structured_spec.max_output_tokens,
        )
        return max(*named, structured)
    spec = shape_spec(
        route,
        shape,
        f"plan:{route.id}:{shape}",
        suite="plan",
        workload_key=f"plan:{{route}}:{shape}",
        shape_config=shape_config,
    )
    return route.worst_case_cost(
        reserved_input_tokens(route, spec, input_factor), spec.max_output_tokens
    )


def _load_plan_cell(
    route: RouteConfig,
    shape: str,
    epoch_id: str,
    phase: str,
    *,
    planned_disposition: str = "required",
    shape_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = shape_spec(
        route,
        shape,
        f"plan:{route.id}:{epoch_id}",
        suite="plan",
        workload_key=f"plan:{{route}}:{epoch_id}",
        shape_config=shape_config,
    )
    return {
        "plan_cell_id": f"load_epoch:{epoch_id}",
        "logical_id": None,
        "route_id": route.id,
        "suite": "load",
        "cell_id": (
            f"{shape}:{phase}:in{spec.planned_input_tokens}:out{spec.max_output_tokens}:{epoch_id}"
        ),
        "planned_disposition": planned_disposition,
    }
