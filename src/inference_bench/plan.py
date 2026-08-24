from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import CampaignConfig
from .load import (
    scheduled_offsets,
    soak_rate_rps,
    validate_aimd_config,
    validate_soak_config,
)
from .models import RouteConfig
from .workloads import plan_static_suites, shape_spec


@dataclass(frozen=True, slots=True)
class PlanSummary:
    campaign_hash: str
    static_requests: int
    load_requests_upper_path: int
    total_requests_upper_path: int
    max_attempts_per_logical_request: int
    physical_attempts_upper_bound: int
    static_worst_case_usd: float
    load_worst_case_usd_upper_path: float
    total_worst_case_usd_upper_path: float
    launch_budget_usd: float
    native_placeholder_routes: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_hash": self.campaign_hash,
            "static_requests": self.static_requests,
            "load_requests_upper_path": self.load_requests_upper_path,
            "total_requests_upper_path": self.total_requests_upper_path,
            "max_attempts_per_logical_request": self.max_attempts_per_logical_request,
            "physical_attempts_upper_bound": self.physical_attempts_upper_bound,
            "static_worst_case_usd": self.static_worst_case_usd,
            "load_worst_case_usd_upper_path": self.load_worst_case_usd_upper_path,
            "total_worst_case_usd_upper_path": self.total_worst_case_usd_upper_path,
            "launch_budget_usd": self.launch_budget_usd,
            "native_placeholder_routes": list(self.native_placeholder_routes),
            "notes": list(self.notes),
        }


def build_plan(config: CampaignConfig) -> PlanSummary:
    static_specs = plan_static_suites(config.routes, config.suites, seed=config.seed)
    route_by_id = {route.id: route for route in config.routes}
    attempts_per_logical = config.retries + 1
    static_cost = attempts_per_logical * sum(
        route_by_id[spec.route_id].worst_case_cost(
            math.ceil(spec.planned_input_tokens * config.input_token_reservation_factor),
            spec.max_output_tokens,
        )
        for spec in static_specs
    )
    load_count = 0
    load_cost = 0.0
    shapes = ["short_short", "long_short", "short_long", "mixed"]
    aimd = config.suites.get("aimd")
    soak = config.suites.get("soak")
    if aimd and aimd.get("enabled", True):
        validate_aimd_config(aimd, config.concurrency)
    if soak and soak.get("enabled", True):
        validate_soak_config(soak, config.concurrency)
    for route in config.routes:
        if aimd and aimd.get("enabled", True):
            epochs = int(aimd.get("epochs", 12))
            seconds = float(aimd.get("epoch_seconds", 20))
            additive = float(aimd.get("additive_rps", 0.25))
            for shape in aimd.get("shapes", shapes):
                rate = float(aimd.get("initial_rps", 0.25))
                shape_cost = _shape_cost(route, shape, config.input_token_reservation_factor)
                baseline_rate = float(aimd.get("baseline_rps", min(rate, 0.1)))
                baseline_id = f"aimd-{route.id}-{shape}-baseline"
                n = len(
                    scheduled_offsets(
                        baseline_rate,
                        seconds,
                        seed=config.seed - 1,
                        epoch_id=baseline_id,
                    )
                )
                load_count += n
                load_cost += n * shape_cost * attempts_per_logical
                for index in range(epochs):
                    epoch_id = f"aimd-{route.id}-{shape}-{index:03d}"
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
                    rate += additive  # upper path assumes every epoch is healthy
                best = max(0.01, rate - additive)
                for confirmation in range(3):
                    confirmation_id = (
                        f"aimd-{route.id}-{shape}-confirm-{confirmation}"
                    )
                    n = len(
                        scheduled_offsets(
                            best,
                            seconds,
                            seed=config.seed + confirmation + 1,
                            epoch_id=confirmation_id,
                        )
                    )
                    load_count += n
                    load_cost += n * shape_cost * attempts_per_logical
                    if confirmation < 2:
                        separator_id = (
                            f"aimd-{route.id}-{shape}-separator-{confirmation}"
                        )
                        separator_rate = float(
                            aimd.get("baseline_rps", min(rate, 0.1))
                        )
                        n = len(
                            scheduled_offsets(
                                separator_rate,
                                seconds,
                                seed=config.seed + 50 + confirmation,
                                epoch_id=separator_id,
                            )
                        )
                        load_count += n
                        load_cost += n * shape_cost * attempts_per_logical
                # Recovery is conditional on an observed two-epoch overload at runtime. Include
                # its maximal possible schedule in the upper bound even though the all-healthy
                # trajectory itself would omit it.
                recovery_id = f"aimd-{route.id}-{shape}-recovery"
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
        if soak and soak.get("enabled", True):
            blocks = int(soak.get("blocks", 4))
            seconds = float(soak.get("block_seconds", 30))
            for shape in soak.get("shapes", shapes):
                rate = soak_rate_rps(soak, route.id, shape)
                shape_cost = _shape_cost(route, shape, config.input_token_reservation_factor)
                baseline_rate = float(soak.get("baseline_rps", min(rate, 0.1)))
                baseline_id = f"soak-{route.id}-{shape}-baseline"
                n = len(
                    scheduled_offsets(
                        baseline_rate,
                        seconds,
                        seed=config.seed - 1,
                        epoch_id=baseline_id,
                    )
                )
                load_count += n
                load_cost += n * shape_cost * attempts_per_logical
                for block in range(blocks):
                    epoch_id = f"soak-{route.id}-{shape}-block-{block}"
                    n = len(
                        scheduled_offsets(
                            rate,
                            seconds,
                            seed=config.seed + block,
                            epoch_id=epoch_id,
                        )
                    )
                    load_count += n
                    load_cost += n * shape_cost * attempts_per_logical
    placeholders = tuple(
        route.id
        for route in config.routes
        if route.adapter
        in {
            "bedrock_native",
            "vertex_native",
            "azure_model_inference_native",
            "openrouter",
        }
    )
    total = static_cost + load_cost
    notes = [
        "Costs include every allowed physical retry attempt at the full prompt/output ceiling.",
        "AIMD upper path assumes every additive epoch is healthy; "
        "actual offered rate adapts downward.",
        "Mixed workload cost uses its most expensive possible subtype, not one sampled subtype.",
        "Runtime reserves each deterministic send atomically and stops before the launch guard.",
    ]
    if total > config.max_cost_usd - config.launch_reserve_usd:
        notes.append(
            "Upper-path plan exceeds launch budget; later cells can be honestly budget-censored."
        )
    return PlanSummary(
        campaign_hash=config.identity_hash,
        static_requests=len(static_specs),
        load_requests_upper_path=load_count,
        total_requests_upper_path=len(static_specs) + load_count,
        max_attempts_per_logical_request=attempts_per_logical,
        physical_attempts_upper_bound=(len(static_specs) + load_count) * attempts_per_logical,
        static_worst_case_usd=static_cost,
        load_worst_case_usd_upper_path=load_cost,
        total_worst_case_usd_upper_path=total,
        launch_budget_usd=config.max_cost_usd - config.launch_reserve_usd,
        native_placeholder_routes=placeholders,
        notes=tuple(notes),
    )


def _shape_cost(route: RouteConfig, shape: str, input_factor: float) -> float:
    if shape == "mixed":
        # shape_spec samples one of three named shapes or a 1,024-in/512-out structured task.
        # Planning must reserve the worst subtype rather than whichever one a plan-only hash picks.
        named = [_shape_cost(route, item, input_factor) for item in (
            "short_short",
            "long_short",
            "short_long",
        )]
        structured = route.worst_case_cost(
            math.ceil(1_024 * input_factor), min(512, route.max_output_tokens or 4_096)
        )
        return max(*named, structured)
    spec = shape_spec(route, shape, f"plan:{route.id}:{shape}", suite="plan")
    return route.worst_case_cost(
        math.ceil(spec.planned_input_tokens * input_factor), spec.max_output_tokens
    )
