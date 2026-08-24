from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import CampaignConfig
from .load import poisson_offsets
from .models import RouteConfig
from .workloads import plan_static_suites, shape_spec


@dataclass(frozen=True, slots=True)
class PlanSummary:
    campaign_hash: str
    static_requests: int
    load_requests_upper_path: int
    total_requests_upper_path: int
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
    static_cost = sum(
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
    for route in config.routes:
        if aimd and aimd.get("enabled", True):
            epochs = int(aimd.get("epochs", 12))
            seconds = float(aimd.get("epoch_seconds", 20))
            additive = float(aimd.get("additive_rps", 0.25))
            for shape in aimd.get("shapes", shapes):
                rate = float(aimd.get("initial_rps", 0.25))
                shape_cost = _shape_cost(route, shape, config.input_token_reservation_factor)
                baseline_rate = float(aimd.get("baseline_rps", min(rate, 0.1)))
                n = len(
                    poisson_offsets(
                        baseline_rate,
                        seconds,
                        seed=f"{config.seed}:plan-baseline:{route.id}:{shape}",
                    )
                )
                load_count += n
                load_cost += n * shape_cost
                for index in range(epochs):
                    n = len(
                        poisson_offsets(
                            rate, seconds, seed=f"{config.seed}:plan:{route.id}:{shape}:{index}"
                        )
                    )
                    load_count += n
                    load_cost += n * shape_cost
                    rate += additive  # upper path assumes every epoch is healthy
                best = max(0.01, rate - additive)
                for confirmation in range(3):
                    n = len(
                        poisson_offsets(
                            best,
                            seconds,
                            seed=f"{config.seed}:plan-confirm:{route.id}:{shape}:{confirmation}",
                        )
                    )
                    load_count += n
                    load_cost += n * shape_cost
                n = len(
                    poisson_offsets(
                        best * 0.5, seconds, seed=f"{config.seed}:plan-recovery:{route.id}:{shape}"
                    )
                )
                load_count += n
                load_cost += n * shape_cost
        if soak and soak.get("enabled", True):
            blocks = int(soak.get("blocks", 4))
            seconds = float(soak.get("block_seconds", 30))
            rate_by_route = soak.get("rate_rps_by_route") or {}
            rate = float(rate_by_route.get(route.id, soak.get("rate_rps", 0.25)))
            for shape in soak.get("shapes", shapes):
                shape_cost = _shape_cost(route, shape, config.input_token_reservation_factor)
                baseline_rate = float(soak.get("baseline_rps", min(rate, 0.1)))
                n = len(
                    poisson_offsets(
                        baseline_rate,
                        seconds,
                        seed=f"{config.seed}:plan-soak-baseline:{route.id}:{shape}",
                    )
                )
                load_count += n
                load_cost += n * shape_cost
                for block in range(blocks):
                    n = len(
                        poisson_offsets(
                            rate,
                            seconds,
                            seed=f"{config.seed}:plan-soak:{route.id}:{shape}:{block}",
                        )
                    )
                    load_count += n
                    load_cost += n * shape_cost
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
        "Costs assume every planned request consumes its full prompt estimate and output ceiling.",
        "AIMD upper path assumes every additive epoch is healthy; "
        "actual offered rate adapts downward.",
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
        static_worst_case_usd=static_cost,
        load_worst_case_usd_upper_path=load_cost,
        total_worst_case_usd_upper_path=total,
        launch_budget_usd=config.max_cost_usd - config.launch_reserve_usd,
        native_placeholder_routes=placeholders,
        notes=tuple(notes),
    )


def _shape_cost(route: RouteConfig, shape: str, input_factor: float) -> float:
    spec = shape_spec(route, shape, f"plan:{route.id}:{shape}", suite="plan")
    return route.worst_case_cost(
        math.ceil(spec.planned_input_tokens * input_factor), spec.max_output_tokens
    )
