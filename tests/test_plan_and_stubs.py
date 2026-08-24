from __future__ import annotations

import pytest

from inference_bench.adapters.base import FailClosedAdapter, adapter_for
from inference_bench.config import CampaignConfig
from inference_bench.load import scheduled_offsets, soak_rate_rps
from inference_bench.models import AuthConfig, RouteConfig
from inference_bench.plan import _shape_cost, build_plan


def test_native_stub_is_honestly_labelled(route) -> None:
    stub = RouteConfig(
        id="bedrock",
        provider="amazon-bedrock",
        adapter="bedrock_native",
        model="exact-model",
        base_url="native-sdk",
        auth=AuthConfig(env="AWS_ACCESS_KEY_ID"),
        input_usd_per_million=1,
        output_usd_per_million=1,
    )
    config = CampaignConfig(
        name="stub",
        seed=1,
        max_wall_seconds=100,
        max_cost_usd=10,
        launch_reserve_seconds=10,
        launch_reserve_usd=1,
        concurrency=1,
        retries=0,
        routes=(stub,),
        suites={"latency": {"enabled": True, "repeats": 1, "shapes": ["short_short"]}},
    )
    assert build_plan(config).native_placeholder_routes == ("bedrock",)
    assert isinstance(adapter_for("bedrock_native"), FailClosedAdapter)


def test_plan_uses_runtime_schedule_contract_and_includes_retry_ceiling(route) -> None:
    aimd = {
        "enabled": True,
        "shapes": ["short_short"],
        "epochs": 1,
        "epoch_seconds": 10,
        "initial_rps": 2,
        "additive_rps": 1,
        "baseline_rps": 0.5,
    }
    config = CampaignConfig(
        name="schedule",
        seed=7,
        max_wall_seconds=1_000,
        max_cost_usd=1_000,
        launch_reserve_seconds=10,
        launch_reserve_usd=1,
        concurrency=8,
        retries=2,
        routes=(route,),
        suites={"aimd": aimd},
    )
    ids_and_seeds_and_rates = [
        ("aimd-route-a-short_short-baseline", 6, 0.5),
        ("aimd-route-a-short_short-000", 7, 2.0),
        ("aimd-route-a-short_short-confirm-0", 8, 2.0),
        ("aimd-route-a-short_short-separator-0", 57, 0.5),
        ("aimd-route-a-short_short-confirm-1", 9, 2.0),
        ("aimd-route-a-short_short-separator-1", 58, 0.5),
        ("aimd-route-a-short_short-confirm-2", 10, 2.0),
        ("aimd-route-a-short_short-recovery", 107, 1.0),
    ]
    logical = sum(
        len(scheduled_offsets(rate, 10, seed=seed, epoch_id=epoch_id))
        for epoch_id, seed, rate in ids_and_seeds_and_rates
    )
    plan = build_plan(config)
    assert plan.load_requests_upper_path == logical
    assert plan.max_attempts_per_logical_request == 3
    assert plan.physical_attempts_upper_bound == logical * 3


def test_mixed_plan_cost_is_worst_subtype(route) -> None:
    mixed = _shape_cost(route, "mixed", 1.5)
    assert mixed >= max(
        _shape_cost(route, shape, 1.5)
        for shape in ("short_short", "long_short", "short_long")
    )


def test_soak_rates_can_be_shape_specific() -> None:
    config = {
        "rate_rps": 1,
        "rate_rps_by_route": {"r": 2},
        "rate_rps_by_route_shape": {"r": {"short_long": 0.25}},
    }
    assert soak_rate_rps(config, "r", "short_long") == 0.25
    assert soak_rate_rps(config, "r", "short_short") == 2


def test_invalid_load_controller_contract_fails_in_plan(route) -> None:
    config = CampaignConfig(
        name="invalid-controller",
        seed=1,
        max_wall_seconds=100,
        max_cost_usd=10,
        launch_reserve_seconds=10,
        launch_reserve_usd=1,
        concurrency=1,
        retries=0,
        routes=(route,),
        suites={"aimd": {"enabled": True, "epochs": 0}},
    )
    with pytest.raises(ValueError, match="aimd.epochs"):
        build_plan(config)
