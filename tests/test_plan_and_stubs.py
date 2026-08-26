from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from inference_bench.adapters.base import FailClosedAdapter, adapter_for
from inference_bench.adapters.providers import VertexOpenAIAdapter
from inference_bench.config import NATIVE_PLACEHOLDER_ADAPTERS, CampaignConfig, load_config
from inference_bench.load import scheduled_offsets, soak_rate_rps
from inference_bench.models import AuthConfig, RouteConfig
from inference_bench.plan import _shape_cost, build_plan
from inference_bench.workloads import plan_static_suites


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


def test_vertex_openai_uses_refreshable_oauth_adapter(route) -> None:
    vertex = replace(
        route,
        provider="google-vertex-ai",
        adapter="vertex_openai",
        base_url=(
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/"
            "us-central1/endpoints/openapi/chat/completions"
        ),
        auth=AuthConfig(env="GOOGLE_APPLICATION_CREDENTIALS"),
    )
    adapter = adapter_for("vertex_openai")
    assert isinstance(adapter, VertexOpenAIAdapter)
    assert "vertex_openai" not in NATIVE_PLACEHOLDER_ADAPTERS
    assert vertex.adapter == "vertex_openai"


def test_shipped_digitalocean_template_has_fail_closed_pricing() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "digitalocean.yaml"
    config = load_config(path)
    assert config.routes[0].input_usd_per_million is None
    assert config.routes[0].output_usd_per_million is None
    with pytest.raises(ValueError, match="incomplete pricing"):
        build_plan(config)


def test_plan_rejects_identity_placeholders_after_pricing_is_filled() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "digitalocean.yaml"
    config = load_config(path)
    priced = replace(config.routes[0], input_usd_per_million=1.0, output_usd_per_million=1.0)
    with pytest.raises(ValueError, match="campaign.client_location"):
        build_plan(replace(config, routes=(priced,)))


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
        client_location="test-client",
        suites={"aimd": aimd},
    )
    ids_and_seeds_and_rates = [
        ("aimd-route-a-short_short-000", 7, 2.0),
        ("aimd-route-a-short_short-confirm-0", 8, 2.0),
        ("aimd-route-a-short_short-confirm-1", 9, 2.0),
        ("aimd-route-a-short_short-confirm-2", 10, 2.0),
        ("aimd-route-a-short_short-recovery", 107, 1.0),
    ]
    # Baseline and the two low-load separators use exact 20-sample deterministic schedules.
    logical = 60 + sum(
        len(scheduled_offsets(rate, 10, seed=seed, epoch_id=epoch_id))
        for epoch_id, seed, rate in ids_and_seeds_and_rates
    )
    plan = build_plan(config)
    assert plan.load_requests_upper_path == logical
    assert plan.load_arrival_window_seconds_sequential_upper_path == pytest.approx(170)
    assert plan.max_attempts_per_logical_request == 3
    assert plan.physical_attempts_upper_bound == logical * 3
    assert plan.request_timeout_seconds_by_route == {"route-a": 180.0}
    assert plan.max_single_request_timeout_seconds == 180.0


def test_route_timeout_reaches_every_planned_request_and_changes_identity(route) -> None:
    changed = replace(route, request_timeout_seconds=900)
    suites = {
        "latency": {"enabled": True, "repeats": 1, "shapes": ["short_short"]},
        "context": {"enabled": True, "percentages": [99]},
        "output": {"enabled": True},
    }
    specs = plan_static_suites((changed,), suites, seed=7)
    assert specs
    assert {spec.timeout_seconds for spec in specs} == {900}
    assert changed.identity_hash != route.identity_hash


def test_invalid_aimd_bracket_contract_fails(route) -> None:
    with pytest.raises(ValueError, match="bracket_multiplier"):
        CampaignConfig(
            name="invalid-bracket",
            seed=1,
            max_wall_seconds=100,
            max_cost_usd=10,
            launch_reserve_seconds=10,
            launch_reserve_usd=1,
            concurrency=1,
            retries=0,
            routes=(route,),
            suites={
                "aimd": {
                    "enabled": True,
                    "epochs": 2,
                    "bracket_epochs": 1,
                    "bracket_multiplier": 1,
                }
            },
        )


@pytest.mark.parametrize(
    "yaml_text",
    [
        "campaign:\n  max_cost_usd: 1\n  max_cost_usd: 2\n",
        "routes:\n  - id: first\n    id: second\n",
        "suites:\n  aimd:\n    epochs: 1\n    epochs: 2\n",
    ],
)
def test_config_rejects_duplicate_yaml_keys(tmp_path: Path, yaml_text: str) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate mapping key"):
        load_config(path)


def test_mixed_plan_cost_is_worst_subtype(route) -> None:
    mixed = _shape_cost(route, "mixed", 1.5)
    assert mixed >= max(
        _shape_cost(route, shape, 1.5) for shape in ("short_short", "long_short", "short_long")
    )


def test_plan_identity_and_reservation_include_100k_load_target(route) -> None:
    large = replace(route, context_tokens=131_072, max_output_tokens=65_536)
    aimd = {
        "enabled": True,
        "shapes": ["long_short"],
        "long_input_tokens": 100_000,
        "long_input_overflow": "fail",
        "epochs": 1,
        "epoch_seconds": 1,
        "initial_rps": 0.1,
        "additive_rps": 0.1,
        "baseline_samples": 20,
    }
    configured = CampaignConfig(
        name="100k-plan",
        seed=7,
        max_wall_seconds=1_000,
        max_cost_usd=100,
        launch_reserve_seconds=10,
        launch_reserve_usd=1,
        concurrency=4,
        retries=0,
        routes=(large,),
        client_location="test-client",
        suites={"aimd": aimd},
    )
    plan = build_plan(configured)
    load_cells = [cell for cell in plan.coverage_cells if cell["suite"] == "load"]
    assert load_cells
    assert all(":in100000:out128:" in str(cell["cell_id"]) for cell in load_cells)
    assert _shape_cost(large, "long_short", 1.5, shape_config=aimd) > _shape_cost(
        large, "long_short", 1.5
    )

    changed = replace(
        configured,
        suites={"aimd": {**aimd, "long_input_tokens": 90_000}},
    )
    assert changed.identity_hash != configured.identity_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("long_input_tokens", 0),
        ("long_output_tokens", -1),
        ("long_input_overflow", "silent"),
        ("long_output_overflow", "silent"),
    ],
)
def test_long_shape_suite_configuration_is_strict(route, field, value) -> None:
    suite = {"enabled": True, "shapes": ["long_short", "short_long"], field: value}
    if field.endswith("overflow"):
        suite[field.replace("overflow", "tokens")] = 1_024
    with pytest.raises(ValueError, match=field):
        CampaignConfig(
            name="bad-long-shape",
            seed=1,
            max_wall_seconds=100,
            max_cost_usd=10,
            launch_reserve_seconds=10,
            launch_reserve_usd=1,
            concurrency=1,
            retries=0,
            routes=(route,),
            suites={"aimd": suite},
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
    with pytest.raises(ValueError, match="aimd.epochs"):
        CampaignConfig(
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
