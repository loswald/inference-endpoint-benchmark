from __future__ import annotations

from dataclasses import replace

import pytest

from inference_bench.config import CampaignConfig
from inference_bench.report import summarize_time_variation
from inference_bench.workloads import plan_static_suites


def _config(route, suites):  # type: ignore[no-untyped-def]
    return CampaignConfig(
        name="time-variation",
        seed=17,
        max_wall_seconds=90_000,
        max_cost_usd=20,
        launch_reserve_seconds=60,
        launch_reserve_usd=1,
        concurrency=4,
        retries=0,
        routes=(replace(route, context_tokens=131_072, max_output_tokens=8_192),),
        client_location="test-client",
        suites=suites,
    )


def test_time_variation_builds_matched_fixed_offset_panels(route) -> None:
    config = _config(
        route,
        {
            "time_variation": {
                "enabled": True,
                "panels": 3,
                "interval_minutes": 60,
                "samples_per_route_shape": 2,
                "shapes": ["short_short", "long_short"],
                "offered_rps": 0.2,
            }
        },
    )
    specs = plan_static_suites(config.routes, config.suites, seed=config.seed)
    assert len(specs) == 3 * 2 * 2
    assert {spec.metadata["time_variation_offset_seconds"] for spec in specs} == {
        0.0,
        3_600.0,
        7_200.0,
    }
    # The prompt identity is stable across panels; only panel/repeat labels differ.
    panel_zero = [spec for spec in specs if spec.metadata["time_variation_panel"] == 0]
    panel_one = [spec for spec in specs if spec.metadata["time_variation_panel"] == 1]
    assert [spec.planned_input_tokens for spec in panel_zero] == [
        spec.planned_input_tokens for spec in panel_one
    ]
    assert {
        spec.metadata["cache_condition"] for spec in specs
    } == {"stable_exact_prompt_across_panels"}


def test_time_variation_requires_complete_explicit_cache_repeat_design(route) -> None:
    with pytest.raises(ValueError, match="requires both"):
        _config(
            route,
            {
                "time_variation": {
                    "enabled": True,
                    "samples_per_route_shape": 4,
                    "stable_exact_prompt_repeats": 2,
                }
            },
        )
    with pytest.raises(ValueError, match="must sum"):
        _config(
            route,
            {
                "time_variation": {
                    "enabled": True,
                    "samples_per_route_shape": 4,
                    "stable_exact_prompt_repeats": 1,
                    "panel_unique_cache_cold_repeats": 2,
                }
            },
        )


def test_time_variation_refuses_overlap_with_capacity(route) -> None:
    with pytest.raises(ValueError, match="dedicated low-load campaign"):
        _config(
            route,
            {
                "time_variation": {"enabled": True},
                "aimd": {"enabled": True, "shapes": ["short_short"]},
            },
        )


def test_time_variation_report_projection_is_explicit() -> None:
    summary = [
        {
            "route_id": "route-a",
            "suite": "time_variation",
            "cell_id": "short_short:in256:out128:panel=002",
            "latency_p50": 1.2,
        }
    ]
    projected = summarize_time_variation(summary)
    assert projected[0]["shape"] == "short_short"
    assert projected[0]["panel_index"] == 2
