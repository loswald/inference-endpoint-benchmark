from __future__ import annotations

from dataclasses import replace

import pytest

from inference_bench.models import RequestSpec
from inference_bench.suite_registry import SuitePlugin, register_suite
from inference_bench.workloads import plan_static_suites


def _planner(route, config, *, seed):  # type: ignore[no-untyped-def]
    return [
        RequestSpec(
            logical_id=f"portable:{route.id}:{seed}",
            route_id=route.id,
            suite="portable_probe_test",
            cell_id=f"probe:{config['label']}",
            messages=({"role": "user", "content": "portable probe"},),
            planned_input_tokens=8,
            max_output_tokens=8,
        )
    ]


def _validate(values):  # type: ignore[no-untyped-def]
    if not isinstance(values.get("label"), str) or not values["label"]:
        raise ValueError("portable suite label is required")


def test_new_static_suite_plans_without_core_enum_or_cli_changes(campaign, route) -> None:
    plugin = SuitePlugin(
        id="portable_probe_test",
        version="test/v1",
        planner=_planner,
        validator=_validate,
        public_keys=frozenset({"enabled", "label"}),
    )
    register_suite(plugin, replace=True)
    config = replace(
        campaign,
        routes=(route,),
        suites={"portable_probe_test": {"enabled": True, "label": "works"}},
    )
    requests = plan_static_suites(config.routes, config.suites, seed=config.seed)
    assert [(request.suite, request.cell_id) for request in requests] == [
        ("portable_probe_test", "probe:works")
    ]
    assert config.public_dict()["suites"]["portable_probe_test"] == {
        "enabled": True,
        "label": "works",
    }


def test_plugin_validator_rejects_bad_config_before_planning(campaign, route) -> None:
    plugin = SuitePlugin(
        id="portable_probe_validation_test",
        version="test/v1",
        planner=_planner,
        validator=_validate,
        public_keys=frozenset({"enabled", "label"}),
    )
    register_suite(plugin, replace=True)
    with pytest.raises(ValueError, match="label is required"):
        replace(
            campaign,
            routes=(route,),
            suites={"portable_probe_validation_test": {"enabled": True, "label": ""}},
        )

