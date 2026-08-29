from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from inference_bench.cli import (
    _capacity_job_seconds,
    _run_interleaved_six_hour_study,
    _run_time_variation_panel,
)
from inference_bench.config import load_config, selected_capacity_cells
from inference_bench.digitalocean_closure import _sha256, build_digitalocean_closure_package
from inference_bench.load import LoadRunResult
from inference_bench.models import RequestSpec
from inference_bench.workloads import plan_static_suites

ROOT = Path(__file__).resolve().parents[1]


def test_closure_source_hash_is_stable_across_platform_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    assert _sha256(lf) == _sha256(crlf)


def test_digitalocean_six_hour_closure_compiles_exact_registered_gaps(tmp_path: Path) -> None:
    config_path, manifest_path = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        tmp_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "cache_gap_endpoints": 6,
        "historical_failed_fixed_rate_cells": 38,
        "historical_transport_gated_fixed_rate_cells": 3,
        "missing_capacity_cells": 25,
        "scheduled_fixed_rate_cells": 0,
        "time_panel_requests": 1_232,
    }
    assert manifest["schedule"] == {
        "finalization_reserve_seconds": 660,
        "gap_work_is_serial_and_never_overlaps_a_time_panel": True,
        "hard_send_cutoff_seconds": 21_840,
        "max_wall_seconds": 22_500,
        "measurement_span_seconds": 21_600,
        "panel_deadline_seconds": 600,
        "panel_guard_seconds": 420,
        "panel_offsets_seconds": [0, 3_600, 7_200, 10_800, 14_400, 18_000, 21_600],
    }
    assert manifest["input_identity"] == {
        "historical_fixed_rate_exclusion_reason": (
            "exact workload recipe identity differs from the corrected live recipes"
        ),
        "historical_fixed_rate_rows_are_not_rerun": True,
        "historical_input32k_short_is_distinct": True,
        "public_shape": "input100k_short",
        "runner_shape": "long_short",
        "target_tokens": 100_000,
    }
    assert manifest["timing_proof"] == {
        "configured_panel_concurrency": 176,
        "core_worst_case_usd": 174.757096752,
        "global_offered_rps": 1.0,
        "guaranteed_core": "seven matched low-load panels",
        "load_arrival_window_seconds_sequential_upper_path": 42750.0,
        "maximum_request_timeout_seconds": 360.0,
        "maximum_panel_in_flight": 176,
        "observations_per_endpoint_shape": 28,
        "optional_work": (
            "gap closure runs only when its conservative bound fits before the next guard"
        ),
        "panel_deadline_seconds": 600.0,
        "panel_deadline_slack_seconds": 65.0,
        "panel_launch_plus_timeout_bound_seconds": 535.0,
        "panel_launch_span_seconds": 175.0,
        "panel_prompt_design": (
            "two stable exact-prompt repeats plus two panel-unique cache-cold repeats per cell"
        ),
        "panel_unique_cache_cold_observations_per_endpoint_shape": 14,
        "panels": 7,
        "requests_per_panel": 176,
        "stable_exact_prompt_observations_per_endpoint_shape": 14,
        "success_reason_after_all_panels": "six_hour_window_completed",
    }
    config = load_config(config_path)
    assert len(selected_capacity_cells(config, "aimd")) == 25
    assert selected_capacity_cells(config, "soak") == []
    assert config.suites["soak"] == {"enabled": False}
    excluded_fixed_rate = manifest["selected"][
        "historical_fixed_rate_evidence_not_scheduled"
    ]
    assert len(excluded_fixed_rate) == 41
    assert all(
        row["live_disposition"] == "not_scheduled_exact_recipe_identity_differs"
        for row in excluded_fixed_rate
    )
    assert all("closure_shape" not in row for row in excluded_fixed_rate)
    assert config.max_wall_seconds == 22_500
    assert config.concurrency == 128
    assert config.retries == 0
    assert config.suites["time_variation"]["interleave_gap_work"] is True
    assert _capacity_job_seconds(config, "aimd") == 28_720.0


def test_suite_route_and_capability_probe_selectors_do_not_replay_resolved_cells(
    tmp_path: Path,
) -> None:
    config_path, _ = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        tmp_path,
    )
    config = load_config(config_path)
    specs = plan_static_suites(config.routes, config.suites, seed=config.seed)
    cache_routes = {spec.route_id for spec in specs if spec.suite == "cache"}
    assert cache_routes == set(config.suites["cache"]["route_ids"])
    capability = [spec for spec in specs if spec.suite == "capability"]
    qwen = [spec.logical_id for spec in capability if spec.route_id == "qwen3.8-max"]
    assert any(":vision" in logical_id for logical_id in qwen)
    assert any(":tool" in logical_id or ":parallel-tools" in logical_id for logical_id in qwen)
    assert not any(":temperature:" in logical_id for logical_id in qwen)


def test_closure_config_is_byte_stable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_paths = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        first,
    )
    second_paths = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        second,
    )
    assert first_paths[0].read_bytes() == second_paths[0].read_bytes()
    assert first_paths[1].read_bytes() == second_paths[1].read_bytes()


def test_open_loop_panel_launches_concurrently_and_records_deadline() -> None:
    events: list[tuple[str, str, dict[str, object]]] = []

    class FakeLedger:
        def record_event_once(
            self, key: str, kind: str, payload: dict[str, object]
        ) -> bool:
            events.append((key, kind, payload))
            return True

        def attempts_for_logical(self, _logical_id: str) -> list[dict[str, object]]:
            return []

    active = 0
    peak = 0

    class FakeEngine:
        ledger = FakeLedger()
        config = SimpleNamespace(seed=7, retries=0)
        reservation_overrun_latched = False

        async def execute(self, *_args, **_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return SimpleNamespace(http_status=200)

    specs = [
        RequestSpec(
            logical_id=f"panel:{index}",
            route_id="route",
            suite="time_variation",
            cell_id=f"cell:{index}",
            messages=({"role": "user", "content": "x"},),
            planned_input_tokens=1,
            max_output_tokens=1,
            stream=True,
            timeout_seconds=1,
        )
        for index in range(4)
    ]
    reason = asyncio.run(
        _run_time_variation_panel(
            FakeEngine(),
            0,
            specs,
            offered_rps=1_000,
            concurrency=4,
            planned_offset_seconds=0,
            deadline_seconds=2,
        )
    )
    assert reason is None
    assert peak > 1
    start = next(payload for _, kind, payload in events if kind == "time_variation_panel_started")
    assert start["arrival_pattern"] == "deterministic open-loop global schedule"
    assert start["panel_deadline_seconds"] == 2


def test_time_variation_panel_sends_nothing_when_arrivals_cross_hard_cutoff() -> None:
    events: list[tuple[str, str, dict[str, object]]] = []
    executed = 0

    class FakeLedger:
        def record_event_once(
            self, key: str, kind: str, payload: dict[str, object]
        ) -> bool:
            events.append((key, kind, payload))
            return True

        def attempts_for_logical(self, _logical_id: str) -> list[dict[str, object]]:
            return []

    class FakeEngine:
        ledger = FakeLedger()
        config = SimpleNamespace(seed=7, retries=0)
        reservation_overrun_latched = False

        async def execute(self, *_args, **_kwargs):
            nonlocal executed
            executed += 1
            return SimpleNamespace(http_status=200)

    specs = [
        RequestSpec(
            logical_id=f"cutoff:{index}",
            route_id="route",
            suite="time_variation",
            cell_id=f"cutoff:{index}",
            messages=({"role": "user", "content": "x"},),
            planned_input_tokens=1,
            max_output_tokens=1,
            stream=True,
            timeout_seconds=1,
        )
        for index in range(2)
    ]

    async def run() -> str | None:
        loop = asyncio.get_running_loop()
        return await _run_time_variation_panel(
            FakeEngine(),
            6,
            specs,
            offered_rps=1,
            concurrency=2,
            planned_offset_seconds=21_600,
            deadline_seconds=600,
            not_after_monotonic=loop.time() + 0.01,
        )

    assert asyncio.run(run()) == "time_guard"
    assert executed == 0
    assert not events


def test_interleaved_scheduler_requeues_paused_controller_after_protected_panel(
    campaign, route, monkeypatch
) -> None:
    clock = [0.0]
    anchor = datetime(2026, 8, 28, tzinfo=UTC)
    events: dict[str, tuple[str, dict[str, object]]] = {}
    sequence: list[str] = []

    class FakeDateTime:
        @classmethod
        def now(cls, _timezone):
            return anchor + timedelta(seconds=clock[0])

        @classmethod
        def fromisoformat(cls, value: str):
            return datetime.fromisoformat(value)

    class FakeLedger:
        def meta(self, key: str):
            if key == "started_at_utc":
                return anchor.isoformat().replace("+00:00", "Z")
            return None

        def event_by_key(self, key: str):
            value = events.get(key)
            return None if value is None else {"kind": value[0], "payload": value[1]}

        def record_event_once(
            self, key: str, kind: str, payload: dict[str, object]
        ) -> bool:
            if key in events:
                return False
            events[key] = (kind, payload)
            return True

    guarded_route = replace(route, request_timeout_seconds=0.001)
    config = replace(
        campaign,
        routes=(guarded_route,),
        retries=0,
        suites={
            "time_variation": {
                "enabled": True,
                "interleave_gap_work": True,
                "panels": 2,
                "interval_minutes": 20 / 60,
                "samples_per_route_shape": 1,
                "shapes": ["short_short"],
                "offered_rps": 1,
                "concurrency": 1,
                "panel_guard_seconds": 5,
                "panel_deadline_seconds": 10,
                "send_cutoff_seconds": 40,
            },
            "aimd": {
                "enabled": True,
                "shapes": ["short_short"],
                "epochs": 1,
                "epoch_seconds": 0.001,
            },
            "soak": {"enabled": False},
        },
    )
    engine = SimpleNamespace(ledger=FakeLedger(), config=config)
    specs = [
        RequestSpec(
            logical_id=f"panel:{panel}",
            route_id=route.id,
            suite="time_variation",
            cell_id=f"panel:{panel}",
            messages=({"role": "user", "content": "x"},),
            planned_input_tokens=1,
            max_output_tokens=1,
            stream=True,
            timeout_seconds=1,
            metadata={
                "time_variation_panel": panel,
                "time_variation_offset_seconds": offset,
            },
        )
        for panel, offset in enumerate((0, 20))
    ]

    async def fake_sleep(seconds: float) -> None:
        clock[0] += seconds

    async def fake_panel(engine, panel, *_args, **_kwargs):
        sequence.append(f"panel-{panel}")
        engine.ledger.record_event_once(
            f"time_variation_panel_completed:{panel}",
            "time_variation_panel_completed",
            {"panel": panel},
        )
        return None

    controller_calls = 0

    async def fake_aimd(engine, route, shape, config, **_kwargs):
        nonlocal controller_calls
        controller_calls += 1
        sequence.append(f"aimd-{controller_calls}")
        if controller_calls == 1:
            return LoadRunResult(paused_for_window=True)
        engine.ledger.record_event_once(
            f"aimd_complete:{route.id}:{shape}",
            "aimd_complete",
            {"route_id": route.id, "shape": shape},
        )
        return LoadRunResult()

    monkeypatch.setattr("inference_bench.cli.datetime", FakeDateTime)
    monkeypatch.setattr("inference_bench.cli.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("inference_bench.cli._run_time_variation_panel", fake_panel)
    monkeypatch.setattr("inference_bench.cli.run_aimd", fake_aimd)

    reason = asyncio.run(_run_interleaved_six_hour_study(engine, specs, [], config))
    assert reason == "six_hour_window_completed"
    assert sequence == ["panel-0", "aimd-1", "panel-1", "aimd-2"]
    assert controller_calls == 2
    assert "aimd_complete:route-a:short_short" in events


def test_generated_yaml_has_no_32k_alias_for_corrected_long_input(tmp_path: Path) -> None:
    config_path, _ = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        tmp_path,
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for suite in ("time_variation", "aimd"):
        assert document["suites"][suite]["long_input_tokens"] == 100_000
    assert document["suites"]["soak"] == {"enabled": False}


def test_six_hour_panels_separate_stable_and_cache_cold_prompt_identities(
    tmp_path: Path,
) -> None:
    config_path, _ = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        tmp_path,
    )
    config = load_config(config_path)
    specs = [
        spec
        for spec in plan_static_suites(config.routes, config.suites, seed=config.seed)
        if spec.suite == "time_variation"
        and spec.route_id == "deepseek-v4-flash-0731"
        and spec.metadata["shape"] == "short_short"
    ]
    assert len(specs) == 7 * 4
    by_panel = {
        panel: [spec for spec in specs if spec.metadata["time_variation_panel"] == panel]
        for panel in range(7)
    }
    for panel_specs in by_panel.values():
        assert [
            spec.metadata["cache_condition"] for spec in sorted(
                panel_specs, key=lambda item: item.metadata["time_variation_repeat"]
            )
        ] == [
            "stable_exact_prompt_across_panels",
            "stable_exact_prompt_across_panels",
            "panel_unique_cache_cold",
            "panel_unique_cache_cold",
        ]
    stable_identities = {
        spec.metadata["time_variation_prompt_identity"]
        for spec in specs
        if spec.metadata["cache_condition"] == "stable_exact_prompt_across_panels"
    }
    cold_identities = [
        spec.metadata["time_variation_prompt_identity"]
        for spec in specs
        if spec.metadata["cache_condition"] == "panel_unique_cache_cold"
    ]
    assert stable_identities == {"stable-000", "stable-001"}
    assert len(cold_identities) == len(set(cold_identities)) == 14
    stable_messages = {
        repeat: {
            repr(spec.messages)
            for spec in specs
            if spec.metadata["time_variation_repeat"] == repeat
        }
        for repeat in (0, 1)
    }
    cold_messages = {
        repeat: {
            repr(spec.messages)
            for spec in specs
            if spec.metadata["time_variation_repeat"] == repeat
        }
        for repeat in (2, 3)
    }
    assert all(len(messages) == 1 for messages in stable_messages.values())
    assert all(len(messages) == 7 for messages in cold_messages.values())

    mixed_specs = [
        spec
        for spec in plan_static_suites(config.routes, config.suites, seed=config.seed)
        if spec.suite == "time_variation"
        and spec.route_id == "deepseek-v4-flash-0731"
        and spec.metadata["shape"] == "mixed"
    ]
    for repeat in range(4):
        cell_recipes = {
            spec.cell_id.split(":panel=", 1)[0]
            for spec in mixed_specs
            if spec.metadata["time_variation_repeat"] == repeat
        }
        assert len(cell_recipes) == 1


def test_interleaved_panel_plan_rejects_request_retries_that_break_deadline_proof(
    tmp_path: Path,
) -> None:
    config_path, _ = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        tmp_path,
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["campaign"]["retries"] = 1
    invalid_path = tmp_path / "invalid-retry.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="campaign.retries=0"):
        load_config(invalid_path)


def test_interleaved_panel_plan_rejects_client_queueing(tmp_path: Path) -> None:
    config_path, _ = build_digitalocean_closure_package(
        ROOT / "examples" / "digitalocean-hosted-2026-08-27.yaml",
        ROOT / "reports" / "digitalocean",
        tmp_path,
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["suites"]["time_variation"]["concurrency"] = 175
    invalid_path = tmp_path / "invalid-panel-concurrency.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="without client-side queueing"):
        load_config(invalid_path)
