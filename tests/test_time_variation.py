from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from inference_bench.cli import _run_time_variation, _run_time_variation_panel
from inference_bench.config import CampaignConfig
from inference_bench.ledger import Ledger
from inference_bench.models import InferenceResult, RequestSpec
from inference_bench.profile_config import compose_profile_config, load_profile_config
from inference_bench.report import summarize_rows, summarize_time_variation
from inference_bench.validity import assess_result
from inference_bench.workloads import plan_static_suites

ROOT = Path(__file__).resolve().parents[1]


def _config(route, suites):  # type: ignore[no-untyped-def]
    suites = {name: dict(values) for name, values in suites.items()}
    if "time_variation" in suites:
        suites["time_variation"].setdefault("send_cutoff_seconds", 89_000)
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
            "cell_id": (
                "short_short:in256:out128:variation_stratum=stable_prefix:panel=002"
            ),
            "cache_state": "stable_prefix",
            "latency_p50": 1.2,
        }
    ]
    projected = summarize_time_variation(summary)
    assert projected[0]["shape"] == "short_short"
    assert projected[0]["panel_index"] == 2
    assert projected[0]["variation_stratum"] == "stable_prefix"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "samples_per_route_shape": 4,
                "shapes": ["short_short", "long_short"],
                "concurrency": 7,
            },
            "without client-side queueing",
        ),
        (
            {
                "samples_per_route_shape": 4,
                "shapes": ["short_short", "long_short"],
                "concurrency": 8,
                "offered_rps": 1,
                "panel_deadline_seconds": 186,
            },
            "launch span plus maximum route timeout",
        ),
        (
            {
                "panels": 2,
                "interval_minutes": 1497,
                "samples_per_route_shape": 2,
                "stable_exact_prompt_repeats": 1,
                "panel_unique_cache_cold_repeats": 1,
                "shapes": ["short_short"],
                "concurrency": 2,
                "send_cutoff_seconds": 89_900,
            },
            "cannot drain before campaign.max_wall_seconds",
        ),
    ],
)
def test_time_variation_schedule_validation_fails_before_credentials(
    route, overrides: dict[str, object], message: str
) -> None:
    suite: dict[str, object] = {
        "enabled": True,
        "panels": 7,
        "interval_minutes": 60,
        "samples_per_route_shape": 4,
        "stable_exact_prompt_repeats": 2,
        "panel_unique_cache_cold_repeats": 2,
        "shapes": ["short_short", "long_short", "short_long", "mixed"],
        "offered_rps": 1,
        "concurrency": 16,
        "panel_deadline_seconds": 1200,
        "send_cutoff_seconds": 21_840,
    }
    suite.update(overrides)
    with pytest.raises(ValueError, match=message):
        _config(route, {"time_variation": suite})


def test_dedicated_variation_resume_uses_immutable_ledger_anchor(monkeypatch) -> None:
    anchor = datetime(2026, 8, 30, tzinfo=UTC)
    now = anchor + timedelta(seconds=3_590)
    completed = {"time_variation_panel_completed:0"}
    calls: list[tuple[int, float, float]] = []
    sleeps: list[float] = []

    class FakeDateTime:
        @classmethod
        def now(cls, _timezone):
            return now

        @classmethod
        def fromisoformat(cls, value: str):
            return datetime.fromisoformat(value)

    class FakeLedger:
        def meta(self, key: str):
            return anchor.isoformat().replace("+00:00", "Z") if key == "started_at_utc" else None

        def event_by_key(self, key: str):
            return {"kind": "completed"} if key in completed else None

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def fake_panel(_engine, panel, _specs, **kwargs):
        calls.append(
            (
                panel,
                float(kwargs["planned_offset_seconds"]),
                float(kwargs["panel_start_monotonic"])
                - asyncio.get_running_loop().time(),
            )
        )
        completed.add(f"time_variation_panel_completed:{panel}")
        return None

    specs = [
        RequestSpec(
            logical_id=f"resume:{panel}",
            route_id="route-a",
            suite="time_variation",
            cell_id=f"short_short:variation_stratum=stable_prefix:panel={panel:03d}",
            messages=({"role": "user", "content": "x"},),
            planned_input_tokens=1,
            max_output_tokens=1,
            timeout_seconds=1,
            metadata={
                "time_variation_panel": panel,
                "time_variation_offset_seconds": panel * 3_600,
                "cache_state": "stable_prefix",
            },
        )
        for panel in (0, 1)
    ]
    engine = SimpleNamespace(ledger=FakeLedger())
    config = SimpleNamespace(
        concurrency=2,
        suites={
            "time_variation": {
                "offered_rps": 1,
                "concurrency": 2,
                "panel_deadline_seconds": 600,
                "send_cutoff_seconds": 21_840,
            }
        },
    )
    monkeypatch.setattr("inference_bench.cli.datetime", FakeDateTime)
    monkeypatch.setattr("inference_bench.cli.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("inference_bench.cli._run_time_variation_panel", fake_panel)

    assert (
        asyncio.run(
            _run_time_variation(
                engine,
                specs,
                config,
                resume_invocation=True,
            )
        )
        is None
    )
    assert calls[0][:2] == (1, 3_600.0)
    assert calls[0][2] == pytest.approx(10.0, abs=0.05)
    assert sleeps == [pytest.approx(10.0, abs=0.05)]


def test_ledger_anchored_overdue_panel_without_started_event_is_censored(
    monkeypatch,
) -> None:
    anchor = datetime(2026, 8, 30, tzinfo=UTC)
    now = anchor + timedelta(seconds=1.5)
    events: dict[str, tuple[str, dict[str, object]]] = {}
    censored_cells: list[str] = []
    executed: list[str] = []

    class FakeDateTime:
        @classmethod
        def now(cls, _timezone):
            return now

        @classmethod
        def fromisoformat(cls, value: str):
            return datetime.fromisoformat(value)

    class FakeLedger:
        def meta(self, key: str):
            return anchor.isoformat().replace("+00:00", "Z") if key == "started_at_utc" else None

        def event_by_key(self, key: str):
            return events.get(key)

        def attempts_for_logical(self, _logical_id: str) -> list[dict[str, object]]:
            return []

        def mark_plan_cell_if_planned(
            self, plan_cell_id: str, _state: str, _reason: str | None = None
        ) -> bool:
            censored_cells.append(plan_cell_id)
            return True

        def record_event_once(
            self, key: str, kind: str, payload: dict[str, object]
        ) -> bool:
            events[key] = (kind, payload)
            return True

    class FakeEngine:
        ledger = FakeLedger()
        config = SimpleNamespace(seed=7, retries=0)

        async def execute(self, spec, **_kwargs):
            executed.append(spec.logical_id)
            return SimpleNamespace(http_status=200)

    specs = [
        RequestSpec(
            logical_id=f"overdue:{index}",
            route_id="route-a",
            suite="time_variation",
            cell_id=f"short_short:variation_stratum=stable_prefix:panel=000:{index}",
            messages=({"role": "user", "content": "x"},),
            planned_input_tokens=1,
            max_output_tokens=1,
            timeout_seconds=1,
            metadata={
                "time_variation_panel": 0,
                "time_variation_offset_seconds": 0,
                "cache_state": "stable_prefix",
            },
        )
        for index in range(2)
    ]
    config = SimpleNamespace(
        concurrency=2,
        suites={
            "time_variation": {
                "offered_rps": 1,
                "concurrency": 2,
                "panel_deadline_seconds": 10,
                "arrival_lateness_tolerance_seconds": 0.25,
                "send_cutoff_seconds": 100,
            }
        },
    )
    monkeypatch.setattr("inference_bench.cli.datetime", FakeDateTime)

    assert (
        asyncio.run(
            _run_time_variation(
                FakeEngine(),
                specs,
                config,
                resume_invocation=True,
            )
        )
        is None
    )
    assert executed == []
    assert set(censored_cells) == {"request:overdue:0", "request:overdue:1"}
    assert "time_variation_panel_started:0" not in events
    assert events["time_variation_panel_censored:0"][1]["reason"] == (
        "resume_missed_registered_panel_arrival"
    )


@pytest.mark.parametrize(
    "started_event_present",
    [True, False],
    ids=("started-event-present", "process-down-before-started-event"),
)
def test_mid_panel_resume_censors_elapsed_arrivals_without_replay(
    started_event_present: bool,
) -> None:
    events: dict[str, tuple[str, dict[str, object]]] = {}
    if started_event_present:
        events["time_variation_panel_started:2"] = (
            "time_variation_panel_started",
            {"panel": 2},
        )
    coverage_updates: list[tuple[str, str, str | None]] = []
    executed: list[str] = []
    specs = [
        RequestSpec(
            logical_id=f"partial:{index}",
            route_id="route-a",
            suite="time_variation",
            cell_id=f"short_short:variation_stratum=stable_prefix:panel=002:{index}",
            messages=({"role": "user", "content": "x"},),
            planned_input_tokens=1,
            max_output_tokens=1,
            timeout_seconds=1,
        )
        for index in range(4)
    ]
    unknown_logical_id = specs[0].logical_id

    class FakeLedger:
        def event_by_key(self, key: str):
            return events.get(key)

        def attempts_for_logical(self, logical_id: str) -> list[dict[str, object]]:
            if logical_id != unknown_logical_id:
                return []
            return [
                {
                    "attempt_index": 1,
                    "state": "unknown",
                    "status": "unknown",
                }
            ]

        def mark_plan_cell(
            self, plan_cell_id: str, state: str, reason: str | None = None
        ) -> None:
            coverage_updates.append((plan_cell_id, state, reason))

        def mark_plan_cell_if_planned(
            self, plan_cell_id: str, state: str, reason: str | None = None
        ) -> bool:
            coverage_updates.append((plan_cell_id, state, reason))
            return True

        def record_event_once(
            self, key: str, kind: str, payload: dict[str, object]
        ) -> bool:
            events[key] = (kind, payload)
            return True

    class FakeEngine:
        ledger = FakeLedger()
        config = SimpleNamespace(seed=7, retries=0)

        async def execute(self, spec, **_kwargs):
            executed.append(spec.logical_id)
            return SimpleNamespace(http_status=200)

    async def run() -> str | None:
        loop = asyncio.get_running_loop()
        arrival_expiry = loop.time()
        return await _run_time_variation_panel(
            FakeEngine(),
            2,
            specs,
            offered_rps=1,
            concurrency=4,
            planned_offset_seconds=7_200,
            deadline_seconds=10,
            panel_start_monotonic=arrival_expiry - 1.5,
            arrival_expiry_monotonic=arrival_expiry,
            resume_invocation=True,
            not_after_monotonic=loop.time() + 20,
        )

    assert asyncio.run(run()) is None
    assert executed == []
    assert unknown_logical_id not in {
        update[0].removeprefix("request:")
        for update in coverage_updates
        if update[1] == "time_censored"
    }
    censored = events["time_variation_panel_censored:2"]
    assert censored[0] == "time_variation_panel_censored"
    assert censored[1]["reason"] == "resume_missed_registered_panel_arrival"
    assert censored[1]["previously_attempted_requests"] == 1
    assert censored[1]["remaining_unsent_requests_censored"] == 3
    assert int(censored[1]["elapsed_unsent_arrivals"]) >= 1
    assert "time_variation_panel_completed:2" not in events


def test_provider_profiles_compile_to_exact_six_hour_panel_rows() -> None:
    experiment = ROOT / "examples" / "experiment-profiles" / "six-hour-variation.yaml"
    expected = {
        "digitalocean-hosted-open-models.yaml": 1_232,
        "azure-ai-foundry-eastus2.yaml": 560,
        "amazon-bedrock-mantle-us-east-1.yaml": 112,
        "google-vertex-gemini36-flash-global.yaml": 112,
        "alibaba-model-studio-singapore.yaml": 896,
    }
    for profile_name, expected_rows in expected.items():
        compilation = load_profile_config(
            ROOT / "examples" / "provider-profiles" / profile_name,
            experiment,
        )
        specs = plan_static_suites(
            compilation.config.routes,
            compilation.config.suites,
            seed=compilation.config.seed,
        )
        assert len(specs) == expected_rows


def test_alibaba_variation_rejects_catalog_timeout_and_accepts_experiment_override() -> None:
    provider = yaml.safe_load(
        (ROOT / "examples" / "provider-profiles" / "alibaba-model-studio-singapore.yaml")
        .read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        (ROOT / "examples" / "experiment-profiles" / "six-hour-variation.yaml").read_text(
            encoding="utf-8"
        )
    )
    without_override = copy.deepcopy(experiment)
    del without_override["provider_route_overrides"]["alibaba-model-studio"]
    with pytest.raises(ValueError, match="launch span plus maximum route timeout"):
        compose_profile_config(provider, without_override)

    compiled = compose_profile_config(provider, experiment)
    assert {route.request_timeout_seconds for route in compiled.config.routes} == {720.0}


def test_variation_repeats_persist_as_two_unpooled_strata(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    for stratum, base_seconds in (("stable_prefix", 1.0), ("panel_unique_cold", 2.0)):
        for repeat in range(2):
            logical = f"time-variation:route-a:panel-000:short_short:{stratum}:{repeat:03d}"
            spec = RequestSpec(
                logical_id=logical,
                route_id="route-a",
                suite="time_variation",
                cell_id=(
                    "short_short:in256:out128:"
                    f"variation_stratum={stratum}:panel=000"
                ),
                messages=({"role": "user", "content": "x"},),
                planned_input_tokens=1,
                max_output_tokens=8,
                timeout_seconds=10,
                metadata={"cache_state": stratum},
            )
            result = InferenceResult(
                logical_id=logical,
                status="success",
                http_status=200,
                started_at_utc="2026-08-30T00:00:00Z",
                ended_at_utc="2026-08-30T00:00:02Z",
                total_seconds=base_seconds + repeat * 0.1,
                time_to_headers_seconds=0.1,
                ttft_seconds=0.2,
                decode_seconds=base_seconds - 0.2,
                output_event_offsets_seconds=(0.2, base_seconds),
                input_tokens=1,
                output_tokens=8,
                reasoning_tokens=0,
                cache_state=stratum,  # type: ignore[arg-type]
                cost_usd=0.001,
                cost_basis="provider_usage",
                arrival_to_completion_seconds=base_seconds + repeat * 0.1,
            )
            ledger.claim(
                request_id=f"{stratum}-{repeat}",
                attempt_index=1,
                spec=spec,
                route=route,
                reserved_usd=0.01,
                max_cost_usd=10,
                cost_reserve_usd=0,
                scheduled_at_utc="2026-08-30T00:00:00Z",
            )
            ledger.finish(
                request_id=f"{stratum}-{repeat}",
                result=result,
                validity=assess_result(result),
                quality_score=None,
            )
    summary = summarize_rows(ledger.rows())
    ledger.close()
    assert len(summary) == 2
    assert {row["cache_state"] for row in summary} == {
        "stable_prefix",
        "panel_unique_cold",
    }
    assert {row["logical_requests_n"] for row in summary} == {2}
    projected = summarize_time_variation(summary)
    assert len(projected) == 2
    stable = next(row for row in projected if row["variation_stratum"] == "stable_prefix")
    cold = next(row for row in projected if row["variation_stratum"] == "panel_unique_cold")
    assert stable["paired_contrast_state"] == "matched_reference_row"
    assert cold["paired_contrast_state"] == "matched_contrast_reported_here"
    assert cold["paired_difference_direction"] == "panel_unique_cold_minus_stable_prefix"
    assert cold["paired_cold_minus_stable_latency_p50"] == pytest.approx(1.0)
    assert cold["paired_cold_minus_stable_latency_p50_state"] == "estimated"
