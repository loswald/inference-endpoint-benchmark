from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import replace

import pytest

from inference_bench.engine import BenchmarkEngine, PaymentRequiredLatched
from inference_bench.ledger import Ledger
from inference_bench.load import (
    EpochSummary,
    baseline_design,
    fixed_count_offsets,
    run_aimd,
    run_open_loop_epoch,
    run_soak,
    scheduled_offsets,
)
from inference_bench.models import InferenceResult, RequestSpec
from inference_bench.report import summarize_rows


def _spec(logical_id: str) -> RequestSpec:
    return RequestSpec(
        logical_id=logical_id,
        route_id="route-a",
        suite="test",
        cell_id="cell",
        messages=({"role": "user", "content": "hello"},),
        planned_input_tokens=8,
        max_output_tokens=8,
    )


def test_baseline_design_has_exact_nonzero_sample_count_and_truthful_rate() -> None:
    samples, duration, offered_rps = baseline_design({}, 20)
    assert (samples, duration, offered_rps) == (20, 20.0, 1.0)
    offsets = fixed_count_offsets(samples, duration)
    assert len(offsets) == 20
    assert len(set(offsets)) == 20
    assert 0 < offsets[0] < offsets[-1] < duration

    limited_samples, limited_duration, limited_rps = baseline_design({"baseline_rps": 0.1}, 20)
    assert limited_samples == 20
    assert limited_duration == pytest.approx(200)
    assert limited_rps == pytest.approx(0.1)

    with pytest.raises(ValueError, match="at least 20"):
        baseline_design({"baseline_samples": 19}, 20)


def _result(logical_id: str, *, status: str = "success", http_status: int = 200):
    success = status == "success"
    return InferenceResult(
        logical_id=logical_id,
        status=status,  # type: ignore[arg-type]
        http_status=http_status,
        started_at_utc="2026-01-01T00:00:00Z",
        ended_at_utc="2026-01-01T00:00:00.100000Z",
        total_seconds=0.1,
        time_to_headers_seconds=0.01,
        ttft_seconds=0.02 if success else None,
        decode_seconds=0.08 if success else None,
        output_event_offsets_seconds=(0.02, 0.08) if success else (),
        input_tokens=8 if success else None,
        output_tokens=8 if success else None,
        reasoning_tokens=0 if success else None,
    )


def _epoch_summary(
    *,
    epoch_id: str,
    phase: str,
    offered_rps: float,
    healthy: bool,
    controller_eligible: bool = True,
    scientific_censor_reason: str | None = None,
    launch_guard_reason: str | None = None,
) -> EpochSummary:
    return EpochSummary(
        epoch_id=epoch_id,
        route_id="route-a",
        shape="short_short",
        phase=phase,
        offered_rps=offered_rps,
        duration_seconds=1,
        actual_elapsed_seconds=1,
        scheduled=1,
        launched_logical=1,
        completed=1,
        unknown=0,
        physical_attempts=1,
        physical_successes=int(healthy),
        successful=int(healthy),
        rate_limited=0,
        server_errors=int(not healthy and controller_eligible),
        timeouts=0,
        transport_errors=0,
        queue_end_seconds=0,
        healthy=healthy,
        successful_input_tokens=8 if healthy else None,
        successful_output_tokens=8 if healthy else None,
        usage_complete_successful=int(healthy),
        ttft_observed_n=int(healthy),
        p95_ttft_seconds=0.02 if healthy else None,
        p95_service_seconds=0.1 if healthy else None,
        p95_total_seconds=0.1 if healthy else None,
        launch_guard_triggered=launch_guard_reason is not None,
        launch_guard_reason=launch_guard_reason,
        controller_eligible=controller_eligible,
        scientific_censor_reason=scientific_censor_reason,
    )


class SequenceAdapter:
    def __init__(self, *, fail_first: bool = False, payment_required: bool = False) -> None:
        self.fail_first = fail_first
        self.payment_required = payment_required
        self.calls: dict[str, int] = defaultdict(int)
        self.closed = False

    async def infer(self, route, request):
        self.calls[request.logical_id] += 1
        if self.payment_required:
            return _result(request.logical_id, status="client_error", http_status=402)
        if self.fail_first and self.calls[request.logical_id] == 1:
            return _result(request.logical_id, status="server_error", http_status=503)
        return _result(request.logical_id)

    async def close(self) -> None:
        self.closed = True


def _engine(tmp_path, campaign, adapter: SequenceAdapter, *, retries: int = 0):
    config = replace(campaign, retries=retries)
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=config.identity_hash, config_json="{}")
    engine = BenchmarkEngine(config, ledger)
    engine.adapters["openai_compatible"] = adapter
    return engine, ledger


def test_http_402_latches_and_blocks_every_later_send(tmp_path, campaign) -> None:
    async def run() -> tuple[int, int]:
        adapter = SequenceAdapter(payment_required=True)
        engine, ledger = _engine(tmp_path, campaign, adapter)
        first = await engine.execute(_spec("first"))
        assert first is not None and first.http_status == 402
        with pytest.raises(PaymentRequiredLatched):
            await engine.execute(_spec("second"))
        event_count = sum(row["kind"] == "http_402_latched" for row in ledger.event_rows())
        rows = len(ledger.rows())
        await engine.close()
        ledger.close()
        return rows, event_count

    rows, event_count = asyncio.run(run())
    assert rows == 1
    assert event_count == 1

    async def resume() -> None:
        ledger = Ledger(tmp_path)
        engine = BenchmarkEngine(campaign, ledger)
        assert engine.payment_required_latched
        with pytest.raises(PaymentRequiredLatched):
            await engine.execute(_spec("after-restart"))
        ledger.close()

    asyncio.run(resume())


def test_concurrent_same_logical_request_never_skips_to_retry_attempt(tmp_path, campaign) -> None:
    class SlowAdapter(SequenceAdapter):
        async def infer(self, route, request):
            self.calls[request.logical_id] += 1
            await asyncio.sleep(0.05)
            return _result(request.logical_id)

    async def run() -> tuple[int, list[dict[str, object]]]:
        adapter = SlowAdapter()
        engine, ledger = _engine(tmp_path, campaign, adapter, retries=1)
        await asyncio.gather(
            engine.execute(_spec("same-logical")),
            engine.execute(_spec("same-logical")),
        )
        rows = ledger.rows()
        calls = adapter.calls["same-logical"]
        await engine.close()
        ledger.close()
        return calls, rows

    calls, rows = asyncio.run(run())
    assert calls == 1
    assert len(rows) == 1
    assert rows[0]["attempt_index"] == 1
    assert rows[0]["state"] == "terminal"


def test_load_epoch_counts_every_physical_retry_and_includes_drain(
    tmp_path, campaign, route
) -> None:
    async def run():
        adapter = SequenceAdapter(fail_first=True)
        engine, ledger = _engine(tmp_path, campaign, adapter, retries=1)
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape="short_short",
            epoch_id="retry-epoch",
            phase="aimd",
            offered_rps=30,
            duration_seconds=0.2,
            concurrency=64,
            seed=11,
        )
        rows = ledger.rows()
        await engine.close()
        ledger.close()
        return summary, rows

    summary, rows = asyncio.run(run())
    assert summary.scheduled > 0
    assert summary.physical_attempts == 2 * summary.scheduled
    assert summary.server_errors == summary.scheduled
    assert summary.physical_successes == summary.scheduled
    assert summary.actual_elapsed_seconds > summary.duration_seconds
    assert summary.successful_input_tokens == 8 * summary.scheduled
    assert all("rps=30" in row["cell_id"] and "sample_seed=11" in row["cell_id"] for row in rows)
    matched = summarize_rows(rows)
    assert len(matched) == 1
    assert sum(item["attempts_n"] for item in matched) == 2 * summary.scheduled
    successful_stratum = next(
        item for item in matched if item["reasoning_token_state"] == "unconditional_base_cell"
    )
    assert successful_stratum["logical_requests_n"] == summary.scheduled
    assert successful_stratum["success_rate"] == 1.0
    assert successful_stratum["request_sampling_unit"] == (
        "persisted final logical outcome per request; incomplete retry sequences "
        "remain excluded from non-quality request-level estimands"
    )
    assert successful_stratum["latency_p50_ci95_low"] is None
    assert successful_stratum["latency_p50_ci95_high"] is None
    assert (
        successful_stratum["latency_p50_ci_method"] == "descriptive_correlated_load_requests_no_CI"
    )


def test_normal_response_drain_is_not_misclassified_as_queue_growth(
    tmp_path, campaign, route
) -> None:
    class SlowServiceAdapter(SequenceAdapter):
        async def infer(self, route, request):
            self.calls[request.logical_id] += 1
            await asyncio.sleep(1.1)
            return _result(request.logical_id)

    async def run() -> EpochSummary:
        engine, ledger = _engine(tmp_path, campaign, SlowServiceAdapter())
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape="short_short",
            epoch_id="slow-service-without-queue",
            phase="baseline",
            offered_rps=20,
            duration_seconds=0.05,
            concurrency=1,
            seed=11,
            deterministic_scheduled_count=1,
        )
        await engine.close()
        ledger.close()
        return summary

    summary = asyncio.run(run())
    assert summary.queue_end_seconds > 1.0
    assert summary.healthy


def test_zero_arrival_epoch_is_scientifically_censored(tmp_path, campaign, route) -> None:
    kwargs = {
        "shape": "short_short",
        "epoch_id": "zero-arrival",
        "phase": "aimd",
        "offered_rps": 0.000001,
        "duration_seconds": 0.001,
        "concurrency": 1,
        "seed": 4,
    }
    assert (
        scheduled_offsets(
            kwargs["offered_rps"],
            kwargs["duration_seconds"],
            seed=kwargs["seed"],
            epoch_id=kwargs["epoch_id"],
        )
        == []
    )

    async def run() -> EpochSummary:
        adapter = SequenceAdapter()
        engine, ledger = _engine(tmp_path, campaign, adapter)
        summary = await run_open_loop_epoch(engine, route, **kwargs)
        assert adapter.calls == {}
        await engine.close()
        ledger.close()
        return summary

    summary = asyncio.run(run())
    assert summary.scheduled == 0
    assert not summary.controller_eligible
    assert summary.scientific_censor_reason == "zero_scheduled_poisson_arrivals"


def test_epoch_holds_registered_arrival_window_after_early_final_arrival(
    tmp_path, campaign, route
) -> None:
    async def run() -> tuple[EpochSummary, float]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        loop = asyncio.get_running_loop()
        started = loop.time()
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape="short_short",
            epoch_id="fixed-window",
            phase="baseline",
            offered_rps=20,
            duration_seconds=0.05,
            concurrency=1,
            seed=11,
            deterministic_scheduled_count=1,
        )
        observed = loop.time() - started
        await engine.close()
        ledger.close()
        return summary, observed

    summary, observed = asyncio.run(run())
    assert summary.scheduled == 1
    assert observed >= 0.045
    assert summary.actual_elapsed_seconds >= 0.045


def test_open_loop_execution_materializes_configured_100k_workload(
    tmp_path, campaign, route
) -> None:
    class CapturingAdapter(SequenceAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[RequestSpec] = []

        async def infer(self, route, request):
            self.requests.append(request)
            result = _result(request.logical_id)
            result.input_tokens = request.planned_input_tokens
            return result

    async def run():
        large = replace(route, context_tokens=131_072, max_output_tokens=65_536)
        configured = replace(campaign, routes=(large,), retries=0)
        adapter = CapturingAdapter()
        engine, ledger = _engine(tmp_path, configured, adapter, retries=0)
        summary = await run_open_loop_epoch(
            engine,
            large,
            shape="long_short",
            shape_config={
                "long_input_tokens": 100_000,
                "long_input_overflow": "fail",
            },
            epoch_id="100k-execution",
            phase="soak_block",
            offered_rps=50,
            duration_seconds=0.02,
            concurrency=1,
            seed=11,
            deterministic_scheduled_count=1,
        )
        rows = ledger.rows()
        await engine.close()
        ledger.close()
        return adapter.requests, rows, summary

    requests, rows, summary = asyncio.run(run())
    assert summary.scheduled == 1
    assert len(requests) == len(rows) == 1
    assert requests[0].planned_input_tokens == 100_000
    assert requests[0].metadata["workload_input_was_clipped"] is False
    assert ":in100000:out128:" in requests[0].cell_id
    assert rows[0]["reserved_input_tokens"] >= 100_000


def test_aimd_and_soak_forward_identity_bound_long_shape_targets(
    tmp_path, campaign, route, monkeypatch
) -> None:
    observed: list[tuple[str, dict[str, object] | None]] = []

    async def healthy_epoch(engine, route, **kwargs):
        observed.append((kwargs["phase"], kwargs.get("shape_config")))
        return replace(
            _epoch_summary(
                epoch_id=kwargs["epoch_id"],
                phase=kwargs["phase"],
                offered_rps=kwargs["offered_rps"],
                healthy=True,
            ),
            route_id=route.id,
            shape=kwargs["shape"],
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", healthy_epoch)
    aimd = {
        "epochs": 1,
        "epoch_seconds": 1,
        "initial_rps": 1,
        "additive_rps": 1,
        "long_input_tokens": 100_000,
        "long_input_overflow": "fail",
    }
    soak = {
        "blocks": 1,
        "block_seconds": 1,
        "rate_rps": 1,
        "long_output_tokens": 32_768,
        "long_output_overflow": "fail",
    }

    async def run() -> None:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_aimd(engine, route, "long_short", aimd, seed=7)
        await run_soak(engine, route, "short_long", soak, seed=7)
        await engine.close()
        ledger.close()

    asyncio.run(run())
    assert observed
    assert all(
        config is aimd
        for phase, config in observed
        if phase
        in {
            "baseline",
            "aimd",
            "confirmation",
            "confirmation_separator",
            "recovery_after_observed_overload",
        }
    )
    assert all(
        config is soak for phase, config in observed if phase in {"soak_baseline", "soak_block"}
    )


@pytest.mark.parametrize(
    "censor_reason",
    ["interrupted_epoch_incomplete_no_replay", "zero_scheduled_poisson_arrivals"],
)
def test_interrupted_baseline_censors_controllers_without_thresholds(
    tmp_path, campaign, route, monkeypatch, censor_reason
) -> None:
    calls: list[tuple[str, float]] = []

    async def censored_epoch(engine, route, **kwargs):
        calls.append((kwargs["phase"], kwargs["offered_rps"]))
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=kwargs["phase"],
            offered_rps=kwargs["offered_rps"],
            healthy=False,
            controller_eligible=False,
            scientific_censor_reason=censor_reason,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", censored_epoch)

    async def run() -> tuple[list[EpochSummary], list[EpochSummary], list[str]]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        aimd = await run_aimd(
            engine,
            route,
            "short_short",
            {
                "epochs": 2,
                "epoch_seconds": 1,
                "initial_rps": 1,
                "additive_rps": 1,
                "multiplicative_decrease": 0.5,
            },
            seed=9,
        )
        soak = await run_soak(
            engine,
            route,
            "short_short",
            {"blocks": 2, "block_seconds": 1, "rate_rps": 1},
            seed=9,
        )
        event_kinds = [event["kind"] for event in ledger.event_rows()]
        await engine.close()
        ledger.close()
        return aimd, soak, event_kinds

    aimd, soak, event_kinds = asyncio.run(run())
    assert len(aimd) == len(soak) == 1
    assert calls == [("baseline", 0.1), ("soak_baseline", 0.1)]
    assert "aimd_controller_censored" in event_kinds
    assert "soak_controller_censored" in event_kinds


def test_unhealthy_low_load_baseline_stops_aimd_and_soak(
    tmp_path, campaign, route, monkeypatch
) -> None:
    calls: list[str] = []

    async def unhealthy_baseline(engine, route, **kwargs):
        calls.append(kwargs["phase"])
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=kwargs["phase"],
            offered_rps=kwargs["offered_rps"],
            healthy=False,
            controller_eligible=True,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", unhealthy_baseline)

    async def run() -> tuple[list[EpochSummary], list[EpochSummary], list[dict[str, object]]]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        aimd = await run_aimd(
            engine,
            route,
            "short_short",
            {"epochs": 2, "epoch_seconds": 1, "initial_rps": 1, "additive_rps": 1},
            seed=9,
        )
        soak = await run_soak(
            engine,
            route,
            "short_short",
            {"blocks": 2, "block_seconds": 1, "rate_rps": 1},
            seed=9,
        )
        events = ledger.event_rows()
        await engine.close()
        ledger.close()
        return aimd, soak, events

    aimd, soak, events = asyncio.run(run())
    assert len(aimd) == len(soak) == 1
    assert calls == ["baseline", "soak_baseline"]
    censored = [event for event in events if event["kind"].endswith("controller_censored")]
    assert len(censored) == 2
    assert all("unhealthy_low_load_baseline" in event["payload_json"] for event in censored)


@pytest.mark.parametrize(
    "censor_reason",
    ["interrupted_epoch_incomplete_no_replay", "zero_scheduled_poisson_arrivals"],
)
def test_interrupted_ramp_epoch_does_not_change_aimd_state(
    tmp_path, campaign, route, monkeypatch, censor_reason
) -> None:
    aimd_rates: list[float] = []

    async def sequenced_epoch(engine, route, **kwargs):
        phase = kwargs["phase"]
        offered_rps = kwargs["offered_rps"]
        if phase == "baseline":
            healthy, eligible, reason = True, True, None
        elif phase == "aimd":
            aimd_rates.append(offered_rps)
            index = len(aimd_rates)
            if index == 1:
                healthy, eligible, reason = (
                    False,
                    False,
                    censor_reason,
                )
            elif index == 2:
                healthy, eligible, reason = False, True, None
            else:
                healthy, eligible, reason = True, True, None
        else:
            healthy, eligible, reason = True, True, None
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=phase,
            offered_rps=offered_rps,
            healthy=healthy,
            controller_eligible=eligible,
            scientific_censor_reason=reason,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", sequenced_epoch)

    async def run() -> None:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_aimd(
            engine,
            route,
            "short_short",
            {
                "epochs": 3,
                "epoch_seconds": 1,
                "initial_rps": 1,
                "additive_rps": 0.5,
                "multiplicative_decrease": 0.5,
            },
            seed=9,
        )
        await engine.close()
        ledger.close()

    asyncio.run(run())
    assert aimd_rates == [1, 1, 1]


def test_aimd_geometric_bracket_is_bounded_and_no_overload_is_right_censored(
    tmp_path, campaign, route, monkeypatch
) -> None:
    ramp_rates: list[float] = []

    async def healthy_epoch(engine, route, **kwargs):
        if kwargs["phase"] == "aimd":
            ramp_rates.append(kwargs["offered_rps"])
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=kwargs["phase"],
            offered_rps=kwargs["offered_rps"],
            healthy=True,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", healthy_epoch)

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_aimd(
            engine,
            route,
            "short_short",
            {
                "epochs": 5,
                "epoch_seconds": 1,
                "initial_rps": 1,
                "additive_rps": 0.5,
                "bracket_epochs": 3,
                "bracket_multiplier": 2,
                "max_rps": 4,
            },
            seed=9,
        )
        event = next(item for item in ledger.event_rows() if item["kind"] == "aimd_complete")
        payload = json.loads(event["payload_json"])
        await engine.close()
        ledger.close()
        return payload

    payload = asyncio.run(run())
    assert ramp_rates == [1, 2, 4, 4, 4]
    assert payload["highest_observed_healthy_rps"] == 4
    assert payload["overload_observed"] is False
    assert payload["capacity_bound_state"] == "right_censored_highest_tested_healthy_no_overload"


def test_censored_epoch_breaks_unhealthy_consecutiveness(
    tmp_path, campaign, route, monkeypatch
) -> None:
    ramp_rates: list[float] = []

    async def sequenced_epoch(engine, route, **kwargs):
        phase = kwargs["phase"]
        if phase == "baseline":
            healthy, eligible, reason = True, True, None
        elif phase == "aimd":
            ramp_rates.append(kwargs["offered_rps"])
            index = len(ramp_rates)
            if index in {1, 3}:
                healthy, eligible, reason = False, True, None
            elif index == 2:
                healthy, eligible, reason = (
                    False,
                    False,
                    "interrupted_epoch_incomplete_no_replay",
                )
            else:
                healthy, eligible, reason = True, True, None
        else:
            healthy, eligible, reason = True, True, None
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=phase,
            offered_rps=kwargs["offered_rps"],
            healthy=healthy,
            controller_eligible=eligible,
            scientific_censor_reason=reason,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", sequenced_epoch)

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_aimd(
            engine,
            route,
            "short_short",
            {
                "epochs": 4,
                "epoch_seconds": 1,
                "initial_rps": 1,
                "additive_rps": 0.5,
            },
            seed=9,
        )
        event = next(item for item in ledger.event_rows() if item["kind"] == "aimd_complete")
        payload = json.loads(event["payload_json"])
        await engine.close()
        ledger.close()
        return payload

    payload = asyncio.run(run())
    assert ramp_rates == [1, 1, 1, 1]
    assert payload["overload_observed"] is False
    assert payload["recovery_run"] is False
    assert payload["capacity_bound_state"] == "right_censored_highest_tested_healthy_no_overload"


def test_no_healthy_ramp_candidate_is_explicitly_left_censored(
    tmp_path, campaign, route, monkeypatch
) -> None:
    calls: list[str] = []

    async def unhealthy_ramp(engine, route, **kwargs):
        phase = kwargs["phase"]
        calls.append(phase)
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=phase,
            offered_rps=kwargs["offered_rps"],
            healthy=phase == "baseline",
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", unhealthy_ramp)

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_aimd(
            engine,
            route,
            "short_short",
            {
                "epochs": 2,
                "epoch_seconds": 1,
                "initial_rps": 1,
                "additive_rps": 0.5,
            },
            seed=9,
        )
        event = next(item for item in ledger.event_rows() if item["kind"] == "aimd_complete")
        payload = json.loads(event["payload_json"])
        await engine.close()
        ledger.close()
        return payload

    payload = asyncio.run(run())
    assert calls == ["baseline", "aimd", "aimd"]
    assert payload["highest_observed_healthy_rps"] is None
    assert payload["capacity_bound_state"] == "left_censored_no_healthy_candidate"
    assert payload["confirmation_healthy"] == []
    assert payload["recovery_run"] is False


def test_aimd_nonmonotonic_evidence_permanently_invalidates_knee_bracket(
    tmp_path, campaign, route, monkeypatch
) -> None:
    ramp_outcomes = iter([False, False, True, True, False, False])

    async def sequenced_epoch(engine, route, **kwargs):
        phase = kwargs["phase"]
        healthy = next(ramp_outcomes) if phase == "aimd" else True
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=phase,
            offered_rps=kwargs["offered_rps"],
            healthy=healthy,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", sequenced_epoch)

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_aimd(
            engine,
            route,
            "short_short",
            {
                "epochs": 6,
                "epoch_seconds": 1,
                "initial_rps": 1,
                "additive_rps": 1,
                "multiplicative_decrease": 0.5,
            },
            seed=9,
        )
        event = next(item for item in ledger.event_rows() if item["kind"] == "aimd_complete")
        payload = json.loads(event["payload_json"])
        await engine.close()
        ledger.close()
        return payload

    payload = asyncio.run(run())
    assert payload["overload_observed"] is True
    assert payload["nonmonotonic_overload_observed"] is True
    assert payload["capacity_bound_state"] == "nonmonotonic_overload_no_current_bracket"
    assert payload["unhealthy_upper_bound_rps"] is None


def test_censored_confirmation_is_not_mislabeled_unhealthy(
    tmp_path, campaign, route, monkeypatch
) -> None:
    confirmation_index = 0

    async def sequenced_epoch(engine, route, **kwargs):
        nonlocal confirmation_index
        phase = kwargs["phase"]
        eligible = True
        reason = None
        if phase == "confirmation":
            confirmation_index += 1
            if confirmation_index == 2:
                eligible = False
                reason = "interrupted_epoch_incomplete_no_replay"
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=phase,
            offered_rps=kwargs["offered_rps"],
            healthy=True,
            controller_eligible=eligible,
            scientific_censor_reason=reason,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", sequenced_epoch)

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_aimd(
            engine,
            route,
            "short_short",
            {"epochs": 1, "epoch_seconds": 1, "initial_rps": 1, "additive_rps": 1},
            seed=9,
        )
        event = next(item for item in ledger.event_rows() if item["kind"] == "aimd_complete")
        payload = json.loads(event["payload_json"])
        await engine.close()
        ledger.close()
        return payload

    payload = asyncio.run(run())
    assert payload["controller_completion_state"] == "confirmations_inconclusive"
    assert payload["confirmation_execution_complete"] is True
    assert payload["confirmation_complete"] is False
    assert payload["confirmation_healthy"] == [True, None, True]
    assert payload["confirmation_eligible"] == [True, False, True]
    assert payload["confirmation_all_healthy"] is None


def test_final_guarded_soak_block_is_execution_complete_but_scientifically_censored(
    tmp_path, campaign, route, monkeypatch
) -> None:
    block_index = 0

    async def guarded_final_block(engine, route, **kwargs):
        nonlocal block_index
        phase = kwargs["phase"]
        guard = None
        healthy = True
        if phase == "soak_block":
            block_index += 1
            if block_index == 2:
                guard = "cost_guard"
                healthy = False
        return _epoch_summary(
            epoch_id=kwargs["epoch_id"],
            phase=phase,
            offered_rps=kwargs["offered_rps"],
            healthy=healthy,
            launch_guard_reason=guard,
        )

    monkeypatch.setattr("inference_bench.load.run_open_loop_epoch", guarded_final_block)

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, SequenceAdapter())
        await run_soak(
            engine,
            route,
            "short_short",
            {"blocks": 2, "block_seconds": 1, "rate_rps": 1},
            seed=9,
        )
        event = next(item for item in ledger.event_rows() if item["kind"] == "soak_complete")
        payload = json.loads(event["payload_json"])
        await engine.close()
        ledger.close()
        return payload

    payload = asyncio.run(run())
    assert payload["completed_blocks"] == 2
    assert payload["execution_complete"] is True
    assert payload["scientifically_complete"] is False
    assert payload["all_blocks_healthy"] is None
    assert payload["block_eligible"] == [True, False]
    assert payload["block_censor_reasons"] == [None, "cost_guard"]
    assert payload["controller_completion_state"] == "campaign_guard_censored"


def test_adapter_factory_is_not_eagerly_reinvoked(tmp_path, campaign, monkeypatch) -> None:
    created: list[SequenceAdapter] = []

    def factory(name: str):
        adapter = SequenceAdapter()
        created.append(adapter)
        return adapter

    monkeypatch.setattr("inference_bench.engine.adapter_for", factory)

    async def run() -> None:
        config = replace(campaign, retries=0)
        ledger = Ledger(tmp_path)
        ledger.initialize(campaign_hash=config.identity_hash, config_json="{}")
        engine = BenchmarkEngine(config, ledger)
        await engine.execute(_spec("one"))
        await engine.execute(_spec("two"))
        await engine.close()
        ledger.close()

    asyncio.run(run())
    assert len(created) == 1
    assert created[0].closed


def test_impossible_provider_usage_settles_conservative_reservation(tmp_path, campaign) -> None:
    class BadUsageAdapter(SequenceAdapter):
        async def infer(self, route, request):
            result = _result(request.logical_id)
            result.output_tokens = -1
            return result

    async def run():
        engine, ledger = _engine(tmp_path, campaign, BadUsageAdapter())
        result = await engine.execute(_spec("bad-usage"))
        row = ledger.rows()[0]
        await engine.close()
        ledger.close()
        return result, row

    result, row = asyncio.run(run())
    assert result is not None
    assert result.cost_basis == "reserved_upper_bound"
    assert row["reserved_usd"] == 0
    assert row["state"] == "terminal"
    assert row["settled_usd"] > 0.000028
    assert row["validity_class"] == "invalid"


def test_reasoning_tokens_cannot_exceed_billed_output_tokens(tmp_path, campaign) -> None:
    class ImpossibleReasoningUsageAdapter(SequenceAdapter):
        async def infer(self, route, request):
            result = _result(request.logical_id)
            result.output_tokens = 1
            result.reasoning_tokens = 100
            return result

    async def run():
        engine, ledger = _engine(tmp_path, campaign, ImpossibleReasoningUsageAdapter())
        result = await engine.execute(_spec("reasoning-exceeds-output"))
        row = ledger.rows()[0]
        await engine.close()
        ledger.close()
        return result, row

    result, row = asyncio.run(run())
    assert result is not None
    assert result.cost_basis == "reserved_upper_bound"
    assert row["validity_class"] == "invalid"
    assert row["usage_eligible"] == 0
    assert row["settled_usd"] > campaign.routes[0].actual_cost(8, 1)


def test_usage_parse_conflict_settles_conservative_reservation(tmp_path, campaign) -> None:
    class ConflictingUsageAdapter(SequenceAdapter):
        async def infer(self, route, request):
            result = _result(request.logical_id)
            result.usage_parse_errors = ("input_tokens_alias_conflict",)
            return result

    async def run():
        engine, ledger = _engine(tmp_path, campaign, ConflictingUsageAdapter())
        result = await engine.execute(_spec("usage-conflict"))
        row = ledger.rows()[0]
        await engine.close()
        ledger.close()
        return result, row

    result, row = asyncio.run(run())
    assert result is not None
    assert result.cost_basis == "reserved_upper_bound"
    assert row["validity_class"] == "invalid"
    assert row["settled_usd"] > campaign.routes[0].actual_cost(8, 8)


@pytest.mark.parametrize(
    ("zero_field", "expected_error"),
    [
        ("input_tokens", "provider_input_tokens_zero_for_nonempty_request"),
        ("output_tokens", "provider_output_tokens_zero_for_nonempty_response"),
    ],
)
def test_zero_usage_cannot_underprice_nonempty_content(
    tmp_path, campaign, zero_field, expected_error
) -> None:
    class ImpossibleZeroUsageAdapter(SequenceAdapter):
        async def infer(self, route, request):
            result = _result(request.logical_id)
            result.output_text = "visible provider output"
            setattr(result, zero_field, 0)
            return result

    async def run():
        engine, ledger = _engine(tmp_path, campaign, ImpossibleZeroUsageAdapter())
        result = await engine.execute(_spec(f"zero-{zero_field}"))
        row = ledger.rows()[0]
        await engine.close()
        ledger.close()
        return result, row

    result, row = asyncio.run(run())
    assert result is not None
    assert expected_error in result.usage_parse_errors
    assert result.cost_basis == "reserved_upper_bound"
    assert row["settled_usd"] > 0
    assert row["usage_eligible"] == 0
    assert row["validity_class"] == "invalid"


def test_quality_estimand_includes_failures_and_is_independent_of_usage_validity(
    tmp_path, campaign
) -> None:
    class QualityAdapter(SequenceAdapter):
        async def infer(self, route, request):
            if request.logical_id == "quality-failure":
                return _result(request.logical_id, status="server_error", http_status=503)
            result = _result(request.logical_id)
            result.output_text = "OK"
            if request.logical_id == "quality-invalid-usage":
                result.usage_parse_errors = ("total_tokens_mismatch_input_plus_output",)
            return result

    def quality_spec(logical_id: str) -> RequestSpec:
        return replace(
            _spec(logical_id),
            metadata={"scorer": "exact", "expected": "OK"},
        )

    async def run():
        engine, ledger = _engine(tmp_path, campaign, QualityAdapter())
        for logical_id in ("quality-success", "quality-failure", "quality-invalid-usage"):
            await engine.execute(quality_spec(logical_id))
        rows = ledger.rows()
        summary = summarize_rows(rows)
        await engine.close()
        ledger.close()
        return rows, summary

    rows, summary = asyncio.run(run())
    assert len(summary) == 1
    cell = summary[0]
    assert cell["quality_estimand"] == "end_to_end_all_predeclared_trials_non_success_is_zero"
    assert cell["quality_trials_n"] == 3
    assert cell["quality_successful_response_n"] == 2
    assert cell["quality_non_success_zero_n"] == 1
    assert cell["quality_mean"] == pytest.approx(2 / 3)
    invalid_usage = next(row for row in rows if row["logical_id"] == "quality-invalid-usage")
    assert invalid_usage["validity_class"] == "invalid"
    assert invalid_usage["quality_predeclared"] == 1
    assert invalid_usage["quality_eligible"] == 1


def test_unknown_claimed_quality_trials_are_durable_zeroes(tmp_path, campaign) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=campaign.identity_hash, config_json="{}")

    for index, logical_id in enumerate(("ambiguous", "crash"), start=1):
        spec = replace(
            _spec(logical_id),
            metadata={"scorer": "exact", "expected": "OK"},
        )
        assert ledger.claim(
            request_id=f"quality-unknown-{index}",
            attempt_index=1,
            spec=spec,
            route=campaign.routes[0],
            reserved_usd=0.001,
            max_cost_usd=10,
            cost_reserve_usd=1,
            scheduled_at_utc=None,
        )
    assert ledger.mark_unknown_if_in_flight(
        "quality-unknown-1", error_kind="ambiguous_transport_outcome"
    )
    assert ledger.recover_in_flight() == 1

    rows = ledger.rows()
    summary = summarize_rows(rows)
    ledger.close()

    assert all(row["quality_predeclared"] == 1 for row in rows)
    assert all(row["quality_eligible"] == 1 for row in rows)
    assert all(row["quality_score"] == 0 for row in rows)
    assert summary[0]["quality_trials_n"] == 2
    assert summary[0]["quality_non_success_zero_n"] == 2
    assert summary[0]["quality_mean"] == 0
