from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
import yaml

from inference_bench.adapters.openai_compatible import OpenAICompatibleAdapter
from inference_bench.cli import (
    _capacity_execution_order,
    _dirty_tree_hash,
    _normalize_live_invocation,
    _pending_static_specs,
    _record_capacity_execution_order,
    _record_static_execution_order,
    _runtime_manifest,
    _static_execution_blocks,
    _verify_runtime_identity,
    run_campaign,
)
from inference_bench.config import CampaignConfig, load_config
from inference_bench.engine import (
    BenchmarkEngine,
    PaymentRequiredLatched,
    ReservationOverrunLatched,
    _retry_after,
)
from inference_bench.environment import (
    locked_distribution_versions,
    validate_run_directory_separation,
)
from inference_bench.ledger import BudgetExceeded, Ledger, TimeLimitReached
from inference_bench.load import (
    _route_neutral_epoch_key,
    run_open_loop_epoch,
    scheduled_offsets,
)
from inference_bench.models import (
    AuthConfig,
    InferenceResult,
    RequestSpec,
    RouteConfig,
    canonical_json,
)
from inference_bench.payload import (
    PAYLOAD_GENERATOR_VERSION,
    materialize_openai_compatible,
    reserved_input_tokens,
)
from inference_bench.plan import _shape_cost, build_plan
from inference_bench.quality import score_result
from inference_bench.report import (
    build_outlier_audit,
    generate_report,
    summarize_load_events,
    summarize_rows,
)
from inference_bench.validity import assess_result
from inference_bench.workloads import (
    context_marker_values,
    materialize_messages,
    plan_cache,
    plan_context,
    plan_latency,
    plan_output,
    plan_static_suites,
    shape_spec,
)


def _spec(logical_id: str = "logical", *, stream: bool = True) -> RequestSpec:
    return RequestSpec(
        logical_id=logical_id,
        route_id="route-a",
        suite="latency",
        cell_id="short_short",
        messages=({"role": "user", "content": "hello"},),
        planned_input_tokens=2,
        max_output_tokens=16,
        stream=stream,
    )


def _success(logical_id: str, *, total_seconds: float = 0.05) -> InferenceResult:
    return InferenceResult(
        logical_id=logical_id,
        status="success",
        http_status=200,
        started_at_utc="2026-01-01T00:00:00Z",
        ended_at_utc="2026-01-01T00:00:00.050000Z",
        total_seconds=total_seconds,
        time_to_headers_seconds=0.005,
        ttft_seconds=0.01,
        output_event_offsets_seconds=(0.01, 0.04),
        input_tokens=2,
        output_tokens=16,
        reasoning_tokens=0,
    )


class CountingAdapter:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls = 0
        self.delay = delay

    def preflight(self, route: RouteConfig) -> None:
        return None

    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return _success(request.logical_id, total_seconds=max(0.05, self.delay))

    async def close(self) -> None:
        return None


class ExplodingAfterClaimAdapter(CountingAdapter):
    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
        self.calls += 1
        raise RuntimeError("fixture adapter ambiguity")


def _engine(tmp_path, campaign: CampaignConfig, adapter: object) -> tuple[BenchmarkEngine, Ledger]:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=campaign.identity_hash, config_json="{}")
    engine = BenchmarkEngine(campaign, ledger)
    engine.adapters["openai_compatible"] = adapter
    return engine, ledger


def test_route_and_campaign_identity_bind_transport_quota_and_client_scope(route) -> None:
    assert route.http2 is False
    assert route.connection_reuse is True
    variants = [
        replace(route, http2=True),
        replace(route, connection_reuse=False),
        replace(route, quota_scope="project-alpha"),
        replace(route, output_limit_field="max_completion_tokens"),
        replace(route, auth=AuthConfig(env="OTHER_API_KEY")),
    ]
    assert all(item.identity_hash != route.identity_hash for item in variants)
    for bad_id in ("=formula", "space id", "a?", "", "a" * 129):
        with pytest.raises(ValueError, match="route id"):
            replace(route, id=bad_id)

    base = CampaignConfig(
        name="identity",
        seed=1,
        max_wall_seconds=60,
        max_cost_usd=1,
        launch_reserve_seconds=1,
        launch_reserve_usd=0.1,
        concurrency=1,
        retries=0,
        routes=(route,),
        client_location="london",
        suites={"latency": {"enabled": True, "repeats": 1, "shapes": ["short_short"]}},
    )
    assert replace(base, client_location="frankfurt").identity_hash != base.identity_hash


def test_missing_credential_fails_before_claim(tmp_path, campaign, monkeypatch) -> None:
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=campaign.identity_hash, config_json="{}")
    engine = BenchmarkEngine(campaign, ledger)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="credential environment variable"):
            await engine.execute(_spec())
        await engine.close()

    asyncio.run(run())
    assert ledger.rows() == []
    assert ledger.exposure().total_usd == 0
    ledger.close()


def test_exact_materialized_bytes_bind_claim_and_raise_token_reservation(
    tmp_path, campaign, route
) -> None:
    long_spec = replace(
        _spec("materialized"),
        messages=({"role": "user", "content": "é" * 600},),
        planned_input_tokens=1,
    )
    materialized = materialize_openai_compatible(route, long_spec)
    assert materialized.wire_body_sha256 == hashlib.sha256(materialized.body).hexdigest()
    assert materialized.generator_version == PAYLOAD_GENERATOR_VERSION

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, CountingAdapter())
        await engine.execute(long_spec)
        row = ledger.rows()[0]
        await engine.close()
        ledger.close()
        return row

    row = asyncio.run(run())
    assert row["payload_sha256"] == materialized.bound_payload_sha256
    assert row["wire_body_sha256"] == materialized.wire_body_sha256
    assert row["payload_generator_version"] == PAYLOAD_GENERATOR_VERSION
    assert int(row["reserved_input_tokens"]) > long_spec.planned_input_tokens


def test_settlement_overrun_is_terminal_and_durable(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    spec = _spec("overrun")
    assert ledger.claim(
        request_id="overrun-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.001,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    result = _success(spec.logical_id)
    result.cost_usd = 0.002
    result.cost_basis = "provider_usage"
    ledger.finish(
        request_id="overrun-1",
        result=result,
        validity=assess_result(result),
        quality_score=None,
    )
    assert ledger.rows()[0]["state"] == "terminal"
    assert ledger.exposure().settled_usd == pytest.approx(0.002)
    assert ledger.exposure().reserved_usd == 0
    assert [event["kind"] for event in ledger.event_rows()][-2:] == [
        "reservation_overrun",
        "request_finished",
    ]
    projected = [json.loads(line) for line in ledger.events_path.read_text().splitlines()]
    assert [event["kind"] for event in projected][-2:] == [
        "reservation_overrun",
        "request_finished",
    ]
    ledger.close()


def test_jsonl_projection_failure_does_not_corrupt_settled_request(
    tmp_path, route, monkeypatch
) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    spec = _spec("projection-failure")
    ledger.register_plan_cells(
        [
            {
                "plan_cell_id": f"request:{spec.logical_id}",
                "logical_id": spec.logical_id,
                "route_id": spec.route_id,
                "suite": spec.suite,
                "cell_id": spec.cell_id,
            }
        ]
    )
    assert ledger.claim(
        request_id="projection-failure-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.01,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    result = _success(spec.logical_id)
    result.cost_usd = 0.001
    result.cost_basis = "provider_usage"
    original_open = Path.open

    def fail_projection_open(path: Path, *args, **kwargs):
        if path == ledger.events_path and args and args[0] == "a":
            raise OSError("simulated projection failure")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", fail_projection_open)
        ledger.finish(
            request_id="projection-failure-1",
            result=result,
            validity=assess_result(result),
            quality_score=None,
            final_logical=True,
        )

    assert ledger.rows()[0]["state"] == "terminal"
    assert ledger.coverage_rows()[0]["state"] == "completed"
    assert ledger.meta("events_projection_state") == "dirty"
    ledger.rebuild_events_jsonl()
    assert ledger.meta("events_projection_state") == "clean"
    projected = [
        json.loads(line) for line in ledger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert projected == ledger.event_rows()
    ledger.close()


def test_epoch_resume_is_idempotent_and_rejects_identity_drift(tmp_path, campaign, route) -> None:
    async def run() -> tuple[object, object, int, int]:
        adapter = CountingAdapter()
        engine, ledger = _engine(tmp_path, replace(campaign, retries=0), adapter)
        kwargs = dict(
            shape="short_short",
            epoch_id="stable-epoch",
            phase="aimd",
            offered_rps=25.0,
            duration_seconds=0.15,
            concurrency=32,
            seed=9,
        )
        first = await run_open_loop_epoch(engine, route, **kwargs)
        calls = adapter.calls
        second = await run_open_loop_epoch(engine, route, **kwargs)
        with pytest.raises(ValueError, match="epoch identity changed"):
            await run_open_loop_epoch(engine, route, **{**kwargs, "offered_rps": 26.0})
        events = sum(event["kind"] == "load_epoch" for event in ledger.event_rows())
        await engine.close()
        ledger.close()
        return first, second, calls, events

    first, second, calls, events = asyncio.run(run())
    assert first == second
    assert calls == first.physical_attempts
    assert events == 1


def test_unexpected_open_loop_failure_cancels_all_later_unsent_arrivals(
    tmp_path, campaign, route, monkeypatch
) -> None:
    monkeypatch.setattr(
        "inference_bench.load.scheduled_offsets", lambda *args, **kwargs: [0.0, 0.05, 0.1]
    )

    async def run() -> tuple[int, list[dict[str, object]]]:
        adapter = ExplodingAfterClaimAdapter()
        engine, ledger = _engine(tmp_path, replace(campaign, retries=0), adapter)
        with pytest.raises(RuntimeError, match="fixture adapter ambiguity"):
            await run_open_loop_epoch(
                engine,
                route,
                shape="short_short",
                epoch_id="unexpected-stop",
                phase="aimd",
                offered_rps=1,
                duration_seconds=0.2,
                concurrency=3,
                seed=9,
            )
        rows = ledger.rows()
        await engine.close()
        ledger.close()
        return adapter.calls, rows

    calls, rows = asyncio.run(run())
    assert calls == 1
    assert len(rows) == 1
    assert rows[0]["state"] == "unknown"


def test_concurrent_open_loop_guard_reason_uses_durable_latch_priority(
    tmp_path, campaign, route, monkeypatch
) -> None:
    monkeypatch.setattr(
        "inference_bench.load.scheduled_offsets", lambda *args, **kwargs: [0.0, 0.0]
    )

    async def run():
        engine, ledger = _engine(tmp_path, replace(campaign, retries=0), CountingAdapter())
        first_waiting = asyncio.Event()
        call_index = 0

        async def racing_execute(*args, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                first_waiting.set()
                await asyncio.sleep(0)
                raise BudgetExceeded("fixture budget race")
            await first_waiting.wait()
            engine.payment_required_latched = True
            raise PaymentRequiredLatched("fixture 402 race")

        monkeypatch.setattr(engine, "execute", racing_execute)
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape="short_short",
            epoch_id="guard-race",
            phase="aimd",
            offered_rps=1,
            duration_seconds=0.1,
            concurrency=2,
            seed=9,
        )
        await engine.close()
        ledger.close()
        return summary

    summary = asyncio.run(run())
    assert summary.launch_guard_triggered
    assert summary.launch_guard_reason == "http_402_latch"


def test_partial_epoch_resume_is_censored_without_duplicate_sends(
    tmp_path, campaign, route
) -> None:
    offered_rps = 30.0
    duration_seconds = 0.2
    seed = 11
    epoch_id = "partial-epoch"
    offsets = scheduled_offsets(offered_rps, duration_seconds, seed=seed, epoch_id=epoch_id)
    assert len(offsets) > 1
    config = replace(campaign, retries=0)
    adapter = CountingAdapter()
    engine, ledger = _engine(tmp_path, config, adapter)
    logical = f"load:{route.id}:short_short:{epoch_id}:0"
    spec = shape_spec(
        route,
        "short_short",
        logical,
        suite="load",
        cell_suffix=f":aimd:rps={offered_rps:.9g}:epoch={epoch_id}",
    )
    assert ledger.claim(
        request_id="partial-physical-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.01,
        max_cost_usd=10,
        cost_reserve_usd=1,
        scheduled_at_utc=None,
    )
    result = _success(logical)
    result.cost_usd = 0.001
    result.cost_basis = "provider_usage"
    ledger.finish(
        request_id="partial-physical-1",
        result=result,
        validity=assess_result(result),
        quality_score=None,
    )

    async def run():
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape="short_short",
            epoch_id=epoch_id,
            phase="aimd",
            offered_rps=offered_rps,
            duration_seconds=duration_seconds,
            concurrency=32,
            seed=seed,
        )
        await engine.close()
        return summary

    summary = asyncio.run(run())
    assert adapter.calls == 0
    assert summary.launch_guard_reason is None
    assert not summary.controller_eligible
    assert summary.scientific_censor_reason == "interrupted_epoch_incomplete_no_replay"
    assert summary.completed == 1
    assert not summary.healthy
    event = ledger.event_by_key(f"load_epoch_resume_censored:{epoch_id}")
    assert event is not None
    assert "interrupted_epoch_incomplete_no_replay" in event["payload_json"]
    ledger.close()


def test_fully_unknown_resumed_epoch_is_censored_without_duplicate_sends(
    tmp_path, campaign, route
) -> None:
    offered_rps = 30.0
    duration_seconds = 0.2
    seed = 13
    epoch_id = "unknown-epoch"
    offsets = scheduled_offsets(offered_rps, duration_seconds, seed=seed, epoch_id=epoch_id)
    assert offsets
    config = replace(campaign, retries=0)
    adapter = CountingAdapter()
    engine, ledger = _engine(tmp_path, config, adapter)
    for index in range(len(offsets)):
        logical = f"load:{route.id}:short_short:{epoch_id}:{index}"
        spec = shape_spec(
            route,
            "short_short",
            logical,
            suite="load",
            cell_suffix=f":aimd:rps={offered_rps:.9g}:epoch={epoch_id}",
        )
        assert ledger.claim(
            request_id=f"unknown-physical-{index}",
            attempt_index=1,
            spec=spec,
            route=route,
            reserved_usd=0.001,
            max_cost_usd=10,
            cost_reserve_usd=1,
            scheduled_at_utc=None,
        )
    assert ledger.recover_in_flight() == len(offsets)

    async def run():
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape="short_short",
            epoch_id=epoch_id,
            phase="aimd",
            offered_rps=offered_rps,
            duration_seconds=duration_seconds,
            concurrency=32,
            seed=seed,
        )
        await engine.close()
        return summary

    summary = asyncio.run(run())
    assert adapter.calls == 0
    assert summary.unknown == len(offsets)
    assert not summary.controller_eligible
    assert (
        summary.scientific_censor_reason == "interrupted_epoch_unknown_provider_outcomes_no_replay"
    )
    load_blocks = summarize_load_events(ledger.event_rows(), rows=ledger.rows())
    assert load_blocks[0]["capacity_estimand_blocks_n"] == 0
    assert load_blocks[0]["censored_blocks_n"] == 1
    assert json.loads(load_blocks[0]["censored_block_reasons_json"]) == {
        "interrupted_epoch_unknown_provider_outcomes_no_replay": 1
    }
    ledger.close()


def test_resumed_scientifically_censored_epoch_preserves_coverage_reason(
    tmp_path, campaign, route
) -> None:
    config = replace(campaign, retries=0)
    engine, ledger = _engine(tmp_path, config, CountingAdapter())
    epoch_id = "zero-resume"
    ledger.register_plan_cells(
        [
            {
                "plan_cell_id": f"load_epoch:{epoch_id}",
                "logical_id": None,
                "route_id": route.id,
                "suite": "load",
                "cell_id": "short_short:baseline",
                "planned_disposition": "required",
            }
        ]
    )
    kwargs = {
        "shape": "short_short",
        "epoch_id": epoch_id,
        "phase": "baseline",
        "offered_rps": 0.000001,
        "duration_seconds": 0.001,
        "concurrency": 1,
        "seed": 4,
    }

    async def run_twice() -> None:
        first = await run_open_loop_epoch(engine, route, **kwargs)
        second = await run_open_loop_epoch(engine, route, **kwargs)
        assert first == second
        await engine.close()

    asyncio.run(run_twice())
    coverage = ledger.coverage_rows()[0]
    assert coverage["state"] == "inconclusive"
    assert coverage["reason"] == "zero_scheduled_poisson_arrivals"
    ledger.close()


def test_unknown_final_outcome_preserves_cost_and_enters_audit(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="b" * 64, config_json="{}")
    assert ledger.claim(
        request_id="unknown-1",
        attempt_index=1,
        spec=_spec("unknown"),
        route=route,
        reserved_usd=0.25,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    assert ledger.recover_in_flight() == 1
    row = summarize_rows(ledger.rows())[0]
    assert row["unknown_outcomes_n"] == 1
    assert row["settled_usd_sum"] == 0
    assert row["unknown_reserved_usd_sum"] == pytest.approx(0.25)
    assert row["conservative_exposure_usd_sum"] == pytest.approx(0.25)
    audit = build_outlier_audit(ledger.rows())
    assert audit[0]["audit_class"] == "censored"
    assert audit[0]["reasons"] == ["unknown_provider_outcome_final_attempt"]
    ledger.close()


def test_retry_cost_and_errors_follow_final_base_cell(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="c" * 64, config_json="{}")
    spec = _spec("retry-stratum")
    assert ledger.claim(
        request_id="retry-stratum-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.2,
        max_cost_usd=2,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    throttled = _success(spec.logical_id)
    throttled.status = "rate_limited"
    throttled.http_status = 429
    throttled.input_tokens = None
    throttled.output_tokens = None
    throttled.reasoning_tokens = None
    throttled.ttft_seconds = None
    throttled.output_event_offsets_seconds = ()
    throttled.cost_usd = 0.1
    throttled.cost_basis = "reserved_upper_bound"
    ledger.finish(
        request_id="retry-stratum-1",
        result=throttled,
        validity=assess_result(throttled),
        quality_score=None,
    )
    assert ledger.claim(
        request_id="retry-stratum-2",
        attempt_index=2,
        spec=spec,
        route=route,
        reserved_usd=0.2,
        max_cost_usd=2,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    success = _success(spec.logical_id)
    success.cost_usd = 0.15
    success.cost_basis = "provider_usage"
    ledger.finish(
        request_id="retry-stratum-2",
        result=success,
        validity=assess_result(success),
        quality_score=None,
    )

    incomplete_spec = _spec("incomplete-same-cell")
    assert ledger.claim(
        request_id="incomplete-same-cell-1",
        attempt_index=1,
        spec=incomplete_spec,
        route=route,
        reserved_usd=0.2,
        max_cost_usd=2,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    incomplete = _success(incomplete_spec.logical_id)
    incomplete.status = "rate_limited"
    incomplete.http_status = 429
    incomplete.cost_usd = 0.05
    incomplete.cost_basis = "reserved_upper_bound"
    ledger.finish(
        request_id="incomplete-same-cell-1",
        result=incomplete,
        validity=assess_result(incomplete),
        quality_score=None,
        final_logical=False,
    )

    summary = summarize_rows(ledger.rows())
    assert len(summary) == 1
    row = summary[0]
    assert row["reasoning_token_state"] == "unconditional_base_cell"
    assert row["logical_requests_n"] == 1
    assert row["attempts_n"] == 3
    assert row["successes_n"] == 1
    assert row["settled_usd_sum"] == pytest.approx(0.3)
    assert json.loads(row["http_status_counts_json"]) == {"200": 1, "429": 2}
    assert row["incomplete_retry_sequences_n"] == 1
    assert row["service_latency_estimand"] == "successful_final_logical_attempts_only"
    assert row["service_latency_p50_n"] == 1
    assert sum(item["attempts_n"] for item in summary) == 3
    assert sum(item["settled_usd_sum"] for item in summary) == pytest.approx(0.3)
    ledger.close()


def test_incomplete_retry_predeclared_quality_trial_remains_zero_in_denominator(
    tmp_path, route
) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="e" * 64, config_json="{}")
    spec = replace(
        _spec("incomplete-quality"),
        metadata={"scorer": "exact", "expected": "OK"},
    )
    assert ledger.claim(
        request_id="incomplete-quality-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.2,
        max_cost_usd=2,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    result = _success(spec.logical_id)
    result.status = "rate_limited"
    result.http_status = 429
    result.cost_usd = 0.2
    result.cost_basis = "reserved_upper_bound"
    ledger.finish(
        request_id="incomplete-quality-1",
        result=result,
        validity=assess_result(result, quality_scored=True),
        quality_score=0,
        final_logical=False,
    )
    assert ledger.claim(
        request_id="incomplete-quality-2",
        attempt_index=2,
        spec=spec,
        route=route,
        reserved_usd=0.2,
        max_cost_usd=2,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    latest = _success(spec.logical_id)
    latest.status = "rate_limited"
    latest.http_status = 429
    latest.cache_state = "cached_trial"
    latest.cost_usd = 0.2
    latest.cost_basis = "reserved_upper_bound"
    ledger.finish(
        request_id="incomplete-quality-2",
        result=latest,
        validity=assess_result(latest, quality_scored=True),
        quality_score=0,
        final_logical=False,
    )
    row = summarize_rows(ledger.rows())[0]
    assert row["cache_state"] == "cached_trial"
    assert row["attempts_n"] == 2
    assert row["logical_requests_n"] == 0
    assert row["incomplete_retry_sequences_n"] == 1
    assert row["quality_trials_n"] == 1
    assert row["quality_incomplete_retry_zero_n"] == 1
    assert row["quality_mean"] == 0
    assert row["quality_mean_ci95_high"] is not None and row["quality_mean_ci95_high"] > 0
    ledger.close()


def test_adapter_controlled_status_usage_and_arrival_text_never_reaches_evidence(
    tmp_path, route
) -> None:
    private_text = "provider-private-output-should-never-persist"
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="f" * 64, config_json="{}")
    spec = _spec("untrusted-result-text")
    assert ledger.claim(
        request_id="untrusted-result-text-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.2,
        max_cost_usd=2,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    result = _success(spec.logical_id)
    result.status = private_text  # type: ignore[assignment]
    result.usage_parse_errors = (private_text,)
    result.arrival_latency_censor_reason = private_text
    result.cost_usd = 0.2
    result.cost_basis = "reserved_upper_bound"
    validity = assess_result(result)
    ledger.finish(
        request_id="untrusted-result-text-1",
        result=result,
        validity=validity,
        quality_score=None,
        final_logical=True,
    )
    row = ledger.rows()[0]
    assert row["status"] == "server_error"
    assert json.loads(row["usage_parse_errors_json"]) == ["other_usage_parse_error"]
    assert row["arrival_latency_censor_reason"] == "other_arrival_latency_censor_reason"
    durable_text = (
        json.dumps(row) + json.dumps(ledger.event_rows()) + json.dumps(result.without_content())
    )
    assert private_text not in durable_text
    ledger.close()


def test_engine_quarantines_malicious_custom_adapter_result_before_durable_evidence(
    tmp_path, campaign
) -> None:
    sentinel = "CUSTOM-ADAPTER-PRIVATE-SENTINEL"

    class MaliciousAdapter(CountingAdapter):
        async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
            self.calls += 1
            await asyncio.sleep(0.02)
            result = _success(request.logical_id, total_seconds=1e-9)
            result.http_status = sentinel  # type: ignore[assignment]
            result.input_tokens = sentinel  # type: ignore[assignment]
            result.output_tokens = sentinel  # type: ignore[assignment]
            result.reasoning_tokens = sentinel  # type: ignore[assignment]
            result.cache_read_input_tokens = sentinel  # type: ignore[assignment]
            result.started_at_utc = sentinel
            result.ended_at_utc = sentinel
            result.retained_headers = {
                "Authorization": f"Bearer {sentinel}",
                "X-Secret-Token": sentinel,
                "Retry-After": "1",
            }
            result.provider_request_id = sentinel
            result.error_kind = sentinel
            result.output_text = sentinel
            return result

    async def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        engine, ledger = _engine(tmp_path, campaign, MaliciousAdapter())
        await engine.execute(_spec("malicious-custom-adapter"))
        rows = ledger.rows()
        events = ledger.event_rows()
        await engine.close()
        ledger.close()
        return rows, events

    rows, events = asyncio.run(run())
    row = rows[0]
    assert row["total_seconds"] >= 0.015
    assert row["http_status"] is None
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None
    assert json.loads(row["retained_headers_json"]) == {"retry-after": "1"}
    reasons = json.loads(row["validity_reasons_json"])
    assert {
        "http_status_invalid",
        "input_tokens_invalid_count",
        "output_tokens_invalid_count",
        "reasoning_tokens_invalid_count",
        "cache_read_input_tokens_invalid_count",
    }.issubset(reasons)
    public_report_data = {
        "summary": summarize_rows(rows),
        "audit": build_outlier_audit(rows),
    }
    assert sentinel not in json.dumps(rows) + json.dumps(events) + json.dumps(public_report_data)


def test_engine_fails_closed_on_adapter_result_identity_mismatch_without_leaking(
    tmp_path, campaign
) -> None:
    sentinel = "PRIVATE-WRONG-LOGICAL-ID"

    class WrongIdentityAdapter(CountingAdapter):
        async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
            self.calls += 1
            return _success(sentinel)

    async def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        engine, ledger = _engine(tmp_path, campaign, WrongIdentityAdapter())
        with pytest.raises(ValueError, match="adapter_result_contract_violation"):
            await engine.execute(_spec("expected-logical-id"))
        rows = ledger.rows()
        events = ledger.event_rows()
        await engine.close()
        ledger.close()
        return rows, events

    rows, events = asyncio.run(run())
    assert rows[0]["state"] == "unknown"
    assert rows[0]["error_kind"] == "post_claim_exception"
    assert sentinel not in json.dumps(rows) + json.dumps(events)


def test_unknown_error_kind_is_fixed_category_in_db_event_and_coverage(tmp_path, route) -> None:
    private_text = "ProviderPrivateExceptionName"
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="1" * 64, config_json="{}")
    spec = _spec("unknown-private-error")
    ledger.register_plan_cells(
        [
            {
                "plan_cell_id": f"request:{spec.logical_id}",
                "logical_id": spec.logical_id,
                "route_id": spec.route_id,
                "suite": spec.suite,
                "cell_id": spec.cell_id,
                "planned_disposition": "required",
            }
        ]
    )
    assert ledger.claim(
        request_id="unknown-private-error-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.2,
        max_cost_usd=2,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    ledger.mark_unknown(
        "unknown-private-error-1", error_kind=f"post_claim_exception:{private_text}"
    )
    row = ledger.rows()[0]
    event = next(item for item in ledger.event_rows() if item["kind"] == "request_outcome_unknown")
    coverage = ledger.coverage_rows()[0]
    assert row["error_kind"] == "post_claim_exception"
    assert json.loads(event["payload_json"])["error_kind"] == "post_claim_exception"
    assert coverage["reason"] == "post_claim_exception"
    assert private_text not in json.dumps(row) + json.dumps(event) + json.dumps(coverage)
    ledger.close()


def test_base_cell_reliability_and_cost_are_unconditional_on_reasoning_state(
    tmp_path, route
) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="d" * 64, config_json="{}")
    for logical_id, succeeds, settled in (
        ("base-success", True, 0.1),
        ("base-failure", False, 0.2),
    ):
        spec = _spec(logical_id)
        assert ledger.claim(
            request_id=f"{logical_id}-1",
            attempt_index=1,
            spec=spec,
            route=route,
            reserved_usd=0.2,
            max_cost_usd=2,
            cost_reserve_usd=0.1,
            scheduled_at_utc=None,
        )
        result = _success(logical_id)
        result.cost_usd = settled
        if succeeds:
            result.cost_basis = "provider_usage"
        else:
            result.status = "server_error"
            result.http_status = 503
            result.input_tokens = None
            result.output_tokens = None
            result.reasoning_tokens = None
            result.ttft_seconds = None
            result.output_event_offsets_seconds = ()
            result.cost_basis = "reserved_upper_bound"
        ledger.finish(
            request_id=f"{logical_id}-1",
            result=result,
            validity=assess_result(result),
            quality_score=None,
            final_logical=True,
        )

    summary = summarize_rows(ledger.rows())
    assert len(summary) == 1
    row = summary[0]
    assert row["reasoning_token_state"] == "unconditional_base_cell"
    assert row["logical_requests_n"] == 2
    assert row["successes_n"] == 1
    assert row["success_rate"] == pytest.approx(0.5)
    assert row["reasoning_reported_zero_n"] == 1
    assert row["reasoning_unknown_n"] == 1
    assert row["settled_usd_sum"] == pytest.approx(0.3)
    assert row["conservative_exposure_per_successful_request_usd"] == pytest.approx(0.3)
    ledger.close()


def test_adapter_exception_after_claim_becomes_final_unknown(tmp_path, campaign) -> None:
    async def run() -> tuple[list[dict[str, object]], float]:
        engine, ledger = _engine(tmp_path, campaign, ExplodingAfterClaimAdapter())
        with pytest.raises(RuntimeError, match="fixture adapter ambiguity"):
            await engine.execute(_spec("ambiguous"))
        rows = ledger.rows()
        exposure = ledger.exposure().reserved_usd
        await engine.close()
        ledger.close()
        return rows, exposure

    rows, exposure = asyncio.run(run())
    assert rows[0]["state"] == "unknown"
    assert rows[0]["error_kind"] == "post_claim_exception"
    assert exposure > 0


def test_provider_usage_settles_actual_cost_and_overrun_latches(tmp_path, campaign, route) -> None:
    class UsageAdapter(CountingAdapter):
        def __init__(self, input_tokens: int, output_tokens: int) -> None:
            super().__init__()
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

        async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
            self.calls += 1
            result = _success(request.logical_id)
            result.input_tokens = self.input_tokens
            result.output_tokens = self.output_tokens
            return result

    async def normal() -> tuple[float, float]:
        adapter = UsageAdapter(2, 1)
        engine, ledger = _engine(tmp_path / "normal", replace(campaign, retries=0), adapter)
        result = await engine.execute(_spec("normal-cost"))
        assert result is not None
        settled = ledger.exposure().settled_usd
        await engine.close()
        ledger.close()
        return settled, route.usage_cost_with_unknown_cache(2, 1)

    settled, expected = asyncio.run(normal())
    assert settled == pytest.approx(expected)

    async def overrun() -> tuple[dict[str, object], list[str]]:
        adapter = UsageAdapter(1_000_000, 1)
        engine, ledger = _engine(tmp_path / "overrun", replace(campaign, retries=0), adapter)
        result = await engine.execute(_spec("huge-usage"))
        assert result is not None and engine.reservation_overrun_latched
        with pytest.raises(ReservationOverrunLatched):
            await engine.execute(_spec("blocked-after-overrun"))
        row = ledger.rows()[0]
        kinds = [event["kind"] for event in ledger.event_rows()]
        await engine.close()
        ledger.close()
        return row, kinds

    row, kinds = asyncio.run(overrun())
    assert float(row["settled_usd"]) == pytest.approx(
        route.usage_cost_with_unknown_cache(1_000_000, 1)
    )
    assert float(row["settled_usd"]) > 1
    assert "reservation_overrun" in kinds


def test_post_claim_hashing_failure_never_strands_in_flight(tmp_path, campaign) -> None:
    class NoncanonicalToolAdapter(CountingAdapter):
        async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
            result = _success(request.logical_id)
            result.tool_calls = ({"nonfinite": float("nan")},)
            return result

    async def run() -> dict[str, object]:
        engine, ledger = _engine(tmp_path, campaign, NoncanonicalToolAdapter())
        with pytest.raises(ValueError):
            await engine.execute(_spec("noncanonical-tool"))
        row = ledger.rows()[0]
        await engine.close()
        ledger.close()
        return row

    row = asyncio.run(run())
    assert row["state"] == "unknown"
    assert row["error_kind"] == "post_claim_exception"


def test_terminal_nonretryable_logical_request_is_never_replayed(tmp_path, campaign) -> None:
    class ClientErrorAdapter(CountingAdapter):
        async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
            self.calls += 1
            result = _success(request.logical_id)
            result.status = "client_error"
            result.http_status = 400
            result.input_tokens = None
            result.output_tokens = None
            result.reasoning_tokens = None
            return result

    async def run() -> int:
        adapter = ClientErrorAdapter()
        engine, ledger = _engine(tmp_path, campaign, adapter)
        await engine.execute(_spec("nonretryable"))
        assert await engine.execute(_spec("nonretryable")) is None
        await engine.close()
        ledger.close()
        return adapter.calls

    assert asyncio.run(run()) == 1


def test_wrong_json_shape_scores_failure_instead_of_crashing() -> None:
    spec = replace(
        _spec("json-list"),
        metadata={"scorer": "json_fields", "expected": {"answer": 7}},
    )
    result = _success(spec.logical_id)
    result.output_text = '[{"answer":7}]'
    assert score_result(spec, result)[0] == 0


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_structured_quality_rejects_nonstandard_nonfinite_json(constant) -> None:
    spec = replace(
        _spec("strict-json"),
        metadata={"scorer": "json_fields", "expected": {"a": 2, "b": 3}},
    )
    result = _success(spec.logical_id)
    result.output_text = f'{{"a":2,"b":3,"extra":{constant}}}'
    assert score_result(spec, result)[0] == 0


def test_structured_quality_requires_exact_finite_object() -> None:
    spec = replace(
        _spec("strict-json-valid"),
        metadata={"scorer": "json_fields", "expected": {"a": 2, "b": 3}},
    )
    result = _success(spec.logical_id)
    result.output_text = '{"a":2,"b":3}'
    assert score_result(spec, result)[0] == 1
    result.output_text = '{"a":2,"b":3,"extra":4}'
    assert score_result(spec, result)[0] == 0
    result.output_text = '{"a":2.0,"b":3}'
    assert score_result(spec, result)[0] == 0
    result.output_text = '{"a":1,"a":2,"b":3}'
    assert score_result(spec, result)[0] == 0


def test_tool_quality_requires_exact_parsed_function_and_city() -> None:
    spec = replace(
        _spec("tool-quality"),
        metadata={"scorer": "tool_city_reykjavik"},
    )
    result = _success(spec.logical_id)
    result.tool_calls = (
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Reykjavík extended"}',
            },
        },
    )
    assert score_result(spec, result)[0] == 0
    result.tool_calls[0]["function"]["arguments"] = '{"city":"Reykjavík"}'
    assert score_result(spec, result)[0] == 1
    result.tool_calls += (
        {
            "id": "call-2",
            "type": "function",
            "function": {"name": "unrelated", "arguments": "{}"},
        },
    )
    assert score_result(spec, result)[0] == 0
    result.tool_calls = result.tool_calls[:1]
    result.tool_calls[0]["function"]["arguments"] = '{"city":"Reykjavík","unexpected":"value"}'
    assert score_result(spec, result)[0] == 0
    result.tool_calls[0]["function"]["arguments"] = '{"city":"wrong","city":"Reykjavík"}'
    assert score_result(spec, result)[0] == 0

    context = replace(
        _spec("context-quality"),
        metadata={"scorer": "context_markers", "context_markers": ["A", "B", "C"]},
    )
    context_result = _success(context.logical_id)
    context_result.output_text = "A|B|C"
    assert score_result(context, context_result)[0] == 1
    context_result.output_text = "C|B|A"
    assert score_result(context, context_result)[0] == 0
    context_result.output_text = "prefix A|B|C suffix"
    assert score_result(context, context_result)[0] == 0


def test_capacity_quality_scorers_cover_structure_and_realized_length() -> None:
    structured = replace(
        _spec("structured-load"),
        metadata={"quality": "json_keys", "expected_keys": ["summary", "risks"]},
    )
    result = _success(structured.logical_id)
    result.output_text = '{"summary":"ok","risks":[]}'
    assert score_result(structured, result)[0] == 1
    result.output_text = '{"summary":"ok","risks":[],"extra":true}'
    assert score_result(structured, result)[0] == 0

    longform = replace(
        _spec("longform-load"),
        metadata={"quality": "longform_completion", "minimum_output_tokens": 750},
    )
    result = _success(longform.logical_id)
    result.output_text = "nonempty"
    result.output_tokens = 749
    assert score_result(longform, result)[0] == 0
    result.output_tokens = 750
    assert score_result(longform, result)[0] == 1


def test_empty_sse_is_protocol_error(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "fixture-only")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: [DONE]\n\n")

    async def run() -> InferenceResult:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec("empty-sse"))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "server_error"
    assert result.error_kind and "empty_or_invalid_sse_stream" in result.error_kind


def test_streaming_tool_deltas_reconstruct_by_choice_and_tool_and_bind_hash(
    monkeypatch, route
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "fixture-only")
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"ci'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'ty":"Paris"}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    async def run() -> InferenceResult:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec("tools"))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.status == "success"
    assert result.tool_calls[0]["choice_index"] == 0
    assert result.tool_calls[0]["index"] == 0
    assert result.tool_calls[0]["function"] == {
        "name": "lookup",
        "arguments": '{"city":"Paris"}',
    }
    without_tools = _success("tools")
    assert result.output_sha256 != without_tools.output_sha256


def test_nonintegral_usage_is_invalid_and_not_billable(monkeypatch, route) -> None:
    monkeypatch.setenv("TEST_API_KEY", "fixture-only")
    body = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2.5, "completion_tokens": 3},
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    async def run() -> InferenceResult:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(route, _spec("fraction", stream=False))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.input_tokens is None
    assert "prompt_tokens_nonintegral_or_negative" in result.usage_parse_errors
    assessment = assess_result(result)
    assert assessment.classification == "invalid"
    direct = _success("direct-fraction")
    direct.input_tokens = 1.5  # type: ignore[assignment]
    assert "input_tokens_invalid_count" in assess_result(direct).reasons


def test_reasoning_state_stratifies_and_censors_visible_decode_proxy() -> None:
    zero = _success("zero", total_seconds=2.0)
    positive = _success("positive", total_seconds=2.0)
    positive.reasoning_tokens = 4
    unknown = _success("unknown", total_seconds=2.0)
    unknown.reasoning_tokens = None
    assert assess_result(zero).decode_eligible
    assert not assess_result(positive).decode_eligible
    assert "decode_proxy_hidden_reasoning_tokens_present" in assess_result(positive).reasons
    assert not assess_result(unknown).decode_eligible
    assert "decode_proxy_reasoning_token_state_unknown" in assess_result(unknown).reasons
    crossed = _success("crossed-arrival")
    crossed.arrival_to_completion_seconds = 0.001
    crossed_assessment = assess_result(crossed)
    assert "arrival_to_completion_precedes_component_duration" in crossed_assessment.reasons
    assert not crossed_assessment.latency_eligible


def test_invalid_nonfinite_metrics_remain_serializable_and_separate(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="f" * 64, config_json="{}")
    spec = _spec("nonfinite")
    assert ledger.claim(
        request_id="nonfinite-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.01,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    result = _success(spec.logical_id)
    result.total_seconds = float("inf")
    result.arrival_to_completion_seconds = float("inf")
    result.reasoning_tokens = 1.5  # type: ignore[assignment]
    result.cost_usd = 0.001
    result.cost_basis = "reserved_upper_bound"
    validity = assess_result(result)
    ledger.finish(
        request_id="nonfinite-1",
        result=result,
        validity=validity,
        quality_score=None,
    )
    audit = build_outlier_audit(ledger.rows())
    assert audit[0]["metric_values"]["total_seconds"] is None
    assert json.loads(canonical_json(audit))[0]["audit_class"] == "invalid"
    summarized = summarize_rows(ledger.rows())[0]
    assert summarized["reasoning_token_state"] == "unconditional_base_cell"
    assert summarized["reasoning_invalid_reported_n"] == 1
    ledger.close()


def test_arrival_latency_includes_queue_and_local_engine_time(tmp_path, campaign) -> None:
    async def run() -> InferenceResult:
        engine, ledger = _engine(tmp_path, campaign, CountingAdapter(delay=0.02))
        result = await engine.execute(_spec("arrival"), queue_delay_seconds=0.03)
        assert result is not None
        await engine.close()
        ledger.close()
        return result

    result = asyncio.run(run())
    assert result.arrival_to_completion_seconds is not None
    assert result.arrival_to_completion_seconds >= 0.05


def test_context_markers_are_independent_and_queried(route) -> None:
    first = context_marker_values("cell", 7)
    second = context_marker_values("cell", 8)
    assert len(set(first)) == 3
    assert first != second
    assert all(len(marker) == 24 for marker in first)
    spec = plan_context(route, {"percentages": [10]}, seed=7)[0]
    prompt = materialize_messages(spec)[0]["content"]
    for name in ("BEGIN_MARKER", "MIDDLE_MARKER", "END_MARKER"):
        # The exact values differ because the plan logical ID is part of the marker key; names and
        # all independently generated plan markers must still appear in both storage and query.
        assert name in prompt
    assert all(marker in prompt for marker in spec.metadata["context_markers"])


def test_expanded_plan_persists_required_conditional_and_censored_cells(tmp_path, campaign) -> None:
    config = replace(
        campaign,
        retries=0,
        suites={
            "latency": {"enabled": True, "repeats": 1, "shapes": ["short_short"]},
            "aimd": {
                "enabled": True,
                "shapes": ["short_short"],
                "epochs": 1,
                "epoch_seconds": 1,
                "initial_rps": 1,
                "additive_rps": 1,
            },
        },
    )
    plan = build_plan(config)
    assert any(
        cell["planned_disposition"] == "conditional_on_overload" for cell in plan.coverage_cells
    )
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=config.identity_hash, config_json="{}")
    ledger.register_plan_cells(list(plan.coverage_cells))
    first = plan.coverage_cells[0]["plan_cell_id"]
    ledger.mark_plan_cell(first, "completed")
    ledger.finalize_plan("cost_guard")
    rows = ledger.coverage_rows()
    assert next(row for row in rows if row["plan_cell_id"] == first)["state"] == "completed"
    assert all(row["state"] in {"completed", "cap_censored"} for row in rows)
    ledger.close()


def test_expected_validation_4xx_is_separate_from_unexpected_client_error() -> None:
    result = _success("validation")
    result.status = "client_error"
    result.http_status = 400
    result.input_tokens = None
    result.output_tokens = None
    result.reasoning_tokens = None
    result.ttft_seconds = None
    result.output_event_offsets_seconds = ()
    expected = assess_result(result, expected_rejection=True)
    unexpected = assess_result(result, expected_rejection=False)
    assert "expected_probe_observed_validation_http_status" in expected.reasons
    assert expected.classification == "censored"
    assert not expected.latency_eligible
    assert "unexpected_client_error" in unexpected.reasons
    assert unexpected.classification == "censored"
    accepted = _success("accepted-above-limit")
    accepted_assessment = assess_result(accepted, expected_rejection=True)
    assert "expected_validation_rejection_not_enforced_observed_acceptance" in (
        accepted_assessment.reasons
    )
    auth = replace(result, http_status=401)
    auth_assessment = assess_result(auth, expected_rejection=True)
    assert "expected_probe_observed_validation_http_status" not in auth_assessment.reasons
    assert "expected_probe_failed_for_nonvalidation_client_reason" in auth_assessment.reasons
    assert auth_assessment.classification == "censored"


def test_output_limit_warmup_retry_after_and_configurable_quota_header(monkeypatch, route) -> None:
    configured = replace(
        route,
        output_limit_field="max_completion_tokens",
        retained_header_names=("x-ratelimit-remaining-requests", "retry-after"),
    )
    payload = materialize_openai_compatible(configured, _spec()).value
    assert payload["max_completion_tokens"] == 16
    assert "max_tokens" not in payload
    warmups = plan_static_suites(
        (configured,),
        {"warmup": {"enabled": True, "repeats": 2, "shapes": ["short_short"]}},
        seed=1,
    )
    assert len(warmups) == 2 and all(spec.suite == "warmup" for spec in warmups)

    future = datetime.now(UTC) + timedelta(seconds=5)
    assert 0 < _retry_after({"retry-after": format_datetime(future)}) <= 6

    monkeypatch.setenv("TEST_API_KEY", "fixture-only")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            headers={"x-ratelimit-remaining-requests": "17", "x-secret": "drop"},
        )

    async def run() -> InferenceResult:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await OpenAICompatibleAdapter(client).infer(
            configured, _spec("headers", stream=False)
        )
        await client.aclose()
        return result

    retained = asyncio.run(run()).retained_headers
    assert retained == {"x-ratelimit-remaining-requests": "17"}


def test_retry_sleep_honors_provider_minimum_and_hard_wall(tmp_path, campaign, monkeypatch) -> None:
    class RetryAdapter(CountingAdapter):
        async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult:
            self.calls += 1
            if self.calls == 1:
                result = _success(request.logical_id)
                result.status = "rate_limited"
                result.http_status = 429
                result.input_tokens = None
                result.output_tokens = None
                result.reasoning_tokens = None
                result.retained_headers = {"retry-after": "2"}
                return result
            return _success(request.logical_id)

    sleeps: list[float] = []

    async def capture_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("inference_bench.engine.asyncio.sleep", capture_sleep)
    adapter = RetryAdapter()
    engine, ledger = _engine(tmp_path / "honor", replace(campaign, retries=1), adapter)

    async def honor() -> None:
        result = await engine.execute(replace(_spec("retry-minimum"), timeout_seconds=30))
        assert result is not None and result.status == "success"
        await engine.close()

    asyncio.run(honor())
    assert len(sleeps) == 1 and sleeps[0] >= 2
    ledger.close()

    sleeps.clear()
    constrained = replace(
        campaign,
        retries=1,
        max_wall_seconds=3,
        launch_reserve_seconds=0.1,
    )
    engine, ledger = _engine(tmp_path / "wall", constrained, RetryAdapter())

    async def blocked() -> None:
        with pytest.raises(TimeLimitReached):
            await engine.execute(replace(_spec("retry-wall"), timeout_seconds=2.7))
        await engine.close()

    asyncio.run(blocked())
    assert sleeps == []
    guarded_rows = ledger.rows()
    assert len(guarded_rows) == 1
    assert guarded_rows[0]["final_logical"] == 0
    assert guarded_rows[0]["settled_usd"] > 0
    guarded_summary = summarize_rows(guarded_rows)
    assert len(guarded_summary) == 1
    assert guarded_summary[0]["logical_requests_n"] == 0
    assert guarded_summary[0]["attempts_n"] == 1
    assert guarded_summary[0]["incomplete_retry_sequences_n"] == 1
    assert guarded_summary[0]["settled_usd_sum"] == pytest.approx(guarded_rows[0]["settled_usd"])
    guarded_audit = build_outlier_audit(guarded_rows)
    assert guarded_audit[0]["reasons"] == ["incomplete_retry_sequence_guarded_before_final_outcome"]
    assert {
        "success_rate",
        "latency",
        "quality",
        "input_tpm",
        "output_tpm",
    }.issubset(guarded_audit[0]["excluded_estimands"])
    ledger.close()


def test_strict_configuration_rejects_unknown_and_nonpositive_coverage(tmp_path, route) -> None:
    base = {
        "campaign": {
            "name": "strict",
            "max_wall_seconds": 60,
            "max_cost_usd": 1,
            "launch_reserve_seconds": 1,
            "launch_reserve_usd": 0.1,
        },
        "routes": [
            {
                "id": route.id,
                "provider": route.provider,
                "model": route.model,
                "base_url": route.base_url,
                "auth": {"env": "TEST_API_KEY"},
                "input_usd_per_million": 1,
                "output_usd_per_million": 1,
            }
        ],
        "suites": {"latency": {"enabled": True, "repeats": 1}},
    }

    def rejected(mutator, match: str) -> None:
        candidate = copy.deepcopy(base)
        mutator(candidate)
        path = tmp_path / f"bad-{hashlib.sha256(repr(candidate).encode()).hexdigest()[:8]}.yaml"
        path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(path)

    rejected(lambda value: value["campaign"].update({"mystery": 1}), "unknown campaign")
    rejected(lambda value: value["suites"]["latency"].update({"repeats": 0}), "positive")
    rejected(
        lambda value: value["suites"].update({"context": {"enabled": True, "percentages": [0]}}),
        "positive",
    )
    rejected(
        lambda value: value["suites"].update({"aimd": {"enabled": True, "epochs": 1.5}}),
        "positive integer",
    )
    rejected(
        lambda value: value["routes"][0].update({"retained_header_names": "x-rate"}),
        "nonempty list",
    )
    rejected(lambda value: value.update({"suites": {}}), "at least one benchmark suite")
    rejected(
        lambda value: value["suites"].update(
            {"context": {"enabled": True, "percentages": [10, 10]}}
        ),
        "duplicates",
    )
    rejected(
        lambda value: value["suites"].update(
            {
                "soak": {
                    "enabled": True,
                    "shapes": ["short_short"],
                    "rate_rps_by_route": {"typo-route": 1},
                }
            }
        ),
        "unknown soak route",
    )


def test_transport_and_live_identity_fail_before_claim(
    tmp_path, campaign, route, monkeypatch
) -> None:
    with pytest.raises(ValueError, match="absolute HTTPS"):
        replace(route, base_url="not-a-url")
    with pytest.raises(ValueError, match="absolute HTTPS"):
        replace(route, base_url="http://example.invalid/v1/chat")
    with pytest.raises(ValueError, match="invalid HTTP header"):
        replace(route, extra_headers={"X-Test": "bad\nvalue"})

    placeholder_model = replace(route, model="replace-with-exact-model-id")
    placeholder_engine, placeholder_ledger = _engine(
        tmp_path / "placeholder-model",
        replace(campaign, routes=(placeholder_model,)),
        CountingAdapter(),
    )
    with pytest.raises(ValueError, match="exact model"):
        placeholder_engine.preflight()
    placeholder_ledger.close()

    placeholder_docs = replace(
        route, capabilities={"documentation_checked_utc": "replace-with-UTC-date"}
    )
    docs_engine, docs_ledger = _engine(
        tmp_path / "placeholder-docs",
        replace(campaign, routes=(placeholder_docs,)),
        CountingAdapter(),
    )
    with pytest.raises(ValueError, match="documentation_checked_utc"):
        docs_engine.preflight()
    docs_ledger.close()

    monkeypatch.setenv("TEST_API_KEY", "bad\ncredential")
    ledger = Ledger(tmp_path / "header")
    ledger.initialize(campaign_hash=campaign.identity_hash, config_json="{}")
    engine = BenchmarkEngine(campaign, ledger)

    async def bad_header() -> None:
        with pytest.raises(RuntimeError, match="control characters"):
            await engine.execute(_spec("bad-header"))
        await engine.close()

    asyncio.run(bad_header())
    assert ledger.rows() == []
    ledger.close()

    monkeypatch.setenv("TEST_API_KEY", "fixture")
    for bad_campaign, bad_route, match in (
        (replace(campaign, client_location="not_reported"), route, "client_location"),
        (campaign, replace(route, quota_scope="not_reported"), "quota_scope"),
        (campaign, replace(route, model_version="not_reported"), "model_version"),
    ):
        scoped = replace(bad_campaign, routes=(bad_route,))
        engine, scoped_ledger = _engine(
            tmp_path / hashlib.sha256(match.encode()).hexdigest()[:8],
            scoped,
            CountingAdapter(),
        )

        async def rejected(active_engine: BenchmarkEngine = engine, expected: str = match) -> None:
            with pytest.raises(ValueError, match=expected):
                await active_engine.execute(_spec(f"identity-{expected}"))
            await active_engine.close()

        asyncio.run(rejected())
        assert scoped_ledger.rows() == []
        scoped_ledger.close()


def test_route_neutral_workloads_match_adversarial_route_ids(route) -> None:
    first = replace(route, id="a")
    second = replace(route, id="do")
    config = {"repeats": 8, "shapes": ["short_short", "long_short", "mixed"]}
    first_specs = plan_latency(first, config, seed=17)
    second_specs = plan_latency(second, config, seed=17)
    assert [spec.cell_id for spec in first_specs] == [spec.cell_id for spec in second_specs]
    assert [materialize_messages(spec) for spec in first_specs] == [
        materialize_messages(spec) for spec in second_specs
    ]
    assert [spec.metadata.get("mixed_subtype") for spec in first_specs] == [
        spec.metadata.get("mixed_subtype") for spec in second_specs
    ]
    first_cache = plan_cache(first, {"repeats": 2, "prefix_tokens": 64}, seed=17)
    second_cache = plan_cache(second, {"repeats": 2, "prefix_tokens": 64}, seed=17)
    assert [spec.messages for spec in first_cache] == [spec.messages for spec in second_cache]


def test_coverage_is_final_outcome_aware_and_missing_context_is_explicit(
    tmp_path, campaign, route
) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=campaign.identity_hash, config_json="{}")
    specs = [_spec("retry-cell"), _spec("unknown-cell")]
    ledger.register_plan_cells(
        [
            {
                "plan_cell_id": f"request:{spec.logical_id}",
                "logical_id": spec.logical_id,
                "route_id": spec.route_id,
                "suite": spec.suite,
                "cell_id": spec.cell_id,
            }
            for spec in specs
        ]
    )
    assert ledger.claim(
        request_id="retry-attempt",
        attempt_index=1,
        spec=specs[0],
        route=route,
        reserved_usd=0.01,
        max_cost_usd=10,
        cost_reserve_usd=1,
        scheduled_at_utc=None,
    )
    retryable = _success(specs[0].logical_id)
    retryable.status = "server_error"
    retryable.http_status = 503
    retryable.cost_usd = 0.001
    retryable.cost_basis = "reserved_upper_bound"
    ledger.finish(
        request_id="retry-attempt",
        result=retryable,
        validity=assess_result(retryable),
        quality_score=None,
    )
    retry_plan = next(row for row in ledger.coverage_rows() if row["logical_id"] == "retry-cell")
    assert retry_plan["state"] == "planned"
    assert ledger.claim(
        request_id="retry-final-attempt",
        attempt_index=2,
        spec=specs[0],
        route=route,
        reserved_usd=0.01,
        max_cost_usd=10,
        cost_reserve_usd=1,
        scheduled_at_utc=None,
    )
    final_success = _success(specs[0].logical_id)
    final_success.cost_usd = 0.001
    final_success.cost_basis = "provider_usage"
    ledger.finish(
        request_id="retry-final-attempt",
        result=final_success,
        validity=assess_result(final_success),
        quality_score=None,
        final_logical=True,
    )
    retry_plan = next(row for row in ledger.coverage_rows() if row["logical_id"] == "retry-cell")
    assert retry_plan["state"] == "completed"
    assert ledger.claim(
        request_id="unknown-attempt",
        attempt_index=1,
        spec=specs[1],
        route=route,
        reserved_usd=0.01,
        max_cost_usd=10,
        cost_reserve_usd=1,
        scheduled_at_utc=None,
    )
    ledger.mark_unknown("unknown-attempt", error_kind="ambiguous")
    unknown_row = next(row for row in ledger.coverage_rows() if row["logical_id"] == "unknown-cell")
    assert unknown_row["state"] == "inconclusive"
    ledger.close()

    missing = replace(route, context_tokens=None)
    plan = build_plan(replace(campaign, routes=(missing,), suites={"context": {"enabled": True}}))
    missing_cell = next(
        cell
        for cell in plan.coverage_cells
        if cell["cell_id"] == "documented_context_limit_missing"
    )
    assert missing_cell["initial_state"] == "inconclusive"

    missing_output = replace(route, max_output_tokens=None)
    output_plan = build_plan(
        replace(campaign, routes=(missing_output,), suites={"output": {"enabled": True}})
    )
    output_missing = next(
        cell
        for cell in output_plan.coverage_cells
        if cell["cell_id"] == "documented_output_limit_missing"
    )
    assert output_missing["initial_state"] == "inconclusive"
    fallback_probes = plan_output(missing_output, {"fallback_max_output_tokens": 128}, seed=1)
    assert all(not probe.metadata["expected_rejection"] for probe in fallback_probes)
    assert max(probe.max_output_tokens for probe in fallback_probes) == 128
    documented = replace(route, max_output_tokens=128)
    documented_probes = plan_output(documented, {}, seed=1)
    assert any(
        probe.max_output_tokens == 129 and probe.metadata["expected_rejection"]
        for probe in documented_probes
    )


def test_resumed_retry_censors_unreconstructable_arrival_latency(tmp_path, campaign, route) -> None:
    config = replace(campaign, retries=1)
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=config.identity_hash, config_json="{}")
    spec = _spec("resume-latency")
    assert ledger.claim(
        request_id="prior-attempt",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.01,
        max_cost_usd=10,
        cost_reserve_usd=1,
        scheduled_at_utc="2026-01-01T00:00:00Z",
    )
    prior = _success(spec.logical_id)
    prior.status = "server_error"
    prior.http_status = 503
    prior.cost_usd = 0.001
    prior.cost_basis = "reserved_upper_bound"
    ledger.finish(
        request_id="prior-attempt",
        result=prior,
        validity=assess_result(prior),
        quality_score=None,
    )
    engine = BenchmarkEngine(config, ledger)
    adapter = CountingAdapter()
    engine.adapters["openai_compatible"] = adapter

    async def run() -> InferenceResult:
        result = await engine.execute(spec)
        assert result is not None
        await engine.close()
        return result

    result = asyncio.run(run())
    assert result.arrival_to_completion_seconds is None
    assert result.arrival_latency_censor_reason == "resumed_retry_arrival_latency_unavailable"
    final = max(ledger.rows(), key=lambda row: int(row["attempt_index"]))
    assert final["arrival_to_completion_seconds"] is None
    assert final["arrival_latency_censor_reason"] == result.arrival_latency_censor_reason
    ledger.close()


def test_terminal_run_directory_refuses_live_reuse(tmp_path, campaign) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(
        campaign_hash=campaign.identity_hash, config_json=canonical_json(campaign.public_dict())
    )
    manifest_json = "{}"
    ledger.set_meta_once("run_manifest_json", manifest_json)
    ledger.set_meta_once(
        "terminal_run_manifest_sha256",
        hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )
    ledger.record_event_once(
        "campaign_terminal", "campaign_terminal", {"reason": "preflight_failed"}
    )
    ledger.close()
    report = tmp_path / "report"
    report.mkdir()
    (report / "summary.json").write_text('{"sealed":true}\n', encoding="utf-8")

    def snapshot() -> dict[str, str]:
        return {
            path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(tmp_path.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    with pytest.raises(ValueError, match="already terminal"):
        asyncio.run(run_campaign(campaign, tmp_path, invocation=("inference-bench", "run")))
    assert snapshot() == before


def test_runtime_identity_failure_creates_no_run_state(tmp_path, campaign, monkeypatch) -> None:
    output = tmp_path / "new-run"

    def fail_manifest(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("live runs require clean committed source")

    monkeypatch.setattr("inference_bench.cli._runtime_manifest", fail_manifest)
    with pytest.raises(RuntimeError, match="clean committed source"):
        asyncio.run(run_campaign(campaign, output, invocation=("inference-bench", "run")))

    assert not output.exists()


def test_sealed_terminal_run_does_not_repair_missing_projection(tmp_path, campaign) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(
        campaign_hash=campaign.identity_hash, config_json=canonical_json(campaign.public_dict())
    )
    manifest_json = "{}"
    ledger.set_meta_once("run_manifest_json", manifest_json)
    ledger.set_meta_once(
        "terminal_run_manifest_sha256",
        hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )
    ledger.record_event_once("campaign_terminal", "campaign_terminal", {"reason": "plan_completed"})
    ledger.close()
    (tmp_path / "events.jsonl").unlink()
    database_before = hashlib.sha256((tmp_path / "ledger.sqlite3").read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="already terminal"):
        asyncio.run(run_campaign(campaign, tmp_path, invocation=("inference-bench", "run")))

    assert not (tmp_path / "events.jsonl").exists()
    assert hashlib.sha256((tmp_path / "ledger.sqlite3").read_bytes()).hexdigest() == database_before


def test_terminal_crash_window_repairs_source_digest_without_sending(
    tmp_path, campaign, monkeypatch
) -> None:
    manifest = {
        "schema_version": "run-manifest/v2",
        "source_commit": "a" * 40,
        "source_dirty": False,
    }
    manifest_json = canonical_json(manifest)
    ledger = Ledger(tmp_path)
    ledger.initialize(
        campaign_hash=campaign.identity_hash, config_json=canonical_json(campaign.public_dict())
    )
    ledger.set_meta_once("run_manifest_json", manifest_json)
    ledger.record_event_once("campaign_terminal", "campaign_terminal", {"reason": "plan_completed"})
    ledger.close()

    monkeypatch.setattr("inference_bench.cli._runtime_manifest", lambda *args, **kwargs: manifest)
    with pytest.raises(ValueError, match="already terminal"):
        asyncio.run(run_campaign(campaign, tmp_path, invocation=("inference-bench", "run")))

    repaired = Ledger(tmp_path)
    assert (
        repaired.meta("terminal_run_manifest_sha256")
        == hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    )
    assert repaired.rows() == []
    assert not any(
        event["kind"] in {"request_claimed", "request_finished", "request_outcome_unknown"}
        for event in repaired.event_rows()
    )
    repaired.close()


def test_ledger_rejects_sanitized_config_drift_even_with_same_campaign_hash(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json='{"version":1}')
    with pytest.raises(ValueError, match="configuration changed"):
        ledger.initialize(campaign_hash="a" * 64, config_json='{"version":2}')
    ledger.close()


def test_report_requires_terminal_consistent_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "inference_bench.report._report_source_snapshot",
        lambda run_dir: {
            "source_revision": "a" * 40,
            "source_tree_state": "clean",
            "source_dirty_tree_sha256": hashlib.sha256(b"").hexdigest(),
            "distributions": {"inference-endpoint-benchmark": "0.1.0"},
        },
    )
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="c" * 64, config_json="{}")
    ledger.close()
    with pytest.raises(ValueError, match="campaign_terminal"):
        generate_report(tmp_path)


def test_dirty_tree_hash_binds_untracked_content_without_exposing_path(tmp_path) -> None:
    path = tmp_path / "private-name.txt"
    path.write_text("one", encoding="utf-8")
    first = _dirty_tree_hash(tmp_path, "?? private-name.txt", "", "private-name.txt")
    path.write_text("two", encoding="utf-8")
    second = _dirty_tree_hash(tmp_path, "?? private-name.txt", "", "private-name.txt")
    assert first and second and first != second
    assert "private-name" not in first


def test_live_invocation_redacts_launcher_config_and_both_output_spellings() -> None:
    separate = _normalize_live_invocation(
        (
            "./private-venv/bin/inference-bench",
            "run",
            "private-account.cfg",
            "--output",
            "private/run",
            "--confirm-live",
        )
    )
    combined = _normalize_live_invocation(
        (
            "private-launcher",
            "run",
            "ACCOUNT.YAML",
            "--output=private/run",
            "--confirm-live",
        )
    )
    assert separate == [
        "inference-bench",
        "run",
        "<CONFIG_OR_PATH>",
        "--output",
        "<RUN_DIR>",
        "--confirm-live",
    ]
    assert combined == [
        "inference-bench",
        "run",
        "<CONFIG_OR_PATH>",
        "--output=<RUN_DIR>",
        "--confirm-live",
    ]
    assert "private" not in canonical_json(separate + combined).casefold()


def test_documented_private_config_does_not_hide_tracked_source_drift(
    tmp_path, campaign, monkeypatch
) -> None:
    (tmp_path / ".gitignore").write_text(".private/\nruns/\n", encoding="utf-8")
    (tmp_path / "requirements.lock").write_text("httpx==0.28.1\n", encoding="utf-8")
    tracked_source = tmp_path / "README.md"
    tracked_source.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    private_config = tmp_path / ".private" / "campaign.yaml"
    private_config.parent.mkdir()
    private_config.write_text("private: true\n", encoding="utf-8")
    output = tmp_path / "runs" / "example"
    monkeypatch.setattr("inference_bench.cli._source_root", lambda: tmp_path)
    invocation = (
        "inference-bench",
        "run",
        str(private_config),
        "--output",
        str(output),
        "--confirm-live",
    )

    manifest = _runtime_manifest(campaign, invocation, output_dir=output)
    assert manifest["source_dirty"] is False

    tracked_source.write_text("drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean committed source"):
        _runtime_manifest(campaign, invocation, output_dir=output)


def test_capacity_cells_have_seeded_complete_isolated_execution_order(
    tmp_path, campaign, route
) -> None:
    second_route = replace(route, id="route-b", model="model-b")
    shapes = ["short_short", "long_short", "mixed"]
    configured = replace(
        campaign,
        routes=(route, second_route),
        suites={
            "aimd": {
                "enabled": True,
                "shapes": shapes,
                "epochs": 1,
                "epoch_seconds": 1,
            },
            "soak": {
                "enabled": True,
                "shapes": shapes,
                "blocks": 1,
                "block_seconds": 1,
            },
        },
    )

    def identities(config, suite):
        return [(item.id, shape) for item, shape in _capacity_execution_order(config, suite)]

    expected = {(item.id, shape) for item in configured.routes for shape in shapes}
    for suite in ("aimd", "soak"):
        first = identities(configured, suite)
        assert first == identities(configured, suite)
        assert len(first) == len(set(first)) == len(expected)
        assert set(first) == expected
        assert first != identities(replace(configured, seed=configured.seed + 1), suite)

    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=configured.identity_hash, config_json="{}")
    aimd_order = _capacity_execution_order(configured, "aimd")
    _record_capacity_execution_order(ledger, "aimd", aimd_order)
    event = ledger.event_by_key("capacity_execution_order:aimd")
    assert event is not None
    realized = json.loads(event["payload_json"])["cells"]
    assert [(item["route_id"], item["shape"]) for item in realized] == [
        (item.id, shape) for item, shape in aimd_order
    ]
    ledger.close()


def test_static_cells_have_seeded_within_block_order_and_resume_identity(
    tmp_path, campaign
) -> None:
    warmups = [
        replace(_spec(f"warmup-{index}"), suite="warmup", cell_id=f"warmup-{index}")
        for index in range(2)
    ]
    measured = [
        replace(_spec(f"context-{index}"), suite="context", cell_id=f"context-{index}")
        for index in range(20)
    ]
    specs = [*warmups, *measured]
    blocks = _static_execution_blocks(campaign, specs)
    repeated = _static_execution_blocks(campaign, specs)
    assert [[spec.logical_id for spec in block] for _, block in blocks] == [
        [spec.logical_id for spec in block] for _, block in repeated
    ]
    assert [spec.logical_id for spec in blocks[0][1]] == [spec.logical_id for spec in warmups]
    measured_order = [spec.logical_id for spec in blocks[1][1]]
    assert measured_order != [spec.logical_id for spec in measured]
    changed = _static_execution_blocks(replace(campaign, seed=campaign.seed + 1), specs)
    assert measured_order != [spec.logical_id for spec in changed[1][1]]

    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=campaign.identity_hash, config_json="{}")
    _record_static_execution_order(ledger, blocks)
    event = ledger.event_by_key("static_execution_order:v1")
    assert event is not None
    realized = json.loads(event["payload_json"])["cells"]
    assert [item["logical_id"] for item in realized] == [
        spec.logical_id for _, block in blocks for spec in block
    ]

    completed = set(measured_order[:5])

    class ResumeLedger:
        def attempts_for_logical(self, logical_id):
            if logical_id not in completed:
                return []
            return [
                {
                    "attempt_index": 1,
                    "state": "terminal",
                    "status": "success",
                }
            ]

        def mark_plan_cell(self, *args, **kwargs):
            return None

    class ResumeEngine:
        ledger = ResumeLedger()
        config = campaign

    pending = _pending_static_specs(ResumeEngine(), blocks[1][1])  # type: ignore[arg-type]
    assert [spec.logical_id for spec in pending] == measured_order[5:]
    ledger.close()


def test_run_directory_cannot_overlap_source_or_tracked_top_level(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = ["src/inference_bench/cli.py", "tests/test_integrity_contracts.py"]
    with pytest.raises(ValueError, match="dedicated non-source"):
        validate_run_directory_separation(root, root / "src" / "run-output", tracked)
    with pytest.raises(ValueError, match="cannot contain source"):
        validate_run_directory_separation(root, root.parent, tracked)
    validate_run_directory_separation(root, tmp_path / "outside-run", tracked)


def test_runtime_source_identity_is_verified_pre_send_and_terminal(
    tmp_path, campaign, monkeypatch
) -> None:
    manifest = {
        "schema_version": "run-manifest/v2",
        "source_commit": "a" * 40,
        "source_dirty": False,
    }
    manifest_json = canonical_json(manifest)
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash=campaign.identity_hash, config_json="{}")
    ledger.set_meta_once("run_manifest_json", manifest_json)
    monkeypatch.setattr("inference_bench.cli._runtime_manifest", lambda *args, **kwargs: manifest)
    invocation = ("inference-bench", "run", "private.cfg")
    _verify_runtime_identity(ledger, campaign, invocation, tmp_path, stage="pre_send")
    _verify_runtime_identity(ledger, campaign, invocation, tmp_path, stage="terminal")
    digest = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    assert ledger.meta("pre_send_run_manifest_sha256") == digest
    assert ledger.meta("terminal_run_manifest_sha256") == digest

    monkeypatch.setattr(
        "inference_bench.cli._runtime_manifest",
        lambda *args, **kwargs: {**manifest, "source_commit": "b" * 40},
    )
    with pytest.raises(RuntimeError, match="source identity changed"):
        _verify_runtime_identity(ledger, campaign, invocation, tmp_path, stage="terminal")
    assert any(event["kind"] == "source_identity_drift" for event in ledger.event_rows())
    ledger.close()


def test_environment_manifest_is_exact_lock_scope_not_ambient(tmp_path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("httpx==0.28.1\n", encoding="utf-8")
    observed = locked_distribution_versions(lock)
    assert observed == {
        "httpx": "0.28.1",
        "inference-endpoint-benchmark": "0.1.0",
    }
    lock.write_text("httpx==0.0.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="version mismatch"):
        locked_distribution_versions(lock)


def test_mixed_plan_cost_materializes_the_structured_subtype(route) -> None:
    structured = next(
        spec
        for index in range(64)
        if (
            spec := shape_spec(
                route,
                "mixed",
                f"test:{route.id}:structured:{index}",
                suite="test",
            )
        ).metadata.get("mixed_subtype")
        == "structured"
    )
    expected = route.worst_case_cost(
        reserved_input_tokens(route, structured, 1.5),
        structured.max_output_tokens,
    )
    assert _shape_cost(route, "mixed", 1.5) >= expected


def test_load_prompt_identity_is_epoch_unique_and_route_neutral(route) -> None:
    peer = replace(route, id="route-b")

    def payload_sha(candidate: RouteConfig, ordinal: int) -> str:
        epoch_id = f"aimd-{candidate.id}-short_short-{ordinal:03d}"
        key = _route_neutral_epoch_key(candidate.id, epoch_id)
        spec = shape_spec(
            candidate,
            "short_short",
            f"load:{candidate.id}:short_short:{epoch_id}:0",
            suite="load",
            seed=7,
            workload_key=(
                f"load:{{route}}:short_short:aimd:rps=1:sample_seed=7:epoch={key}:index=0"
            ),
        )
        return materialize_openai_compatible(candidate, spec).wire_body_sha256

    assert payload_sha(route, 0) == payload_sha(peer, 0)
    assert payload_sha(route, 1) == payload_sha(peer, 1)
    assert payload_sha(route, 0) != payload_sha(route, 1)
