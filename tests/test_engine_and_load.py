from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace

import pytest

from inference_bench.engine import BenchmarkEngine, PaymentRequiredLatched
from inference_bench.ledger import Ledger
from inference_bench.load import run_open_loop_epoch
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
    assert all("rps=30" in row["cell_id"] and "epoch=retry-epoch" in row["cell_id"] for row in rows)
    matched = summarize_rows(rows)[0]
    assert matched["attempts_n"] == 2 * summary.scheduled
    assert matched["logical_requests_n"] == summary.scheduled
    assert matched["success_rate"] == 1.0
    assert matched["request_sampling_unit"] == "final terminal attempt per logical request"


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
    assert row["settled_usd"] == pytest.approx(0.000028)
    assert row["validity_class"] == "invalid"
