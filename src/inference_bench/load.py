from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .engine import BenchmarkEngine
from .ledger import BudgetExceeded, TimeLimitReached
from .models import InferenceResult, RouteConfig
from .statistics import quantile
from .workloads import shape_spec


def _iso_at(base: datetime, seconds: float) -> str:
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def poisson_offsets(rate_rps: float, duration_seconds: float, *, seed: str) -> list[float]:
    if rate_rps <= 0 or duration_seconds <= 0:
        raise ValueError("rate and duration must be positive")
    rng = random.Random(seed)
    offsets: list[float] = []
    current = 0.0
    while True:
        current += rng.expovariate(rate_rps)
        if current >= duration_seconds:
            break
        offsets.append(current)
    return offsets


@dataclass(frozen=True, slots=True)
class EpochSummary:
    epoch_id: str
    route_id: str
    shape: str
    phase: str
    offered_rps: float
    duration_seconds: float
    scheduled: int
    completed: int
    successful: int
    rate_limited: int
    server_errors: int
    timeouts: int
    queue_end_seconds: float
    healthy: bool
    successful_input_tokens: int
    successful_output_tokens: int
    ttft_observed_n: int
    p95_ttft_seconds: float | None
    p95_total_seconds: float | None
    launch_guard_triggered: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_open_loop_epoch(
    engine: BenchmarkEngine,
    route: RouteConfig,
    *,
    shape: str,
    epoch_id: str,
    phase: str,
    offered_rps: float,
    duration_seconds: float,
    concurrency: int,
    seed: int,
    max_p95_ttft_seconds: float | None = None,
    max_p95_total_seconds: float | None = None,
) -> EpochSummary:
    """Open-loop arrivals with a separate concurrency ceiling.

    Every arrival is scheduled from the epoch clock before any request completes. When the ceiling
    is busy, the arrival waits and its queue delay is retained rather than omitted.
    """
    offsets = poisson_offsets(offered_rps, duration_seconds, seed=f"{seed}:{epoch_id}")
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    epoch_started = loop.time()
    wall_started = datetime.now(UTC)
    results: list[InferenceResult] = []
    stop = asyncio.Event()

    async def one(index: int, offset: float) -> None:
        await asyncio.sleep(max(0.0, epoch_started + offset - loop.time()))
        if stop.is_set():
            return
        arrived = loop.time()
        async with semaphore:
            if stop.is_set():
                return
            queue_delay = loop.time() - arrived
            logical = f"load:{route.id}:{shape}:{epoch_id}:{index}"
            spec = shape_spec(route, shape, logical, suite="load", cell_suffix=f":{phase}")
            try:
                result = await engine.execute(
                    spec,
                    scheduled_at_utc=_iso_at(wall_started, offset),
                    queue_delay_seconds=queue_delay,
                )
            except (BudgetExceeded, TimeLimitReached):
                stop.set()
                return
            if result is not None:
                results.append(result)

    tasks = [asyncio.create_task(one(index, offset)) for index, offset in enumerate(offsets)]
    if tasks:
        await asyncio.gather(*tasks)
    ended = loop.time()
    success = [result for result in results if result.status == "success"]
    rate_limited = sum(result.status == "rate_limited" for result in results)
    server_errors = sum(result.status == "server_error" for result in results)
    timeouts = sum(result.status == "timeout" for result in results)
    queue_end = max(0.0, ended - epoch_started - duration_seconds)
    completed = len(results)
    success_rate = len(success) / completed if completed else 0.0
    ttfts = [result.ttft_seconds for result in success if result.ttft_seconds is not None]
    totals = [result.total_seconds for result in success if result.total_seconds > 0]
    p95_ttft = quantile(ttfts, 0.95) if ttfts else None
    p95_total = quantile(totals, 0.95) if totals else None
    healthy = bool(
        completed
        and success_rate >= 0.99
        and rate_limited / completed <= 0.01
        and (server_errors + timeouts) / completed <= 0.01
        and queue_end <= max(1.0, duration_seconds * 0.1)
        and len(ttfts) == len(success)
        and (
            max_p95_ttft_seconds is None
            or (p95_ttft is not None and p95_ttft <= max_p95_ttft_seconds)
        )
        and (
            max_p95_total_seconds is None
            or (p95_total is not None and p95_total <= max_p95_total_seconds)
        )
    )
    summary = EpochSummary(
        epoch_id=epoch_id,
        route_id=route.id,
        shape=shape,
        phase=phase,
        offered_rps=offered_rps,
        duration_seconds=duration_seconds,
        scheduled=len(offsets),
        completed=completed,
        successful=len(success),
        rate_limited=rate_limited,
        server_errors=server_errors,
        timeouts=timeouts,
        queue_end_seconds=queue_end,
        healthy=healthy,
        successful_input_tokens=sum(result.input_tokens or 0 for result in success),
        successful_output_tokens=sum(result.output_tokens or 0 for result in success),
        ttft_observed_n=len(ttfts),
        p95_ttft_seconds=p95_ttft,
        p95_total_seconds=p95_total,
        launch_guard_triggered=stop.is_set(),
    )
    engine.ledger.record_event("load_epoch", summary.to_dict())
    return summary


async def run_aimd(
    engine: BenchmarkEngine,
    route: RouteConfig,
    shape: str,
    config: dict[str, Any],
    *,
    seed: int,
) -> list[EpochSummary]:
    epochs = int(config.get("epochs", 12))
    duration = float(config.get("epoch_seconds", 20))
    rate = float(config.get("initial_rps", 0.25))
    additive = float(config.get("additive_rps", 0.25))
    decrease = float(config.get("multiplicative_decrease", 0.5))
    ceiling = int(config.get("concurrency", engine.config.concurrency))
    baseline = await run_open_loop_epoch(
        engine,
        route,
        shape=shape,
        epoch_id=f"aimd-{route.id}-{shape}-baseline",
        phase="baseline",
        offered_rps=float(config.get("baseline_rps", min(rate, 0.1))),
        duration_seconds=duration,
        concurrency=ceiling,
        seed=seed - 1,
    )
    results: list[EpochSummary] = [baseline]
    if baseline.launch_guard_triggered:
        return results
    ttft_limit = None if baseline.p95_ttft_seconds is None else 2 * baseline.p95_ttft_seconds
    total_limit = None if baseline.p95_total_seconds is None else 2 * baseline.p95_total_seconds
    unhealthy_streak = 0
    best_healthy = 0.0
    for index in range(epochs):
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape=shape,
            epoch_id=f"aimd-{route.id}-{shape}-{index:03d}",
            phase="aimd",
            offered_rps=rate,
            duration_seconds=duration,
            concurrency=ceiling,
            seed=seed,
            max_p95_ttft_seconds=ttft_limit,
            max_p95_total_seconds=total_limit,
        )
        results.append(summary)
        if summary.launch_guard_triggered:
            break
        if summary.healthy:
            best_healthy = max(best_healthy, rate)
            unhealthy_streak = 0
            rate += additive
        else:
            unhealthy_streak += 1
            if unhealthy_streak >= 2:
                rate = max(0.01, rate * decrease)
                unhealthy_streak = 0

    # Three separated confirmations are explicit; they are not silently inferred from ramp epochs.
    if best_healthy > 0:
        for confirmation in range(3):
            results.append(
                await run_open_loop_epoch(
                    engine,
                    route,
                    shape=shape,
                    epoch_id=f"aimd-{route.id}-{shape}-confirm-{confirmation}",
                    phase="confirmation",
                    offered_rps=best_healthy,
                    duration_seconds=duration,
                    concurrency=ceiling,
                    seed=seed + confirmation + 1,
                    max_p95_ttft_seconds=ttft_limit,
                    max_p95_total_seconds=total_limit,
                )
            )
        recovery_rate = max(0.01, best_healthy * 0.5)
        results.append(
            await run_open_loop_epoch(
                engine,
                route,
                shape=shape,
                epoch_id=f"aimd-{route.id}-{shape}-recovery",
                phase="recovery",
                offered_rps=recovery_rate,
                duration_seconds=duration,
                concurrency=ceiling,
                seed=seed + 100,
                max_p95_ttft_seconds=ttft_limit,
                max_p95_total_seconds=total_limit,
            )
        )
    return results


async def run_soak(
    engine: BenchmarkEngine,
    route: RouteConfig,
    shape: str,
    config: dict[str, Any],
    *,
    seed: int,
) -> list[EpochSummary]:
    rate_by_route = config.get("rate_rps_by_route") or {}
    rate = float(rate_by_route.get(route.id, config.get("rate_rps", 0.25)))
    blocks = int(config.get("blocks", 4))
    block_seconds = float(config.get("block_seconds", 30))
    ceiling = int(config.get("concurrency", engine.config.concurrency))
    baseline = await run_open_loop_epoch(
        engine,
        route,
        shape=shape,
        epoch_id=f"soak-{route.id}-{shape}-baseline",
        phase="soak_baseline",
        offered_rps=float(config.get("baseline_rps", min(rate, 0.1))),
        duration_seconds=block_seconds,
        concurrency=ceiling,
        seed=seed - 1,
    )
    ttft_limit = None if baseline.p95_ttft_seconds is None else 2 * baseline.p95_ttft_seconds
    total_limit = None if baseline.p95_total_seconds is None else 2 * baseline.p95_total_seconds
    soak_blocks: list[EpochSummary] = []
    if baseline.launch_guard_triggered:
        return [baseline]
    for block in range(blocks):
        soak_blocks.append(
            await run_open_loop_epoch(
                engine,
                route,
                shape=shape,
                epoch_id=f"soak-{route.id}-{shape}-block-{block}",
                phase="soak_block",
                offered_rps=rate,
                duration_seconds=block_seconds,
                concurrency=ceiling,
                seed=seed + block,
                max_p95_ttft_seconds=ttft_limit,
                max_p95_total_seconds=total_limit,
            )
        )
        if soak_blocks[-1].launch_guard_triggered:
            break
    engine.ledger.record_event(
        "soak_complete",
        {
            "route_id": route.id,
            "shape": shape,
            "rate_rps": rate,
            "blocks": blocks,
            "all_blocks_healthy": all(block.healthy for block in soak_blocks),
        },
    )
    return [baseline, *soak_blocks]
