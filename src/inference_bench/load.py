from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .engine import BenchmarkEngine, PaymentRequiredLatched
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


def scheduled_offsets(
    rate_rps: float, duration_seconds: float, *, seed: int, epoch_id: str
) -> list[float]:
    """The single schedule contract shared by planning and live execution."""
    return poisson_offsets(rate_rps, duration_seconds, seed=f"{seed}:{epoch_id}")


def soak_rate_rps(config: dict[str, Any], route_id: str, shape: str) -> float:
    """Resolve a soak rate at endpoint × shape granularity, with explicit fallbacks."""
    by_cell = config.get("rate_rps_by_route_shape") or {}
    if f"{route_id}:{shape}" in by_cell:
        return float(by_cell[f"{route_id}:{shape}"])
    route_cells = by_cell.get(route_id)
    if isinstance(route_cells, dict) and shape in route_cells:
        return float(route_cells[shape])
    by_route = config.get("rate_rps_by_route") or {}
    return float(by_route.get(route_id, config.get("rate_rps", 0.25)))


def validate_aimd_config(config: dict[str, Any], default_concurrency: int) -> None:
    if int(config.get("epochs", 12)) < 1:
        raise ValueError("aimd.epochs must be at least 1")
    if float(config.get("epoch_seconds", 20)) <= 0:
        raise ValueError("aimd.epoch_seconds must be positive")
    if float(config.get("initial_rps", 0.25)) <= 0:
        raise ValueError("aimd.initial_rps must be positive")
    if float(config.get("additive_rps", 0.25)) <= 0:
        raise ValueError("aimd.additive_rps must be positive")
    decrease = float(config.get("multiplicative_decrease", 0.5))
    if not 0 < decrease < 1:
        raise ValueError("aimd.multiplicative_decrease must lie strictly between 0 and 1")
    if int(config.get("concurrency", default_concurrency)) < 1:
        raise ValueError("aimd.concurrency must be at least 1")


def validate_soak_config(config: dict[str, Any], default_concurrency: int) -> None:
    if int(config.get("blocks", 4)) < 1:
        raise ValueError("soak.blocks must be at least 1")
    if float(config.get("block_seconds", 30)) <= 0:
        raise ValueError("soak.block_seconds must be positive")
    if int(config.get("concurrency", default_concurrency)) < 1:
        raise ValueError("soak.concurrency must be at least 1")


@dataclass(frozen=True, slots=True)
class EpochSummary:
    epoch_id: str
    route_id: str
    shape: str
    phase: str
    offered_rps: float
    duration_seconds: float
    actual_elapsed_seconds: float
    scheduled: int
    launched_logical: int
    completed: int
    physical_attempts: int
    physical_successes: int
    successful: int
    rate_limited: int
    server_errors: int
    timeouts: int
    transport_errors: int
    queue_end_seconds: float
    healthy: bool
    successful_input_tokens: int | None
    successful_output_tokens: int | None
    usage_complete_successful: int
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
    offsets = scheduled_offsets(
        offered_rps, duration_seconds, seed=seed, epoch_id=epoch_id
    )
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    epoch_started = loop.time()
    wall_started = datetime.now(UTC)
    results: list[InferenceResult] = []
    attempt_rows: list[dict[str, Any]] = []
    stop = asyncio.Event()
    executing_tasks: set[asyncio.Task[Any]] = set()

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
            spec = shape_spec(
                route,
                shape,
                logical,
                suite="load",
                cell_suffix=f":{phase}:rps={offered_rps:.9g}:epoch={epoch_id}",
            )
            current = asyncio.current_task()
            if current is not None:
                executing_tasks.add(current)
            try:
                result = await engine.execute(
                    spec,
                    scheduled_at_utc=_iso_at(wall_started, offset),
                    queue_delay_seconds=queue_delay,
                )
            except (BudgetExceeded, TimeLimitReached, PaymentRequiredLatched):
                stop.set()
                return
            finally:
                attempt_rows.extend(engine.ledger.attempts_for_logical(logical))
                if current is not None:
                    executing_tasks.discard(current)
            if result is not None:
                results.append(result)
                if result.http_status == 402:
                    stop.set()

    tasks = [asyncio.create_task(one(index, offset)) for index, offset in enumerate(offsets)]
    if tasks:
        all_done = asyncio.gather(*tasks, return_exceptions=True)
        stop_waiter = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait(
            {all_done, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        if stop_waiter in done and stop.is_set() and not all_done.done():
            for task in tasks:
                # Cancel only not-yet-sent arrivals/semaphore waiters. Requests already inside
                # execute must drain so their claimed ledger rows cannot become false unknowns.
                if not task.done() and task not in executing_tasks:
                    task.cancel()
        outcomes = await all_done
        stop_waiter.cancel()
        await asyncio.gather(stop_waiter, return_exceptions=True)
        unexpected = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, BaseException)
            and not isinstance(outcome, asyncio.CancelledError)
        ]
        if unexpected:
            raise unexpected[0]
    ended = loop.time()
    success = [result for result in results if result.status == "success"]
    terminal_attempts = [row for row in attempt_rows if row["state"] == "terminal"]
    rate_limited = sum(row["status"] == "rate_limited" for row in terminal_attempts)
    server_errors = sum(row["status"] == "server_error" for row in terminal_attempts)
    timeouts = sum(row["status"] == "timeout" for row in terminal_attempts)
    transport_errors = sum(row["status"] == "transport_error" for row in terminal_attempts)
    queue_end = max(0.0, ended - epoch_started - duration_seconds)
    actual_elapsed = max(0.0, ended - epoch_started)
    completed = len(results)
    success_rate = len(success) / completed if completed else 0.0
    physical_count = len(terminal_attempts)
    ttfts = [result.ttft_seconds for result in success if result.ttft_seconds is not None]
    totals = [result.total_seconds for result in success if result.total_seconds > 0]
    p95_ttft = quantile(ttfts, 0.95) if ttfts else None
    p95_total = quantile(totals, 0.95) if totals else None
    healthy = bool(
        completed
        and success_rate >= 0.99
        and physical_count > 0
        and rate_limited / physical_count <= 0.01
        and (server_errors + timeouts + transport_errors) / physical_count <= 0.01
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
        actual_elapsed_seconds=actual_elapsed,
        scheduled=len(offsets),
        launched_logical=len({row["logical_id"] for row in attempt_rows}),
        completed=completed,
        physical_attempts=physical_count,
        physical_successes=sum(row["status"] == "success" for row in terminal_attempts),
        successful=len(success),
        rate_limited=rate_limited,
        server_errors=server_errors,
        timeouts=timeouts,
        transport_errors=transport_errors,
        queue_end_seconds=queue_end,
        healthy=healthy,
        successful_input_tokens=(
            sum(int(result.input_tokens) for result in success if result.input_tokens is not None)
            if all(result.input_tokens is not None for result in success)
            else None
        ),
        successful_output_tokens=(
            sum(int(result.output_tokens) for result in success if result.output_tokens is not None)
            if all(result.output_tokens is not None for result in success)
            else None
        ),
        usage_complete_successful=sum(result.usage_complete for result in success),
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
    validate_aimd_config(config, engine.config.concurrency)
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
    overload_observed = False
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
                overload_observed = True
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
            if results[-1].launch_guard_triggered:
                break
            if confirmation < 2 and not results[-1].launch_guard_triggered:
                # A low-load separator makes confirmations distinct observations rather than one
                # contiguous long epoch cut into three labels.
                results.append(
                    await run_open_loop_epoch(
                        engine,
                        route,
                        shape=shape,
                        epoch_id=f"aimd-{route.id}-{shape}-separator-{confirmation}",
                        phase="confirmation_separator",
                        offered_rps=float(config.get("baseline_rps", min(rate, 0.1))),
                        duration_seconds=duration,
                        concurrency=ceiling,
                        seed=seed + 50 + confirmation,
                        max_p95_ttft_seconds=ttft_limit,
                        max_p95_total_seconds=total_limit,
                    )
                )
        if overload_observed and not results[-1].launch_guard_triggered:
            recovery_rate = max(0.01, best_healthy * 0.5)
            results.append(
                await run_open_loop_epoch(
                    engine,
                    route,
                    shape=shape,
                    epoch_id=f"aimd-{route.id}-{shape}-recovery",
                    phase="recovery_after_observed_overload",
                    offered_rps=recovery_rate,
                    duration_seconds=duration,
                    concurrency=ceiling,
                    seed=seed + 100,
                    max_p95_ttft_seconds=ttft_limit,
                    max_p95_total_seconds=total_limit,
                )
            )
        engine.ledger.record_event(
            "aimd_complete",
            {
                "route_id": route.id,
                "shape": shape,
                "candidate_rps": best_healthy,
                "overload_observed": overload_observed,
                "confirmation_healthy": [
                    epoch.healthy for epoch in results if epoch.phase == "confirmation"
                ],
                "recovery_run": overload_observed,
            },
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
    validate_soak_config(config, engine.config.concurrency)
    rate = soak_rate_rps(config, route.id, shape)
    if rate <= 0:
        raise ValueError(f"soak rate must be positive for {route.id}/{shape}")
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
