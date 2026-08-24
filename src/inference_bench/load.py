from __future__ import annotations

import asyncio
import random
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .engine import BenchmarkEngine, PaymentRequiredLatched, ReservationOverrunLatched
from .ledger import BudgetExceeded, TimeLimitReached
from .models import RouteConfig
from .statistics import quantile
from .workloads import shape_spec

MIN_BASELINE_SAMPLES = 20


def _strict_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if result <= 0 or result == float("inf") or result == float("-inf") or result != result:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _strict_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _strict_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


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


def fixed_count_offsets(count: int, duration_seconds: float) -> list[float]:
    """Deterministic open-loop arrivals centered in equal-width time bins."""

    _strict_positive_int(count, "baseline_samples")
    _strict_positive_float(duration_seconds, "baseline_duration_seconds")
    return [(index + 0.5) * duration_seconds / count for index in range(count)]


def baseline_design(
    config: dict[str, Any], nominal_duration: float, *, default_rps: float | None = None
) -> tuple[int, float, float]:
    """Return exact sample count, duration, and truthful offered RPS for a low-load baseline."""

    samples = _strict_positive_int(
        config.get("baseline_samples", MIN_BASELINE_SAMPLES), "baseline_samples"
    )
    if samples < MIN_BASELINE_SAMPLES:
        raise ValueError(f"baseline_samples must be at least {MIN_BASELINE_SAMPLES}")
    duration = _strict_positive_float(nominal_duration, "baseline nominal duration")
    requested_rate_value = config.get("baseline_rps", default_rps)
    if requested_rate_value is not None:
        requested_rate = _strict_positive_float(requested_rate_value, "baseline_rps")
        duration = max(duration, samples / requested_rate)
    return samples, duration, samples / duration


def next_healthy_aimd_rate(
    current_rps: float,
    *,
    healthy_increases: int,
    overload_observed: bool,
    additive_rps: float,
    bracket_epochs: int,
    bracket_multiplier: float,
    max_rps: float | None,
) -> float:
    """Shared all-healthy rate transition for live AIMD and conservative planning."""

    if not overload_observed and healthy_increases < bracket_epochs:
        candidate = current_rps * bracket_multiplier
    else:
        candidate = current_rps + additive_rps
    if max_rps is not None:
        candidate = min(max_rps, candidate)
    return candidate


def _route_neutral_epoch_key(route_id: str, epoch_id: str) -> str:
    """Remove only the controller's exact route slot while preserving epoch/block identity."""

    for controller in ("aimd", "soak"):
        prefix = f"{controller}-{route_id}-"
        if epoch_id.startswith(prefix):
            return f"{controller}-{{route}}-{epoch_id.removeprefix(prefix)}"
    return epoch_id


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
    epochs = _strict_positive_int(config.get("epochs", 12), "aimd.epochs")
    _strict_positive_float(config.get("epoch_seconds", 20), "aimd.epoch_seconds")
    initial_rps = _strict_positive_float(config.get("initial_rps", 0.25), "aimd.initial_rps")
    _strict_positive_float(config.get("additive_rps", 0.25), "aimd.additive_rps")
    decrease = _strict_positive_float(
        config.get("multiplicative_decrease", 0.5), "aimd.multiplicative_decrease"
    )
    if not 0 < decrease < 1:
        raise ValueError("aimd.multiplicative_decrease must lie strictly between 0 and 1")
    bracket_epochs = _strict_nonnegative_int(
        config.get("bracket_epochs", min(6, epochs)), "aimd.bracket_epochs"
    )
    if bracket_epochs > epochs:
        raise ValueError("aimd.bracket_epochs must not exceed aimd.epochs")
    bracket_multiplier = _strict_positive_float(
        config.get("bracket_multiplier", 2.0), "aimd.bracket_multiplier"
    )
    if bracket_epochs and bracket_multiplier <= 1:
        raise ValueError("aimd.bracket_multiplier must exceed 1 when bracketing is enabled")
    if "max_rps" in config:
        max_rps = _strict_positive_float(config["max_rps"], "aimd.max_rps")
        if max_rps < initial_rps:
            raise ValueError("aimd.max_rps must be at least aimd.initial_rps")
    _strict_positive_int(config.get("concurrency", default_concurrency), "aimd.concurrency")
    if "baseline_rps" in config:
        _strict_positive_float(config["baseline_rps"], "aimd.baseline_rps")
    samples = _strict_positive_int(
        config.get("baseline_samples", MIN_BASELINE_SAMPLES), "aimd.baseline_samples"
    )
    if samples < MIN_BASELINE_SAMPLES:
        raise ValueError(f"aimd.baseline_samples must be at least {MIN_BASELINE_SAMPLES}")


def validate_soak_config(config: dict[str, Any], default_concurrency: int) -> None:
    _strict_positive_int(config.get("blocks", 4), "soak.blocks")
    _strict_positive_float(config.get("block_seconds", 30), "soak.block_seconds")
    _strict_positive_int(config.get("concurrency", default_concurrency), "soak.concurrency")
    if "baseline_rps" in config:
        _strict_positive_float(config["baseline_rps"], "soak.baseline_rps")
    samples = _strict_positive_int(
        config.get("baseline_samples", MIN_BASELINE_SAMPLES), "soak.baseline_samples"
    )
    if samples < MIN_BASELINE_SAMPLES:
        raise ValueError(f"soak.baseline_samples must be at least {MIN_BASELINE_SAMPLES}")


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
    unknown: int
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
    p95_service_seconds: float | None
    p95_total_seconds: float | None
    launch_guard_triggered: bool
    launch_guard_reason: str | None
    controller_eligible: bool
    scientific_censor_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_RETRYABLE_STATUSES = {"rate_limited", "server_error", "timeout", "transport_error"}


def _is_final_logical_attempt(row: dict[str, Any], retries: int) -> bool:
    if "final_logical" in row:
        return bool(row["final_logical"])
    return bool(
        row["state"] == "unknown"
        or row.get("status") == "success"
        or row.get("status") not in _RETRYABLE_STATUSES
        or int(row["attempt_index"]) >= retries + 1
    )


def _build_epoch_summary(
    engine: BenchmarkEngine,
    route: RouteConfig,
    *,
    shape: str,
    epoch_id: str,
    phase: str,
    offered_rps: float,
    duration_seconds: float,
    logical_ids: list[str],
    actual_elapsed: float,
    stop_reason: str | None,
    scientific_censor_reason: str | None,
    max_p95_ttft_seconds: float | None,
    max_p95_total_seconds: float | None,
) -> EpochSummary:
    if not logical_ids and scientific_censor_reason is None:
        scientific_censor_reason = "zero_scheduled_poisson_arrivals"
    attempts_by_logical = {
        logical: engine.ledger.attempts_for_logical(logical) for logical in logical_ids
    }
    attempt_rows = [row for rows in attempts_by_logical.values() for row in rows]
    final_rows = []
    for rows in attempts_by_logical.values():
        if not rows:
            continue
        latest = max(rows, key=lambda row: int(row["attempt_index"]))
        if _is_final_logical_attempt(latest, engine.config.retries):
            final_rows.append(latest)
    physical_attempts = [row for row in attempt_rows if row["state"] in {"terminal", "unknown"}]
    success = [
        row for row in final_rows if row["state"] == "terminal" and row["status"] == "success"
    ]
    unknown = [row for row in final_rows if row["state"] == "unknown"]
    rate_limited = sum(row["status"] == "rate_limited" for row in physical_attempts)
    server_errors = sum(row["status"] == "server_error" for row in physical_attempts)
    timeouts = sum(row["status"] == "timeout" for row in physical_attempts)
    transport_errors = sum(row["status"] == "transport_error" for row in physical_attempts)
    queue_end = max(0.0, actual_elapsed - duration_seconds)
    completed = sum(row["state"] == "terminal" for row in final_rows)
    logical_observed = len(final_rows)
    success_rate = len(success) / len(logical_ids) if logical_ids else 0.0
    physical_count = len(physical_attempts)
    ttfts = [float(row["ttft_seconds"]) for row in success if row["ttft_seconds"] is not None]
    services = [
        float(row["total_seconds"])
        for row in success
        if row["total_seconds"] is not None and float(row["total_seconds"]) > 0
    ]
    arrivals = [
        float(row["arrival_to_completion_seconds"])
        for row in success
        if row["arrival_to_completion_seconds"] is not None
        and float(row["arrival_to_completion_seconds"]) > 0
    ]
    p95_ttft = quantile(ttfts, 0.95) if ttfts else None
    p95_service = quantile(services, 0.95) if services else None
    p95_total = quantile(arrivals, 0.95) if arrivals else None
    healthy = bool(
        completed
        and stop_reason is None
        and not unknown
        and logical_observed == len(logical_ids)
        and success_rate >= 0.99
        and physical_count > 0
        and rate_limited / physical_count <= 0.01
        and (server_errors + timeouts + transport_errors) / physical_count <= 0.01
        and queue_end <= max(1.0, duration_seconds * 0.1)
        and len(ttfts) == len(success)
        and len(arrivals) == len(success)
        and (
            max_p95_ttft_seconds is None
            or (p95_ttft is not None and p95_ttft <= max_p95_ttft_seconds)
        )
        and (
            max_p95_total_seconds is None
            or (p95_total is not None and p95_total <= max_p95_total_seconds)
        )
    )
    return EpochSummary(
        epoch_id=epoch_id,
        route_id=route.id,
        shape=shape,
        phase=phase,
        offered_rps=offered_rps,
        duration_seconds=duration_seconds,
        actual_elapsed_seconds=max(0.0, actual_elapsed),
        scheduled=len(logical_ids),
        launched_logical=len({row["logical_id"] for row in attempt_rows}),
        completed=completed,
        unknown=len(unknown),
        physical_attempts=physical_count,
        physical_successes=sum(row["status"] == "success" for row in physical_attempts),
        successful=len(success),
        rate_limited=rate_limited,
        server_errors=server_errors,
        timeouts=timeouts,
        transport_errors=transport_errors,
        queue_end_seconds=queue_end,
        healthy=healthy,
        successful_input_tokens=(
            sum(int(row["input_tokens"]) for row in success if row["input_tokens"] is not None)
            if all(row["input_tokens"] is not None and row["usage_eligible"] for row in success)
            else None
        ),
        successful_output_tokens=(
            sum(int(row["output_tokens"]) for row in success if row["output_tokens"] is not None)
            if all(row["output_tokens"] is not None and row["usage_eligible"] for row in success)
            else None
        ),
        usage_complete_successful=sum(bool(row["usage_eligible"]) for row in success),
        ttft_observed_n=len(ttfts),
        p95_ttft_seconds=p95_ttft,
        p95_service_seconds=p95_service,
        p95_total_seconds=p95_total,
        launch_guard_triggered=stop_reason is not None,
        launch_guard_reason=stop_reason,
        controller_eligible=scientific_censor_reason is None,
        scientific_censor_reason=scientific_censor_reason,
    )


def _record_epoch_summary(engine: BenchmarkEngine, summary: EpochSummary) -> None:
    engine.ledger.record_event_once(
        f"load_epoch:{summary.epoch_id}", "load_epoch", summary.to_dict()
    )
    plan_cell_id = f"load_epoch:{summary.epoch_id}"
    try:
        if summary.launch_guard_reason == "cost_guard":
            engine.ledger.mark_plan_cell(plan_cell_id, "cap_censored", summary.launch_guard_reason)
        elif summary.launch_guard_reason == "time_guard":
            engine.ledger.mark_plan_cell(plan_cell_id, "time_censored", summary.launch_guard_reason)
        elif summary.launch_guard_reason:
            engine.ledger.mark_plan_cell(plan_cell_id, "inconclusive", summary.launch_guard_reason)
        elif summary.scientific_censor_reason:
            engine.ledger.mark_plan_cell(
                plan_cell_id, "inconclusive", summary.scientific_censor_reason
            )
        elif summary.scheduled == 0:
            engine.ledger.mark_plan_cell(
                plan_cell_id, "inconclusive", "zero_scheduled_poisson_arrivals"
            )
        elif summary.unknown == 0 and summary.completed == summary.scheduled:
            engine.ledger.mark_plan_cell(plan_cell_id, "completed")
        else:
            engine.ledger.mark_plan_cell(plan_cell_id, "inconclusive", "missing_scheduled_arrivals")
    except KeyError:
        # Unit-level direct epoch use may intentionally omit a campaign coverage plan.
        pass


def _censor_unstarted_load_cells(
    engine: BenchmarkEngine, epoch_ids: list[str], reason: str
) -> None:
    for epoch_id in epoch_ids:
        engine.ledger.mark_plan_cell_if_planned(f"load_epoch:{epoch_id}", "inconclusive", reason)


def _reconstructed_epoch_elapsed(
    attempts_by_logical: dict[str, list[dict[str, Any]]],
    logical_ids: list[str],
    offsets: list[float],
    duration_seconds: float,
) -> float:
    starts: list[datetime] = []
    ends: list[datetime] = []
    offset_by_logical = dict(zip(logical_ids, offsets, strict=True))
    for logical_id, rows in attempts_by_logical.items():
        for row in rows:
            scheduled = row.get("scheduled_at_utc")
            ended = row.get("ended_at_utc")
            if scheduled:
                starts.append(
                    datetime.fromisoformat(str(scheduled).replace("Z", "+00:00"))
                    - timedelta(seconds=offset_by_logical[logical_id])
                )
            if ended:
                ends.append(datetime.fromisoformat(str(ended).replace("Z", "+00:00")))
    if not starts or not ends:
        return duration_seconds
    return max(0.0, (max(ends) - min(starts)).total_seconds())


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
    shape_config: dict[str, Any] | None = None,
    deterministic_scheduled_count: int | None = None,
    max_p95_ttft_seconds: float | None = None,
    max_p95_total_seconds: float | None = None,
) -> EpochSummary:
    """Open-loop arrivals with a separate concurrency ceiling.

    Every arrival is scheduled from the epoch clock before any request completes. When the ceiling
    is busy, the arrival waits and its queue delay is retained rather than omitted.
    """
    event_key = f"load_epoch:{epoch_id}"
    existing_summary = engine.ledger.event_by_key(event_key)
    if existing_summary is not None:
        import json

        stored_payload = json.loads(str(existing_summary["payload_json"]))
        stored_payload.setdefault("controller_eligible", True)
        stored_payload.setdefault("scientific_censor_reason", None)
        restored = EpochSummary(**stored_payload)
        expected_identity = (
            route.id,
            shape,
            phase,
            offered_rps,
            duration_seconds,
            deterministic_scheduled_count,
        )
        restored_identity = (
            restored.route_id,
            restored.shape,
            restored.phase,
            restored.offered_rps,
            restored.duration_seconds,
            restored.scheduled if deterministic_scheduled_count is not None else None,
        )
        if restored_identity != expected_identity:
            raise ValueError(
                f"epoch identity changed for {epoch_id}: "
                f"stored={restored_identity!r}, requested={expected_identity!r}"
            )
        try:
            restored_state = (
                "cap_censored"
                if restored.launch_guard_reason == "cost_guard"
                else "time_censored"
                if restored.launch_guard_reason == "time_guard"
                else "inconclusive"
                if restored.launch_guard_reason
                else "completed"
                if restored.unknown == 0
                and restored.completed == restored.scheduled
                and restored.scheduled > 0
                else "inconclusive"
            )
            engine.ledger.mark_plan_cell(
                f"load_epoch:{epoch_id}",
                restored_state,
                restored.launch_guard_reason or restored.scientific_censor_reason,
            )
        except KeyError:
            pass
        return restored

    offsets = (
        fixed_count_offsets(deterministic_scheduled_count, duration_seconds)
        if deterministic_scheduled_count is not None
        else scheduled_offsets(offered_rps, duration_seconds, seed=seed, epoch_id=epoch_id)
    )
    logical_ids = [f"load:{route.id}:{shape}:{epoch_id}:{index}" for index in range(len(offsets))]
    preexisting = {logical: engine.ledger.attempts_for_logical(logical) for logical in logical_ids}
    if any(preexisting.values()):
        reconstructed_elapsed = _reconstructed_epoch_elapsed(
            preexisting, logical_ids, offsets, duration_seconds
        )
        final_count = 0
        unknown_count = 0
        for rows in preexisting.values():
            if not rows:
                continue
            latest = max(rows, key=lambda row: int(row["attempt_index"]))
            final_count += int(_is_final_logical_attempt(latest, engine.config.retries))
            unknown_count += int(latest["state"] == "unknown")
        scientific_censor_reason = None
        if unknown_count:
            scientific_censor_reason = "interrupted_epoch_unknown_provider_outcomes_no_replay"
        elif final_count != len(logical_ids):
            scientific_censor_reason = "interrupted_epoch_incomplete_no_replay"
        if scientific_censor_reason is not None:
            # Missing arrivals from an interrupted epoch are never replayed, but they are a
            # censored scientific cell rather than a global launch guard. Continue with later
            # pre-registered epochs/routes so one crash does not terminalize the whole campaign.
            engine.ledger.record_event_once(
                f"load_epoch_resume_censored:{epoch_id}",
                "load_epoch_resume_censored",
                {
                    "epoch_id": epoch_id,
                    "route_id": route.id,
                    "shape": shape,
                    "final_logical_outcomes": final_count,
                    "unknown_logical_outcomes": unknown_count,
                    "scheduled": len(logical_ids),
                    "reason": scientific_censor_reason,
                },
            )
        reconstructed = _build_epoch_summary(
            engine,
            route,
            shape=shape,
            epoch_id=epoch_id,
            phase=phase,
            offered_rps=offered_rps,
            duration_seconds=duration_seconds,
            logical_ids=logical_ids,
            actual_elapsed=reconstructed_elapsed,
            stop_reason=None,
            scientific_censor_reason=scientific_censor_reason,
            max_p95_ttft_seconds=max_p95_ttft_seconds,
            max_p95_total_seconds=max_p95_total_seconds,
        )
        _record_epoch_summary(engine, reconstructed)
        return reconstructed
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    epoch_started = loop.time()
    wall_started = datetime.now(UTC)
    stop = asyncio.Event()
    stop_reason: str | None = None
    executing_tasks: set[asyncio.Task[Any]] = set()
    stop_priority = {
        "unexpected_runner_error": 1,
        "time_guard": 2,
        "cost_guard": 3,
        "reservation_overrun_latch": 4,
        "http_402_latch": 5,
    }

    def set_stop_reason(reason: str) -> None:
        nonlocal stop_reason
        if stop_reason is None or stop_priority.get(reason, 0) > stop_priority.get(stop_reason, 0):
            stop_reason = reason
        stop.set()

    async def one(index: int, offset: float) -> None:
        nonlocal stop_reason
        await asyncio.sleep(max(0.0, epoch_started + offset - loop.time()))
        if stop.is_set():
            return
        scheduled_monotonic = epoch_started + offset
        async with semaphore:
            if stop.is_set():
                return
            queue_delay = max(0.0, loop.time() - scheduled_monotonic)
            logical = logical_ids[index]
            epoch_workload_key = _route_neutral_epoch_key(route.id, epoch_id)
            spec = shape_spec(
                route,
                shape,
                logical,
                suite="load",
                cell_suffix=f":{phase}:rps={offered_rps:.9g}:epoch={epoch_id}",
                seed=seed,
                workload_key=(
                    f"load:{{route}}:{shape}:{phase}:rps={offered_rps:.9g}:"
                    f"sample_seed={seed}:epoch={epoch_workload_key}:index={index}"
                ),
                matched_cell_suffix=(f":{phase}:rps={offered_rps:.9g}:sample_seed={seed}"),
                shape_config=shape_config,
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
            except BudgetExceeded:
                set_stop_reason("cost_guard")
                return
            except TimeLimitReached:
                set_stop_reason("time_guard")
                return
            except PaymentRequiredLatched:
                set_stop_reason("http_402_latch")
                return
            except ReservationOverrunLatched:
                set_stop_reason("reservation_overrun_latch")
                return
            except Exception:
                # Stop all not-yet-sent arrivals immediately. The failing claimed attempt is
                # already marked unknown by the engine; other claimed sends drain before this
                # exception is re-raised below.
                set_stop_reason("unexpected_runner_error")
                raise
            finally:
                if current is not None:
                    executing_tasks.discard(current)
            if result is not None and result.http_status == 402:
                set_stop_reason("http_402_latch")

    tasks = [asyncio.create_task(one(index, offset)) for index, offset in enumerate(offsets)]
    if tasks:
        all_done = asyncio.gather(*tasks, return_exceptions=True)
        stop_waiter = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait({all_done, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
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
            if engine.payment_required_latched:
                raise PaymentRequiredLatched("HTTP 402 latch won a concurrent epoch failure")
            if engine.reservation_overrun_latched:
                raise ReservationOverrunLatched(
                    "reservation overrun latch won a concurrent epoch failure"
                )
            raise unexpected[0]
    # Durable fatal latches outrank task scheduling order when multiple concurrent sends fail in
    # different ways. This makes coverage/censor labels stable across equivalent races.
    if engine.payment_required_latched:
        stop_reason = "http_402_latch"
    elif engine.reservation_overrun_latched:
        stop_reason = "reservation_overrun_latch"
    if stop_reason is None:
        # Preserve the complete registered arrival window, including a quiet Poisson tail. Without
        # this hold, the next endpoint-isolated epoch can begin inside the prior block's conceptual
        # window and condition away sparse no-arrival time. Response drain may extend past it.
        await asyncio.sleep(max(0.0, epoch_started + duration_seconds - loop.time()))
    summary = _build_epoch_summary(
        engine,
        route,
        shape=shape,
        epoch_id=epoch_id,
        phase=phase,
        offered_rps=offered_rps,
        duration_seconds=duration_seconds,
        logical_ids=logical_ids,
        actual_elapsed=loop.time() - epoch_started,
        stop_reason=stop_reason,
        scientific_censor_reason=None,
        max_p95_ttft_seconds=max_p95_ttft_seconds,
        max_p95_total_seconds=max_p95_total_seconds,
    )
    _record_epoch_summary(engine, summary)
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
    bracket_epochs = int(config.get("bracket_epochs", min(6, epochs)))
    bracket_multiplier = float(config.get("bracket_multiplier", 2.0))
    max_rps = float(config["max_rps"]) if "max_rps" in config else None
    ceiling = int(config.get("concurrency", engine.config.concurrency))
    baseline_samples, baseline_duration, baseline_rate = baseline_design(
        config, duration, default_rps=min(rate, 0.1)
    )
    baseline = await run_open_loop_epoch(
        engine,
        route,
        shape=shape,
        epoch_id=f"aimd-{route.id}-{shape}-baseline",
        phase="baseline",
        offered_rps=baseline_rate,
        duration_seconds=baseline_duration,
        concurrency=ceiling,
        seed=seed - 1,
        shape_config=config,
        deterministic_scheduled_count=baseline_samples,
    )
    results: list[EpochSummary] = [baseline]
    if baseline.launch_guard_triggered:
        reason = baseline.launch_guard_reason or "launch_guard"
        engine.ledger.record_event_once(
            f"aimd_controller_censored:{route.id}:{shape}",
            "aimd_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": baseline.epoch_id,
            },
        )
        downstream = [f"aimd-{route.id}-{shape}-{index:03d}" for index in range(epochs)]
        downstream.extend(
            f"aimd-{route.id}-{shape}-confirm-{confirmation}" for confirmation in range(3)
        )
        downstream.extend(
            f"aimd-{route.id}-{shape}-separator-{separator}" for separator in range(2)
        )
        downstream.append(f"aimd-{route.id}-{shape}-recovery")
        _censor_unstarted_load_cells(engine, downstream, reason)
        return results
    if not baseline.controller_eligible or not baseline.healthy:
        reason = (
            baseline.scientific_censor_reason
            if not baseline.controller_eligible
            else "unhealthy_low_load_baseline"
        )
        engine.ledger.record_event_once(
            f"aimd_controller_censored:{route.id}:{shape}",
            "aimd_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": baseline.epoch_id,
            },
        )
        downstream = [f"aimd-{route.id}-{shape}-{index:03d}" for index in range(epochs)]
        downstream.extend(
            f"aimd-{route.id}-{shape}-confirm-{confirmation}" for confirmation in range(3)
        )
        downstream.extend(
            f"aimd-{route.id}-{shape}-separator-{separator}" for separator in range(2)
        )
        downstream.append(f"aimd-{route.id}-{shape}-recovery")
        _censor_unstarted_load_cells(engine, downstream, reason)
        return results
    ttft_limit = None if baseline.p95_ttft_seconds is None else 2 * baseline.p95_ttft_seconds
    total_limit = None if baseline.p95_total_seconds is None else 2 * baseline.p95_total_seconds
    unhealthy_streak = 0
    best_healthy = 0.0
    overload_observed = False
    unhealthy_upper_bound_rps: float | None = None
    nonmonotonic_overload_observed = False
    healthy_increases = 0
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
            shape_config=config,
            max_p95_ttft_seconds=ttft_limit,
            max_p95_total_seconds=total_limit,
        )
        results.append(summary)
        if summary.launch_guard_triggered:
            break
        if not summary.controller_eligible:
            # A pre-crash partial epoch is preserved as censored evidence but is neither healthy
            # nor congestion evidence. It also breaks consecutiveness of unhealthy evidence.
            # Continue at the same offered rate without otherwise changing AIMD state.
            unhealthy_streak = 0
            continue
        if summary.healthy:
            best_healthy = max(best_healthy, rate)
            if unhealthy_upper_bound_rps is not None and rate >= unhealthy_upper_bound_rps:
                # A later healthy epoch at or above the previously unhealthy rate destroys the
                # monotone knee bracket. Preserve the conflicting evidence, but never publish the
                # stale unhealthy rate as a current upper bound.
                unhealthy_upper_bound_rps = None
                nonmonotonic_overload_observed = True
            unhealthy_streak = 0
            rate = next_healthy_aimd_rate(
                rate,
                healthy_increases=healthy_increases,
                overload_observed=overload_observed,
                additive_rps=additive,
                bracket_epochs=bracket_epochs,
                bracket_multiplier=bracket_multiplier,
                max_rps=max_rps,
            )
            healthy_increases += 1
        else:
            unhealthy_streak += 1
            if unhealthy_streak >= 2:
                overload_observed = True
                if rate > best_healthy:
                    unhealthy_upper_bound_rps = rate
                else:
                    unhealthy_upper_bound_rps = None
                    nonmonotonic_overload_observed = True
                rate = max(0.01, rate * decrease)
                unhealthy_streak = 0

    ramp_guard = next((epoch for epoch in results if epoch.launch_guard_triggered), None)
    if ramp_guard is not None:
        reason = ramp_guard.launch_guard_reason or "launch_guard"
        downstream = [
            f"aimd-{route.id}-{shape}-confirm-{confirmation}" for confirmation in range(3)
        ]
        downstream.extend(
            f"aimd-{route.id}-{shape}-separator-{separator}" for separator in range(2)
        )
        downstream.append(f"aimd-{route.id}-{shape}-recovery")
        _censor_unstarted_load_cells(engine, downstream, reason)
        engine.ledger.record_event_once(
            f"aimd_complete:{route.id}:{shape}",
            "aimd_complete",
            {
                "route_id": route.id,
                "shape": shape,
                "highest_observed_healthy_rps": best_healthy or None,
                "healthy_lower_bound_rps": best_healthy or None,
                "unhealthy_upper_bound_rps": unhealthy_upper_bound_rps,
                "overload_observed": overload_observed,
                "nonmonotonic_overload_observed": nonmonotonic_overload_observed,
                "capacity_bound_state": "campaign_guard_censored_before_confirmation",
                "controller_completion_state": "campaign_guard_censored",
                "censor_reason": reason,
                "confirmations_required": 3,
                "confirmation_healthy": [],
                "confirmation_eligible": [],
                "confirmation_censor_reasons": [],
                "confirmation_execution_complete": False,
                "confirmation_complete": False,
                "confirmation_all_healthy": None,
                "recovery_run": False,
                "recovery_healthy": None,
                "recovery_eligible": None,
                "recovery_censor_reason": None,
            },
        )
        return results

    # Three separated confirmations are explicit; they are not silently inferred from ramp epochs.
    if best_healthy <= 0:
        reason = "no_healthy_capacity_candidate_observed"
        downstream = [
            f"aimd-{route.id}-{shape}-confirm-{confirmation}" for confirmation in range(3)
        ]
        downstream.extend(
            f"aimd-{route.id}-{shape}-separator-{separator}" for separator in range(2)
        )
        downstream.append(f"aimd-{route.id}-{shape}-recovery")
        _censor_unstarted_load_cells(engine, downstream, reason)
        engine.ledger.record_event_once(
            f"aimd_complete:{route.id}:{shape}",
            "aimd_complete",
            {
                "route_id": route.id,
                "shape": shape,
                "highest_observed_healthy_rps": None,
                "healthy_lower_bound_rps": None,
                "unhealthy_upper_bound_rps": unhealthy_upper_bound_rps,
                "overload_observed": overload_observed,
                "nonmonotonic_overload_observed": nonmonotonic_overload_observed,
                "capacity_bound_state": "left_censored_no_healthy_candidate",
                "controller_completion_state": "left_censored_no_healthy_candidate",
                "censor_reason": None,
                "confirmations_required": 3,
                "confirmation_healthy": [],
                "confirmation_eligible": [],
                "confirmation_censor_reasons": [],
                "confirmation_execution_complete": False,
                "confirmation_complete": False,
                "confirmation_all_healthy": None,
                "recovery_run": False,
                "recovery_healthy": None,
                "recovery_eligible": None,
                "recovery_censor_reason": None,
            },
        )
        return results
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
                    shape_config=config,
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
                        offered_rps=baseline_rate,
                        duration_seconds=baseline_duration,
                        concurrency=ceiling,
                        seed=seed + 50 + confirmation,
                        shape_config=config,
                        deterministic_scheduled_count=baseline_samples,
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
                    shape_config=config,
                    max_p95_ttft_seconds=ttft_limit,
                    max_p95_total_seconds=total_limit,
                )
            )
        elif not overload_observed:
            with suppress(KeyError):
                engine.ledger.mark_plan_cell(
                    f"load_epoch:aimd-{route.id}-{shape}-recovery",
                    "not_applicable",
                    "no_two_epoch_overload_observed",
                )
        confirmations = [epoch for epoch in results if epoch.phase == "confirmation"]
        recovery_epochs = [
            epoch for epoch in results if epoch.phase == "recovery_after_observed_overload"
        ]
        guarded = next((epoch for epoch in results if epoch.launch_guard_triggered), None)
        confirmation_eligible = [
            epoch.controller_eligible and not epoch.launch_guard_triggered
            for epoch in confirmations
        ]
        confirmation_execution_complete = len(confirmations) == 3
        confirmation_complete = confirmation_execution_complete and all(confirmation_eligible)
        confirmation_all_healthy = (
            all(epoch.healthy for epoch in confirmations) if confirmation_complete else None
        )
        controller_completion_state = (
            "campaign_guard_censored"
            if guarded is not None
            else "confirmations_inconclusive"
            if not confirmation_complete
            else "completed_confirmations_healthy"
            if confirmation_all_healthy
            else "completed_confirmations_unhealthy"
        )
        if nonmonotonic_overload_observed:
            capacity_bound_state = "nonmonotonic_overload_no_current_bracket"
            unhealthy_upper_bound_rps = None
        elif unhealthy_upper_bound_rps is not None and unhealthy_upper_bound_rps > best_healthy:
            capacity_bound_state = "bracketed_healthy_lower_unhealthy_upper"
        elif overload_observed:
            capacity_bound_state = "nonmonotonic_overload_no_current_bracket"
        else:
            capacity_bound_state = "right_censored_highest_tested_healthy_no_overload"
        recovery = recovery_epochs[-1] if recovery_epochs else None
        engine.ledger.record_event_once(
            f"aimd_complete:{route.id}:{shape}",
            "aimd_complete",
            {
                "route_id": route.id,
                "shape": shape,
                "highest_observed_healthy_rps": best_healthy,
                "healthy_lower_bound_rps": best_healthy,
                "unhealthy_upper_bound_rps": unhealthy_upper_bound_rps,
                "overload_observed": overload_observed,
                "nonmonotonic_overload_observed": nonmonotonic_overload_observed,
                "capacity_bound_state": capacity_bound_state,
                "controller_completion_state": controller_completion_state,
                "censor_reason": (
                    guarded.launch_guard_reason or "launch_guard" if guarded is not None else None
                ),
                "confirmations_required": 3,
                "confirmation_healthy": [
                    epoch.healthy if eligible else None
                    for epoch, eligible in zip(confirmations, confirmation_eligible, strict=True)
                ],
                "confirmation_eligible": confirmation_eligible,
                "confirmation_censor_reasons": [
                    (
                        epoch.launch_guard_reason
                        or epoch.scientific_censor_reason
                        or "launch_guard"
                        if not eligible
                        else None
                    )
                    for epoch, eligible in zip(confirmations, confirmation_eligible, strict=True)
                ],
                "confirmation_execution_complete": confirmation_execution_complete,
                "confirmation_complete": confirmation_complete,
                "confirmation_all_healthy": confirmation_all_healthy,
                "recovery_run": bool(recovery_epochs),
                "recovery_healthy": (
                    recovery.healthy
                    if recovery is not None
                    and recovery.controller_eligible
                    and not recovery.launch_guard_triggered
                    else None
                ),
                "recovery_eligible": (
                    recovery.controller_eligible and not recovery.launch_guard_triggered
                    if recovery is not None
                    else None
                ),
                "recovery_censor_reason": (
                    recovery.launch_guard_reason
                    or recovery.scientific_censor_reason
                    or "launch_guard"
                    if recovery is not None
                    and (not recovery.controller_eligible or recovery.launch_guard_triggered)
                    else None
                ),
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
    baseline_samples, baseline_duration, baseline_rate = baseline_design(
        config, block_seconds, default_rps=min(rate, 0.1)
    )
    baseline = await run_open_loop_epoch(
        engine,
        route,
        shape=shape,
        epoch_id=f"soak-{route.id}-{shape}-baseline",
        phase="soak_baseline",
        offered_rps=baseline_rate,
        duration_seconds=baseline_duration,
        concurrency=ceiling,
        seed=seed - 1,
        shape_config=config,
        deterministic_scheduled_count=baseline_samples,
    )
    soak_blocks: list[EpochSummary] = []
    if baseline.launch_guard_triggered:
        reason = baseline.launch_guard_reason or "launch_guard"
        engine.ledger.record_event_once(
            f"soak_controller_censored:{route.id}:{shape}",
            "soak_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": baseline.epoch_id,
            },
        )
        _censor_unstarted_load_cells(
            engine,
            [f"soak-{route.id}-{shape}-block-{block}" for block in range(blocks)],
            reason,
        )
        return [baseline]
    if not baseline.controller_eligible or not baseline.healthy:
        reason = (
            baseline.scientific_censor_reason
            if not baseline.controller_eligible
            else "unhealthy_low_load_baseline"
        )
        engine.ledger.record_event_once(
            f"soak_controller_censored:{route.id}:{shape}",
            "soak_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": baseline.epoch_id,
            },
        )
        _censor_unstarted_load_cells(
            engine,
            [f"soak-{route.id}-{shape}-block-{block}" for block in range(blocks)],
            reason,
        )
        return [baseline]
    ttft_limit = None if baseline.p95_ttft_seconds is None else 2 * baseline.p95_ttft_seconds
    total_limit = None if baseline.p95_total_seconds is None else 2 * baseline.p95_total_seconds
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
                shape_config=config,
                max_p95_ttft_seconds=ttft_limit,
                max_p95_total_seconds=total_limit,
            )
        )
        if soak_blocks[-1].launch_guard_triggered:
            break
    engine.ledger.record_event_once(
        f"soak_complete:{route.id}:{shape}",
        "soak_complete",
        {
            "route_id": route.id,
            "shape": shape,
            "rate_rps": rate,
            "blocks": blocks,
            "completed_blocks": len(soak_blocks),
            "block_eligible": [
                block.controller_eligible and not block.launch_guard_triggered
                for block in soak_blocks
            ],
            "block_healthy": [
                block.healthy
                if block.controller_eligible and not block.launch_guard_triggered
                else None
                for block in soak_blocks
            ],
            "block_censor_reasons": [
                block.launch_guard_reason or block.scientific_censor_reason or "launch_guard"
                if not block.controller_eligible or block.launch_guard_triggered
                else None
                for block in soak_blocks
            ],
            "execution_complete": len(soak_blocks) == blocks,
            "scientifically_complete": len(soak_blocks) == blocks
            and all(
                block.controller_eligible and not block.launch_guard_triggered
                for block in soak_blocks
            ),
            "all_blocks_healthy": (
                all(block.healthy for block in soak_blocks)
                if len(soak_blocks) == blocks
                and all(
                    block.controller_eligible and not block.launch_guard_triggered
                    for block in soak_blocks
                )
                else None
            ),
            "controller_completion_state": (
                "campaign_guard_censored"
                if soak_blocks and soak_blocks[-1].launch_guard_triggered
                else "execution_complete_inconclusive"
                if len(soak_blocks) == blocks
                and not all(
                    block.controller_eligible and not block.launch_guard_triggered
                    for block in soak_blocks
                )
                else "completed_healthy"
                if len(soak_blocks) == blocks and all(block.healthy for block in soak_blocks)
                else "completed_unhealthy"
                if len(soak_blocks) == blocks
                else "partial_incomplete"
            ),
            "censor_reason": (
                soak_blocks[-1].launch_guard_reason or "launch_guard"
                if soak_blocks and soak_blocks[-1].launch_guard_triggered
                else None
            ),
        },
    )
    return [baseline, *soak_blocks]
