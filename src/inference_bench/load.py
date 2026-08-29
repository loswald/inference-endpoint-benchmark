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


def aimd_max_rps(config: dict[str, Any], shape: str) -> float | None:
    """Resolve a shape-specific AIMD ceiling without conflating RPM and TPM stress."""
    by_shape = config.get("max_rps_by_shape") or {}
    if shape in by_shape:
        return float(by_shape[shape])
    return float(config["max_rps"]) if "max_rps" in config else None


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
    max_rps: float | None = None
    if "max_rps" in config:
        max_rps = _strict_positive_float(config["max_rps"], "aimd.max_rps")
        if max_rps < initial_rps:
            raise ValueError("aimd.max_rps must be at least aimd.initial_rps")
    by_shape = config.get("max_rps_by_shape") or {}
    if not isinstance(by_shape, dict):
        raise ValueError("aimd.max_rps_by_shape must be a mapping")
    unknown_shapes = set(by_shape) - {"short_short", "long_short", "short_long", "mixed"}
    if unknown_shapes:
        raise ValueError(
            "aimd.max_rps_by_shape has unknown shapes: " + ", ".join(sorted(unknown_shapes))
        )
    for shape, value in by_shape.items():
        ceiling = _strict_positive_float(value, f"aimd.max_rps_by_shape.{shape}")
        if ceiling < initial_rps:
            raise ValueError(
                f"aimd.max_rps_by_shape.{shape} must be at least aimd.initial_rps"
            )
    _strict_positive_int(config.get("concurrency", default_concurrency), "aimd.concurrency")
    if "baseline_rps" in config:
        _strict_positive_float(config["baseline_rps"], "aimd.baseline_rps")
    samples = _strict_positive_int(
        config.get("baseline_samples", MIN_BASELINE_SAMPLES), "aimd.baseline_samples"
    )
    if samples < MIN_BASELINE_SAMPLES:
        raise ValueError(f"aimd.baseline_samples must be at least {MIN_BASELINE_SAMPLES}")
    _strict_positive_int(config.get("baseline_attempts", 3), "aimd.baseline_attempts")
    baseline_decrease = _strict_positive_float(
        config.get("baseline_multiplicative_decrease", 0.5),
        "aimd.baseline_multiplicative_decrease",
    )
    if not 0 < baseline_decrease <= 1:
        raise ValueError("aimd.baseline_multiplicative_decrease must lie in (0, 1]")
    _strict_positive_int(config.get("confirmation_max_stages", 4), "aimd.confirmation_max_stages")
    _strict_positive_int(
        config.get("confirmation_separator_samples", samples),
        "aimd.confirmation_separator_samples",
    )
    confirmation_decrease = _strict_positive_float(
        config.get("confirmation_multiplicative_decrease", decrease),
        "aimd.confirmation_multiplicative_decrease",
    )
    if not 0 < confirmation_decrease < 1:
        raise ValueError("aimd.confirmation_multiplicative_decrease must lie in (0, 1)")
    minimum_rps = _strict_positive_float(config.get("minimum_rps", 0.01), "aimd.minimum_rps")
    if max_rps is not None and minimum_rps > max_rps:
        raise ValueError("aimd.minimum_rps must not exceed aimd.max_rps")


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
    _strict_positive_int(config.get("baseline_attempts", 3), "soak.baseline_attempts")
    baseline_decrease = _strict_positive_float(
        config.get("baseline_multiplicative_decrease", 0.5),
        "soak.baseline_multiplicative_decrease",
    )
    if not 0 < baseline_decrease <= 1:
        raise ValueError("soak.baseline_multiplicative_decrease must lie in (0, 1]")
    _strict_positive_int(config.get("max_rate_stages", 4), "soak.max_rate_stages")
    rate_decrease = _strict_positive_float(
        config.get("rate_multiplicative_decrease", 0.5),
        "soak.rate_multiplicative_decrease",
    )
    if not 0 < rate_decrease < 1:
        raise ValueError("soak.rate_multiplicative_decrease must lie in (0, 1)")
    _strict_positive_float(config.get("minimum_rps", 0.01), "soak.minimum_rps")


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


class LoadRunResult(list[EpochSummary]):
    """Epoch summaries plus a non-terminal scheduling-pause signal.

    A panel-boundary pause is deliberately not written as controller evidence: it is neither a
    failed experiment nor a campaign guard. The same controller call can replay completed epoch
    summaries from the immutable ledger and continue with the first unstarted phase in a later
    scheduling interval.
    """

    def __init__(
        self,
        values: list[EpochSummary] | None = None,
        *,
        paused_for_window: bool = False,
    ) -> None:
        super().__init__(values or [])
        self.paused_for_window = paused_for_window


def _monotonic_time() -> float:
    return asyncio.get_running_loop().time()


def _phase_fits_before(
    engine: BenchmarkEngine,
    route: RouteConfig,
    *,
    epoch_id: str,
    arrival_window_seconds: float,
    not_after_monotonic: float | None,
) -> bool:
    """Return whether a new provider-bearing phase can drain before a local guard.

    Completed phases are safe to restore because ``run_open_loop_epoch`` returns their immutable
    summary without a provider call. For a new phase, the registered arrival window and the
    route's full-stream request timeout form the conservative drain bound.
    """

    if engine.ledger.event_by_key(f"load_epoch:{epoch_id}") is not None:
        return True
    if not_after_monotonic is None:
        return True
    now = _monotonic_time()
    return now + arrival_window_seconds + route.request_timeout_seconds <= not_after_monotonic


def _paused_result(values: list[EpochSummary]) -> LoadRunResult:
    return LoadRunResult(values, paused_for_window=True)


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
    queue_delays = [
        float(row["queue_delay_seconds"])
        for row in final_rows
        if row["queue_delay_seconds"] is not None
    ]
    p95_queue_delay = quantile(queue_delays, 0.95) if queue_delays else None
    healthy = bool(
        completed
        and stop_reason is None
        and not unknown
        and logical_observed == len(logical_ids)
        and success_rate >= 0.99
        and physical_count > 0
        and rate_limited / physical_count <= 0.01
        and (server_errors + timeouts + transport_errors) / physical_count <= 0.01
        # Response drain after the registered arrival window is not queue growth. A slow final
        # request may finish well after the window even when every arrival starts immediately.
        # Coordinated-omission-safe congestion is instead detected from the measured delay between
        # each registered arrival and admission through the independent concurrency ceiling.
        and len(queue_delays) == logical_observed
        and p95_queue_delay is not None
        and p95_queue_delay <= max(1.0, duration_seconds * 0.1)
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


def adaptive_baseline_epoch_id(controller: str, route_id: str, shape: str, attempt: int) -> str:
    base = f"{controller}-{route_id}-{shape}-baseline"
    return base if attempt == 0 else f"{base}-attempt-{attempt}"


def aimd_confirmation_epoch_id(route_id: str, shape: str, stage: int, index: int) -> str:
    if stage == 0:
        return f"aimd-{route_id}-{shape}-confirm-{index}"
    return f"aimd-{route_id}-{shape}-confirm-stage-{stage}-{index}"


def aimd_separator_epoch_id(route_id: str, shape: str, stage: int, index: int) -> str:
    if stage == 0:
        return f"aimd-{route_id}-{shape}-separator-{index}"
    return f"aimd-{route_id}-{shape}-separator-stage-{stage}-{index}"


def soak_block_epoch_id(route_id: str, shape: str, stage: int, block: int) -> str:
    if stage == 0:
        return f"soak-{route_id}-{shape}-block-{block}"
    return f"soak-{route_id}-{shape}-stage-{stage}-block-{block}"


async def _run_adaptive_baselines(
    engine: BenchmarkEngine,
    route: RouteConfig,
    *,
    controller: str,
    phase: str,
    shape: str,
    config: dict[str, Any],
    nominal_duration: float,
    default_rps: float,
    concurrency: int,
    seed: int,
    not_after_monotonic: float | None = None,
) -> tuple[list[EpochSummary], EpochSummary | None, bool]:
    """Repeat a low-load reference at progressively lower rates until it is healthy.

    A single transport wobble must not erase an endpoint/shape cell.  Every attempt remains
    visible as evidence, while only an eligible healthy attempt supplies latency thresholds.
    """

    samples, initial_duration, initial_rate = baseline_design(
        config, nominal_duration, default_rps=default_rps
    )
    attempts = int(config.get("baseline_attempts", 3))
    decrease = float(config.get("baseline_multiplicative_decrease", 0.5))
    minimum_rps = float(config.get("minimum_rps", 0.01))
    summaries: list[EpochSummary] = []
    for attempt in range(attempts):
        rate = max(minimum_rps, initial_rate * decrease**attempt)
        duration = max(initial_duration, samples / rate)
        epoch_id = adaptive_baseline_epoch_id(controller, route.id, shape, attempt)
        if not _phase_fits_before(
            engine,
            route,
            epoch_id=epoch_id,
            arrival_window_seconds=duration,
            not_after_monotonic=not_after_monotonic,
        ):
            return summaries, None, True
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape=shape,
            epoch_id=epoch_id,
            phase=phase,
            offered_rps=samples / duration,
            duration_seconds=duration,
            concurrency=concurrency,
            seed=seed - 1 - attempt,
            shape_config=config,
            deterministic_scheduled_count=samples,
        )
        summaries.append(summary)
        if summary.launch_guard_triggered:
            break
        if summary.controller_eligible and summary.healthy:
            for unused_attempt in range(attempt + 1, attempts):
                with suppress(KeyError):
                    unused_id = adaptive_baseline_epoch_id(
                        controller, route.id, shape, unused_attempt
                    )
                    engine.ledger.mark_plan_cell(
                        f"load_epoch:{unused_id}",
                        "not_applicable",
                        "healthy_baseline_already_established",
                    )
            return summaries, summary, False
    return summaries, None, False


async def run_aimd(
    engine: BenchmarkEngine,
    route: RouteConfig,
    shape: str,
    config: dict[str, Any],
    *,
    seed: int,
    not_after_monotonic: float | None = None,
) -> LoadRunResult:
    validate_aimd_config(config, engine.config.concurrency)
    epochs = int(config.get("epochs", 12))
    duration = float(config.get("epoch_seconds", 20))
    rate = float(config.get("initial_rps", 0.25))
    additive = float(config.get("additive_rps", 0.25))
    decrease = float(config.get("multiplicative_decrease", 0.5))
    bracket_epochs = int(config.get("bracket_epochs", min(6, epochs)))
    bracket_multiplier = float(config.get("bracket_multiplier", 2.0))
    max_rps = aimd_max_rps(config, shape)
    ceiling = int(config.get("concurrency", engine.config.concurrency))
    confirmation_max_stages = int(config.get("confirmation_max_stages", 4))
    confirmation_decrease = float(config.get("confirmation_multiplicative_decrease", decrease))
    minimum_rps = float(config.get("minimum_rps", 0.01))
    baseline_attempts, baseline, paused = await _run_adaptive_baselines(
        engine,
        route,
        controller="aimd",
        phase="baseline",
        shape=shape,
        config=config,
        nominal_duration=duration,
        default_rps=min(rate, 0.5),
        concurrency=ceiling,
        seed=seed,
        not_after_monotonic=not_after_monotonic,
    )
    results = LoadRunResult(list(baseline_attempts))
    if paused:
        return _paused_result(results)
    baseline_guard = next(
        (summary for summary in baseline_attempts if summary.launch_guard_triggered), None
    )
    if baseline_guard is not None:
        reason = baseline_guard.launch_guard_reason or "launch_guard"
        engine.ledger.record_event_once(
            f"aimd_controller_censored:{route.id}:{shape}",
            "aimd_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": baseline_guard.epoch_id,
            },
        )
        downstream = [f"aimd-{route.id}-{shape}-{index:03d}" for index in range(epochs)]
        downstream.extend(
            aimd_confirmation_epoch_id(route.id, shape, stage, confirmation)
            for stage in range(confirmation_max_stages)
            for confirmation in range(3)
        )
        downstream.extend(
            aimd_separator_epoch_id(route.id, shape, stage, separator)
            for stage in range(confirmation_max_stages)
            for separator in range(2)
        )
        downstream.append(f"aimd-{route.id}-{shape}-recovery")
        _censor_unstarted_load_cells(engine, downstream, reason)
        return results
    if baseline is None:
        last_baseline = baseline_attempts[-1]
        all_measured_unhealthy = all(
            item.controller_eligible and not item.launch_guard_triggered and not item.healthy
            for item in baseline_attempts
        )
        reason = (
            "measured_unhealthy_at_all_baseline_rates"
            if all_measured_unhealthy
            else last_baseline.scientific_censor_reason or "baseline_attempts_inconclusive"
        )
        engine.ledger.record_event_once(
            f"aimd_controller_censored:{route.id}:{shape}",
            "aimd_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": last_baseline.epoch_id,
            },
        )
        downstream = [f"aimd-{route.id}-{shape}-{index:03d}" for index in range(epochs)]
        downstream.extend(
            aimd_confirmation_epoch_id(route.id, shape, stage, confirmation)
            for stage in range(confirmation_max_stages)
            for confirmation in range(3)
        )
        downstream.extend(
            aimd_separator_epoch_id(route.id, shape, stage, separator)
            for stage in range(confirmation_max_stages)
            for separator in range(2)
        )
        downstream.append(f"aimd-{route.id}-{shape}-recovery")
        if all_measured_unhealthy:
            for epoch_id in downstream:
                with suppress(KeyError):
                    engine.ledger.mark_plan_cell(f"load_epoch:{epoch_id}", "not_applicable", reason)
            engine.ledger.record_event_once(
                f"aimd_complete:{route.id}:{shape}",
                "aimd_complete",
                {
                    "route_id": route.id,
                    "shape": shape,
                    "highest_observed_healthy_rps": None,
                    "healthy_lower_bound_rps": None,
                    "unhealthy_upper_bound_rps": baseline_attempts[-1].offered_rps,
                    "overload_observed": True,
                    "nonmonotonic_overload_observed": False,
                    "capacity_bound_state": "left_censored_no_healthy_at_lowest_tested_rate",
                    "controller_completion_state": "completed_no_healthy_at_lowest_tested_rate",
                    "censor_reason": None,
                    "confirmations_required": 0,
                    "confirmation_stage": None,
                    "confirmation_stage_history": [],
                    "confirmation_healthy": [],
                    "confirmation_eligible": [],
                    "confirmation_censor_reasons": [],
                    "confirmation_execution_complete": True,
                    "confirmation_complete": True,
                    "confirmation_all_healthy": False,
                    "recovery_run": False,
                    "recovery_healthy": None,
                    "recovery_eligible": None,
                    "recovery_censor_reason": None,
                },
            )
        else:
            _censor_unstarted_load_cells(engine, downstream, reason)
        return results
    baseline_samples = baseline.scheduled
    baseline_rate = baseline.offered_rps
    separator_samples = int(config.get("confirmation_separator_samples", baseline_samples))
    separator_duration = max(duration, separator_samples / baseline_rate)
    ttft_limit = None if baseline.p95_ttft_seconds is None else 2 * baseline.p95_ttft_seconds
    total_limit = None if baseline.p95_total_seconds is None else 2 * baseline.p95_total_seconds
    unhealthy_streak = 0
    best_healthy = 0.0
    overload_observed = False
    unhealthy_upper_bound_rps: float | None = None
    nonmonotonic_overload_observed = False
    healthy_increases = 0
    for index in range(epochs):
        epoch_id = f"aimd-{route.id}-{shape}-{index:03d}"
        if not _phase_fits_before(
            engine,
            route,
            epoch_id=epoch_id,
            arrival_window_seconds=duration,
            not_after_monotonic=not_after_monotonic,
        ):
            return _paused_result(results)
        summary = await run_open_loop_epoch(
            engine,
            route,
            shape=shape,
            epoch_id=epoch_id,
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
            aimd_confirmation_epoch_id(route.id, shape, stage, confirmation)
            for stage in range(confirmation_max_stages)
            for confirmation in range(3)
        ]
        downstream.extend(
            aimd_separator_epoch_id(route.id, shape, stage, separator)
            for stage in range(confirmation_max_stages)
            for separator in range(2)
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

    # Confirm the best ramp observation.  If it does not reproduce, step down and repeat the
    # entire three-epoch separated confirmation.  A ramp with no healthy point is not abandoned:
    # the configured floor is tested so the result becomes a measured negative rather than blank.
    candidate_rate = max(minimum_rps, best_healthy or minimum_rps)
    confirmation_history: list[dict[str, Any]] = []
    selected_confirmations: list[EpochSummary] = []
    selected_confirmation_stage: int | None = None
    accepted_rate: float | None = None
    guarded: EpochSummary | None = None
    reached_unhealthy_floor = False
    for stage in range(confirmation_max_stages):
        stage_confirmations: list[EpochSummary] = []
        for confirmation in range(3):
            confirmation_id = aimd_confirmation_epoch_id(
                route.id, shape, stage, confirmation
            )
            if not _phase_fits_before(
                engine,
                route,
                epoch_id=confirmation_id,
                arrival_window_seconds=duration,
                not_after_monotonic=not_after_monotonic,
            ):
                return _paused_result(results)
            summary = await run_open_loop_epoch(
                engine,
                route,
                shape=shape,
                epoch_id=confirmation_id,
                phase="confirmation",
                offered_rps=candidate_rate,
                duration_seconds=duration,
                concurrency=ceiling,
                seed=seed + stage * 1000 + confirmation + 1,
                shape_config=config,
                max_p95_ttft_seconds=ttft_limit,
                max_p95_total_seconds=total_limit,
            )
            results.append(summary)
            stage_confirmations.append(summary)
            if summary.launch_guard_triggered:
                guarded = summary
                break
            if confirmation < 2:
                separator_id = aimd_separator_epoch_id(
                    route.id, shape, stage, confirmation
                )
                if not _phase_fits_before(
                    engine,
                    route,
                    epoch_id=separator_id,
                    arrival_window_seconds=separator_duration,
                    not_after_monotonic=not_after_monotonic,
                ):
                    return _paused_result(results)
                separator = await run_open_loop_epoch(
                    engine,
                    route,
                    shape=shape,
                    epoch_id=separator_id,
                    phase="confirmation_separator",
                    offered_rps=separator_samples / separator_duration,
                    duration_seconds=separator_duration,
                    concurrency=ceiling,
                    seed=seed + stage * 1000 + 50 + confirmation,
                    shape_config=config,
                    deterministic_scheduled_count=separator_samples,
                    max_p95_ttft_seconds=ttft_limit,
                    max_p95_total_seconds=total_limit,
                )
                results.append(separator)
                if separator.launch_guard_triggered:
                    guarded = separator
                    break
        eligible = [
            epoch.controller_eligible and not epoch.launch_guard_triggered
            for epoch in stage_confirmations
        ]
        complete = len(stage_confirmations) == 3 and all(eligible)
        all_healthy = all(epoch.healthy for epoch in stage_confirmations) if complete else None
        confirmation_history.append(
            {
                "stage": stage,
                "rate_rps": candidate_rate,
                "execution_complete": len(stage_confirmations) == 3,
                "scientifically_complete": complete,
                "healthy": all_healthy,
            }
        )
        selected_confirmations = stage_confirmations
        selected_confirmation_stage = stage
        if guarded is not None:
            break
        if complete and all_healthy:
            accepted_rate = candidate_rate
            best_healthy = max(best_healthy, candidate_rate)
            break
        if complete and all_healthy is False:
            overload_observed = True
            if unhealthy_upper_bound_rps is None or candidate_rate < unhealthy_upper_bound_rps:
                unhealthy_upper_bound_rps = candidate_rate
            next_rate = max(minimum_rps, candidate_rate * confirmation_decrease)
            if next_rate >= candidate_rate:
                reached_unhealthy_floor = True
                break
            candidate_rate = next_rate
        # An interrupted/ineligible stage is retried at exactly the same rate with fresh IDs.

    for stage in range((selected_confirmation_stage or 0) + 1, confirmation_max_stages):
        for confirmation in range(3):
            with suppress(KeyError):
                confirmation_id = aimd_confirmation_epoch_id(route.id, shape, stage, confirmation)
                engine.ledger.mark_plan_cell(
                    f"load_epoch:{confirmation_id}",
                    "not_applicable",
                    "lower_rate_confirmation_not_needed",
                )
            if confirmation < 2:
                with suppress(KeyError):
                    separator_id = aimd_separator_epoch_id(route.id, shape, stage, confirmation)
                    engine.ledger.mark_plan_cell(
                        f"load_epoch:{separator_id}",
                        "not_applicable",
                        "lower_rate_confirmation_not_needed",
                    )

    recovery_epochs: list[EpochSummary] = []
    if overload_observed and accepted_rate is not None and guarded is None:
        recovery_rate = max(minimum_rps, accepted_rate * 0.5)
        recovery_id = f"aimd-{route.id}-{shape}-recovery"
        if not _phase_fits_before(
            engine,
            route,
            epoch_id=recovery_id,
            arrival_window_seconds=duration,
            not_after_monotonic=not_after_monotonic,
        ):
            return _paused_result(results)
        recovery = await run_open_loop_epoch(
            engine,
            route,
            shape=shape,
            epoch_id=recovery_id,
            phase="recovery_after_observed_overload",
            offered_rps=recovery_rate,
            duration_seconds=duration,
            concurrency=ceiling,
            seed=seed + 100,
            shape_config=config,
            max_p95_ttft_seconds=ttft_limit,
            max_p95_total_seconds=total_limit,
        )
        results.append(recovery)
        recovery_epochs.append(recovery)
        if recovery.launch_guard_triggered:
            guarded = recovery
    else:
        with suppress(KeyError):
            engine.ledger.mark_plan_cell(
                f"load_epoch:aimd-{route.id}-{shape}-recovery",
                "not_applicable",
                "no_confirmed_rate_after_overload"
                if overload_observed
                else "no_two_epoch_overload_observed",
            )

    confirmation_eligible = [
        epoch.controller_eligible and not epoch.launch_guard_triggered
        for epoch in selected_confirmations
    ]
    confirmation_execution_complete = len(selected_confirmations) == 3
    confirmation_complete = confirmation_execution_complete and all(confirmation_eligible)
    confirmation_all_healthy = (
        all(epoch.healthy for epoch in selected_confirmations) if confirmation_complete else None
    )
    if guarded is not None:
        controller_completion_state = "campaign_guard_censored"
    elif accepted_rate is not None:
        controller_completion_state = "completed_confirmations_healthy"
    elif reached_unhealthy_floor:
        controller_completion_state = "completed_no_healthy_rate_at_floor"
    else:
        controller_completion_state = "confirmations_inconclusive_after_retries"

    if accepted_rate is None:
        capacity_bound_state = (
            "left_censored_no_healthy_rate_at_floor"
            if reached_unhealthy_floor
            else "confirmation_attempts_inconclusive"
        )
    elif unhealthy_upper_bound_rps is not None and unhealthy_upper_bound_rps > accepted_rate:
        capacity_bound_state = "bracketed_confirmed_healthy_lower_unhealthy_upper"
    elif overload_observed:
        capacity_bound_state = "confirmed_healthy_after_nonmonotonic_overload"
    else:
        capacity_bound_state = "right_censored_highest_tested_confirmed_healthy_no_overload"
    recovery = recovery_epochs[-1] if recovery_epochs else None
    engine.ledger.record_event_once(
        f"aimd_complete:{route.id}:{shape}",
        "aimd_complete",
        {
            "route_id": route.id,
            "shape": shape,
            "highest_observed_healthy_rps": best_healthy or None,
            "healthy_lower_bound_rps": accepted_rate,
            "unhealthy_upper_bound_rps": unhealthy_upper_bound_rps,
            "overload_observed": overload_observed,
            "nonmonotonic_overload_observed": nonmonotonic_overload_observed,
            "capacity_bound_state": capacity_bound_state,
            "controller_completion_state": controller_completion_state,
            "censor_reason": (
                guarded.launch_guard_reason or "launch_guard" if guarded is not None else None
            ),
            "confirmations_required": 3,
            "confirmation_stage": selected_confirmation_stage,
            "confirmation_stage_history": confirmation_history,
            "confirmation_healthy": [
                epoch.healthy if eligible else None
                for epoch, eligible in zip(
                    selected_confirmations, confirmation_eligible, strict=True
                )
            ],
            "confirmation_eligible": confirmation_eligible,
            "confirmation_censor_reasons": [
                (
                    epoch.launch_guard_reason or epoch.scientific_censor_reason or "launch_guard"
                    if not eligible
                    else None
                )
                for epoch, eligible in zip(
                    selected_confirmations, confirmation_eligible, strict=True
                )
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
                recovery.launch_guard_reason or recovery.scientific_censor_reason or "launch_guard"
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
    not_after_monotonic: float | None = None,
) -> LoadRunResult:
    validate_soak_config(config, engine.config.concurrency)
    rate = soak_rate_rps(config, route.id, shape)
    if rate <= 0:
        raise ValueError(f"soak rate must be positive for {route.id}/{shape}")
    blocks = int(config.get("blocks", 4))
    block_seconds = float(config.get("block_seconds", 30))
    ceiling = int(config.get("concurrency", engine.config.concurrency))
    baseline_attempts, baseline, paused = await _run_adaptive_baselines(
        engine,
        route,
        controller="soak",
        phase="soak_baseline",
        shape=shape,
        config=config,
        nominal_duration=block_seconds,
        default_rps=min(rate, 0.5),
        concurrency=ceiling,
        seed=seed,
        not_after_monotonic=not_after_monotonic,
    )
    results = LoadRunResult(list(baseline_attempts))
    if paused:
        return _paused_result(results)
    baseline_guard = next(
        (summary for summary in baseline_attempts if summary.launch_guard_triggered), None
    )
    max_rate_stages = int(config.get("max_rate_stages", 4))
    if baseline_guard is not None:
        reason = baseline_guard.launch_guard_reason or "launch_guard"
        engine.ledger.record_event_once(
            f"soak_controller_censored:{route.id}:{shape}",
            "soak_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": baseline_guard.epoch_id,
            },
        )
        _censor_unstarted_load_cells(
            engine,
            [
                soak_block_epoch_id(route.id, shape, stage, block)
                for stage in range(max_rate_stages)
                for block in range(blocks)
            ],
            reason,
        )
        return results
    if baseline is None:
        last_baseline = baseline_attempts[-1]
        all_measured_unhealthy = all(
            item.controller_eligible and not item.launch_guard_triggered and not item.healthy
            for item in baseline_attempts
        )
        reason = (
            "measured_unhealthy_at_all_baseline_rates"
            if all_measured_unhealthy
            else last_baseline.scientific_censor_reason or "baseline_attempts_inconclusive"
        )
        engine.ledger.record_event_once(
            f"soak_controller_censored:{route.id}:{shape}",
            "soak_controller_censored",
            {
                "route_id": route.id,
                "shape": shape,
                "reason": reason,
                "source_epoch_id": last_baseline.epoch_id,
            },
        )
        for stage in range(max_rate_stages):
            for block in range(blocks):
                with suppress(KeyError):
                    engine.ledger.mark_plan_cell(
                        f"load_epoch:{soak_block_epoch_id(route.id, shape, stage, block)}",
                        "not_applicable" if all_measured_unhealthy else "inconclusive",
                        reason,
                    )
        engine.ledger.record_event_once(
            f"soak_complete:{route.id}:{shape}",
            "soak_complete",
            {
                "route_id": route.id,
                "shape": shape,
                "requested_rate_rps": rate,
                "rate_rps": None,
                "blocks": blocks,
                "completed_blocks": 0,
                "block_eligible": [],
                "block_healthy": [],
                "block_censor_reasons": [],
                "execution_complete": all_measured_unhealthy,
                "scientifically_complete": all_measured_unhealthy,
                "all_blocks_healthy": False if all_measured_unhealthy else None,
                "rate_stage_history": [],
                "controller_completion_state": (
                    "completed_no_healthy_at_lowest_tested_rate"
                    if all_measured_unhealthy
                    else "baseline_attempts_inconclusive"
                ),
                "censor_reason": None if all_measured_unhealthy else reason,
            },
        )
        return results
    ttft_limit = None if baseline.p95_ttft_seconds is None else 2 * baseline.p95_ttft_seconds
    total_limit = None if baseline.p95_total_seconds is None else 2 * baseline.p95_total_seconds
    rate_decrease = float(config.get("rate_multiplicative_decrease", 0.5))
    minimum_rps = float(config.get("minimum_rps", 0.01))
    candidate_rate = max(minimum_rps, rate)
    selected_blocks: list[EpochSummary] = []
    selected_stage: int | None = None
    accepted_rate: float | None = None
    guarded: EpochSummary | None = None
    reached_unhealthy_floor = False
    rate_stage_history: list[dict[str, Any]] = []
    for stage in range(max_rate_stages):
        stage_blocks: list[EpochSummary] = []
        for block in range(blocks):
            block_id = soak_block_epoch_id(route.id, shape, stage, block)
            if not _phase_fits_before(
                engine,
                route,
                epoch_id=block_id,
                arrival_window_seconds=block_seconds,
                not_after_monotonic=not_after_monotonic,
            ):
                return _paused_result(results)
            summary = await run_open_loop_epoch(
                engine,
                route,
                shape=shape,
                epoch_id=block_id,
                phase="soak_block",
                offered_rps=candidate_rate,
                duration_seconds=block_seconds,
                concurrency=ceiling,
                seed=seed + stage * 1000 + block,
                shape_config=config,
                max_p95_ttft_seconds=ttft_limit,
                max_p95_total_seconds=total_limit,
            )
            results.append(summary)
            stage_blocks.append(summary)
            if summary.launch_guard_triggered:
                guarded = summary
                break
        eligible = [
            block.controller_eligible and not block.launch_guard_triggered for block in stage_blocks
        ]
        scientifically_complete = len(stage_blocks) == blocks and all(eligible)
        all_healthy = (
            all(block.healthy for block in stage_blocks) if scientifically_complete else None
        )
        rate_stage_history.append(
            {
                "stage": stage,
                "rate_rps": candidate_rate,
                "execution_complete": len(stage_blocks) == blocks,
                "scientifically_complete": scientifically_complete,
                "healthy": all_healthy,
            }
        )
        selected_blocks = stage_blocks
        selected_stage = stage
        if guarded is not None:
            break
        if scientifically_complete and all_healthy:
            accepted_rate = candidate_rate
            break
        if scientifically_complete and all_healthy is False:
            next_rate = max(minimum_rps, candidate_rate * rate_decrease)
            if next_rate >= candidate_rate:
                reached_unhealthy_floor = True
                break
            candidate_rate = next_rate
        # Ineligible transport evidence is repeated at the same offered rate with fresh IDs.

    for stage in range((selected_stage or 0) + 1, max_rate_stages):
        for block in range(blocks):
            with suppress(KeyError):
                engine.ledger.mark_plan_cell(
                    f"load_epoch:{soak_block_epoch_id(route.id, shape, stage, block)}",
                    "not_applicable",
                    "lower_rate_soak_not_needed",
                )

    block_eligible = [
        block.controller_eligible and not block.launch_guard_triggered for block in selected_blocks
    ]
    execution_complete = len(selected_blocks) == blocks
    scientifically_complete = execution_complete and all(block_eligible)
    all_blocks_healthy = (
        all(block.healthy for block in selected_blocks) if scientifically_complete else None
    )
    if guarded is not None:
        controller_completion_state = "campaign_guard_censored"
    elif accepted_rate is not None:
        controller_completion_state = "completed_healthy"
    elif reached_unhealthy_floor:
        controller_completion_state = "completed_unhealthy_at_floor"
    else:
        controller_completion_state = "rate_stages_inconclusive_after_retries"
    engine.ledger.record_event_once(
        f"soak_complete:{route.id}:{shape}",
        "soak_complete",
        {
            "route_id": route.id,
            "shape": shape,
            "requested_rate_rps": rate,
            "rate_rps": accepted_rate if accepted_rate is not None else candidate_rate,
            "accepted_rate_rps": accepted_rate,
            "rate_stage": selected_stage,
            "rate_stage_history": rate_stage_history,
            "blocks": blocks,
            "completed_blocks": len(selected_blocks),
            "block_eligible": block_eligible,
            "block_healthy": [
                block.healthy
                if block.controller_eligible and not block.launch_guard_triggered
                else None
                for block in selected_blocks
            ],
            "block_censor_reasons": [
                block.launch_guard_reason or block.scientific_censor_reason or "launch_guard"
                if not block.controller_eligible or block.launch_guard_triggered
                else None
                for block in selected_blocks
            ],
            "execution_complete": execution_complete,
            "scientifically_complete": scientifically_complete,
            "all_blocks_healthy": all_blocks_healthy,
            "controller_completion_state": controller_completion_state,
            "censor_reason": (
                guarded.launch_guard_reason or "launch_guard" if guarded is not None else None
            ),
        },
    )
    return results
