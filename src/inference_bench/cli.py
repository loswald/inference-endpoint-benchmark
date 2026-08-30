from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import sqlite3
import sys
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .atlas import generate_atlas
from .capacity_closure import build_capacity_closure_package_from_files
from .config import CampaignConfig, load_config, selected_capacity_cells
from .digitalocean_atlas import generate_digitalocean_atlas
from .digitalocean_closure import build_digitalocean_closure_package
from .engine import BenchmarkEngine, PaymentRequiredLatched, ReservationOverrunLatched
from .environment import (
    find_source_root,
    locked_distribution_versions,
    resolve_build_identity,
    source_tree_state_hash,
    validate_run_directory_separation,
)
from .ledger import BudgetExceeded, Ledger, TimeLimitReached
from .load import baseline_attempt_count, baseline_design, run_aimd, run_soak
from .matrix import load_matrix, matrix_plan, run_matrix
from .models import TRANSPORT_HEADER_PROFILE, RequestSpec, RouteConfig, canonical_json
from .plan import build_plan
from .profile_config import compile_profile_files
from .report import generate_report
from .soak_config import derive_soak_config
from .workloads import plan_static_suites

_RETRYABLE_STATUSES = {"rate_limited", "server_error", "timeout", "transport_error"}
_DEFAULT_CAPACITY_SHAPES = ("short_short", "long_short", "short_long", "mixed")


def _terminal_run_is_fully_sealed(output: Path) -> bool:
    """Inspect a completed ledger without creating SQLite sidecars or taking a writer lease.

    A terminal event plus the terminal source-manifest digest is the commit marker for an
    immutable run directory.  This read uses SQLite's immutable URI mode deliberately: an
    accidental second ``run`` invocation must not create a WAL/SHM file, rewrite the public
    projection, refresh owner diagnostics, or otherwise perturb the evidence package it refuses.
    A crash-window ledger that has not committed both markers returns false and is repaired by the
    normal exclusive-owner path below.
    """

    database = output / "ledger.sqlite3"
    if not database.is_file():
        return False
    try:
        with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as db:
            terminal = db.execute(
                "SELECT 1 FROM events WHERE event_key='campaign_terminal' LIMIT 1"
            ).fetchone()
            digest = db.execute(
                "SELECT value FROM meta WHERE key='terminal_run_manifest_sha256'"
            ).fetchone()
    except sqlite3.Error:
        # Missing/legacy tables and crash-window WAL state are handled under the exclusive owner
        # lease.  Never infer terminality from a partially readable file.
        return False
    return terminal is not None and digest is not None and bool(str(digest[0]))


def _capacity_execution_order(
    config: CampaignConfig, suite_name: str
) -> list[tuple[RouteConfig, str]]:
    if suite_name not in {"aimd", "soak"}:
        raise ValueError("capacity execution order supports only aimd or soak")
    suite = config.suites.get(suite_name)
    if not suite or not suite.get("enabled", True):
        return []
    cells = selected_capacity_cells(config, suite_name)
    random.Random(f"capacity-order/v1:{config.seed}:{suite_name}").shuffle(cells)
    return cells


def _record_capacity_execution_order(
    ledger: Ledger, suite_name: str, order: list[tuple[RouteConfig, str]]
) -> None:
    ledger.record_event_once(
        f"capacity_execution_order:{suite_name}",
        "capacity_execution_order",
        {
            "suite": suite_name,
            "randomization": "deterministic seeded shuffle; capacity cells execute sequentially",
            "cells": [
                {"position": index, "route_id": route.id, "shape": shape}
                for index, (route, shape) in enumerate(order)
            ],
        },
    )


def _static_execution_blocks(
    config: CampaignConfig, specs: list[RequestSpec]
) -> list[tuple[tuple[bool, str, str], list[RequestSpec]]]:
    """Build a deterministic, resume-stable order without time-confounding cell levels."""

    grouped: dict[tuple[bool, str, str], list[RequestSpec]] = {}
    for spec in specs:
        key = (spec.suite != "warmup", spec.route_id, spec.suite)
        grouped.setdefault(key, []).append(spec)
    warmup_keys = sorted(key for key in grouped if not key[0])
    measured_keys = sorted(key for key in grouped if key[0])
    random.Random(f"static-block-order/v1:{config.seed}").shuffle(measured_keys)
    blocks: list[tuple[tuple[bool, str, str], list[RequestSpec]]] = []
    for key in (*warmup_keys, *measured_keys):
        block = list(grouped[key])
        if key[0]:
            random.Random(f"static-cell-order/v1:{config.seed}:{key[1]}:{key[2]}").shuffle(block)
        blocks.append((key, block))
    return blocks


def _record_static_execution_order(
    ledger: Ledger, blocks: list[tuple[tuple[bool, str, str], list[RequestSpec]]]
) -> None:
    position = 0
    realized: list[dict[str, object]] = []
    for block_position, (key, specs) in enumerate(blocks):
        for cell_position, spec in enumerate(specs):
            realized.append(
                {
                    "position": position,
                    "block_position": block_position,
                    "cell_position": cell_position,
                    "route_id": spec.route_id,
                    "suite": spec.suite,
                    "cell_id": spec.cell_id,
                    "logical_id": spec.logical_id,
                    "warmup_diagnostic": not key[0],
                }
            )
            position += 1
    ledger.record_event_once(
        "static_execution_order:v1",
        "static_execution_order",
        {
            "randomization": (
                "warmup diagnostics first; measured blocks and cells use deterministic seeded "
                "shuffles; blocks execute sequentially"
            ),
            "cells": realized,
        },
    )


def _pending_static_specs(engine: BenchmarkEngine, specs: list[RequestSpec]) -> list[RequestSpec]:
    pending: list[RequestSpec] = []
    for spec in specs:
        attempts = engine.ledger.attempts_for_logical(spec.logical_id)
        if not attempts:
            pending.append(spec)
            continue
        latest = max(attempts, key=lambda row: int(row["attempt_index"]))
        final = bool(
            latest["state"] == "unknown"
            or latest.get("status") == "success"
            or latest.get("status") not in _RETRYABLE_STATUSES
            or int(latest["attempt_index"]) >= engine.config.retries + 1
        )
        if final:
            if latest["state"] == "unknown":
                with suppress(KeyError):
                    engine.ledger.mark_plan_cell(
                        f"request:{spec.logical_id}",
                        "inconclusive",
                        "unknown_provider_outcome",
                    )
            else:
                with suppress(KeyError):
                    engine.ledger.mark_plan_cell(f"request:{spec.logical_id}", "completed")
            continue
        if (
            latest.get("status") in _RETRYABLE_STATUSES
            and int(latest["attempt_index"]) < engine.config.retries + 1
        ):
            pending.append(spec)
    return pending


def _time_variation_panel_is_terminal(ledger: Ledger, panel: int) -> bool:
    return any(
        ledger.event_by_key(f"time_variation_panel_{state}:{panel}") is not None
        for state in ("completed", "censored")
    )


async def _run_static(
    engine: BenchmarkEngine,
    specs: list[RequestSpec],
    config: CampaignConfig,
    *,
    offered_rps: float | None = None,
) -> str | None:
    """Run one endpoint × suite block serially with no coordinated-omission claim.

    Static cells are low-load measurements and validation probes, not capacity tests. Starting the
    next cell only after the prior one drains prevents long context/output probes from queueing
    behind each other or contaminating another endpoint's latency baseline.
    """

    static_rps = (
        float(offered_rps)
        if offered_rps is not None
        else float(config.suites.get("static", {}).get("offered_rps", 1.0))
    )
    loop = asyncio.get_running_loop()
    not_before = loop.time()
    for spec in specs:
        await asyncio.sleep(max(0.0, not_before - loop.time()))
        scheduled_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        try:
            result = await engine.execute(
                spec,
                scheduled_at_utc=scheduled_at,
                queue_delay_seconds=0.0,
            )
        except BudgetExceeded:
            return "cost_guard"
        except TimeLimitReached:
            return "time_guard"
        except PaymentRequiredLatched:
            return "http_402_latch"
        except ReservationOverrunLatched:
            return "reservation_overrun_latch"
        if result is not None and result.http_status == 402:
            return "http_402_latch"
        if engine.reservation_overrun_latched:
            return "reservation_overrun_latch"
        not_before = loop.time() + 1.0 / static_rps
    return None


async def _run_time_variation(
    engine: BenchmarkEngine,
    specs: list[RequestSpec],
    config: CampaignConfig,
    *,
    resume_invocation: bool = False,
) -> str | None:
    """Execute resume-safe fixed-offset panels without optional gap traffic.

    Dedicated and interleaved studies share the same concurrent open-loop panel executor.  The
    persisted campaign start is the only schedule anchor: restarting the process cannot slide the
    six-hour window or turn overdue panels into a fresh experiment.
    """

    suite = config.suites["time_variation"]
    offered_rps = float(suite.get("offered_rps", 0.2))
    panel_concurrency = int(suite.get("concurrency", config.concurrency))
    deadline_seconds = float(suite.get("panel_deadline_seconds", 600))
    arrival_lateness_tolerance = float(
        suite.get("arrival_lateness_tolerance_seconds", 0.25)
    )
    cutoff_seconds = float(suite["send_cutoff_seconds"])
    started_text = engine.ledger.meta("started_at_utc")
    if not started_text:
        raise RuntimeError("time variation requires the immutable campaign start time")
    anchor = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
    by_panel: dict[int, list[RequestSpec]] = {}
    for spec in specs:
        panel = int(spec.metadata["time_variation_panel"])
        by_panel.setdefault(panel, []).append(spec)

    def elapsed() -> float:
        return max(0.0, (datetime.now(UTC) - anchor).total_seconds())

    for panel in sorted(by_panel):
        if _time_variation_panel_is_terminal(engine.ledger, panel):
            continue
        panel_specs = by_panel[panel]
        offset = float(panel_specs[0].metadata["time_variation_offset_seconds"])
        loop = asyncio.get_running_loop()
        panel_admission_monotonic = loop.time()
        arrival_expiry_monotonic = (
            panel_admission_monotonic - arrival_lateness_tolerance
        )
        panel_start_monotonic = panel_admission_monotonic + offset - elapsed()
        await asyncio.sleep(max(0.0, panel_start_monotonic - loop.time()))
        reason = await _run_time_variation_panel(
            engine,
            panel,
            panel_specs,
            offered_rps=offered_rps,
            concurrency=panel_concurrency,
            planned_offset_seconds=offset,
            deadline_seconds=deadline_seconds,
            panel_start_monotonic=panel_start_monotonic,
            arrival_expiry_monotonic=arrival_expiry_monotonic,
            resume_invocation=resume_invocation,
            not_after_monotonic=loop.time() + max(0.0, cutoff_seconds - elapsed()),
        )
        if reason:
            return reason
    return None


async def _run_time_variation_panel(
    engine: BenchmarkEngine,
    panel: int,
    specs: list[RequestSpec],
    *,
    offered_rps: float,
    concurrency: int,
    planned_offset_seconds: float,
    deadline_seconds: float,
    panel_start_monotonic: float,
    arrival_expiry_monotonic: float,
    resume_invocation: bool,
    not_after_monotonic: float | None = None,
) -> str | None:
    """Run one resume-safe open-loop panel on its immutable absolute arrival schedule."""

    censored_key = f"time_variation_panel_censored:{panel}"
    if engine.ledger.event_by_key(censored_key) is not None:
        return None
    started_key = f"time_variation_panel_started:{panel}"
    ordered = sorted(specs, key=lambda spec: spec.logical_id)
    random.Random(f"time-variation-panel/v2:{engine.config.seed}:{panel}").shuffle(ordered)
    pending_ids = {spec.logical_id for spec in _pending_static_specs(engine, specs)}
    pending = [
        (index, spec)
        for index, spec in enumerate(ordered)
        if spec.logical_id in pending_ids
    ]
    if concurrency < len(ordered):
        raise ValueError(
            f"time variation panel {panel} concurrency is below its registered arrival count"
        )
    launch_span = max(0.0, (len(ordered) - 1) / offered_rps)
    maximum_timeout = max((spec.timeout_seconds for spec in ordered), default=0.0)
    if launch_span + maximum_timeout > deadline_seconds:
        raise ValueError(
            f"time variation panel {panel} cannot drain inside its explicit deadline"
        )
    loop = asyncio.get_running_loop()
    panel_deadline_monotonic = panel_start_monotonic + deadline_seconds
    elapsed_pending = [
        spec
        for index, spec in pending
        if panel_start_monotonic + index / offered_rps < arrival_expiry_monotonic
    ]
    if elapsed_pending:
        censor_reason = (
            "resume_missed_registered_panel_arrival"
            if resume_invocation
            else "missed_registered_panel_arrival"
        )
        for _, spec in pending:
            engine.ledger.mark_plan_cell_if_planned(
                f"request:{spec.logical_id}",
                "time_censored",
                censor_reason,
            )
        engine.ledger.record_event_once(
            censored_key,
            "time_variation_panel_censored",
            {
                "panel": panel,
                "planned_offset_seconds": planned_offset_seconds,
                "reason": censor_reason,
                "resume_invocation": resume_invocation,
                "registered_requests": len(ordered),
                "previously_attempted_requests": len(ordered) - len(pending),
                "elapsed_unsent_arrivals": len(elapsed_pending),
                "remaining_unsent_requests_censored": len(pending),
            },
        )
        return None
    if (
        not_after_monotonic is not None
        and panel_start_monotonic + launch_span > not_after_monotonic
    ):
        # A matched panel is indivisible. If all registered arrivals cannot launch before
        # the provider cutoff, send none of it and preserve an honest time-guard result.
        return "time_guard"
    engine.ledger.record_event_once(
        started_key,
        "time_variation_panel_started",
        {
            "panel": panel,
            "planned_offset_seconds": planned_offset_seconds,
            "requests": len(specs),
            "offered_rps": offered_rps,
            "concurrency": concurrency,
            "panel_deadline_seconds": deadline_seconds,
            "arrival_pattern": "deterministic open-loop global schedule",
        },
    )

    stop = asyncio.Event()
    semaphore = asyncio.Semaphore(concurrency)

    async def execute(index: int, spec: RequestSpec) -> str | None:
        due_monotonic = panel_start_monotonic + index / offered_rps
        await asyncio.sleep(max(0.0, due_monotonic - loop.time()))
        async with semaphore:
            if stop.is_set():
                return None
            if not_after_monotonic is not None and loop.time() > not_after_monotonic:
                stop.set()
                return "time_guard"
            if loop.time() + spec.timeout_seconds > panel_deadline_monotonic:
                stop.set()
                return "time_guard"
            scheduled_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            try:
                result = await engine.execute(
                    spec,
                    scheduled_at_utc=scheduled_at,
                    queue_delay_seconds=max(0.0, loop.time() - due_monotonic),
                )
            except BudgetExceeded:
                reason = "cost_guard"
            except TimeLimitReached:
                reason = "time_guard"
            except PaymentRequiredLatched:
                reason = "http_402_latch"
            except ReservationOverrunLatched:
                reason = "reservation_overrun_latch"
            else:
                reason = (
                    "http_402_latch"
                    if result is not None and result.http_status == 402
                    else None
                )
        if reason:
            stop.set()
        return reason

    results = await asyncio.gather(
        *(execute(index, spec) for index, spec in pending)
    )
    reason = next((value for value in results if value), None)
    if reason:
        return reason
    engine.ledger.record_event_once(
        f"time_variation_panel_completed:{panel}",
        "time_variation_panel_completed",
        {"panel": panel, "planned_offset_seconds": planned_offset_seconds},
    )
    return None


def _capacity_job_seconds(config: CampaignConfig, suite_name: str) -> float:
    """Strict phase-by-phase wall bound used only to protect matched time panels."""

    suite = config.suites[suite_name]
    minimum = float(suite.get("minimum_rps", 0.01))
    baseline_samples = int(suite.get("baseline_samples", 5))
    baseline_decrease = float(suite.get("baseline_multiplicative_decrease", 0.5))
    timeout = max(route.request_timeout_seconds for route in config.routes)
    baseline_nominal = float(
        suite.get("block_seconds", 30)
        if suite_name == "soak"
        else suite.get("epoch_seconds", 20)
    )
    _, _, baseline_rate = baseline_design(
        suite,
        baseline_nominal,
        default_rps=float(suite.get("rate_rps", suite.get("initial_rps", minimum))),
    )
    baseline_attempts = baseline_attempt_count(
        suite, baseline_rate, field_prefix=suite_name
    )
    baseline_seconds = sum(
        max(
            baseline_nominal,
            baseline_samples / max(minimum, baseline_rate * baseline_decrease**attempt),
        )
        + timeout
        for attempt in range(baseline_attempts)
    )
    if suite_name == "soak":
        scheduled = (
            int(suite.get("blocks", 4))
            * int(suite.get("max_rate_stages", 4))
            * (float(suite.get("block_seconds", 30)) + timeout)
        )
    else:
        epoch_seconds = float(suite.get("epoch_seconds", 20))
        separator_samples = int(suite.get("confirmation_separator_samples", baseline_samples))
        separator_seconds = max(epoch_seconds, separator_samples / baseline_rate)
        scheduled = (
            int(suite.get("epochs", 12)) * (epoch_seconds + timeout)
            + int(suite.get("confirmation_max_stages", 4))
            * (3 * (epoch_seconds + timeout) + 2 * (separator_seconds + timeout))
            + epoch_seconds
            + timeout
        )
    return baseline_seconds + scheduled + 10.0


async def _run_interleaved_six_hour_study(
    engine: BenchmarkEngine,
    variation_specs: list[RequestSpec],
    gap_static_specs: list[RequestSpec],
    config: CampaignConfig,
    *,
    resume_invocation: bool = False,
) -> str | None:
    """Protect matched time panels while filling every safe interval with gap work.

    Only one provider-bearing job runs at a time. Static work starts only when its conservative
    request bound fits; capacity work checks every provider-bearing phase against the next panel
    guard and pauses between phases when necessary. The immutable ledger makes every panel,
    static request, AIMD cell, and fixed-rate cell resume-safe without duplicate sends.
    """

    suite = config.suites["time_variation"]
    offered_rps = float(suite.get("offered_rps", 0.2))
    panel_concurrency = int(suite.get("concurrency", config.concurrency))
    guard_seconds = float(suite.get("panel_guard_seconds", 300))
    deadline_seconds = float(suite.get("panel_deadline_seconds", 600))
    arrival_lateness_tolerance = float(
        suite.get("arrival_lateness_tolerance_seconds", 0.25)
    )
    cutoff_seconds = float(suite["send_cutoff_seconds"])
    started_text = engine.ledger.meta("started_at_utc")
    if not started_text:
        raise RuntimeError("interleaved study requires the immutable campaign start time")
    anchor = datetime.fromisoformat(started_text.replace("Z", "+00:00"))

    by_panel: dict[int, list[RequestSpec]] = {}
    offsets: dict[int, float] = {}
    for spec in variation_specs:
        panel = int(spec.metadata["time_variation_panel"])
        by_panel.setdefault(panel, []).append(spec)
        offsets[panel] = float(spec.metadata["time_variation_offset_seconds"])

    static_jobs = list(gap_static_specs)
    random.Random(f"six-hour-static-order/v1:{config.seed}").shuffle(static_jobs)
    capacity_jobs: list[tuple[str, RouteConfig, str]] = [
        (suite_name, route, shape)
        for suite_name in ("aimd", "soak")
        for route, shape in _capacity_execution_order(config, suite_name)
    ]
    quick_static = [spec for spec in static_jobs if spec.suite in {"capability", "cache"}]
    long_static = [spec for spec in static_jobs if spec.suite not in {"capability", "cache"}]
    aimd_jobs = [job for job in capacity_jobs if job[0] == "aimd"]
    soak_jobs = [job for job in capacity_jobs if job[0] == "soak"]
    gap_jobs: list[tuple[str, object]] = [
        *(("static", spec) for spec in quick_static),
        *(("capacity", job) for job in aimd_jobs),
        *(("capacity", job) for job in soak_jobs),
        *(("static", spec) for spec in long_static),
    ]
    awaited_panel_schedules: dict[int, tuple[float, float]] = {}

    def elapsed() -> float:
        return max(0.0, (datetime.now(UTC) - anchor).total_seconds())

    def unfinished_panels() -> list[int]:
        return [
            panel
            for panel in sorted(by_panel)
            if not _time_variation_panel_is_terminal(engine.ledger, panel)
        ]

    def pending_job(job: tuple[str, object]) -> bool:
        kind, value = job
        if kind == "static":
            return bool(_pending_static_specs(engine, [value]))  # type: ignore[list-item]
        suite_name, route, shape = value  # type: ignore[misc]
        return engine.ledger.event_by_key(f"{suite_name}_complete:{route.id}:{shape}") is None

    while elapsed() < cutoff_seconds:
        panels = unfinished_panels()
        due = [panel for panel in panels if offsets[panel] <= elapsed()]
        if due:
            panel = due[0]
            loop = asyncio.get_running_loop()
            stored_schedule = awaited_panel_schedules.pop(panel, None)
            if stored_schedule is None:
                panel_admission_monotonic = loop.time()
                arrival_expiry_monotonic = (
                    panel_admission_monotonic - arrival_lateness_tolerance
                )
                panel_start_monotonic = (
                    panel_admission_monotonic + offsets[panel] - elapsed()
                )
            else:
                panel_start_monotonic, arrival_expiry_monotonic = stored_schedule
            not_after_monotonic = loop.time() + max(0.0, cutoff_seconds - elapsed())
            reason = await _run_time_variation_panel(
                engine,
                panel,
                by_panel[panel],
                offered_rps=offered_rps,
                concurrency=panel_concurrency,
                planned_offset_seconds=offsets[panel],
                deadline_seconds=deadline_seconds,
                panel_start_monotonic=panel_start_monotonic,
                arrival_expiry_monotonic=arrival_expiry_monotonic,
                resume_invocation=resume_invocation,
                not_after_monotonic=not_after_monotonic,
            )
            if reason:
                return reason
            continue

        gap_jobs = [job for job in gap_jobs if pending_job(job)]
        next_panel = min((offsets[panel] for panel in panels), default=cutoff_seconds)
        usable_seconds = min(next_panel - guard_seconds, cutoff_seconds) - elapsed()
        selected_index: int | None = None
        for index, (kind, value) in enumerate(gap_jobs):
            if kind == "static":
                spec = value
                estimate = (  # type: ignore[union-attr]
                    float(spec.timeout_seconds) + 1.0 / offered_rps + 10.0
                )
            else:
                suite_name, route, _ = value  # type: ignore[misc]
                capacity_suite = config.suites[suite_name]
                # A controller is deliberately resumable at every baseline/epoch/block boundary.
                # Admit a useful slice when one nominal phase can fit; the controller itself
                # performs the exact duration-plus-route-timeout check before every new phase.
                nominal_seconds = float(
                    capacity_suite.get("block_seconds", 30)
                    if suite_name == "soak"
                    else capacity_suite.get("epoch_seconds", 20)
                )
                estimate = nominal_seconds + route.request_timeout_seconds + 10.0
            if estimate <= usable_seconds:
                selected_index = index
                break
        if selected_index is None:
            if panels:
                panel = min(panels, key=lambda value: offsets[value])
                loop = asyncio.get_running_loop()
                panel_admission_monotonic = loop.time()
                arrival_expiry_monotonic = (
                    panel_admission_monotonic - arrival_lateness_tolerance
                )
                awaited_panel_schedules[panel] = (
                    panel_admission_monotonic + next_panel - elapsed(),
                    arrival_expiry_monotonic,
                )
                await asyncio.sleep(max(0.0, next_panel - elapsed()))
                continue
            break

        kind, value = gap_jobs.pop(selected_index)
        if kind == "static":
            reason = await _run_static(
                engine,
                [value],  # type: ignore[list-item]
                config,
                offered_rps=offered_rps,
            )
        else:
            suite_name, route, shape = value  # type: ignore[misc]
            loop = asyncio.get_running_loop()
            boundary_elapsed = min(next_panel - guard_seconds, cutoff_seconds)
            not_after_monotonic = loop.time() + max(0.0, boundary_elapsed - elapsed())
            blocks = (
                await run_aimd(
                    engine,
                    route,
                    shape,
                    config.suites[suite_name],
                    seed=config.seed,
                    not_after_monotonic=not_after_monotonic,
                )
                if suite_name == "aimd"
                else await run_soak(
                    engine,
                    route,
                    shape,
                    config.suites[suite_name],
                    seed=config.seed,
                    not_after_monotonic=not_after_monotonic,
                )
            )
            if blocks.paused_for_window:
                # Keep the exact controller cell pending. Completed epoch IDs restore from the
                # ledger in the next interval, so reinserting the job resumes rather than
                # duplicates it.
                gap_jobs.insert(selected_index, (kind, value))
                if panels:
                    await asyncio.sleep(max(0.0, next_panel - elapsed()))
                    continue
                break
            reason = next(
                (
                    block.launch_guard_reason
                    for block in blocks
                    if block.launch_guard_triggered
                ),
                None,
            )
        if reason:
            if reason == "time_guard" and not unfinished_panels():
                break
            return str(reason)

    if unfinished_panels():
        return "time_guard"
    remaining_gap_jobs = sum(pending_job(job) for job in gap_jobs)
    completed_panels = sum(
        engine.ledger.event_by_key(f"time_variation_panel_completed:{panel}") is not None
        for panel in by_panel
    )
    censored_panels = sum(
        engine.ledger.event_by_key(f"time_variation_panel_censored:{panel}") is not None
        for panel in by_panel
    )
    engine.ledger.record_event_once(
        "six_hour_window_completed",
        "six_hour_window_completed",
        {
            "time_panels_terminal": completed_panels + censored_panels,
            "time_panels_completed": completed_panels,
            "time_panels_censored": censored_panels,
            "optional_gap_jobs_remaining": remaining_gap_jobs,
            "send_cutoff_seconds": cutoff_seconds,
        },
    )
    return "six_hour_window_completed"


async def run_campaign(
    config: CampaignConfig, output: Path, *, invocation: tuple[str, ...] = ()
) -> None:
    if _terminal_run_is_fully_sealed(output):
        raise ValueError(
            "run directory is already terminal; reports are immutable and live execution "
            "cannot resume"
        )
    plan = build_plan(config)
    placeholders = plan.native_placeholder_routes
    if placeholders:
        raise ValueError(
            "live run contains fail-closed native adapter placeholders: " + ", ".join(placeholders)
        )
    # Validate the source/runtime identity before creating any run state. A dirty checkout,
    # missing dependency, or other local provenance failure must leave the requested output path
    # untouched so the operator can fix the checkout and retry normally.
    run_manifest = _runtime_manifest(config, invocation, output_dir=output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output, exclusive_owner=True)
    resume_invocation = ledger.meta("started_at_utc") is not None
    try:
        ledger.initialize(
            campaign_hash=config.identity_hash, config_json=canonical_json(config.public_dict())
        )
        ledger.set_meta_once("run_manifest_json", canonical_json(run_manifest))
        if ledger.event_by_key("campaign_terminal") is not None:
            # A crash may commit the canonical terminal event just before its prompt-free JSONL
            # projection or terminal source digest is fsynced. Repairing either derived artifact
            # is safe and sends no traffic. Once the digest exists, an accidental live invocation
            # must remain a read-free refusal: rechecking against a later checkout would mutate an
            # already complete evidence package with a spurious drift event.
            if ledger.meta("terminal_run_manifest_sha256") is None:
                _verify_runtime_identity(
                    ledger,
                    config,
                    invocation,
                    output,
                    stage="terminal",
                )
                ledger.rebuild_events_jsonl()
            raise ValueError(
                "run directory is already terminal; reports are immutable and live execution "
                "cannot resume"
            )
        ledger.register_plan_cells(list(plan.coverage_cells))
    except Exception:
        ledger.close()
        raise
    try:
        recovered = ledger.recover_in_flight()
        if recovered:
            ledger.record_event("resume_notice", {"unknown_in_flight_count": recovered})
        (output / "campaign.public.json").write_text(
            json.dumps(config.public_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        engine = BenchmarkEngine(config, ledger)
    except Exception:
        ledger.close()
        raise
    try:
        try:
            engine.preflight()
            _verify_runtime_identity(
                ledger,
                config,
                invocation,
                output,
                stage="pre_send",
            )
        except Exception:
            ledger.finalize_plan("preflight_failed")
            ledger.record_event_once(
                "campaign_terminal",
                "campaign_terminal",
                {"reason": "preflight_failed", "error_kind": "adapter_or_identity_preflight_error"},
            )
            raise
        static_specs = plan_static_suites(config.routes, config.suites, seed=config.seed)
        time_variation_specs = [spec for spec in static_specs if spec.suite == "time_variation"]
        static_specs = [spec for spec in static_specs if spec.suite != "time_variation"]
        time_variation = config.suites.get("time_variation")
        if time_variation and time_variation.get("interleave_gap_work", False):
            static_blocks = _static_execution_blocks(config, static_specs)
            _record_static_execution_order(ledger, static_blocks)
            for suite_name in ("aimd", "soak"):
                order = _capacity_execution_order(config, suite_name)
                if order:
                    _record_capacity_execution_order(ledger, suite_name, order)
            reason = await _run_interleaved_six_hour_study(
                engine,
                time_variation_specs,
                [spec for _, block in static_blocks for spec in block],
                config,
                resume_invocation=resume_invocation,
            )
            if reason == "six_hour_window_completed":
                ledger.finalize_plan(reason)
                ledger.record_event_once(
                    "campaign_terminal", "campaign_terminal", {"reason": reason}
                )
                return
            if reason:
                ledger.finalize_plan(reason)
                ledger.record_event_once(
                    "campaign_terminal", "campaign_terminal", {"reason": reason}
                )
                return
            ledger.finalize_plan("plan_completed")
            ledger.record_event_once(
                "campaign_terminal", "campaign_terminal", {"reason": "plan_completed"}
            )
            return
        static_blocks = _static_execution_blocks(config, static_specs)
        _record_static_execution_order(ledger, static_blocks)
        for _, block in static_blocks:
            pending = _pending_static_specs(engine, block)
            if pending:
                static_reason = await _run_static(engine, pending, config)
                if static_reason:
                    ledger.finalize_plan(static_reason)
                    ledger.record_event_once(
                        "campaign_terminal",
                        "campaign_terminal",
                        {"reason": static_reason},
                    )
                    return
        if time_variation_specs:
            time_variation_reason = await _run_time_variation(
                engine,
                time_variation_specs,
                config,
                resume_invocation=resume_invocation,
            )
            if time_variation_reason:
                ledger.finalize_plan(time_variation_reason)
                ledger.record_event_once(
                    "campaign_terminal",
                    "campaign_terminal",
                    {"reason": time_variation_reason},
                )
                return
        aimd = config.suites.get("aimd")
        if aimd and aimd.get("enabled", True):
            aimd_order = _capacity_execution_order(config, "aimd")
            _record_capacity_execution_order(ledger, "aimd", aimd_order)
            for route, shape in aimd_order:  # sequential: endpoint-isolated capacity sweeps
                epochs = await run_aimd(engine, route, shape, aimd, seed=config.seed)
                if any(epoch.launch_guard_triggered for epoch in epochs):
                    reason = next(
                        (
                            epoch.launch_guard_reason
                            for epoch in epochs
                            if epoch.launch_guard_triggered
                        ),
                        "launch_guard",
                    )
                    ledger.finalize_plan(str(reason))
                    ledger.record_event_once(
                        "campaign_terminal", "campaign_terminal", {"reason": reason}
                    )
                    return
        soak = config.suites.get("soak")
        if soak and soak.get("enabled", True):
            soak_order = _capacity_execution_order(config, "soak")
            _record_capacity_execution_order(ledger, "soak", soak_order)
            for route, shape in soak_order:  # sequential: endpoint-isolated sustained workloads
                blocks = await run_soak(engine, route, shape, soak, seed=config.seed)
                if any(block.launch_guard_triggered for block in blocks):
                    reason = next(
                        (
                            block.launch_guard_reason
                            for block in blocks
                            if block.launch_guard_triggered
                        ),
                        "launch_guard",
                    )
                    ledger.finalize_plan(str(reason))
                    ledger.record_event_once(
                        "campaign_terminal", "campaign_terminal", {"reason": reason}
                    )
                    return
        ledger.finalize_plan("plan_completed")
        ledger.record_event_once(
            "campaign_terminal", "campaign_terminal", {"reason": "plan_completed"}
        )
    except (
        BudgetExceeded,
        TimeLimitReached,
        PaymentRequiredLatched,
        ReservationOverrunLatched,
    ) as exc:
        reason = (
            "cost_guard"
            if isinstance(exc, BudgetExceeded)
            else "time_guard"
            if isinstance(exc, TimeLimitReached)
            else "http_402_latch"
            if isinstance(exc, PaymentRequiredLatched)
            else "reservation_overrun_latch"
        )
        ledger.finalize_plan(reason)
        ledger.record_event_once(
            "campaign_terminal",
            "campaign_terminal",
            {"reason": reason},
        )
    except Exception:
        ledger.finalize_plan("unexpected_runner_error")
        ledger.record_event_once(
            "campaign_terminal",
            "campaign_terminal",
            {"reason": "unexpected_runner_error", "error_kind": "unexpected_runner_error"},
        )
        raise
    finally:
        terminal_identity_error: Exception | None = None
        try:
            _verify_runtime_identity(
                ledger,
                config,
                invocation,
                output,
                stage="terminal",
            )
        except Exception as exc:
            terminal_identity_error = exc
        await engine.close()
        try:
            ledger.rebuild_events_jsonl()
        finally:
            ledger.close()
        if terminal_identity_error is not None:
            raise terminal_identity_error


def _verify_runtime_identity(
    ledger: Ledger,
    config: CampaignConfig,
    invocation: tuple[str, ...],
    output: Path,
    *,
    stage: str,
) -> None:
    if stage not in {"pre_send", "terminal"}:
        raise ValueError("runtime identity stage must be pre_send or terminal")
    expected = ledger.meta("run_manifest_json")
    if expected is None:
        raise RuntimeError("runtime identity cannot be verified without the immutable manifest")
    try:
        observed = canonical_json(_runtime_manifest(config, invocation, output_dir=output))
    except Exception as exc:
        ledger.record_event_once(
            f"source_identity_drift:{stage}",
            "source_identity_drift",
            {"stage": stage, "error_kind": "runtime_identity_observation_error"},
        )
        raise RuntimeError(f"{stage} source identity verification failed") from exc
    if observed != expected:
        ledger.record_event_once(
            f"source_identity_drift:{stage}",
            "source_identity_drift",
            {"stage": stage, "error_kind": "runtime_manifest_changed"},
        )
        raise RuntimeError(f"{stage} source identity changed")
    digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    ledger.set_meta_once(f"{stage}_run_manifest_sha256", digest)


def _runtime_manifest(
    config: CampaignConfig,
    invocation: tuple[str, ...],
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    root = _source_root()
    identity = resolve_build_identity(root, output_dir=output_dir)
    if output_dir is not None:
        validate_run_directory_separation(
            identity.source_root, output_dir, list(identity.tracked_files)
        )
    lock = identity.dependency_lock
    normalized_invocation = _normalize_live_invocation(invocation)
    return {
        "schema_version": "run-manifest/v2",
        "normalized_exact_invocation": normalized_invocation,
        "raw_invocation_sha256": hashlib.sha256(
            canonical_json(list(invocation)).encode("utf-8")
        ).hexdigest(),
        "client_location": config.client_location,
        "connection_reuse_by_route": {route.id: route.connection_reuse for route in config.routes},
        "http2_by_route": {route.id: route.http2 for route in config.routes},
        "transport_max_connections_by_route": {
            route.id: route.transport_max_connections for route in config.routes
        },
        "transport_header_profile_by_route": {
            route.id: TRANSPORT_HEADER_PROFILE for route in config.routes
        },
        "request_timeout_seconds_by_route": {
            route.id: route.request_timeout_seconds for route in config.routes
        },
        "adapter_plugins": list(config.adapter_plugin_identities()),
        "provider_documentation_declarations": [
            {
                "route_id": route.id,
                "documentation_source_url": route.documentation_source_url,
                "pricing_source_url": route.pricing_source_url,
                "evidence_retrieved_at_utc": route.evidence_retrieved_at_utc,
                "declared_evidence_bundle_sha256": route.evidence_bundle_sha256,
                "verification_status": "declared_unverified_by_harness",
            }
            for route in config.routes
        ],
        "transport_trust_env": False,
        "source_commit": identity.revision,
        "source_identity_kind": identity.kind,
        "source_dirty": False,
        "source_dirty_tree_sha256": identity.tree_sha256,
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "dependency_lock_file": "requirements.lock",
        "execution_environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine_architecture": platform.machine(),
            "distributions": locked_distribution_versions(lock),
        },
    }


def _normalize_live_invocation(invocation: tuple[str, ...]) -> list[str]:
    """Redact path-bearing argv positions by CLI role, never filename heuristics."""

    if not invocation:
        return []
    normalized = ["inference-bench"]
    config_redacted = False
    index = 1
    while index < len(invocation):
        item = invocation[index]
        if item == "--output":
            normalized.append("--output")
            if index + 1 >= len(invocation):
                raise ValueError("live invocation has --output without a value")
            normalized.append("<RUN_DIR>")
            index += 2
            continue
        if item.startswith("--output="):
            normalized.append("--output=<RUN_DIR>")
            index += 1
            continue
        if item == "run":
            normalized.append(item)
        elif not item.startswith("-") and not config_redacted:
            normalized.append("<CONFIG_OR_PATH>")
            config_redacted = True
        else:
            normalized.append(item)
        index += 1
    return normalized


def _source_root() -> Path:
    return find_source_root(Path(__file__))


def _dirty_tree_hash(
    root: Path,
    status: str | None,
    diff: str | None,
    untracked: str | None,
) -> str | None:
    return source_tree_state_hash(root, status, diff, untracked)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="credential-free plan and conservative cost calculation")
    plan.add_argument("config", type=Path)
    plan_matrix = sub.add_parser(
        "plan-matrix", help="credential-free plan for parallel provider campaigns"
    )
    plan_matrix.add_argument("matrix", type=Path)
    run = sub.add_parser("run", help="execute a live campaign")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--confirm-live", action="store_true")
    run.add_argument(
        "--only-suite",
        choices=(
            "warmup",
            "latency",
            "capability",
            "interactions",
            "context",
            "output",
            "quality",
            "cache",
            "time_variation",
            "aimd",
            "soak",
        ),
        help="run exactly one configured suite while retaining the same route evidence",
    )
    run.add_argument("--max-wall-seconds", type=float)
    run.add_argument("--max-cost-usd", type=float)
    run_matrix_parser = sub.add_parser(
        "run-matrix", help="run providers in parallel; isolate endpoint capacity within provider"
    )
    run_matrix_parser.add_argument("matrix", type=Path)
    run_matrix_parser.add_argument("--output-root", type=Path, required=True)
    run_matrix_parser.add_argument("--confirm-live", action="store_true")
    report = sub.add_parser("report", help="build matched-cell tables, audit, plots, and Markdown")
    report.add_argument("run_dir", type=Path)
    report_matrix = sub.add_parser(
        "report-matrix", help="combine terminal provider runs into a readable PDF evidence atlas"
    )
    report_matrix.add_argument("matrix", type=Path)
    report_matrix.add_argument("--run-root", action="append", type=Path, required=True)
    report_matrix.add_argument("--output", type=Path, required=True)
    derive_soak = sub.add_parser(
        "derive-soak", help="build a two-minute soak config from observed AIMD bounds"
    )
    derive_soak.add_argument("source_config", type=Path)
    derive_soak.add_argument("controller_summary", type=Path)
    derive_soak.add_argument("--output", type=Path, required=True)
    derive_soak.add_argument("--fallback-rps", type=float)
    derive_soak.add_argument(
        "--route-profile-overrides",
        type=Path,
        help=(
            "apply a typed, identity-bound route profile while retaining the exact AIMD "
            "configuration as the rate-evidence source"
        ),
    )
    derive_soak.add_argument(
        "--censor-incomplete",
        action="store_true",
        help=(
            "omit cells without a contract-complete confirmed healthy AIMD bound and record "
            "their measured terminal disposition; never substitutes a fallback rate"
        ),
    )
    digitalocean = sub.add_parser(
        "report-digitalocean-summary",
        help="render a clean atlas from a sanitized DigitalOcean direct summary package",
    )
    digitalocean.add_argument("summary_dir", type=Path)
    digitalocean.add_argument("--output", type=Path, required=True)
    digitalocean.add_argument("--capacity-source", required=True)
    digitalocean.add_argument("--soak-source", required=True)
    digitalocean.add_argument(
        "--exclude-endpoint",
        action="append",
        default=[],
        help="omit one exact endpoint identifier from every atlas panel; repeat as needed",
    )
    closure = sub.add_parser(
        "plan-digitalocean-closure",
        help="compile the credential-free six-hour DigitalOcean gap-closure package",
    )
    closure.add_argument("base_config", type=Path)
    closure.add_argument("summary_dir", type=Path)
    closure.add_argument("--output", type=Path, required=True)
    closure.add_argument(
        "--capacity-source", default="do-combined-capacity-20260828"
    )
    closure.add_argument(
        "--fixed-rate-source", default="do-direct-soak-20260823-r1"
    )
    capacity_closure = sub.add_parser(
        "plan-capacity-closure",
        help="compile a provider-neutral capacity-closure plan from evidence and a profile",
    )
    capacity_closure.add_argument("base_config", type=Path)
    capacity_closure.add_argument("capacity_csv", type=Path)
    capacity_closure.add_argument("profile", type=Path)
    capacity_closure.add_argument("--output", type=Path, required=True)
    compile_profile = sub.add_parser(
        "compile-profile",
        help="compile a provider profile and experiment profile into canonical campaign YAML",
    )
    compile_profile.add_argument("provider_profile", type=Path)
    compile_profile.add_argument("experiment_profile", type=Path)
    compile_profile.add_argument("--output", type=Path, required=True)
    return parser


def _apply_live_overrides(config: CampaignConfig, args: argparse.Namespace) -> CampaignConfig:
    """Apply the shared plan/run scope and guard overrides exactly once."""

    if args.only_suite:
        if args.only_suite not in config.suites:
            raise ValueError(f"suite is not configured: {args.only_suite}")
        config = replace(config, suites={args.only_suite: config.suites[args.only_suite]})
    if args.max_wall_seconds is not None:
        if args.max_wall_seconds <= 0:
            raise ValueError("--max-wall-seconds must be positive")
        config = replace(config, max_wall_seconds=args.max_wall_seconds)
    if args.max_cost_usd is not None:
        if args.max_cost_usd <= 0:
            raise ValueError("--max-cost-usd must be positive")
        config = replace(config, max_cost_usd=args.max_cost_usd)
    return config


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        config = load_config(args.config)
        print(json.dumps(build_plan(config).to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "plan-matrix":
        print(json.dumps(matrix_plan(load_matrix(args.matrix)), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        if not args.confirm_live:
            print("refusing live traffic without --confirm-live", file=sys.stderr)
            return 2
        config = _apply_live_overrides(load_config(args.config), args)
        raw_argv = tuple(sys.argv if argv is None else ("inference-bench", *argv))
        asyncio.run(run_campaign(config, args.output, invocation=raw_argv))
        return 0
    if args.command == "run-matrix":
        if not args.confirm_live:
            print("refusing live traffic without --confirm-live", file=sys.stderr)
            return 2
        raw_argv = tuple(sys.argv if argv is None else ("inference-bench", *argv))

        async def matrix_runner(
            config: CampaignConfig, output: Path, invocation: tuple[str, ...]
        ) -> None:
            await run_campaign(config, output, invocation=invocation)

        asyncio.run(
            run_matrix(
                load_matrix(args.matrix),
                args.output_root,
                matrix_runner,
                invocation=raw_argv,
            )
        )
        return 0
    if args.command == "report":
        print(generate_report(args.run_dir))
        return 0
    if args.command == "report-matrix":
        print(generate_atlas(load_matrix(args.matrix), args.run_root, args.output))
        return 0
    if args.command == "derive-soak":
        print(
            derive_soak_config(
                args.source_config,
                args.controller_summary,
                args.output,
                fallback_rps=args.fallback_rps,
                route_profile_overrides=args.route_profile_overrides,
                censor_incomplete=args.censor_incomplete,
            )
        )
        return 0
    if args.command == "report-digitalocean-summary":
        print(
            generate_digitalocean_atlas(
                args.summary_dir,
                args.output,
                capacity_source=args.capacity_source,
                soak_source=args.soak_source,
                exclude_endpoints=tuple(args.exclude_endpoint),
            )
        )
        return 0
    if args.command == "plan-digitalocean-closure":
        config_path, manifest_path = build_digitalocean_closure_package(
            args.base_config,
            args.summary_dir,
            args.output,
            capacity_source=args.capacity_source,
            fixed_rate_source=args.fixed_rate_source,
        )
        print(json.dumps({"config": str(config_path), "plan": str(manifest_path)}))
        return 0
    if args.command == "plan-capacity-closure":
        config_path, manifest_path = build_capacity_closure_package_from_files(
            args.base_config,
            args.capacity_csv,
            args.profile,
            args.output,
        )
        print(json.dumps({"config": str(config_path), "plan": str(manifest_path)}))
        return 0
    if args.command == "compile-profile":
        compilation = compile_profile_files(
            args.provider_profile,
            args.experiment_profile,
            args.output,
        )
        print(
            json.dumps(
                {
                    "config": str(args.output),
                    "config_identity_sha256": compilation.config.identity_hash,
                    "compiled_sha256": compilation.compiled_sha256,
                    "experiment_profile_sha256": compilation.experiment_profile_sha256,
                    "provider_profile_sha256": compilation.provider_profile_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
