from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import sqlite3
import subprocess
import sys
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .atlas import generate_atlas
from .config import CampaignConfig, load_config
from .digitalocean_atlas import generate_digitalocean_atlas
from .engine import BenchmarkEngine, PaymentRequiredLatched, ReservationOverrunLatched
from .environment import locked_distribution_versions, validate_run_directory_separation
from .ledger import BudgetExceeded, Ledger, TimeLimitReached
from .load import run_aimd, run_soak
from .matrix import load_matrix, matrix_plan, run_matrix
from .models import TRANSPORT_HEADER_PROFILE, RequestSpec, RouteConfig, canonical_json
from .plan import build_plan
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
    cells = [
        (route, shape)
        for route in config.routes
        for shape in suite.get("shapes", _DEFAULT_CAPACITY_SHAPES)
    ]
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


async def _run_static(
    engine: BenchmarkEngine, specs: list[RequestSpec], config: CampaignConfig
) -> str | None:
    """Run one endpoint × suite block serially with no coordinated-omission claim.

    Static cells are low-load measurements and validation probes, not capacity tests. Starting the
    next cell only after the prior one drains prevents long context/output probes from queueing
    behind each other or contaminating another endpoint's latency baseline.
    """

    static_rps = float(config.suites.get("static", {}).get("offered_rps", 1.0))
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
    engine: BenchmarkEngine, specs: list[RequestSpec], config: CampaignConfig
) -> str | None:
    """Execute fixed-offset, low-load panels without overlapping other benchmark traffic."""

    suite = config.suites["time_variation"]
    offered_rps = float(suite.get("offered_rps", 0.2))
    by_panel: dict[int, list[RequestSpec]] = {}
    for spec in specs:
        panel = int(spec.metadata["time_variation_panel"])
        by_panel.setdefault(panel, []).append(spec)
    loop = asyncio.get_running_loop()
    campaign_anchor = loop.time()
    for panel in sorted(by_panel):
        panel_specs = by_panel[panel]
        offset = float(panel_specs[0].metadata["time_variation_offset_seconds"])
        await asyncio.sleep(max(0.0, campaign_anchor + offset - loop.time()))
        random.Random(f"time-variation-panel/v1:{config.seed}:{panel}").shuffle(panel_specs)
        engine.ledger.record_event_once(
            f"time_variation_panel_started:{panel}",
            "time_variation_panel_started",
            {"panel": panel, "planned_offset_seconds": offset, "requests": len(panel_specs)},
        )
        pending = _pending_static_specs(engine, panel_specs)
        for spec in pending:
            scheduled_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            try:
                result = await engine.execute(
                    spec, scheduled_at_utc=scheduled_at, queue_delay_seconds=0.0
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
            await asyncio.sleep(1.0 / offered_rps)
        engine.ledger.record_event_once(
            f"time_variation_panel_completed:{panel}",
            "time_variation_panel_completed",
            {"panel": panel, "planned_offset_seconds": offset},
        )
    return None


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
            time_variation_reason = await _run_time_variation(engine, time_variation_specs, config)
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

    def git(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="surrogateescape",
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    pathspec = ["--", "."]
    if output_dir is not None:
        try:
            output_relative = output_dir.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            output_relative = None
        if output_relative and output_relative != ".":
            pathspec.append(f":(exclude){output_relative}/**")
    status = git("status", "--porcelain=v1", "--untracked-files=all", *pathspec)
    diff = git("diff", "--binary", "HEAD", *pathspec)
    untracked = git("ls-files", "--others", "--exclude-standard", *pathspec)
    lock = root / "requirements.lock"
    source_commit = git("rev-parse", "HEAD")
    if source_commit is None or status is None or diff is None or untracked is None:
        raise RuntimeError(
            "live runs require an accessible git source identity and dirty-tree state"
        )
    if not lock.is_file():
        raise RuntimeError("live runs require the repository requirements.lock")
    if status:
        raise RuntimeError("live runs require clean committed source")
    tracked = git("ls-files")
    if tracked is None:
        raise RuntimeError("live runs require a complete tracked-source inventory")
    if output_dir is not None:
        validate_run_directory_separation(root, output_dir, tracked.splitlines())
    dirty_tree_sha256 = _dirty_tree_hash(root, status, diff, untracked)
    if dirty_tree_sha256 is None:
        raise RuntimeError("live runs require a hash-bound source tree state")
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
        "source_commit": source_commit,
        "source_dirty": bool(status),
        "source_dirty_tree_sha256": dirty_tree_sha256,
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
    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if not (candidate / "pyproject.toml").is_file():
            continue
        try:
            root = Path(
                subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=candidate,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ).resolve()
        except (OSError, subprocess.CalledProcessError):
            continue
        if root == candidate.resolve():
            return root
    raise RuntimeError("cannot resolve the benchmark source git root")


def _dirty_tree_hash(
    root: Path,
    status: str | None,
    diff: str | None,
    untracked: str | None,
) -> str | None:
    """Bind tracked changes and untracked bytes without publishing local paths."""

    if status is None or diff is None or untracked is None:
        return None
    untracked_digests: list[dict[str, str]] = []
    for relative in sorted(line for line in untracked.splitlines() if line):
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except (OSError, ValueError):
            digest = "unreadable"
        untracked_digests.append(
            {
                "path_sha256": hashlib.sha256(
                    relative.encode("utf-8", errors="surrogateescape")
                ).hexdigest(),
                "content_sha256": digest,
            }
        )
    material = {
        "status_sha256": hashlib.sha256(
            status.encode("utf-8", errors="surrogateescape")
        ).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(
            diff.encode("utf-8", errors="surrogateescape")
        ).hexdigest(),
        "untracked": untracked_digests,
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
