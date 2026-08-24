from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from .config import CampaignConfig, load_config
from .engine import BenchmarkEngine
from .ledger import BudgetExceeded, Ledger, TimeLimitReached
from .load import run_aimd, run_soak
from .models import RequestSpec, canonical_json
from .plan import build_plan
from .report import generate_report
from .workloads import plan_static_suites


async def _run_static(
    engine: BenchmarkEngine, specs: list[RequestSpec], config: CampaignConfig
) -> bool:
    rng = random.Random(config.seed)
    rng.shuffle(specs)
    static_rps = float(config.suites.get("static", {}).get("offered_rps", 1.0))
    semaphore = asyncio.Semaphore(config.concurrency)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    started = loop.time()

    async def one(index: int, spec: RequestSpec) -> None:
        await asyncio.sleep(max(0.0, started + index / static_rps - loop.time()))
        if stop.is_set():
            return
        arrived = loop.time()
        async with semaphore:
            if stop.is_set():
                return
            try:
                await engine.execute(spec, queue_delay_seconds=loop.time() - arrived)
            except (BudgetExceeded, TimeLimitReached):
                stop.set()

    tasks = [asyncio.create_task(one(index, spec)) for index, spec in enumerate(specs)]
    if tasks:
        await asyncio.gather(*tasks)
    return stop.is_set()


async def run_campaign(config: CampaignConfig, output: Path) -> None:
    placeholders = build_plan(config).native_placeholder_routes
    if placeholders:
        raise ValueError(
            "live run contains fail-closed native adapter placeholders: " + ", ".join(placeholders)
        )
    output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output)
    ledger.initialize(
        campaign_hash=config.identity_hash, config_json=canonical_json(config.public_dict())
    )
    recovered = ledger.recover_in_flight()
    if recovered:
        ledger.record_event("resume_notice", {"unknown_in_flight_count": recovered})
    (output / "campaign.public.json").write_text(
        json.dumps(config.public_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    engine = BenchmarkEngine(config, ledger)
    try:
        static_specs = plan_static_suites(config.routes, config.suites, seed=config.seed)
        if await _run_static(engine, static_specs, config):
            ledger.record_event("campaign_terminal", {"reason": "launch_guard"})
            return
        shapes = ["short_short", "long_short", "short_long", "mixed"]
        aimd = config.suites.get("aimd")
        if aimd and aimd.get("enabled", True):
            for route in config.routes:  # endpoint-isolated capacity sweeps
                for shape in aimd.get("shapes", shapes):
                    epochs = await run_aimd(engine, route, shape, aimd, seed=config.seed)
                    if any(epoch.launch_guard_triggered for epoch in epochs):
                        ledger.record_event("campaign_terminal", {"reason": "launch_guard"})
                        return
        soak = config.suites.get("soak")
        if soak and soak.get("enabled", True):
            for route in config.routes:  # endpoint-isolated sustained workloads
                for shape in soak.get("shapes", shapes):
                    blocks = await run_soak(engine, route, shape, soak, seed=config.seed)
                    if any(block.launch_guard_triggered for block in blocks):
                        ledger.record_event("campaign_terminal", {"reason": "launch_guard"})
                        return
        ledger.record_event("campaign_terminal", {"reason": "plan_completed"})
    except (BudgetExceeded, TimeLimitReached) as exc:
        ledger.record_event(
            "campaign_terminal", {"reason": type(exc).__name__, "message": str(exc)}
        )
    finally:
        await engine.close()
        ledger.rebuild_events_jsonl()
        ledger.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="credential-free plan and conservative cost calculation")
    plan.add_argument("config", type=Path)
    run = sub.add_parser("run", help="execute a live campaign")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--confirm-live", action="store_true")
    report = sub.add_parser("report", help="build matched-cell tables, audit, plots, and Markdown")
    report.add_argument("run_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        config = load_config(args.config)
        print(json.dumps(build_plan(config).to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        if not args.confirm_live:
            print("refusing live traffic without --confirm-live", file=sys.stderr)
            return 2
        config = load_config(args.config)
        asyncio.run(run_campaign(config, args.output))
        return 0
    if args.command == "report":
        print(generate_report(args.run_dir))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
