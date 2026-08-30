"""Standalone CLI for the embedding benchmark lane.

Kept separate from the chat-generation CLI so embedding-only users do not inherit generation
flags or accidentally run an embedding route through a chat suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from .embedding_benchmark import (
    load_embedding_config,
    run_embedding_campaign,
    write_embedding_plan,
    write_embedding_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inference_bench.embedding_cli")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="compile a credential-free embedding request plan")
    plan.add_argument("profile", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    canary_plan = commands.add_parser(
        "plan-canary", help="compile an exact one-request route-admission plan"
    )
    canary_plan.add_argument("profile", type=Path)
    canary_plan.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="run or safely resume an embedding benchmark")
    run.add_argument("profile", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    canary = commands.add_parser(
        "canary", help="run or safely resume one exact route-admission request"
    )
    canary.add_argument("profile", type=Path)
    canary.add_argument("--output-dir", type=Path, required=True)
    report = commands.add_parser("report", help="rebuild reports from plan and receipt journal")
    report.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"plan", "plan-canary"}:
        config = load_embedding_config(args.profile)
        plan_kind = "canary" if args.command == "plan-canary" else "benchmark"
        plan = write_embedding_plan(config, args.output, plan_kind=plan_kind)
        print(
            json.dumps(
                {
                    "plan": str(args.output),
                    "campaign_identity_sha256": plan.campaign_identity_sha256,
                    "route_identity_sha256": plan.route_identity_sha256,
                    "plan_kind": plan.plan_kind,
                    "request_count": len(plan.requests),
                    "worst_case_cost_usd": plan.worst_case_cost_usd,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"run", "canary"}:
        plan_kind = "canary" if args.command == "canary" else "benchmark"
        report = asyncio.run(
            run_embedding_campaign(
                load_embedding_config(args.profile),
                args.output_dir,
                plan_kind=plan_kind,
            )
        )
    else:
        report = write_embedding_report(args.output_dir)
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "embedding-report.json"),
                "planned_cells": report["planned_cells"],
                "state_counts": report["state_counts"],
                "settled_cost_usd": report["settled_cost_usd"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
