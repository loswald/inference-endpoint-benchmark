from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import CampaignConfig, load_config


@dataclass(frozen=True, slots=True)
class MatrixCampaign:
    name: str
    provider: str
    config_path: Path
    output_name: str
    config: CampaignConfig


@dataclass(frozen=True, slots=True)
class CampaignMatrix:
    path: Path
    max_parallel_providers: int
    campaigns: tuple[MatrixCampaign, ...]


def load_matrix(path: Path) -> CampaignMatrix:
    """Load a small provider-level orchestration file.

    A provider can appear only once. This makes provider-level parallelism explicit while
    preserving endpoint isolation inside each campaign's AIMD and soak phases.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {"version", "max_parallel_providers", "campaigns"}:
        raise ValueError("matrix must contain only version, max_parallel_providers, campaigns")
    if raw.get("version") != 1:
        raise ValueError("matrix.version must be 1")
    parallel = raw.get("max_parallel_providers", 4)
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel <= 0:
        raise ValueError("max_parallel_providers must be a positive integer")
    rows = raw.get("campaigns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("matrix.campaigns must be a nonempty list")
    base = path.resolve().parent
    campaigns: list[MatrixCampaign] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "provider", "config", "output"}:
            raise ValueError(
                f"matrix.campaigns[{index}] requires exactly name, provider, config, output"
            )
        values = {key: row[key] for key in ("name", "provider", "config", "output")}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError(f"matrix.campaigns[{index}] fields must be nonempty strings")
        config_path = (base / values["config"]).resolve()
        config = load_config(config_path)
        route_providers = {route.provider for route in config.routes}
        if route_providers != {values["provider"]}:
            raise ValueError(
                f"matrix campaign {values['name']} must contain only provider "
                f"{values['provider']!r}; observed {sorted(route_providers)!r}"
            )
        output_name = values["output"]
        if Path(output_name).is_absolute() or ".." in Path(output_name).parts:
            raise ValueError("matrix output names must be safe relative paths")
        campaigns.append(
            MatrixCampaign(
                name=values["name"],
                provider=values["provider"],
                config_path=config_path,
                output_name=output_name,
                config=config,
            )
        )
    providers = [campaign.provider for campaign in campaigns]
    names = [campaign.name for campaign in campaigns]
    outputs = [campaign.output_name for campaign in campaigns]
    if len(set(providers)) != len(providers):
        raise ValueError("each provider may appear only once in a matrix")
    if len(set(names)) != len(names) or len(set(outputs)) != len(outputs):
        raise ValueError("matrix campaign names and output names must be unique")
    return CampaignMatrix(path.resolve(), parallel, tuple(campaigns))


def matrix_plan(matrix: CampaignMatrix) -> dict[str, Any]:
    from .plan import build_plan

    campaigns = []
    for item in matrix.campaigns:
        plan = build_plan(item.config).to_dict()
        campaigns.append(
            {
                "name": item.name,
                "provider": item.provider,
                "config": item.config_path.name,
                "output": item.output_name,
                "campaign_identity": item.config.identity_hash,
                "plan": plan,
            }
        )
    return {
        "version": 1,
        "parallelism": {
            "unit": "provider",
            "maximum": matrix.max_parallel_providers,
            "endpoint_capacity_within_provider": "isolated",
        },
        "campaigns": campaigns,
    }


async def run_matrix(
    matrix: CampaignMatrix,
    output_root: Path,
    runner: Callable[[CampaignConfig, Path, tuple[str, ...]], Awaitable[None]],
    *,
    invocation: tuple[str, ...],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(matrix.max_parallel_providers)

    async def one(item: MatrixCampaign) -> None:
        async with semaphore:
            await runner(item.config, output_root / item.output_name, invocation)

    outcomes = await asyncio.gather(
        *(one(item) for item in matrix.campaigns), return_exceptions=True
    )
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if failures:
        raise ExceptionGroup("one or more provider campaigns failed", failures)
