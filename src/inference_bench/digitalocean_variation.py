from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .statistics import (
    Estimate,
    block_median_interval,
    median_interval,
    quantile_interval,
    wilson_interval,
)

_LOGICAL_RE = re.compile(
    r"^time-variation:(?P<route>.+):panel-(?P<panel>\d{3}):"
    r"(?P<shape>short_short|long_short|short_long|mixed):"
    r"(?:(?P<stratum>stable_prefix|panel_unique_cold):)?(?P<repeat>\d{3})$"
)
_PANEL_RE = re.compile(r"(?:^|:)panel=(?P<panel>\d{3})$")
_SAFE_COLUMNS = (
    "logical_id",
    "route_id",
    "cell_id",
    "cache_state",
    "state",
    "status",
    "http_status",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "total_seconds",
    "ttft_seconds",
    "settled_usd",
    "latency_eligible",
    "usage_eligible",
    "decode_eligible",
)


def summarize_variation_run(run_dir: str | Path, *, seed: int = 1) -> dict[str, Any]:
    """Summarize the registered six-hour DigitalOcean variation panels.

    The returned structure deliberately contains no request IDs, logical IDs, response content,
    headers, or bodies. Sampling units for panel summaries are final logical outcomes. Repeats 0/1
    are the registered stable exact-prompt stratum; repeats 2/3 are panel-unique cold prompts.
    """

    root = Path(run_dir)
    config = _load_config(root / "ledger.sqlite3")
    variation = config.get("suites", {}).get("time_variation", {})
    if int(variation.get("samples_per_route_shape", 0)) != 4:
        raise ValueError("variation summary requires exactly four registered repeats")
    if int(variation.get("stable_exact_prompt_repeats", 0)) != 2:
        raise ValueError("variation summary requires registered stable repeats 0 and 1")
    if int(variation.get("panel_unique_cache_cold_repeats", 0)) != 2:
        raise ValueError("variation summary requires registered cold repeats 2 and 3")

    database = root / "ledger.sqlite3"
    rows = _load_rows(database)
    panel_times = _load_panel_times(database)
    parsed = [_parse_row(row) for row in rows]
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        key = (
            row["route_id"],
            row["shape"],
            row["mixed_subtype"],
            row["panel"],
            row["cache_stratum"],
        )
        grouped[key].append(row)

    panels = [
        _summarize_group(key, items, seed=_stable_seed(seed, key))
        for key, items in sorted(grouped.items())
    ]
    for row in panels:
        timing = panel_times.get(int(row["panel"]))
        if timing is None:
            raise ValueError(f"missing start event for variation panel {row['panel']}")
        row.update(timing)
    return {
        "schema_version": "digitalocean-variation-summary/v1",
        "estimand": "final logical outcomes within each registered route/shape/panel/cache stratum",
        "cache_strata": {
            "stable_exact_prompt": (
                "registered repeats 0 and 1; identical prompt bytes across panels"
            ),
            "panel_unique_cold": "registered repeats 2 and 3; panel-unique prompt bytes",
        },
        "panel_summaries": panels,
        "across_panel_summaries": _across_panel_summaries(panels, seed=seed),
        "stable_vs_cold": _paired_summaries(panels, seed=seed),
    }


def build_variation_tables(
    run_dir: str | Path, output_dir: str | Path, *, seed: int = 1
) -> dict[str, Path]:
    """Write identifier-free JSON and CSV tables for the six-hour variation study."""

    summary = summarize_variation_run(run_dir, seed=seed)
    root = Path(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": destination / "variation-summary.json",
        "panel_csv": destination / "variation-panel-summary.csv",
        "across_panel_csv": destination / "variation-across-panel-summary.csv",
        "paired_cache_csv": destination / "variation-paired-cache-effects.csv",
        "provenance_json": destination / "variation-provenance-manifest.json",
    }
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(paths["panel_csv"], summary["panel_summaries"])
    _write_csv(paths["across_panel_csv"], summary["across_panel_summaries"])
    _write_csv(paths["paired_cache_csv"], summary["stable_vs_cold"])
    panels = summary["panel_summaries"]
    primary_cells = {
        (row["route_id"], row["shape"], row["panel"], row["cache_stratum"]) for row in panels
    }
    source_files = {
        name: _sha256(root / name)
        for name in ("campaign.public.json", "ledger.sqlite3", "events.jsonl", "SHA256SUMS")
        if (root / name).is_file()
    }
    provenance = {
        "schema_version": "digitalocean-variation-provenance/v1",
        "source_run_id": "do-six-hour-variation-20260828-r1",
        "source_file_sha256": source_files,
        "cache_stratum_reconstruction": (
            "validated hash-bound four-repeat plan plus registered logical repeat suffix; "
            "repeats 0-1 stable exact prompt, repeats 2-3 panel-unique cold"
        ),
        "panel_summary_rows": len(panels),
        "route_shape_panel_stratum_cells": len(primary_cells),
        "across_panel_rows": len(summary["across_panel_summaries"]),
        "paired_cache_rows": len(summary["stable_vs_cold"]),
        "requests_attempted": sum(int(row["attempted_n"]) for row in panels),
        "requests_successful": sum(int(row["success_n"]) for row in panels),
        "panels": sorted({int(row["panel"]) for row in panels}),
        "routes": sorted({str(row["route_id"]) for row in panels}),
        "output_sha256": {
            key: _sha256(path) for key, path in paths.items() if key != "provenance_json"
        },
    }
    paths["provenance_json"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _load_config(database: Path) -> dict[str, Any]:
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT value FROM meta WHERE key='config_json'").fetchone()
    if row is None:
        raise ValueError("ledger is missing config_json")
    value = json.loads(row[0])
    if not isinstance(value, dict):
        raise ValueError("config_json must be an object")
    return value


def _load_rows(database: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        available = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(attempts)").fetchall()
        }
        columns = ", ".join(
            column if column in available else f"'uncontrolled' AS {column}"
            for column in _SAFE_COLUMNS
        )
        rows = connection.execute(
            f"SELECT {columns} FROM attempts "
            "WHERE suite='time_variation' AND final_logical=1 "
            "AND state IN ('terminal','unknown')"
        ).fetchall()
    return [dict(row) for row in rows]


def _load_panel_times(database: Path) -> dict[int, dict[str, Any]]:
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT recorded_at_utc, payload_json FROM events "
            "WHERE kind='time_variation_panel_started' ORDER BY event_id"
        ).fetchall()
    starts: dict[int, datetime] = {}
    rendered: dict[int, str] = {}
    for recorded_at, payload_json in rows:
        payload = json.loads(payload_json)
        panel = payload.get("panel")
        if isinstance(panel, bool) or not isinstance(panel, int) or panel < 0:
            raise ValueError("invalid variation panel start event")
        if panel in starts:
            raise ValueError(f"duplicate start event for variation panel {panel}")
        timestamp = datetime.fromisoformat(str(recorded_at).replace("Z", "+00:00"))
        starts[panel] = timestamp
        rendered[panel] = str(recorded_at)
    if not starts:
        raise ValueError("ledger has no variation panel start events")
    origin = starts[min(starts)]
    return {
        panel: {
            "panel_started_at_utc": rendered[panel],
            "elapsed_hours": (timestamp - origin).total_seconds() / 3600.0,
        }
        for panel, timestamp in starts.items()
    }


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    match = _LOGICAL_RE.fullmatch(str(row["logical_id"]))
    if match is None:
        raise ValueError("unrecognized time-variation logical identity")
    if match["route"] != row["route_id"]:
        raise ValueError("route identity disagrees with registered logical identity")
    panel = int(match["panel"])
    cell_panel = _PANEL_RE.search(str(row["cell_id"]))
    if cell_panel is None or int(cell_panel["panel"]) != panel:
        raise ValueError("cell panel disagrees with registered logical identity")
    repeat = int(match["repeat"])
    if repeat not in {0, 1, 2, 3}:
        raise ValueError("unregistered time-variation repeat")
    shape = match["shape"]
    cell_prefix = str(row["cell_id"]).rsplit(":panel=", 1)[0]
    cell_prefix = cell_prefix.split(":variation_stratum=", 1)[0]
    mixed_subtype = cell_prefix.removeprefix("mixed:") if shape == "mixed" else "not_applicable"
    stratum = match["stratum"]
    if stratum is None:
        stratum = "stable_prefix" if repeat < 2 else "panel_unique_cold"
    persisted = str(row.get("cache_state") or "uncontrolled")
    if persisted != "uncontrolled" and persisted != stratum:
        raise ValueError("persisted variation stratum disagrees with logical identity")
    return {
        **{key: value for key, value in row.items() if key != "logical_id"},
        "panel": panel,
        "shape": shape,
        "mixed_subtype": mixed_subtype,
        "cache_stratum": (
            "stable_exact_prompt" if stratum == "stable_prefix" else "panel_unique_cold"
        ),
    }


def _summarize_group(
    key: tuple[str, str, str, int, str], rows: Sequence[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    route, shape, mixed_subtype, panel, stratum = key
    statuses = Counter(str(row["status"] or row["state"]) for row in rows)
    successes = [row for row in rows if row["state"] == "terminal" and row["status"] == "success"]
    latency = [
        value
        for row in successes
        if bool(row["latency_eligible"])
        and (value := _finite_nonnegative(row["total_seconds"])) is not None
    ]
    ttft = [
        value
        for row in successes
        if bool(row["latency_eligible"])
        and (value := _finite_nonnegative(row["ttft_seconds"])) is not None
    ]
    output_rates = [rate for row in successes if (rate := _output_rate(row)) is not None]
    usage = [row for row in successes if bool(row["usage_eligible"])]
    cache_values = [
        int(row["cache_read_input_tokens"])
        for row in usage
        if _valid_count(row["cache_read_input_tokens"])
    ]
    costs = [
        float(row["settled_usd"])
        for row in rows
        if _finite_nonnegative(row["settled_usd"]) is not None
    ]
    success_ci = wilson_interval(len(successes), len(rows))
    latency_median = median_interval(latency, unit="seconds", seed=seed)
    latency_p95 = quantile_interval(latency, 0.95, unit="seconds", seed=seed + 1)
    ttft_median = median_interval(ttft, unit="seconds", seed=seed + 2)
    ttft_p95 = quantile_interval(ttft, 0.95, unit="seconds", seed=seed + 3)
    output_median = median_interval(output_rates, unit="tokens/second", seed=seed + 4)
    return {
        "route_id": route,
        "shape": shape,
        "mixed_subtype": mixed_subtype,
        "panel": panel,
        "cache_stratum": stratum,
        "attempted_n": len(rows),
        "success_n": len(successes),
        "status_counts": dict(sorted(statuses.items())),
        **_estimate("success_rate", success_ci),
        **_estimate("request_latency_median", latency_median),
        **_estimate("request_latency_p95", latency_p95),
        **_estimate("ttft_median", ttft_median),
        **_estimate("ttft_p95", ttft_p95),
        **_estimate("eligible_output_rate_median", output_median),
        "eligible_output_rate_n": len(output_rates),
        "output_rate_eligibility_rule": (
            "ledger decode_eligible plus >=16 realized output tokens; completion tokens divided "
            "by request latency minus TTFT; also requires >=2 content events, >=1 second "
            "post-TTFT, known zero reasoning tokens, valid complete usage, and <=10000 "
            "tokens/second"
        ),
        "usage_eligible_n": len(usage),
        "prompt_tokens_sum": _sum_counts(usage, "input_tokens"),
        "completion_tokens_sum": _sum_counts(usage, "output_tokens"),
        "cache_read_tokens_reported_n": len(cache_values),
        "cache_read_tokens_sum": sum(cache_values),
        "settled_cost_reported_n": len(costs),
        "settled_cost_usd_sum": sum(costs),
    }


def _paired_summaries(panels: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in panels:
        key = (row["route_id"], row["shape"], row["mixed_subtype"], row["panel"])
        by_cell[key][row["cache_stratum"]] = row

    metrics = (
        "success_rate",
        "request_latency_median",
        "ttft_median",
        "eligible_output_rate_median",
        "prompt_tokens_sum",
        "completion_tokens_sum",
        "cache_read_tokens_sum",
        "settled_cost_usd_sum",
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (route, shape, subtype, panel), strata in sorted(by_cell.items()):
        if set(strata) != {"stable_exact_prompt", "panel_unique_cold"}:
            continue
        stable, cold = strata["stable_exact_prompt"], strata["panel_unique_cold"]
        differences = {
            metric: _difference(cold.get(metric), stable.get(metric)) for metric in metrics
        }
        grouped[(route, shape, subtype)].append({"panel": panel, **differences})

    result: list[dict[str, Any]] = []
    for key, pairs in sorted(grouped.items()):
        route, shape, subtype = key
        row: dict[str, Any] = {
            "route_id": route,
            "shape": shape,
            "mixed_subtype": subtype,
            "paired_panels_n": len(pairs),
            "difference_direction": "panel_unique_cold minus stable_exact_prompt",
            "panel_differences": pairs,
        }
        for index, metric in enumerate(metrics):
            values = [float(pair[metric]) for pair in pairs if pair[metric] is not None]
            unit = "proportion" if metric == "success_rate" else _metric_unit(metric)
            estimate = block_median_interval(
                values, unit=unit, seed=_stable_seed(seed + index, key + (metric,))
            )
            row.update(_estimate(f"paired_{metric}_difference_median", estimate))
        result.append(row)
    return result


def _across_panel_summaries(panels: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in panels:
        grouped[(row["route_id"], row["shape"], row["mixed_subtype"], row["cache_stratum"])].append(
            row
        )
    metric_units = {
        "success_rate": "proportion",
        "request_latency_median": "seconds",
        "request_latency_p95": "seconds",
        "ttft_median": "seconds",
        "ttft_p95": "seconds",
        "eligible_output_rate_median": "tokens/second",
    }
    result: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        route, shape, subtype, stratum = key
        row: dict[str, Any] = {
            "route_id": route,
            "shape": shape,
            "mixed_subtype": subtype,
            "cache_stratum": stratum,
            "panels_n": len(rows),
            "attempted_n": sum(int(item["attempted_n"]) for item in rows),
            "success_n": sum(int(item["success_n"]) for item in rows),
            "first_panel_started_at_utc": min(item["panel_started_at_utc"] for item in rows),
            "last_panel_started_at_utc": max(item["panel_started_at_utc"] for item in rows),
            "elapsed_hours_max": max(float(item["elapsed_hours"]) for item in rows),
            "aggregation_rule": "median across registered panels; panels are sampling units",
        }
        for index, (metric, unit) in enumerate(metric_units.items()):
            values = [
                float(item[metric])
                for item in rows
                if _finite_nonnegative(item.get(metric)) is not None
            ]
            estimate = block_median_interval(
                values, unit=unit, seed=_stable_seed(seed + index, key + (metric,))
            )
            row.update(_estimate(f"across_panel_{metric}_median", estimate))
        result.append(row)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _output_rate(row: Mapping[str, Any]) -> float | None:
    if not bool(row["decode_eligible"]):
        return None
    output = row["output_tokens"]
    total = _finite_nonnegative(row["total_seconds"])
    ttft = _finite_nonnegative(row["ttft_seconds"])
    if (
        not _valid_count(output)
        or int(output) < 16
        or total is None
        or ttft is None
        or total - ttft <= 0
    ):
        return None
    return int(output) / (total - ttft)


def _estimate(prefix: str, estimate: Estimate) -> dict[str, Any]:
    return {
        prefix: estimate.estimate,
        f"{prefix}_ci95_low": estimate.lower_95,
        f"{prefix}_ci95_high": estimate.upper_95,
        f"{prefix}_n": estimate.n,
        f"{prefix}_unit": estimate.unit,
        f"{prefix}_ci_method": estimate.method,
    }


def _sum_counts(rows: Iterable[Mapping[str, Any]], field: str) -> int:
    return sum(int(row[field]) for row in rows if _valid_count(row[field]))


def _valid_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _difference(left: Any, right: Any) -> float | None:
    left_value, right_value = _finite_nonnegative(left), _finite_nonnegative(right)
    return None if left_value is None or right_value is None else left_value - right_value


def _metric_unit(metric: str) -> str:
    if "latency" in metric or "ttft" in metric:
        return "seconds"
    if "output_rate" in metric:
        return "tokens/second"
    if "cost" in metric:
        return "USD"
    return "tokens"


def _stable_seed(seed: int, key: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(json.dumps([seed, *key], separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
