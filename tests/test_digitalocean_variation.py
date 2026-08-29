from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from inference_bench.digitalocean_variation import build_variation_tables, summarize_variation_run


def _run(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE attempts (
          logical_id TEXT, route_id TEXT, suite TEXT, cell_id TEXT, state TEXT,
          final_logical INTEGER, status TEXT, http_status INTEGER, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_input_tokens INTEGER, total_seconds REAL,
          ttft_seconds REAL, settled_usd REAL, latency_eligible INTEGER,
          usage_eligible INTEGER, decode_eligible INTEGER
        );
        CREATE TABLE events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at_utc TEXT,
          kind TEXT, payload_json TEXT
        );
        """
    )
    config = {
        "suites": {
            "time_variation": {
                "samples_per_route_shape": 4,
                "stable_exact_prompt_repeats": 2,
                "panel_unique_cache_cold_repeats": 2,
            }
        }
    }
    connection.execute("INSERT INTO meta VALUES ('config_json', ?)", (json.dumps(config),))
    for panel in range(2):
        connection.execute(
            "INSERT INTO events(recorded_at_utc,kind,payload_json) VALUES (?,?,?)",
            (
                f"2026-08-28T2{panel}:00:00Z",
                "time_variation_panel_started",
                json.dumps({"panel": panel}),
            ),
        )
        for repeat in range(4):
            stable = repeat < 2
            success = not (panel == 1 and repeat == 3)
            connection.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"time-variation:route-a:panel-{panel:03d}:short_short:{repeat:03d}",
                    "route-a",
                    "time_variation",
                    f"short_short:in256:out128:panel={panel:03d}",
                    "terminal",
                    1,
                    "success" if success else "timeout",
                    200 if success else None,
                    100,
                    20 if success else None,
                    10 if stable and success else 0 if success else None,
                    3.0 if stable else 5.0,
                    1.0 if success else None,
                    0.01,
                    int(success),
                    int(success),
                    int(success),
                ),
            )
    # A non-final retry must not increase any denominator.
    connection.execute(
        "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "time-variation:route-a:panel-000:short_short:000",
            "route-a",
            "time_variation",
            "short_short:in256:out128:panel=000",
            "terminal",
            0,
            "timeout",
            None,
            None,
            None,
            None,
            9.0,
            None,
            0.0,
            0,
            0,
            0,
        ),
    )
    # Preserve a mixed subtype as part of the grouping identity.
    for repeat in range(4):
        connection.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"time-variation:route-a:panel-000:mixed:{repeat:03d}",
                "route-a",
                "time_variation",
                "mixed:tool:in1024:out512:panel=000",
                "terminal",
                1,
                "success",
                200,
                50,
                10,
                0,
                2.0,
                0.5,
                0.02,
                1,
                1,
                1,
            ),
        )
    connection.commit()
    connection.close()
    return tmp_path


def test_summarizes_registered_cache_strata_without_identifiers(tmp_path: Path) -> None:
    summary = summarize_variation_run(_run(tmp_path), seed=7)
    panels = [row for row in summary["panel_summaries"] if row["shape"] == "short_short"]
    assert len(panels) == 4
    stable0 = next(
        row for row in panels if row["panel"] == 0 and row["cache_stratum"] == "stable_exact_prompt"
    )
    cold1 = next(
        row for row in panels if row["panel"] == 1 and row["cache_stratum"] == "panel_unique_cold"
    )
    assert stable0["attempted_n"] == stable0["success_n"] == 2
    assert stable0["success_rate_ci_method"] == "Wilson-95"
    assert stable0["eligible_output_rate_median"] == pytest.approx(10.0)
    assert stable0["cache_read_tokens_sum"] == 20
    assert cold1["attempted_n"] == 2 and cold1["success_n"] == 1
    assert cold1["success_rate"] == pytest.approx(0.5)
    assert stable0["panel_started_at_utc"] == "2026-08-28T20:00:00Z"
    assert cold1["elapsed_hours"] == pytest.approx(1.0)
    assert len(summary["across_panel_summaries"]) == 4
    serialized = json.dumps(summary)
    assert "logical_id" not in serialized and "request_id" not in serialized


def test_pairs_stable_and_cold_by_panel_and_preserves_mixed_subtype(tmp_path: Path) -> None:
    summary = summarize_variation_run(_run(tmp_path), seed=11)
    short = next(row for row in summary["stable_vs_cold"] if row["shape"] == "short_short")
    assert short["paired_panels_n"] == 2
    assert short["difference_direction"] == "panel_unique_cold minus stable_exact_prompt"
    assert short["paired_request_latency_median_difference_median"] == pytest.approx(2.0)
    mixed = [row for row in summary["panel_summaries"] if row["shape"] == "mixed"]
    assert {row["mixed_subtype"] for row in mixed} == {"tool:in1024:out512"}


def test_rejects_unregistered_repeat_contract(tmp_path: Path) -> None:
    root = _run(tmp_path)
    connection = sqlite3.connect(root / "ledger.sqlite3")
    bad = {"suites": {"time_variation": {"samples_per_route_shape": 3}}}
    connection.execute("UPDATE meta SET value=? WHERE key='config_json'", (json.dumps(bad),))
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="exactly four"):
        summarize_variation_run(root)


def test_build_variation_tables_writes_identifier_free_outputs(tmp_path: Path) -> None:
    run = _run(tmp_path / "run")
    paths = build_variation_tables(run, tmp_path / "tables", seed=3)
    assert set(paths) == {
        "summary_json",
        "panel_csv",
        "across_panel_csv",
        "paired_cache_csv",
        "provenance_json",
    }
    assert all(path.is_file() for path in paths.values())
    combined = "".join(path.read_text(encoding="utf-8") for path in paths.values())
    assert "logical_id" not in combined and "request_id" not in combined
    assert "panel_started_at_utc" in paths["panel_csv"].read_text(encoding="utf-8")
