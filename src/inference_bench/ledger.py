from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import InferenceResult, RequestSpec, RouteConfig, ValidityAssessment, canonical_json


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class BudgetExceeded(RuntimeError):
    pass


class TimeLimitReached(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Exposure:
    settled_usd: float
    reserved_usd: float

    @property
    def total_usd(self) -> float:
        return self.settled_usd + self.reserved_usd


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
  request_id TEXT PRIMARY KEY,
  logical_id TEXT NOT NULL,
  attempt_index INTEGER NOT NULL,
  route_id TEXT NOT NULL,
  route_identity_hash TEXT NOT NULL,
  suite TEXT NOT NULL,
  cell_id TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('in_flight','terminal','unknown')),
  reserved_usd REAL NOT NULL CHECK(reserved_usd >= 0),
  settled_usd REAL NOT NULL CHECK(settled_usd >= 0),
  scheduled_at_utc TEXT,
  started_at_utc TEXT NOT NULL,
  ended_at_utc TEXT,
  status TEXT,
  http_status INTEGER,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cache_read_input_tokens INTEGER,
  cache_state TEXT NOT NULL DEFAULT 'uncontrolled',
  total_seconds REAL,
  time_to_headers_seconds REAL,
  ttft_seconds REAL,
  decode_seconds REAL,
  content_event_count INTEGER NOT NULL DEFAULT 0,
  output_event_offsets_json TEXT NOT NULL DEFAULT '[]',
  decode_metric_source TEXT NOT NULL DEFAULT 'request_minus_ttft',
  queue_delay_seconds REAL,
  finish_reason TEXT,
  error_kind TEXT,
  error_body_sha256 TEXT,
  output_sha256 TEXT,
  retained_headers_json TEXT NOT NULL DEFAULT '{}',
  validity_class TEXT,
  validity_reasons_json TEXT NOT NULL DEFAULT '[]',
  latency_eligible INTEGER NOT NULL DEFAULT 0,
  usage_eligible INTEGER NOT NULL DEFAULT 0,
  decode_eligible INTEGER NOT NULL DEFAULT 0,
  quality_eligible INTEGER NOT NULL DEFAULT 0,
  quality_score REAL,
  quality_diagnostics_json TEXT NOT NULL DEFAULT '{}',
  cost_basis TEXT NOT NULL DEFAULT 'unpriced',
  UNIQUE(logical_id, attempt_index)
);
CREATE INDEX IF NOT EXISTS attempts_cell ON attempts(route_id, suite, cell_id);
CREATE INDEX IF NOT EXISTS attempts_state ON attempts(state);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  recorded_at_utc TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
"""


class Ledger:
    """Crash-safe request ledger.

    SQLite is authoritative. JSONL is a secret-free event projection for inspection.
    An existing in-flight ID is never claimed again.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "ledger.sqlite3"
        self.events_path = self.directory / "events.jsonl"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def initialize(self, *, campaign_hash: str, config_json: str) -> None:
        with self._transaction() as db:
            existing = db.execute("SELECT value FROM meta WHERE key='campaign_hash'").fetchone()
            if existing and existing[0] != campaign_hash:
                raise ValueError("run directory belongs to a different campaign")
            db.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('campaign_hash',?)",
                (campaign_hash,),
            )
            db.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('started_at_utc',?)",
                (utc_now(),),
            )
            db.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('config_json',?)",
                (config_json,),
            )

    def meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def exposure(self) -> Exposure:
        row = self._connection.execute(
            """SELECT COALESCE(SUM(settled_usd),0),
                      COALESCE(SUM(CASE WHEN state IN ('in_flight','unknown')
                                       THEN reserved_usd ELSE 0 END),0)
               FROM attempts"""
        ).fetchone()
        return Exposure(float(row[0]), float(row[1]))

    def claim(
        self,
        *,
        request_id: str,
        attempt_index: int,
        spec: RequestSpec,
        route: RouteConfig,
        reserved_usd: float,
        max_cost_usd: float,
        cost_reserve_usd: float,
        scheduled_at_utc: str | None,
    ) -> bool:
        if reserved_usd < 0:
            raise ValueError("reserved_usd must be nonnegative")
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM attempts WHERE request_id=?", (request_id,)).fetchone():
                return False
            settled, reserved = db.execute(
                """SELECT COALESCE(SUM(settled_usd),0),
                          COALESCE(SUM(CASE WHEN state IN ('in_flight','unknown')
                                           THEN reserved_usd ELSE 0 END),0)
                   FROM attempts"""
            ).fetchone()
            launch_limit = max_cost_usd - cost_reserve_usd
            if float(settled) + float(reserved) + reserved_usd > launch_limit + 1e-12:
                raise BudgetExceeded(
                    "launch would exceed cost guard: "
                    f"exposure=${float(settled) + float(reserved):.6f}, "
                    f"reserve=${reserved_usd:.6f}, launch_limit=${launch_limit:.6f}"
                )
            db.execute(
                """INSERT INTO attempts(
                     request_id,logical_id,attempt_index,route_id,route_identity_hash,
                     suite,cell_id,payload_sha256,state,reserved_usd,settled_usd,
                     scheduled_at_utc,started_at_utc)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    request_id,
                    spec.logical_id,
                    attempt_index,
                    route.id,
                    route.identity_hash,
                    spec.suite,
                    spec.cell_id,
                    spec.payload_hash,
                    "in_flight",
                    reserved_usd,
                    0.0,
                    scheduled_at_utc,
                    utc_now(),
                ),
            )
            event_id = self._insert_event(
                db,
                "request_claimed",
                {
                    "request_id": request_id,
                    "logical_id": spec.logical_id,
                    "attempt_index": attempt_index,
                    "route_id": route.id,
                    "suite": spec.suite,
                    "cell_id": spec.cell_id,
                    "payload_sha256": spec.payload_hash,
                    "reserved_usd": reserved_usd,
                },
            )
        self._append_event(event_id)
        return True

    def finish(
        self,
        *,
        request_id: str,
        result: InferenceResult,
        validity: ValidityAssessment,
        quality_score: float | None,
        quality_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        settled = max(0.0, float(result.cost_usd or 0.0))
        with self._transaction() as db:
            row = db.execute(
                "SELECT state,reserved_usd FROM attempts WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"request was not claimed: {request_id}")
            if row["state"] != "in_flight":
                raise ValueError(f"request is already {row['state']}: {request_id}")
            # A successful response cannot settle above its worst-case reservation.
            # Unknown/failure accounting remains conservative through the reservation path.
            if settled > float(row["reserved_usd"]) + 1e-9:
                raise ValueError("settled cost exceeds the pre-send reservation")
            db.execute(
                """UPDATE attempts SET
                     state='terminal', reserved_usd=0, settled_usd=?, ended_at_utc=?,
                     status=?, http_status=?, input_tokens=?, output_tokens=?,
                     cache_read_input_tokens=?,
                     cache_state=?, total_seconds=?,
                     time_to_headers_seconds=?, ttft_seconds=?, decode_seconds=?,
                     content_event_count=?, output_event_offsets_json=?, decode_metric_source=?,
                     queue_delay_seconds=?, finish_reason=?, error_kind=?, error_body_sha256=?,
                     output_sha256=?, retained_headers_json=?, validity_class=?,
                     validity_reasons_json=?, latency_eligible=?, usage_eligible=?,
                     decode_eligible=?, quality_eligible=?, quality_score=?,
                     quality_diagnostics_json=?, cost_basis=? WHERE request_id=?""",
                (
                    settled,
                    result.ended_at_utc,
                    result.status,
                    result.http_status,
                    result.input_tokens,
                    result.output_tokens,
                    result.cache_read_input_tokens,
                    result.cache_state,
                    result.total_seconds,
                    result.time_to_headers_seconds,
                    result.ttft_seconds,
                    result.decode_seconds,
                    result.content_event_count,
                    canonical_json(list(result.output_event_offsets_seconds)),
                    "request_minus_ttft",
                    result.queue_delay_seconds,
                    result.finish_reason,
                    result.error_kind,
                    result.error_body_sha256,
                    result.output_sha256,
                    canonical_json(result.retained_headers),
                    validity.classification,
                    canonical_json(list(validity.reasons)),
                    int(validity.latency_eligible),
                    int(validity.usage_eligible),
                    int(validity.decode_eligible),
                    int(validity.quality_eligible),
                    quality_score,
                    canonical_json(quality_diagnostics or {}),
                    result.cost_basis,
                    request_id,
                ),
            )
            event_id = self._insert_event(
                db,
                "request_finished",
                {
                    "request_id": request_id,
                    **result.without_content(),
                    "validity": validity.classification,
                    "validity_reasons": list(validity.reasons),
                    "quality_score": quality_score,
                },
            )
        self._append_event(event_id)

    def recover_in_flight(self) -> int:
        """Fail closed after a crash: preserve reservation and never replay uncertain sends."""
        with self._transaction() as db:
            rows = db.execute("SELECT request_id FROM attempts WHERE state='in_flight'").fetchall()
            if not rows:
                return 0
            db.execute(
                """UPDATE attempts SET state='unknown', ended_at_utc=?, status='unknown',
                   error_kind='process_interrupted_after_claim', validity_class='censored',
                   validity_reasons_json='["unknown_provider_outcome"]'
                   WHERE state='in_flight'""",
                (utc_now(),),
            )
            event_id = self._insert_event(
                db,
                "in_flight_recovered_as_unknown",
                {"count": len(rows), "request_ids": [row[0] for row in rows]},
            )
        self._append_event(event_id)
        return len(rows)

    def has_terminal_logical(self, logical_id: str) -> bool:
        return bool(
            self._connection.execute(
                "SELECT 1 FROM attempts WHERE logical_id=? AND state='terminal' LIMIT 1",
                (logical_id,),
            ).fetchone()
        )

    def attempts_for_logical(self, logical_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._connection.execute(
                "SELECT * FROM attempts WHERE logical_id=? ORDER BY attempt_index", (logical_id,)
            )
        ]

    def record_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._transaction() as db:
            event_id = self._insert_event(db, kind, payload)
        self._append_event(event_id)

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute("SELECT * FROM attempts")]

    def event_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute("SELECT * FROM events")]

    def _insert_event(self, db: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> int:
        cursor = db.execute(
            "INSERT INTO events(recorded_at_utc,kind,payload_json) VALUES(?,?,?)",
            (utc_now(), kind, canonical_json(payload)),
        )
        return int(cursor.lastrowid)

    def _append_event(self, event_id: int) -> None:
        row = self._connection.execute(
            "SELECT * FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        encoded = canonical_json(dict(row)) + "\n"
        with self._lock, self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def rebuild_events_jsonl(self) -> None:
        temporary = self.events_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self._connection.execute("SELECT * FROM events ORDER BY event_id"):
                handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.events_path)
