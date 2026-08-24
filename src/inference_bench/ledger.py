from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import (
    InferenceResult,
    RequestSpec,
    RouteConfig,
    ValidityAssessment,
    canonical_json,
    json_safe_number,
    normalize_arrival_latency_censor_reason,
    normalize_finish_reason,
    normalize_result_status,
    normalize_usage_parse_errors,
    normalize_validity_reasons,
    public_error_category,
)
from .quality import predeclared_quality_scorer


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class BudgetExceeded(RuntimeError):
    pass


class TimeLimitReached(RuntimeError):
    pass


class CampaignLeaseHeld(RuntimeError):
    pass


class CampaignOwnerLease:
    """An OS-released exclusive owner lock for one live campaign directory.

    The kernel releases the byte-range/file lock if the process dies, so a later process can
    safely take over before recovering in-flight ledger rows. The adjacent JSON document is
    diagnostic only; it never decides ownership and is removed before a clean unlock.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.lock_path = directory / ".campaign-owner.lock"
        self.owner_path = directory / ".campaign-owner.json"
        self.owner_id = uuid.uuid4().hex
        self._handle: Any | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("campaign owner lease is already acquired")
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b", buffering=0)
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise CampaignLeaseHeld(
                "run directory is owned by another live campaign process"
            ) from exc
        self._handle = handle
        metadata = canonical_json(
            {
                "schema_version": "campaign-owner/v1",
                "owner_id": self.owner_id,
                "process_id": os.getpid(),
                "acquired_at_utc": utc_now(),
            }
        )
        temporary = self.owner_path.with_name(f".{self.owner_path.name}.{self.owner_id}.tmp")
        try:
            temporary.write_text(metadata + "\n", encoding="utf-8")
            os.replace(temporary, self.owner_path)
        except Exception:
            with suppress(OSError):
                temporary.unlink()
            self.release()
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        # Remove our diagnostic receipt before unlocking so the next owner cannot have its
        # metadata removed by this process after it acquires the kernel lock.
        try:
            if self.owner_path.is_file():
                try:
                    owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    owner = None
                if isinstance(owner, dict) and owner.get("owner_id") == self.owner_id:
                    with suppress(OSError):
                        self.owner_path.unlink()
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                self._handle = None


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
  wire_body_sha256 TEXT NOT NULL DEFAULT '',
  payload_generator_version TEXT NOT NULL DEFAULT 'legacy-unmaterialized',
  reserved_input_tokens INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL CHECK(state IN ('in_flight','terminal','unknown')),
  final_logical INTEGER NOT NULL DEFAULT 0 CHECK(final_logical IN (0,1)),
  reserved_usd REAL NOT NULL CHECK(reserved_usd >= 0),
  settled_usd REAL NOT NULL CHECK(settled_usd >= 0),
  scheduled_at_utc TEXT,
  started_at_utc TEXT NOT NULL,
  ended_at_utc TEXT,
  status TEXT,
  http_status INTEGER,
  input_tokens INTEGER,
  output_tokens INTEGER,
  reasoning_tokens INTEGER,
  cache_read_input_tokens INTEGER,
  usage_parse_errors_json TEXT NOT NULL DEFAULT '[]',
  cache_state TEXT NOT NULL DEFAULT 'uncontrolled',
  total_seconds REAL,
  time_to_headers_seconds REAL,
  ttft_seconds REAL,
  decode_seconds REAL,
  content_event_count INTEGER NOT NULL DEFAULT 0,
  output_event_offsets_json TEXT NOT NULL DEFAULT '[]',
  decode_metric_source TEXT NOT NULL DEFAULT 'billed_completion_tokens_over_request_minus_ttft',
  queue_delay_seconds REAL,
  arrival_to_completion_seconds REAL,
  arrival_latency_censor_reason TEXT,
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
  quality_predeclared INTEGER NOT NULL DEFAULT 0 CHECK(quality_predeclared IN (0,1)),
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
  event_key TEXT UNIQUE,
  recorded_at_utc TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS coverage_plan (
  plan_cell_id TEXT PRIMARY KEY,
  logical_id TEXT UNIQUE,
  route_id TEXT NOT NULL,
  suite TEXT NOT NULL,
  cell_id TEXT NOT NULL,
  planned_disposition TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN (
    'planned','completed','unsupported','inconclusive','untested','cap_censored',
    'time_censored','not_applicable','preflight_failed'
  )),
  reason TEXT,
  updated_at_utc TEXT NOT NULL
);
"""
LEDGER_PRODUCER_SCHEMA_VERSION = "inference-bench-ledger/v4"


class Ledger:
    """Crash-safe request ledger.

    SQLite is authoritative. JSONL is a secret-free event projection for inspection.
    An existing in-flight ID is never claimed again.
    """

    def __init__(self, directory: str | Path, *, exclusive_owner: bool = False) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "ledger.sqlite3"
        self.events_path = self.directory / "events.jsonl"
        self._lock = threading.RLock()
        self._closed = False
        self._owner_lease = CampaignOwnerLease(self.directory) if exclusive_owner else None
        if self._owner_lease is not None:
            self._owner_lease.acquire()
        try:
            self._connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.executescript(SCHEMA)
            self._migrate_legacy_schema()
            self._connection.commit()
        except Exception:
            if self._owner_lease is not None:
                self._owner_lease.release()
            raise

    def _migrate_legacy_schema(self) -> None:
        additions = {
            "wire_body_sha256": "TEXT NOT NULL DEFAULT ''",
            "payload_generator_version": "TEXT NOT NULL DEFAULT 'legacy-unmaterialized'",
            "reserved_input_tokens": "INTEGER NOT NULL DEFAULT 0",
            "reasoning_tokens": "INTEGER",
            "usage_parse_errors_json": "TEXT NOT NULL DEFAULT '[]'",
            "arrival_to_completion_seconds": "REAL",
            "arrival_latency_censor_reason": "TEXT",
            "final_logical": "INTEGER NOT NULL DEFAULT 0",
            "quality_predeclared": "INTEGER NOT NULL DEFAULT 0",
        }
        attempt_columns = {
            str(row[1]) for row in self._connection.execute("PRAGMA table_info(attempts)")
        }
        for name, declaration in additions.items():
            if name not in attempt_columns:
                self._connection.execute(f"ALTER TABLE attempts ADD COLUMN {name} {declaration}")
        event_columns = {
            str(row[1]) for row in self._connection.execute("PRAGMA table_info(events)")
        }
        if "event_key" not in event_columns:
            self._connection.execute("ALTER TABLE events ADD COLUMN event_key TEXT")
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS events_key_unique ON events(event_key)"
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._connection.close()
        finally:
            if self._owner_lease is not None:
                self._owner_lease.release()
            self._closed = True

    def checkpoint_for_export(self) -> None:
        """Fold the WAL into the main database before hashing a locked terminal snapshot."""

        with self._lock:
            self._connection.commit()
            result = self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or int(result[0]) != 0:
                raise RuntimeError("SQLite WAL checkpoint remained busy during terminal export")
        descriptor = os.open(self.db_path, os.O_RDWR)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

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
            historical_attempts = bool(db.execute("SELECT 1 FROM attempts LIMIT 1").fetchone())
            historical_send_events = bool(
                db.execute(
                    """SELECT 1 FROM events
                       WHERE kind IN ('request_claimed','request_finished',
                                      'request_outcome_unknown') LIMIT 1"""
                ).fetchone()
            )
            historical_coverage = bool(db.execute("SELECT 1 FROM coverage_plan LIMIT 1").fetchone())
            historical_events = bool(db.execute("SELECT 1 FROM events LIMIT 1").fetchone())
            historical_authoritative_state = (
                historical_attempts
                or historical_send_events
                or historical_coverage
                or historical_events
            )
            meta_rows = {
                str(row[0]): str(row[1]) for row in db.execute("SELECT key,value FROM meta")
            }
            if historical_authoritative_state:
                required = {
                    "campaign_hash",
                    "config_json",
                    "producer_schema_version",
                    "run_manifest_json",
                }
                missing = sorted(required - meta_rows.keys())
                if missing:
                    raise ValueError(
                        "historical campaign state is missing immutable provenance: "
                        + ", ".join(missing)
                    )
                if meta_rows["producer_schema_version"] != LEDGER_PRODUCER_SCHEMA_VERSION:
                    raise ValueError("historical provider evidence has an incompatible producer")
            existing = db.execute("SELECT value FROM meta WHERE key='campaign_hash'").fetchone()
            if existing and existing[0] != campaign_hash:
                raise ValueError("run directory belongs to a different campaign")
            existing_config = db.execute(
                "SELECT value FROM meta WHERE key='config_json'"
            ).fetchone()
            if existing_config and str(existing_config[0]) != config_json:
                raise ValueError("run directory sanitized campaign configuration changed")
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
            db.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('producer_schema_version',?)",
                (LEDGER_PRODUCER_SCHEMA_VERSION,),
            )

    def meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta_once(self, key: str, value: str) -> None:
        with self._transaction() as db:
            existing = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            if existing is not None and str(existing[0]) != value:
                raise ValueError(f"immutable run metadata changed: {key}")
            if key == "run_manifest_json" and existing is None:
                provider_evidence = bool(
                    db.execute("SELECT 1 FROM attempts LIMIT 1").fetchone()
                    or db.execute(
                        """SELECT 1 FROM events
                           WHERE kind IN ('request_claimed','request_finished',
                                          'request_outcome_unknown') LIMIT 1"""
                    ).fetchone()
                )
                if provider_evidence:
                    raise ValueError(
                        "run_manifest_json must be immutable before any provider evidence"
                    )
            db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)", (key, value))

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
        payload_sha256: str | None = None,
        wire_body_sha256: str = "",
        payload_generator_version: str = "legacy-unmaterialized",
        reserved_input_tokens: int = 0,
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
                     suite,cell_id,payload_sha256,wire_body_sha256,
                     payload_generator_version,reserved_input_tokens,state,reserved_usd,settled_usd,
                     scheduled_at_utc,started_at_utc,quality_predeclared)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    request_id,
                    spec.logical_id,
                    attempt_index,
                    route.id,
                    route.identity_hash,
                    spec.suite,
                    spec.cell_id,
                    payload_sha256 or spec.payload_hash,
                    wire_body_sha256,
                    payload_generator_version,
                    reserved_input_tokens,
                    "in_flight",
                    reserved_usd,
                    0.0,
                    scheduled_at_utc,
                    utc_now(),
                    int(predeclared_quality_scorer(spec) is not None),
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
                    "payload_sha256": payload_sha256 or spec.payload_hash,
                    "wire_body_sha256": wire_body_sha256,
                    "payload_generator_version": payload_generator_version,
                    "reserved_input_tokens": reserved_input_tokens,
                    "reserved_usd": reserved_usd,
                    "quality_predeclared": predeclared_quality_scorer(spec) is not None,
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
        final_logical: bool | None = None,
    ) -> None:
        if final_logical is None:
            # Safe default for direct ledger users: retryable transport/provider outcomes remain
            # provisional. The engine explicitly passes True on retry exhaustion.
            final_logical = result.status not in {
                "rate_limited",
                "server_error",
                "timeout",
                "transport_error",
            }
        if not isinstance(final_logical, bool):
            raise ValueError("final_logical must be a boolean when supplied")
        settled = max(0.0, float(result.cost_usd or 0.0))
        finish_reason = normalize_finish_reason(result.finish_reason)
        status = normalize_result_status(result.status)
        usage_parse_errors = normalize_usage_parse_errors(result.usage_parse_errors)
        arrival_censor_reason = normalize_arrival_latency_censor_reason(
            result.arrival_latency_censor_reason
        )
        validity_reasons = normalize_validity_reasons(validity.reasons)
        error_kind = public_error_category(result.error_kind)
        with self._transaction() as db:
            row = db.execute(
                "SELECT state,reserved_usd FROM attempts WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"request was not claimed: {request_id}")
            if row["state"] != "in_flight":
                raise ValueError(f"request is already {row['state']}: {request_id}")
            reservation_overrun = settled > float(row["reserved_usd"]) + 1e-9
            db.execute(
                """UPDATE attempts SET
                     state='terminal', final_logical=?, reserved_usd=0,
                     settled_usd=?, ended_at_utc=?,
                     status=?, http_status=?, input_tokens=?, output_tokens=?,
                     reasoning_tokens=?, cache_read_input_tokens=?, usage_parse_errors_json=?,
                     cache_state=?, total_seconds=?,
                     time_to_headers_seconds=?, ttft_seconds=?, decode_seconds=?,
                     content_event_count=?, output_event_offsets_json=?, decode_metric_source=?,
                     queue_delay_seconds=?, arrival_to_completion_seconds=?,
                     arrival_latency_censor_reason=?,
                     finish_reason=?, error_kind=?, error_body_sha256=?,
                     output_sha256=?, retained_headers_json=?, validity_class=?,
                     validity_reasons_json=?, latency_eligible=?, usage_eligible=?,
                     decode_eligible=?, quality_eligible=?, quality_score=?,
                     quality_diagnostics_json=?, cost_basis=? WHERE request_id=?""",
                (
                    int(final_logical),
                    settled,
                    result.ended_at_utc,
                    status,
                    result.http_status,
                    result.input_tokens,
                    result.output_tokens,
                    result.reasoning_tokens,
                    result.cache_read_input_tokens,
                    canonical_json(list(usage_parse_errors)),
                    result.cache_state,
                    result.total_seconds,
                    result.time_to_headers_seconds,
                    result.ttft_seconds,
                    result.decode_seconds,
                    result.content_event_count,
                    canonical_json(
                        [json_safe_number(value) for value in result.output_event_offsets_seconds]
                    ),
                    "billed_completion_tokens_over_request_minus_ttft",
                    result.queue_delay_seconds,
                    result.arrival_to_completion_seconds,
                    arrival_censor_reason,
                    finish_reason,
                    error_kind,
                    result.error_body_sha256,
                    result.output_sha256,
                    canonical_json(result.retained_headers),
                    validity.classification,
                    canonical_json(list(validity_reasons)),
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
            event_ids: list[int] = []
            if final_logical:
                db.execute(
                    """UPDATE coverage_plan SET state='completed',reason=NULL,updated_at_utc=?
                       WHERE logical_id=(SELECT logical_id FROM attempts WHERE request_id=?)""",
                    (utc_now(), request_id),
                )
            if reservation_overrun:
                # Never strand an actually-sent call as in-flight because a provider reported more
                # usage than the conservative materialized-byte reservation. Preserve the actual
                # settled amount and emit an explicit invariant breach for the terminal report.
                event_ids.append(
                    self._insert_event(
                        db,
                        "reservation_overrun",
                        {
                            "request_id": request_id,
                            "reserved_usd": float(row["reserved_usd"]),
                            "settled_usd": settled,
                        },
                    )
                )
            event_id = self._insert_event(
                db,
                "request_finished",
                {
                    "request_id": request_id,
                    **result.without_content(),
                    "final_logical": final_logical,
                    "validity": validity.classification,
                    "validity_reasons": list(validity_reasons),
                    "quality_score": quality_score,
                },
            )
            event_ids.append(event_id)
        for event_id in event_ids:
            self._append_event(event_id)

    def recover_in_flight(self) -> int:
        """Fail closed after a crash: preserve reservation and never replay uncertain sends."""
        with self._transaction() as db:
            rows = db.execute("SELECT request_id FROM attempts WHERE state='in_flight'").fetchall()
            if not rows:
                return 0
            db.execute(
                """UPDATE attempts SET state='unknown', final_logical=1,
                   ended_at_utc=?, status='unknown',
                   error_kind='process_interrupted_after_claim', validity_class='censored',
                   validity_reasons_json='["unknown_provider_outcome"]',
                   quality_eligible=quality_predeclared,
                   quality_score=CASE WHEN quality_predeclared=1 THEN 0 ELSE NULL END,
                   quality_diagnostics_json=CASE WHEN quality_predeclared=1
                     THEN '{"outcome":"unknown_after_claim","scored_as_zero":true}' ELSE '{}' END
                   WHERE state='in_flight'""",
                (utc_now(),),
            )
            db.execute(
                """UPDATE coverage_plan SET state='inconclusive',
                   reason='process_interrupted_after_claim',updated_at_utc=?
                   WHERE logical_id IN (
                     SELECT logical_id FROM attempts WHERE state='unknown'
                     AND error_kind='process_interrupted_after_claim'
                   )""",
                (utc_now(),),
            )
            event_id = self._insert_event(
                db,
                "in_flight_recovered_as_unknown",
                {"count": len(rows), "request_ids": [row[0] for row in rows]},
            )
        self._append_event(event_id)
        return len(rows)

    def mark_unknown(self, request_id: str, *, error_kind: str) -> None:
        """Terminalize one claimed send whose provider outcome cannot be known safely."""

        if not self.mark_unknown_if_in_flight(request_id, error_kind=error_kind):
            row = self._connection.execute(
                "SELECT state FROM attempts WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"request was not claimed: {request_id}")
            raise ValueError(f"request is already {row['state']}: {request_id}")

    def mark_unknown_if_in_flight(self, request_id: str, *, error_kind: str) -> bool:
        """Atomically censor an ambiguous claimed send, or return false if already settled."""

        normalized_error_kind = public_error_category(error_kind) or "other_error"
        with self._transaction() as db:
            row = db.execute(
                "SELECT state FROM attempts WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"request was not claimed: {request_id}")
            if row["state"] != "in_flight":
                return False
            db.execute(
                """UPDATE attempts SET state='unknown',final_logical=1,
                   ended_at_utc=?,status='unknown',
                   error_kind=?,validity_class='censored',
                   validity_reasons_json='["unknown_provider_outcome"]',
                   quality_eligible=quality_predeclared,
                   quality_score=CASE WHEN quality_predeclared=1 THEN 0 ELSE NULL END,
                   quality_diagnostics_json=CASE WHEN quality_predeclared=1
                     THEN '{"outcome":"unknown_after_claim","scored_as_zero":true}' ELSE '{}' END
                   WHERE request_id=?""",
                (utc_now(), normalized_error_kind, request_id),
            )
            db.execute(
                """UPDATE coverage_plan SET state='inconclusive',reason=?,updated_at_utc=?
                   WHERE logical_id=(SELECT logical_id FROM attempts WHERE request_id=?)""",
                (normalized_error_kind, utc_now(), request_id),
            )
            event_id = self._insert_event(
                db,
                "request_outcome_unknown",
                {"request_id": request_id, "error_kind": normalized_error_kind},
            )
        self._append_event(event_id)
        return True

    def has_terminal_logical(self, logical_id: str) -> bool:
        return bool(
            self._connection.execute(
                """SELECT 1 FROM attempts
                   WHERE logical_id=? AND state IN ('terminal','unknown') AND final_logical=1
                   LIMIT 1""",
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

    def record_event_once(self, event_key: str, kind: str, payload: dict[str, Any]) -> bool:
        with self._transaction() as db:
            existing = db.execute(
                "SELECT event_id FROM events WHERE event_key=?", (event_key,)
            ).fetchone()
            if existing:
                return False
            event_id = self._insert_event(db, kind, payload, event_key=event_key)
        self._append_event(event_id)
        return True

    def event_by_key(self, event_key: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM events WHERE event_key=?", (event_key,)
        ).fetchone()
        return None if row is None else dict(row)

    def register_plan_cells(self, cells: list[dict[str, Any]]) -> None:
        with self._transaction() as db:
            for cell in cells:
                initial_state = str(cell.get("initial_state", "planned"))
                if initial_state not in {
                    "planned",
                    "completed",
                    "unsupported",
                    "inconclusive",
                    "untested",
                    "cap_censored",
                    "time_censored",
                    "not_applicable",
                    "preflight_failed",
                }:
                    raise ValueError(f"invalid initial coverage state: {initial_state}")
                values = (
                    str(cell["plan_cell_id"]),
                    cell.get("logical_id"),
                    str(cell["route_id"]),
                    str(cell["suite"]),
                    str(cell["cell_id"]),
                    str(cell.get("planned_disposition", "required")),
                    initial_state,
                    cell.get("reason"),
                    utc_now(),
                )
                existing = db.execute(
                    "SELECT * FROM coverage_plan WHERE plan_cell_id=?", (values[0],)
                ).fetchone()
                if existing:
                    immutable = (
                        existing["logical_id"],
                        existing["route_id"],
                        existing["suite"],
                        existing["cell_id"],
                        existing["planned_disposition"],
                    )
                    if immutable != values[1:6]:
                        raise ValueError(f"coverage plan identity changed: {values[0]}")
                    continue
                db.execute(
                    """INSERT INTO coverage_plan(
                         plan_cell_id,logical_id,route_id,suite,cell_id,planned_disposition,
                         state,reason,updated_at_utc) VALUES(?,?,?,?,?,?,?,?,?)""",
                    values,
                )

    def mark_plan_cell(self, plan_cell_id: str, state: str, reason: str | None = None) -> None:
        allowed = {
            "completed",
            "unsupported",
            "inconclusive",
            "untested",
            "cap_censored",
            "time_censored",
            "not_applicable",
            "preflight_failed",
        }
        if state not in allowed:
            raise ValueError(f"invalid coverage state: {state}")
        with self._transaction() as db:
            cursor = db.execute(
                """UPDATE coverage_plan SET state=?,reason=?,updated_at_utc=?
                   WHERE plan_cell_id=?""",
                (state, reason, utc_now(), plan_cell_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown coverage cell: {plan_cell_id}")

    def mark_plan_cell_if_planned(
        self, plan_cell_id: str, state: str, reason: str | None = None
    ) -> bool:
        """Atomically censor an unstarted plan cell without overwriting resumed evidence."""

        if state not in {"inconclusive", "untested", "cap_censored", "time_censored"}:
            raise ValueError("conditional plan-cell updates require a censoring state")
        with self._transaction() as db:
            cursor = db.execute(
                """UPDATE coverage_plan SET state=?,reason=?,updated_at_utc=?
                   WHERE plan_cell_id=? AND state='planned'""",
                (state, reason, utc_now(), plan_cell_id),
            )
            return cursor.rowcount == 1

    def finalize_plan(self, reason: str) -> None:
        if reason in {"BudgetExceeded", "cost_guard"}:
            state = "cap_censored"
        elif reason in {"TimeLimitReached", "time_guard"}:
            state = "time_censored"
        elif reason == "preflight_failed":
            state = "preflight_failed"
        elif reason in {
            "unexpected_runner_error",
            "http_402_latch",
            "launch_guard",
            "reservation_overrun_latch",
        }:
            state = "inconclusive"
        else:
            state = "untested"
        with self._transaction() as db:
            db.execute(
                """UPDATE coverage_plan SET state=?,reason=?,updated_at_utc=?
                   WHERE state='planned'""",
                (state, reason, utc_now()),
            )

    def coverage_rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._connection.execute(
                "SELECT * FROM coverage_plan ORDER BY route_id,suite,cell_id,plan_cell_id"
            )
        ]

    def rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._connection.execute(
                "SELECT * FROM attempts ORDER BY logical_id,attempt_index,request_id"
            )
        ]

    def event_rows(self) -> list[dict[str, Any]]:
        return [
            dict(row) for row in self._connection.execute("SELECT * FROM events ORDER BY event_id")
        ]

    def _insert_event(
        self,
        db: sqlite3.Connection,
        kind: str,
        payload: dict[str, Any],
        *,
        event_key: str | None = None,
    ) -> int:
        cursor = db.execute(
            "INSERT INTO events(event_key,recorded_at_utc,kind,payload_json) VALUES(?,?,?,?)",
            (event_key, utc_now(), kind, canonical_json(payload)),
        )
        return int(cursor.lastrowid)

    def _append_event(self, event_id: int) -> None:
        if self.meta("events_projection_state") == "dirty":
            try:
                self.rebuild_events_jsonl()
            except OSError:
                return
            return
        row = self._connection.execute(
            "SELECT * FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        encoded = canonical_json(dict(row)) + "\n"
        try:
            with self._lock, self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # SQLite is authoritative. A derived-projection I/O failure must not turn an already
            # settled provider call into an unknown outcome or terminalize the campaign. Mark the
            # projection dirty in SQLite so a later safe rebuild is explicit and deterministic.
            with self._transaction() as db:
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) "
                    "VALUES('events_projection_state','dirty')"
                )

    def rebuild_events_jsonl(self) -> None:
        temporary = self.events_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self._connection.execute("SELECT * FROM events ORDER BY event_id"):
                handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.events_path)
        with self._transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('events_projection_state','clean')"
            )
