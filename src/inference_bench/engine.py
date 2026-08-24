from __future__ import annotations

import asyncio
import hashlib
import math
import random
from datetime import UTC, datetime
from typing import Any

from .adapters import AdapterUnavailable, adapter_for
from .config import CampaignConfig
from .ledger import Ledger, TimeLimitReached
from .models import InferenceResult, RequestSpec
from .quality import score_result
from .validity import assess_result


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def deterministic_request_id(spec: RequestSpec, attempt_index: int) -> str:
    material = f"request/v1\0{spec.logical_id}\0{spec.payload_hash}\0{attempt_index}"
    return "req_" + hashlib.sha256(material.encode()).hexdigest()[:32]


class BenchmarkEngine:
    def __init__(self, config: CampaignConfig, ledger: Ledger) -> None:
        self.config = config
        self.ledger = ledger
        self.routes = {route.id: route for route in config.routes}
        self.adapters: dict[str, Any] = {}

    async def close(self) -> None:
        for adapter in self.adapters.values():
            await adapter.close()

    def elapsed_seconds(self) -> float:
        started = self.ledger.meta("started_at_utc")
        if not started:
            return 0.0
        return max(0.0, (datetime.now(UTC) - _parse_utc(started)).total_seconds())

    def check_time_guard(self, timeout_seconds: float) -> None:
        launch_limit = self.config.max_wall_seconds - self.config.launch_reserve_seconds
        elapsed = self.elapsed_seconds()
        if elapsed >= launch_limit or elapsed + timeout_seconds > self.config.max_wall_seconds:
            raise TimeLimitReached(
                f"new request would cross launch cutoff at {launch_limit:.1f} elapsed seconds"
            )

    async def execute(
        self,
        spec: RequestSpec,
        *,
        scheduled_at_utc: str | None = None,
        queue_delay_seconds: float = 0.0,
    ) -> InferenceResult | None:
        route = self.routes[spec.route_id]
        existing = self.ledger.attempts_for_logical(spec.logical_id)
        if any(row["state"] == "unknown" for row in existing):
            return None
        if any(row["status"] == "success" for row in existing):
            return None
        start_attempt = 1 + max((int(row["attempt_index"]) for row in existing), default=0)
        if start_attempt > self.config.retries + 1:
            return None

        last: InferenceResult | None = None
        for attempt in range(start_attempt, self.config.retries + 2):
            self.check_time_guard(spec.timeout_seconds)
            reserved_input_tokens = math.ceil(
                spec.planned_input_tokens * self.config.input_token_reservation_factor
            )
            reservation = route.worst_case_cost(reserved_input_tokens, spec.max_output_tokens)
            request_id = deterministic_request_id(spec, attempt)
            claimed = self.ledger.claim(
                request_id=request_id,
                attempt_index=attempt,
                spec=spec,
                route=route,
                reserved_usd=reservation,
                max_cost_usd=self.config.max_cost_usd,
                cost_reserve_usd=self.config.launch_reserve_usd,
                scheduled_at_utc=scheduled_at_utc,
            )
            if not claimed:
                continue
            try:
                adapter = self.adapters.setdefault(route.adapter, adapter_for(route.adapter))
                result = await adapter.infer(route, spec)
            except AdapterUnavailable as exc:
                now = _utc_now()
                result = InferenceResult(
                    logical_id=spec.logical_id,
                    status="adapter_unavailable",
                    http_status=None,
                    started_at_utc=now,
                    ended_at_utc=now,
                    total_seconds=0.0,
                    error_kind=type(exc).__name__,
                )
            result.queue_delay_seconds = queue_delay_seconds
            if result.usage_complete:
                result.cost_usd = route.actual_cost(
                    int(result.input_tokens or 0),
                    int(result.output_tokens or 0),
                    int(result.cache_read_input_tokens or 0),
                )
                result.cost_basis = "provider_usage"
            else:
                # Failed and usage-incomplete calls can still be billed. Settle the conservative
                # reservation instead of pretending they cost zero.
                result.cost_usd = reservation
                result.cost_basis = "reserved_upper_bound"
            result.cache_state = str(spec.metadata.get("cache_state", "uncontrolled"))  # type: ignore[assignment]
            validity = assess_result(result)
            quality_score, diagnostics = score_result(spec, result)
            self.ledger.finish(
                request_id=request_id,
                result=result,
                validity=validity,
                quality_score=quality_score,
                quality_diagnostics=diagnostics,
            )
            last = result
            if result.status not in {"rate_limited", "server_error", "timeout", "transport_error"}:
                break
            if attempt <= self.config.retries:
                retry_after = _retry_after(result.retained_headers)
                backoff = max(retry_after, min(30.0, 0.5 * (2 ** (attempt - 1))))
                # Deterministic jitter makes a run reproducible while avoiding a retry herd.
                rng = random.Random(f"{spec.logical_id}:{attempt}:{self.config.seed}")
                await asyncio.sleep(backoff * (0.8 + 0.4 * rng.random()))
        return last


def _retry_after(headers: dict[str, str]) -> float:
    value = headers.get("retry-after")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0
