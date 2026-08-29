from __future__ import annotations

import asyncio
import hashlib
import math
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from .adapters import AdapterUnavailable, PreparedRequest, adapter_for
from .adapters.base import validate_adapter_instance
from .config import CampaignConfig, validate_route_evidence_identity
from .ledger import Ledger, TimeLimitReached
from .models import (
    InferenceResult,
    RequestSpec,
    canonical_json,
    normalize_arrival_latency_censor_reason,
    normalize_result_status,
    normalize_usage_parse_errors,
)
from .quality import score_result
from .validity import assess_result


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _materialized_input_is_nonempty(value: dict[str, Any]) -> bool:
    """Return whether the exact request necessarily contains at least one input token.

    This is deliberately only a lower-bound check. A nonempty message array carries at least role
    framing even if its textual content is empty, while tools also contribute prompt material.
    """

    messages = value.get("messages")
    responses_input = value.get("input")
    tools = value.get("tools")
    return (
        bool(isinstance(messages, list) and messages)
        or bool(isinstance(responses_input, list) and responses_input)
        or bool(isinstance(tools, list) and tools)
    )


def deterministic_request_id(
    spec: RequestSpec, attempt_index: int, *, payload_hash: str | None = None
) -> str:
    material = (
        f"request/v2\0{spec.logical_id}\0{payload_hash or spec.payload_hash}\0{attempt_index}"
    )
    return "req_" + hashlib.sha256(material.encode()).hexdigest()[:32]


_CREDENTIAL_HEADER_NAMES = frozenset(
    {"authorization", "api-key", "x-api-key", "cookie", "set-cookie", "proxy-authorization"}
)


def _safe_retained_headers(route: Any, value: object) -> dict[str, str]:
    """Reapply the route allowlist at the engine's untrusted custom-adapter boundary."""

    if not isinstance(value, dict):
        raise ValueError("adapter_result_contract_violation")
    allowed = {name.casefold() for name in route.retained_header_names}
    blocked = {*_CREDENTIAL_HEADER_NAMES, route.auth.header.casefold()}
    result: dict[str, str] = {}
    duplicate_names: set[str] = set()
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            continue
        name = raw_name.casefold()
        if name not in allowed or name in blocked or name in duplicate_names:
            continue
        if any(character in raw_value for character in "\r\n\0") or len(raw_value) > 1_024:
            continue
        if name in result:
            # Conflicting casing must not let a custom adapter choose which value controls retry
            # behavior or becomes public. Quarantine both copies.
            result.pop(name, None)
            duplicate_names.add(name)
            continue
        result[name] = raw_value
    return result


def _normalize_custom_adapter_result(
    route: Any,
    spec: RequestSpec,
    result: object,
    *,
    safe_started_at_utc: str,
    safe_ended_at_utc: str,
    safe_total_seconds: float,
) -> InferenceResult:
    """Quarantine adapter-controlled structures before any durable/public projection."""

    if not isinstance(result, InferenceResult) or result.logical_id != spec.logical_id:
        raise ValueError("adapter_result_contract_violation")
    if not isinstance(result.output_text, str) or not isinstance(result.tool_calls, (list, tuple)):
        raise ValueError("adapter_result_contract_violation")
    if any(not isinstance(item, dict) for item in result.tool_calls):
        raise ValueError("adapter_result_contract_violation")
    try:
        canonical_json(list(result.tool_calls))
    except (TypeError, ValueError) as exc:
        raise ValueError("adapter_result_contract_violation") from exc
    if result.error_body_sha256 is not None and (
        not isinstance(result.error_body_sha256, str)
        or len(result.error_body_sha256) != 64
        or any(character not in "0123456789abcdef" for character in result.error_body_sha256)
    ):
        raise ValueError("adapter_result_contract_violation")

    contract_errors: list[str] = []
    raw_total_seconds = result.total_seconds
    if (
        isinstance(raw_total_seconds, bool)
        or not isinstance(raw_total_seconds, (int, float))
        or not math.isfinite(float(raw_total_seconds))
        or float(raw_total_seconds) <= 0
    ):
        contract_errors.append("total_seconds_invalid_seconds")
    else:
        local_total = max(float(safe_total_seconds), 1e-9)
        if abs(float(raw_total_seconds) - local_total) > max(0.25, local_total * 0.2):
            contract_errors.append("adapter_total_seconds_disagrees_with_engine_clock")
    if result.http_status is not None and (
        isinstance(result.http_status, bool)
        or not isinstance(result.http_status, int)
        or not 100 <= result.http_status <= 599
    ):
        result.http_status = None
        contract_errors.append("http_status_invalid")
    for name in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(result, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            setattr(result, name, None)
            contract_errors.append(f"{name}_invalid_count")
    for name in ("time_to_headers_seconds", "ttft_seconds", "decode_seconds"):
        value = getattr(result, name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            setattr(result, name, None)
            contract_errors.append(f"{name}_invalid_seconds")
        elif value is not None:
            setattr(result, name, float(value))
    offsets = result.output_event_offsets_seconds
    if not isinstance(offsets, (list, tuple)) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in offsets
    ):
        result.output_event_offsets_seconds = ()
        contract_errors.append("stream_event_offset_invalid_seconds")
    else:
        result.output_event_offsets_seconds = tuple(float(value) for value in offsets)

    # These fields are measured locally around the adapter call, never accepted from an adapter.
    result.started_at_utc = safe_started_at_utc
    result.ended_at_utc = safe_ended_at_utc
    result.total_seconds = max(float(safe_total_seconds), 1e-9)
    result.retained_headers = _safe_retained_headers(route, result.retained_headers)
    if result.provider_request_id is not None and not isinstance(result.provider_request_id, str):
        result.provider_request_id = None
    result.tool_calls = tuple(result.tool_calls)
    result.adapter_contract_errors = tuple(dict.fromkeys(contract_errors))
    return result


class PaymentRequiredLatched(RuntimeError):
    """Raised after one HTTP 402 so the campaign cannot launch additional provider traffic."""


class ReservationOverrunLatched(RuntimeError):
    """Raised after provider usage exceeds the pre-send conservative cost reservation."""


class BenchmarkEngine:
    def __init__(self, config: CampaignConfig, ledger: Ledger) -> None:
        self.config = config
        self.ledger = ledger
        self.routes = {route.id: route for route in config.routes}
        self.adapters: dict[tuple[str, bool, bool, int], Any] = {}
        # The latch is durable across process restarts; a prior 402 is never forgotten merely
        # because the runner was resumed.
        self.payment_required_latched = any(
            row.get("http_status") == 402 for row in self.ledger.rows()
        )
        self.reservation_overrun_latched = any(
            row.get("kind") == "reservation_overrun" for row in self.ledger.event_rows()
        )

    async def close(self) -> None:
        for adapter in self.adapters.values():
            await adapter.close()

    def _adapter(self, route: Any) -> Any:
        key = (
            route.adapter,
            route.http2,
            route.connection_reuse,
            route.transport_max_connections,
        )
        adapter = self.adapters.get(key) or self.adapters.get(route.adapter)  # type: ignore[arg-type]
        if adapter is None:
            adapter = adapter_for(
                route.adapter,
                http2=route.http2,
                connection_reuse=route.connection_reuse,
                transport_max_connections=route.transport_max_connections,
            )
            self.adapters[key] = adapter
        return validate_adapter_instance(route.adapter, adapter)

    def preflight(self) -> None:
        """Validate every adapter, credential, and transport before the first claim."""

        for route in self.config.routes:
            self._preflight_identity(route)
            adapter = self._adapter(route)
            if hasattr(adapter, "preflight"):
                adapter.preflight(route)

    def _preflight_identity(self, route: Any) -> None:
        validate_route_evidence_identity(self.config, route)

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
        # This clock starts at admission to the engine, before credential/transport checks and
        # payload materialization. Combined with the caller-supplied scheduled-arrival delay, the
        # headline latency covers every client-side stage through final retry drain.
        logical_started = time.perf_counter()
        if self.payment_required_latched:
            raise PaymentRequiredLatched("HTTP 402 latch is active; no further sends are allowed")
        if self.reservation_overrun_latched:
            raise ReservationOverrunLatched(
                "cost reservation overrun latch is active; no further sends are allowed"
            )
        route = self.routes[spec.route_id]
        self._preflight_identity(route)
        existing = self.ledger.attempts_for_logical(spec.logical_id)
        if any(row["state"] in {"in_flight", "unknown"} for row in existing):
            return None
        retryable_statuses = {"rate_limited", "server_error", "timeout", "transport_error"}
        terminal = [row for row in existing if row["state"] == "terminal"]
        if terminal and terminal[-1]["status"] not in retryable_statuses:
            return None
        resumed_after_terminal_attempt = bool(terminal)
        start_attempt = 1 + max((int(row["attempt_index"]) for row in existing), default=0)
        if start_attempt > self.config.retries + 1:
            return None

        # Credential lookup, adapter construction, exact payload materialization, JSON encoding,
        # and conservative token-bound calculation all happen before a durable claim. A local
        # pre-send failure therefore creates no ambiguous provider outcome.
        adapter = self._adapter(route)
        adapter.preflight(route)
        prepared = adapter.prepare(route, spec)
        if not isinstance(prepared, PreparedRequest):
            raise AdapterUnavailable(
                f"adapter {route.adapter!r} prepare() must return PreparedRequest"
            )
        materialized = prepared.payload
        reserved_input_tokens = math.ceil(
            max(spec.planned_input_tokens, materialized.input_token_upper_bound)
            * self.config.input_token_reservation_factor
        )
        reservation = route.worst_case_cost(reserved_input_tokens, spec.max_output_tokens)
        last: InferenceResult | None = None
        for attempt in range(start_attempt, self.config.retries + 2):
            if self.payment_required_latched:
                raise PaymentRequiredLatched(
                    "HTTP 402 latch is active; no further retries are allowed"
                )
            if self.reservation_overrun_latched:
                raise ReservationOverrunLatched(
                    "cost reservation overrun latch is active; no further retries are allowed"
                )
            self.check_time_guard(spec.timeout_seconds)
            request_id = deterministic_request_id(
                spec, attempt, payload_hash=materialized.bound_payload_sha256
            )
            claimed = self.ledger.claim(
                request_id=request_id,
                attempt_index=attempt,
                spec=spec,
                route=route,
                reserved_usd=reservation,
                max_cost_usd=self.config.max_cost_usd,
                cost_reserve_usd=self.config.launch_reserve_usd,
                scheduled_at_utc=scheduled_at_utc,
                payload_sha256=materialized.bound_payload_sha256,
                wire_body_sha256=materialized.wire_body_sha256,
                payload_generator_version=materialized.generator_version,
                reserved_input_tokens=reserved_input_tokens,
            )
            if not claimed:
                # Another coroutine/process already owns this exact physical attempt. Never skip
                # forward to attempt N+1: that would duplicate a still-running logical send.
                return None
            settlement_committed = False
            try:
                try:
                    adapter_started_utc = _utc_now()
                    adapter_started = time.perf_counter()
                    try:
                        async with asyncio.timeout(spec.timeout_seconds):
                            result = await adapter.send_prepared(route, spec, prepared)
                    except TimeoutError:
                        result = InferenceResult(
                            logical_id=spec.logical_id,
                            status="timeout",
                            http_status=None,
                            started_at_utc=adapter_started_utc,
                            ended_at_utc=_utc_now(),
                            total_seconds=time.perf_counter() - adapter_started,
                            error_kind="hard_request_deadline",
                        )
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
                result = _normalize_custom_adapter_result(
                    route,
                    spec,
                    result,
                    safe_started_at_utc=adapter_started_utc,
                    safe_ended_at_utc=_utc_now(),
                    safe_total_seconds=time.perf_counter() - adapter_started,
                )
                # A custom adapter is an untrusted boundary even when it returns our dataclass.
                # Normalize every string surface that can reach durable/public evidence before it
                # influences scoring, validity, retry control, or event projection.
                result.status = normalize_result_status(result.status)  # type: ignore[assignment]
                result.usage_parse_errors = normalize_usage_parse_errors(result.usage_parse_errors)
                result.arrival_latency_censor_reason = normalize_arrival_latency_censor_reason(
                    result.arrival_latency_censor_reason
                )
                result.queue_delay_seconds = queue_delay_seconds
                if resumed_after_terminal_attempt:
                    # A new process/invocation cannot reconstruct time spent before the prior
                    # durable attempt. Preserve per-attempt latency but censor the end-to-end
                    # scheduled-arrival estimand instead of resetting its clock silently.
                    result.arrival_to_completion_seconds = None
                    result.arrival_latency_censor_reason = (
                        "resumed_retry_arrival_latency_unavailable"
                    )
                else:
                    result.arrival_to_completion_seconds = (
                        queue_delay_seconds + time.perf_counter() - logical_started
                    )
                usage_integrity_errors = list(result.usage_parse_errors)
                if result.status == "success":
                    if result.input_tokens == 0 and _materialized_input_is_nonempty(
                        materialized.value
                    ):
                        usage_integrity_errors.append(
                            "provider_input_tokens_zero_for_nonempty_request"
                        )
                    if result.output_tokens == 0 and (
                        bool(result.output_text) or bool(result.tool_calls)
                    ):
                        usage_integrity_errors.append(
                            "provider_output_tokens_zero_for_nonempty_response"
                        )
                result.usage_parse_errors = tuple(dict.fromkeys(usage_integrity_errors))
                usage_counts_valid = bool(
                    result.usage_complete
                    and result.input_tokens is not None
                    and result.output_tokens is not None
                    and isinstance(result.input_tokens, int)
                    and not isinstance(result.input_tokens, bool)
                    and isinstance(result.output_tokens, int)
                    and not isinstance(result.output_tokens, bool)
                    and result.input_tokens >= 0
                    and result.output_tokens >= 0
                    and (
                        result.reasoning_tokens is None
                        or isinstance(result.reasoning_tokens, int)
                        and not isinstance(result.reasoning_tokens, bool)
                        and result.reasoning_tokens >= 0
                        and result.reasoning_tokens <= result.output_tokens
                    )
                    and not result.usage_parse_errors
                    and (
                        result.cache_read_input_tokens is None
                        or isinstance(result.cache_read_input_tokens, int)
                        and not isinstance(result.cache_read_input_tokens, bool)
                        and 0 <= result.cache_read_input_tokens <= result.input_tokens
                    )
                )
                provider_cost: float | None = None
                if usage_counts_valid:
                    if result.cache_read_input_tokens is None:
                        provider_cost = route.usage_cost_with_unknown_cache(
                            int(result.input_tokens or 0), int(result.output_tokens or 0)
                        )
                        result.cost_basis = "provider_usage_cache_unknown_upper_bound"
                    else:
                        provider_cost = route.actual_cost(
                            int(result.input_tokens or 0),
                            int(result.output_tokens or 0),
                            int(result.cache_read_input_tokens),
                        )
                        result.cost_basis = "provider_usage"
                    # A reservation is a pre-send ceiling, not a bill. Known valid provider usage
                    # settles at its computed price; an amount above the ceiling is retained and
                    # latches the campaign before any further launch.
                    result.cost_usd = provider_cost
                    usage_bound_errors = list(result.usage_parse_errors)
                    if int(result.input_tokens or 0) > reserved_input_tokens:
                        usage_bound_errors.append("provider_input_tokens_exceed_reservation")
                    if int(result.output_tokens or 0) > spec.max_output_tokens:
                        usage_bound_errors.append("provider_output_tokens_exceed_request_limit")
                    result.usage_parse_errors = tuple(usage_bound_errors)
                else:
                    # Failed and usage-incomplete calls can still be billed. Settle the
                    # conservative reservation instead of pretending they cost zero.
                    result.cost_usd = reservation
                    result.cost_basis = "reserved_upper_bound"
                result.cache_state = str(  # type: ignore[assignment]
                    spec.metadata.get("cache_state", "uncontrolled")
                )
                final_attempt = bool(
                    result.status not in retryable_statuses
                    or attempt >= self.config.retries + 1
                    or provider_cost is not None
                    and provider_cost > reservation + 1e-12
                    or result.http_status == 402
                )
                quality_score, diagnostics = score_result(spec, result)
                validity = assess_result(
                    result,
                    expected_rejection=bool(spec.metadata.get("expected_rejection")),
                    parameter_acceptance_only=(
                        spec.metadata.get("capability_evidence_scope")
                        == "parameter_acceptance_only"
                    ),
                    quality_scored=quality_score is not None,
                )
                self.ledger.finish(
                    request_id=request_id,
                    result=result,
                    validity=validity,
                    quality_score=quality_score,
                    quality_diagnostics=diagnostics,
                    final_logical=final_attempt,
                )
                settlement_committed = True
            except Exception as exc:
                # Everything after a claim is inside this barrier: adapter execution, response
                # normalization, hashing, scoring, validity assessment, and ledger settlement.
                # Any exception before a committed terminal settlement is an ambiguous send.
                if not settlement_committed:
                    self.ledger.mark_unknown_if_in_flight(
                        request_id,
                        error_kind=f"post_claim_exception:{type(exc).__name__}",
                    )
                raise
            last = result
            if provider_cost is not None and provider_cost > reservation + 1e-12:
                self.reservation_overrun_latched = True
                break
            if result.http_status == 402:
                if not self.payment_required_latched:
                    self.payment_required_latched = True
                    self.ledger.record_event(
                        "http_402_latched",
                        {"request_id": request_id, "route_id": route.id},
                    )
                break
            if result.status not in {"rate_limited", "server_error", "timeout", "transport_error"}:
                break
            if attempt <= self.config.retries:
                retry_after = _retry_after(result.retained_headers)
                backoff = max(retry_after, min(30.0, 0.5 * (2 ** (attempt - 1))))
                # Deterministic jitter makes a run reproducible while avoiding a retry herd.
                rng = random.Random(f"{spec.logical_id}:{attempt}:{self.config.seed}")
                delay = backoff * (1.0 + 0.2 * rng.random())
                # The honored Retry-After and local jitter are part of the hard campaign wall
                # clock. Refuse the retry before sleeping if delay + its full request deadline
                # cannot fit; the caller will time-censor all remaining plan cells.
                self.check_time_guard(delay + spec.timeout_seconds)
                await asyncio.sleep(delay)
        return last


def _retry_after(headers: dict[str, str]) -> float:
    value = headers.get("retry-after")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
