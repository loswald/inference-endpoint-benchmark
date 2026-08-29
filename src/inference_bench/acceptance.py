from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .models import sha256_json

ACCEPTANCE_POLICY_SCHEMA = "capacity-acceptance/v1"


def _fraction(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    """Explicit, hash-bound criteria used by adaptive and fixed-rate load tests.

    A policy decides whether one measured epoch is acceptable. It deliberately excludes
    observability-only metrics such as TTFT completeness unless a caller supplies a comparative
    TTFT ceiling from a valid baseline. Missing telemetry therefore stays visible without being
    silently reinterpreted as endpoint failure.
    """

    min_success_fraction: float = 0.99
    max_throttled_attempt_fraction: float = 0.01
    max_retryable_error_attempt_fraction: float = 0.01
    queue_delay_floor_seconds: float = 1.0
    max_p95_queue_delay_fraction_of_window: float = 0.10
    max_p95_latency_multiplier_from_baseline: float = 2.0
    require_all_logical_outcomes: bool = True
    require_all_queue_observations: bool = True
    require_all_success_arrival_latencies: bool = True
    schema_version: str = ACCEPTANCE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        _fraction(self.min_success_fraction, "acceptance_policy.min_success_fraction")
        _fraction(
            self.max_throttled_attempt_fraction,
            "acceptance_policy.max_throttled_attempt_fraction",
        )
        _fraction(
            self.max_retryable_error_attempt_fraction,
            "acceptance_policy.max_retryable_error_attempt_fraction",
        )
        _positive(self.queue_delay_floor_seconds, "acceptance_policy.queue_delay_floor_seconds")
        _fraction(
            self.max_p95_queue_delay_fraction_of_window,
            "acceptance_policy.max_p95_queue_delay_fraction_of_window",
        )
        _positive(
            self.max_p95_latency_multiplier_from_baseline,
            "acceptance_policy.max_p95_latency_multiplier_from_baseline",
        )
        for field in (
            "require_all_logical_outcomes",
            "require_all_queue_observations",
            "require_all_success_arrival_latencies",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"acceptance_policy.{field} must be boolean")
        if self.schema_version != ACCEPTANCE_POLICY_SCHEMA:
            raise ValueError(f"unsupported acceptance policy schema: {self.schema_version}")

    @classmethod
    def from_suite(cls, suite: dict[str, Any]) -> AcceptancePolicy:
        raw = suite.get("acceptance_policy") or {}
        if not isinstance(raw, dict):
            raise ValueError("acceptance_policy must be a mapping")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                "unknown acceptance_policy field(s): " + ", ".join(unknown)
            )
        return cls(**raw)

    @property
    def identity_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def queue_delay_limit(self, duration_seconds: float) -> float:
        return max(
            self.queue_delay_floor_seconds,
            duration_seconds * self.max_p95_queue_delay_fraction_of_window,
        )

    def baseline_latency_limit(self, baseline_p95_seconds: float | None) -> float | None:
        if baseline_p95_seconds is None:
            return None
        return baseline_p95_seconds * self.max_p95_latency_multiplier_from_baseline

