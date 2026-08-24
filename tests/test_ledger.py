from __future__ import annotations

import pytest

from inference_bench.ledger import BudgetExceeded, Ledger
from inference_bench.models import RequestSpec


def _spec() -> RequestSpec:
    return RequestSpec(
        logical_id="logical",
        route_id="route-a",
        suite="latency",
        cell_id="short_short",
        messages=({"role": "user", "content": "PRIVATE PROMPT"},),
        planned_input_tokens=10,
        max_output_tokens=10,
    )


def test_claim_is_idempotent_and_event_is_prompt_free(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    assert ledger.claim(
        request_id="req-1",
        attempt_index=1,
        spec=_spec(),
        route=route,
        reserved_usd=0.01,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    assert not ledger.claim(
        request_id="req-1",
        attempt_index=1,
        spec=_spec(),
        route=route,
        reserved_usd=0.01,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    assert "PRIVATE PROMPT" not in ledger.events_path.read_text(encoding="utf-8")
    ledger.close()


def test_recovery_marks_unknown_and_preserves_reservation(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    ledger.claim(
        request_id="req-1",
        attempt_index=1,
        spec=_spec(),
        route=route,
        reserved_usd=0.2,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    assert ledger.recover_in_flight() == 1
    assert ledger.exposure().reserved_usd == pytest.approx(0.2)
    assert ledger.rows()[0]["state"] == "unknown"
    ledger.close()


def test_budget_guard_is_atomic(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    with pytest.raises(BudgetExceeded):
        ledger.claim(
            request_id="req-1",
            attempt_index=1,
            spec=_spec(),
            route=route,
            reserved_usd=0.91,
            max_cost_usd=1,
            cost_reserve_usd=0.1,
            scheduled_at_utc=None,
        )
    assert ledger.rows() == []
    ledger.close()
