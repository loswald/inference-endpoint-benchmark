from __future__ import annotations

import pytest

from inference_bench.ledger import BudgetExceeded, CampaignLeaseHeld, Ledger
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


def test_reservation_overrun_terminalizes_remaining_plan_as_inconclusive(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    ledger.register_plan_cells(
        [
            {
                "plan_cell_id": "request:later",
                "logical_id": "later",
                "route_id": "route-a",
                "suite": "latency",
                "cell_id": "short_short",
            }
        ]
    )
    ledger.finalize_plan("reservation_overrun_latch")
    row = ledger.coverage_rows()[0]
    assert row["state"] == "inconclusive"
    assert row["reason"] == "reservation_overrun_latch"
    ledger.close()


def test_exclusive_owner_blocks_recovery_until_live_owner_releases(tmp_path, route) -> None:
    first = Ledger(tmp_path, exclusive_owner=True)
    first.initialize(campaign_hash="a" * 64, config_json="{}")
    assert first.claim(
        request_id="owner-send-1",
        attempt_index=1,
        spec=_spec(),
        route=route,
        reserved_usd=0.2,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    with pytest.raises(CampaignLeaseHeld, match="another live campaign"):
        Ledger(tmp_path, exclusive_owner=True)
    assert first.rows()[0]["state"] == "in_flight"
    first.close()

    successor = Ledger(tmp_path, exclusive_owner=True)
    assert successor.recover_in_flight() == 1
    assert successor.rows()[0]["state"] == "unknown"
    successor.close()
    assert not (tmp_path / ".campaign-owner.json").exists()


def test_nonempty_legacy_ledger_cannot_adopt_current_provenance(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    assert ledger.claim(
        request_id="legacy-send",
        attempt_index=1,
        spec=_spec(),
        route=route,
        reserved_usd=0.1,
        max_cost_usd=1,
        cost_reserve_usd=0.1,
        scheduled_at_utc=None,
    )
    with pytest.raises(ValueError, match="before any provider evidence"):
        ledger.set_meta_once("run_manifest_json", "{}")
    ledger.close()

    reopened = Ledger(tmp_path)
    with pytest.raises(ValueError, match="missing immutable provenance.*run_manifest_json"):
        reopened.initialize(campaign_hash="a" * 64, config_json="{}")
    reopened.close()


def test_pre_manifest_coverage_plan_cannot_be_adopted_by_new_source(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    ledger.register_plan_cells(
        [
            {
                "plan_cell_id": "request:old-extra",
                "logical_id": "old-extra",
                "route_id": "route-a",
                "suite": "latency",
                "cell_id": "old-cell",
                "planned_disposition": "required",
            }
        ]
    )
    ledger.close()

    reopened = Ledger(tmp_path)
    with pytest.raises(ValueError, match="historical campaign state.*run_manifest_json"):
        reopened.initialize(campaign_hash="a" * 64, config_json="{}")
    assert reopened.coverage_rows()[0]["plan_cell_id"] == "request:old-extra"
    reopened.close()
