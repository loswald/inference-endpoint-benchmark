from __future__ import annotations

from dataclasses import replace

from inference_bench.cli import _capacity_completion_audit
from inference_bench.ledger import Ledger


def _capacity_campaign(campaign):
    return replace(
        campaign,
        suites={"aimd": {"enabled": True, "shapes": ["short_short"]}},
    )


def test_capacity_completion_audit_rejects_missing_or_inconclusive_measurements(
    tmp_path, campaign, route
) -> None:
    config = _capacity_campaign(campaign)
    missing = Ledger(tmp_path / "missing")
    missing_audit = _capacity_completion_audit(config, missing)
    assert missing_audit["scientifically_complete"] is False
    assert missing_audit["unresolved_cells"] == [
        {
            "suite": "aimd",
            "route_id": route.id,
            "shape": "short_short",
            "state": "measurement_missing",
        }
    ]
    missing.close()

    inconclusive = Ledger(tmp_path / "inconclusive")
    inconclusive.record_event_once(
        f"aimd_complete:{route.id}:short_short",
        "aimd_complete",
        {"controller_completion_state": "confirmations_inconclusive_after_retries"},
    )
    inconclusive_audit = _capacity_completion_audit(config, inconclusive)
    assert inconclusive_audit["scientifically_complete"] is False
    assert inconclusive_audit["unresolved_cell_count"] == 1
    inconclusive.close()


def test_capacity_completion_audit_accepts_healthy_or_measured_negative_floor(
    tmp_path, campaign, route
) -> None:
    config = _capacity_campaign(campaign)
    for index, (state, rate_evidence) in enumerate(
        (
            ("completed_confirmations_healthy", {"healthy_lower_bound_rps": 1.25}),
            (
                "completed_no_healthy_rate_at_floor",
                {"unhealthy_upper_bound_rps": 0.01},
            ),
        )
    ):
        ledger = Ledger(tmp_path / str(index))
        ledger.record_event_once(
            f"aimd_complete:{route.id}:short_short",
            "aimd_complete",
            {"controller_completion_state": state, **rate_evidence},
        )
        audit = _capacity_completion_audit(config, ledger)
        assert audit["scientifically_complete"] is True
        assert audit["resolved_cells"] == 1
        assert audit["unresolved_cells"] == []
        ledger.close()


def test_capacity_completion_audit_rejects_unproven_rate_or_floor(
    tmp_path, campaign, route
) -> None:
    config = _capacity_campaign(campaign)
    for index, payload in enumerate(
        (
            {"controller_completion_state": "completed_confirmations_healthy"},
            {
                "controller_completion_state": "completed_no_healthy_rate_at_floor",
                "unhealthy_upper_bound_rps": 0.02,
            },
        )
    ):
        ledger = Ledger(tmp_path / f"unproven-{index}")
        ledger.record_event_once(
            f"aimd_complete:{route.id}:short_short",
            "aimd_complete",
            payload,
        )
        audit = _capacity_completion_audit(config, ledger)
        assert audit["scientifically_complete"] is False
        assert audit["unresolved_cell_count"] == 1
        ledger.close()
