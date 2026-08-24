from __future__ import annotations

import hashlib
import json

import pytest

from inference_bench.ledger import Ledger
from inference_bench.models import InferenceResult, RequestSpec
from inference_bench.report import (
    _binary_quality_interval,
    _plot_matched_cells,
    build_outlier_audit,
    generate_report,
    summarize_controller_events,
    summarize_load_events,
    summarize_rows,
)
from inference_bench.validity import assess_result


def _spec(index: int) -> RequestSpec:
    return RequestSpec(
        logical_id=f"logical-{index}",
        route_id="route-a",
        suite="latency",
        cell_id="short_short",
        messages=({"role": "user", "content": "secret"},),
        planned_input_tokens=10,
        max_output_tokens=10,
    )


def _result(index: int, seconds: float) -> InferenceResult:
    return InferenceResult(
        logical_id=f"logical-{index}",
        status="success",
        http_status=200,
        started_at_utc="2026-01-01T00:00:00Z",
        ended_at_utc="2026-01-01T00:00:02Z",
        total_seconds=seconds,
        time_to_headers_seconds=0.1,
        ttft_seconds=0.2,
        decode_seconds=seconds - 0.2,
        output_event_offsets_seconds=(0.2, max(0.21, seconds - 0.1)),
        input_tokens=10,
        output_tokens=10,
        reasoning_tokens=0,
        arrival_to_completion_seconds=seconds,
        cost_usd=0.00003,
        cost_basis="provider_usage",
    )


def _attach_run_manifest(ledger: Ledger) -> None:
    manifest_json = json.dumps(
        {
            "schema_version": "run-manifest/v2",
            "normalized_exact_invocation": ["inference-bench", "run", "<CONFIG_OR_PATH>"],
            "raw_invocation_sha256": "a" * 64,
            "client_location": "test-fixture",
            "connection_reuse_by_route": {"route-a": True},
            "http2_by_route": {"route-a": False},
            "transport_max_connections_by_route": {"route-a": 256},
            "transport_header_profile_by_route": {
                "route-a": "openai-json-accept-encoding-identity/v1"
            },
            "request_timeout_seconds_by_route": {"route-a": 180.0},
            "provider_documentation_declarations": [
                {
                    "route_id": "route-a",
                    "documentation_source_url": "https://example.invalid/documentation",
                    "pricing_source_url": "https://example.invalid/pricing",
                    "evidence_retrieved_at_utc": "2026-08-24T00:00:00Z",
                    "declared_evidence_bundle_sha256": "a" * 64,
                    "verification_status": "declared_unverified_by_harness",
                }
            ],
            "transport_trust_env": False,
            "source_commit": "b" * 40,
            "source_dirty": False,
            "source_dirty_tree_sha256": "c" * 64,
            "dependency_lock_sha256": "d" * 64,
            "dependency_lock_file": "requirements.lock",
            "execution_environment": {
                "python": "3.12.0",
                "python_implementation": "CPython",
                "operating_system": "test-os",
                "operating_system_release": "test-release",
                "machine_architecture": "test-arch",
                "distributions": {"httpx": "0.28.1"},
            },
        }
    )
    ledger.set_meta_once("run_manifest_json", manifest_json)
    digest = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    ledger.set_meta_once("pre_send_run_manifest_sha256", digest)
    ledger.set_meta_once("terminal_run_manifest_sha256", digest)


def _empty_terminal_run(path) -> None:
    ledger = Ledger(path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    _attach_run_manifest(ledger)
    ledger.record_event_once("campaign_terminal", "campaign_terminal", {"reason": "completed"})
    ledger.close()
    (path / "campaign.public.json").write_text("{}\n", encoding="utf-8")


def _controller_event(kind: str, payload: dict[str, object]) -> dict[str, object]:
    return {"kind": kind, "payload_json": json.dumps(payload, sort_keys=True)}


def test_controller_summary_preserves_bounds_and_scientific_censor_states() -> None:
    config = {
        "routes": [{"id": "r"}],
        "suites": {
            "aimd": {
                "enabled": True,
                "shapes": ["right", "nonmonotonic", "unhealthy", "censored", "missing"],
            },
            "soak": {
                "enabled": True,
                "shapes": ["partial", "censored", "healthy"],
                "blocks": 4,
                "rate_rps": 1.0,
            },
        },
    }

    def aimd(shape: str, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "route_id": "r",
            "shape": shape,
            "highest_observed_healthy_rps": 4.0,
            "healthy_lower_bound_rps": 4.0,
            "unhealthy_upper_bound_rps": None,
            "overload_observed": False,
            "nonmonotonic_overload_observed": False,
            "capacity_bound_state": "right_censored_highest_tested_healthy_no_overload",
            "controller_completion_state": "completed_confirmations_healthy",
            "censor_reason": None,
            "confirmations_required": 3,
            "confirmation_healthy": [True, True, True],
            "confirmation_eligible": [True, True, True],
            "confirmation_censor_reasons": [None, None, None],
            "confirmation_execution_complete": True,
            "confirmation_complete": True,
            "confirmation_all_healthy": True,
            "recovery_run": False,
            "recovery_healthy": None,
            "recovery_eligible": None,
            "recovery_censor_reason": None,
        }
        payload.update(overrides)
        return payload

    def soak(shape: str, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "route_id": "r",
            "shape": shape,
            "rate_rps": 1.0,
            "blocks": 4,
            "completed_blocks": 4,
            "block_eligible": [True, True, True, True],
            "block_healthy": [True, True, True, True],
            "block_censor_reasons": [None, None, None, None],
            "execution_complete": True,
            "scientifically_complete": True,
            "all_blocks_healthy": True,
            "controller_completion_state": "completed_healthy",
            "censor_reason": None,
        }
        payload.update(overrides)
        return payload

    events = [
        _controller_event("aimd_complete", aimd("right")),
        _controller_event(
            "aimd_complete",
            aimd(
                "nonmonotonic",
                overload_observed=True,
                nonmonotonic_overload_observed=True,
                capacity_bound_state="nonmonotonic_overload_no_current_bracket",
            ),
        ),
        _controller_event(
            "aimd_complete",
            aimd(
                "unhealthy",
                overload_observed=True,
                highest_observed_healthy_rps=1.0,
                healthy_lower_bound_rps=1.0,
                unhealthy_upper_bound_rps=2.0,
                capacity_bound_state="bracketed_healthy_lower_unhealthy_upper",
                controller_completion_state="completed_confirmations_unhealthy",
                confirmation_healthy=[True, False, True],
                confirmation_all_healthy=False,
                recovery_run=True,
                recovery_healthy=True,
                recovery_eligible=True,
            ),
        ),
        _controller_event(
            "aimd_complete",
            aimd(
                "censored",
                controller_completion_state="confirmations_inconclusive",
                confirmation_healthy=[True, None, True],
                confirmation_eligible=[True, False, True],
                confirmation_censor_reasons=[
                    None,
                    "interrupted_epoch_incomplete_no_replay",
                    None,
                ],
                confirmation_complete=False,
                confirmation_all_healthy=None,
            ),
        ),
        _controller_event(
            "soak_complete",
            soak(
                "partial",
                completed_blocks=2,
                block_eligible=[True, True],
                block_healthy=[True, False],
                block_censor_reasons=[None, None],
                execution_complete=False,
                scientifically_complete=False,
                all_blocks_healthy=None,
                controller_completion_state="partial_incomplete",
            ),
        ),
        _controller_event(
            "soak_complete",
            soak(
                "censored",
                block_eligible=[True, False, True, True],
                block_healthy=[True, None, True, True],
                block_censor_reasons=[
                    None,
                    "interrupted_epoch_incomplete_no_replay",
                    None,
                    None,
                ],
                scientifically_complete=False,
                all_blocks_healthy=None,
                controller_completion_state="execution_complete_inconclusive",
            ),
        ),
        _controller_event("soak_complete", soak("healthy")),
        _controller_event("campaign_terminal", {"reason": "reservation_overrun_latch"}),
    ]
    rows = summarize_controller_events(events, public_config=config, coverage_rows=[])
    by_cell = {(row["suite"], row["shape"]): row for row in rows}
    assert by_cell[("aimd", "right")]["capacity_bound_state"].startswith("right_censored")
    assert by_cell[("aimd", "nonmonotonic")]["unhealthy_upper_bound_rps"] is None
    assert by_cell[("aimd", "unhealthy")]["unhealthy_upper_bound_rps"] == 2.0
    assert by_cell[("aimd", "censored")]["confirmation_all_healthy"] is None
    assert by_cell[("soak", "partial")]["execution_complete"] is False
    assert by_cell[("soak", "censored")]["scientifically_complete"] is False
    assert by_cell[("soak", "healthy")]["all_blocks_healthy"] is True
    assert by_cell[("aimd", "missing")]["controller_completion_state"] == (
        "campaign_censored_before_start"
    )
    assert by_cell[("aimd", "missing")]["censor_reason"] == "reservation_overrun_latch"


def _single_controller_config(suite: str) -> dict[str, object]:
    return {
        "routes": [{"id": "r"}],
        "suites": {
            "aimd": {"enabled": suite == "aimd", "shapes": ["cell"]},
            "soak": {
                "enabled": suite == "soak",
                "shapes": ["cell"],
                "blocks": 4,
                "rate_rps": 1.0,
            },
        },
    }


def _valid_aimd_controller_payload() -> dict[str, object]:
    return {
        "route_id": "r",
        "shape": "cell",
        "highest_observed_healthy_rps": 4.0,
        "healthy_lower_bound_rps": 4.0,
        "unhealthy_upper_bound_rps": None,
        "overload_observed": False,
        "nonmonotonic_overload_observed": False,
        "capacity_bound_state": "right_censored_highest_tested_healthy_no_overload",
        "controller_completion_state": "completed_confirmations_healthy",
        "censor_reason": None,
        "confirmations_required": 3,
        "confirmation_healthy": [True, True, True],
        "confirmation_eligible": [True, True, True],
        "confirmation_censor_reasons": [None, None, None],
        "confirmation_execution_complete": True,
        "confirmation_complete": True,
        "confirmation_all_healthy": True,
        "recovery_run": False,
        "recovery_healthy": None,
        "recovery_eligible": None,
        "recovery_censor_reason": None,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "confirmation_healthy": [True, False, True],
                "confirmation_all_healthy": False,
            },
            "completion state contradicts",
        ),
        (
            {
                "confirmations_required": 0,
                "confirmation_healthy": [],
                "confirmation_eligible": [],
                "confirmation_censor_reasons": [],
            },
            "exactly three confirmation",
        ),
        (
            {"highest_observed_healthy_rps": 100.0, "healthy_lower_bound_rps": 1.0},
            "highest healthy rate",
        ),
        (
            {
                "capacity_bound_state": "bracketed_healthy_lower_unhealthy_upper",
                "unhealthy_upper_bound_rps": 8.0,
                "overload_observed": True,
                "nonmonotonic_overload_observed": True,
            },
            "bracket bounds",
        ),
        (
            {
                "capacity_bound_state": "nonmonotonic_overload_no_current_bracket",
                "overload_observed": True,
                "nonmonotonic_overload_observed": False,
            },
            "nonmonotonic AIMD state",
        ),
        (
            {
                "capacity_bound_state": "left_censored_no_healthy_candidate",
                "controller_completion_state": "left_censored_no_healthy_candidate",
                "confirmation_healthy": [],
                "confirmation_eligible": [],
                "confirmation_censor_reasons": [],
                "confirmation_execution_complete": False,
                "confirmation_complete": False,
                "confirmation_all_healthy": None,
            },
            "left-censored AIMD state",
        ),
        ({"censor_reason": "cost_guard"}, "completion state contradicts"),
    ],
)
def test_controller_summary_rejects_self_declared_aimd_state_that_conflicts_with_evidence(
    updates: dict[str, object], message: str
) -> None:
    payload = _valid_aimd_controller_payload()
    payload.update(updates)
    events = [
        _controller_event("aimd_complete", payload),
        _controller_event("campaign_terminal", {"reason": "plan_completed"}),
    ]
    with pytest.raises(ValueError, match=message):
        summarize_controller_events(
            events,
            public_config=_single_controller_config("aimd"),
            coverage_rows=[],
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"block_healthy": [True, False, True, True], "all_blocks_healthy": False},
            "completion state contradicts",
        ),
        (
            {
                "completed_blocks": 2,
                "block_eligible": [True, True],
                "block_healthy": [True, True],
                "block_censor_reasons": [None, None],
                "execution_complete": False,
                "scientifically_complete": False,
                "all_blocks_healthy": None,
            },
            "completion state contradicts",
        ),
        (
            {
                "blocks": 0,
                "completed_blocks": 0,
                "block_eligible": [],
                "block_healthy": [],
                "block_censor_reasons": [],
            },
            "block count contradicts",
        ),
        ({"rate_rps": 2.0}, "tested rate contradicts"),
        (
            {
                "controller_completion_state": "campaign_guard_censored",
                "censor_reason": "cost_guard",
            },
            "lacks matching block evidence",
        ),
    ],
)
def test_controller_summary_rejects_self_declared_soak_state_that_conflicts_with_evidence(
    updates: dict[str, object], message: str
) -> None:
    payload: dict[str, object] = {
        "route_id": "r",
        "shape": "cell",
        "rate_rps": 1.0,
        "blocks": 4,
        "completed_blocks": 4,
        "block_eligible": [True, True, True, True],
        "block_healthy": [True, True, True, True],
        "block_censor_reasons": [None, None, None, None],
        "execution_complete": True,
        "scientifically_complete": True,
        "all_blocks_healthy": True,
        "controller_completion_state": "completed_healthy",
        "censor_reason": None,
    }
    payload.update(updates)
    events = [
        _controller_event("soak_complete", payload),
        _controller_event("campaign_terminal", {"reason": "plan_completed"}),
    ]
    with pytest.raises(ValueError, match=message):
        summarize_controller_events(
            events,
            public_config=_single_controller_config("soak"),
            coverage_rows=[],
        )


def test_request_latency_summary_is_explicitly_success_conditioned(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    for index, succeeds in enumerate((True, False)):
        spec = _spec(index)
        assert ledger.claim(
            request_id=f"latency-{index}",
            attempt_index=1,
            spec=spec,
            route=route,
            reserved_usd=0.001,
            max_cost_usd=10,
            cost_reserve_usd=1,
            scheduled_at_utc=None,
        )
        result = _result(index, 1.0)
        if not succeeds:
            result.status = "server_error"
            result.http_status = 503
            result.cost_basis = "reserved_upper_bound"
        ledger.finish(
            request_id=f"latency-{index}",
            result=result,
            validity=assess_result(result),
            quality_score=None,
            final_logical=True,
        )
    row = summarize_rows(ledger.rows())[0]
    assert row["logical_requests_n"] == 2
    assert row["latency_p50_n"] == 1
    assert row["arrival_latency_estimand"] == "successful_final_logical_outcomes_only"
    assert row["ttft_estimand"] == ("successful_final_logical_outcomes_with_observed_ttft_only")
    ledger.close()


def test_audit_preserves_valid_extreme_and_report_has_contract(
    tmp_path, route, monkeypatch
) -> None:
    monkeypatch.setattr(
        "inference_bench.report._report_source_snapshot",
        lambda run_dir: {
            "source_revision": "f" * 40,
            "source_tree_state": "clean",
            "source_dirty_tree_sha256": hashlib.sha256(b"").hexdigest(),
            "distributions": {"inference-endpoint-benchmark": "0.1.0"},
        },
    )
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    _attach_run_manifest(ledger)
    for index, seconds in enumerate([1, 1, 1, 1, 100]):
        spec = _spec(index)
        result = _result(index, seconds)
        if index == 0:
            result.finish_reason = "provider-controlled-private-text"
        ledger.claim(
            request_id=f"req-{index}",
            attempt_index=1,
            spec=spec,
            route=route,
            reserved_usd=0.001,
            max_cost_usd=10,
            cost_reserve_usd=1,
            scheduled_at_utc=None,
        )
        ledger.finish(
            request_id=f"req-{index}",
            result=result,
            validity=assess_result(result),
            quality_score=None,
            quality_diagnostics={},
        )
    rows = ledger.rows()
    audit = build_outlier_audit(rows)
    extreme = next(item for item in audit if item["request_id"] == "req-4")
    assert extreme["audit_class"] == "valid_extreme"
    assert extreme["excluded_estimands"] == []
    assert extreme["preserved"] is True
    assert extreme["warm_state"] == "uncontrolled_not_paired"
    summary = summarize_rows(rows)
    assert summary == summarize_rows(list(reversed(rows)))
    assert summary[0]["latency_p50_n"] == 5
    assert summary[0]["arrival_latency_estimand"] == ("successful_final_logical_outcomes_only")
    assert summary[0]["cache_read_reported_n"] == 0
    assert summary[0]["cache_read_unknown_n"] == 5
    assert summary[0]["cache_miss_n"] == 0
    assert summary[0]["warm_state"] == "uncontrolled_not_paired"
    ledger.record_event_once("campaign_terminal", "campaign_terminal", {"reason": "completed"})
    ledger.close()
    (tmp_path / "campaign.public.json").write_text("{}\n", encoding="utf-8")
    report = generate_report(tmp_path)
    assert report.exists()
    contract = json.loads((tmp_path / "report" / "metric-contract.json").read_text())
    assert contract["sse_event_span"]["eligible_for_token_rate"] is False
    assert contract["latency"]["sampling_population"] == ("successful_final_logical_outcomes_only")
    assert "no warm- or cold-endpoint latency claim" in contract["warm_state"]["measured_cells"]
    assert (tmp_path / "report" / "outlier-audit.jsonl").exists()
    assert (tmp_path / "report" / "reproducibility-manifest.json").exists()
    summary_csv = (tmp_path / "report" / "matched-cell-summary.csv").read_text(encoding="utf-8")
    assert "provider-controlled-private-text" not in summary_csv
    assert '""other""' in summary_csv
    assert "provider-controlled-private-text" not in (tmp_path / "events.jsonl").read_text(
        encoding="utf-8"
    )

    stale = tmp_path / "report" / "stale-from-prior-render.txt"
    stale.write_text("must disappear", encoding="utf-8")
    generate_report(tmp_path)
    assert not stale.exists()

    public_config = tmp_path / "campaign.public.json"
    public_config.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="exact canonical projection"):
        generate_report(tmp_path)
    public_config.write_text('{"tampered":true,"tampered":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or invalid JSON"):
        generate_report(tmp_path)
    public_config.write_text("{}\n", encoding="utf-8")

    event_path = tmp_path / "events.jsonl"
    canonical_events = event_path.read_text(encoding="utf-8")
    duplicate_event = canonical_events.splitlines()[0].replace(
        '"event_id":', '"event_id":999,"event_id":', 1
    )
    event_path.write_text(
        duplicate_event + "\n" + "\n".join(canonical_events.splitlines()[1:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid or duplicate-key JSON"):
        generate_report(tmp_path)
    event_path.write_text(canonical_events, encoding="utf-8")

    projected = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    projected[0]["payload_json"] = '{"tampered":true}'
    event_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in projected),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact complete ordered projection"):
        generate_report(tmp_path)

    restored = Ledger(tmp_path)
    restored.rebuild_events_jsonl()
    restored.close()
    event_path.unlink()
    with pytest.raises(ValueError, match="durable events.jsonl"):
        generate_report(tmp_path)


def test_report_holds_exclusive_campaign_lease(tmp_path, monkeypatch) -> None:
    snapshot = {
        "source_revision": "f" * 40,
        "source_tree_state": "clean",
        "source_dirty_tree_sha256": hashlib.sha256(b"").hexdigest(),
        "distributions": {"inference-endpoint-benchmark": "0.1.0"},
    }
    monkeypatch.setattr("inference_bench.report._report_source_snapshot", lambda run_dir: snapshot)
    _empty_terminal_run(tmp_path)
    live_owner = Ledger(tmp_path, exclusive_owner=True)
    try:
        with pytest.raises(RuntimeError, match="owned by another live campaign"):
            generate_report(tmp_path)
    finally:
        live_owner.close()


def test_report_rejects_source_transition_during_derivation(tmp_path, monkeypatch) -> None:
    first = {
        "source_revision": "a" * 40,
        "source_tree_state": "clean",
        "source_dirty_tree_sha256": hashlib.sha256(b"").hexdigest(),
        "distributions": {"inference-endpoint-benchmark": "0.1.0"},
    }
    changed = {**first, "source_revision": "b" * 40}
    calls = 0

    def transitioning_snapshot(run_dir):
        nonlocal calls
        calls += 1
        return first if calls <= 2 else changed

    monkeypatch.setattr("inference_bench.report._report_source_snapshot", transitioning_snapshot)
    _empty_terminal_run(tmp_path)
    with pytest.raises(ValueError, match="changed during derivation"):
        generate_report(tmp_path)
    assert not (tmp_path / "report").exists()


def test_outlier_schema_requires_preservation() -> None:
    schema = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1]
            / "schemas"
            / "outlier-audit.schema.json"
        ).read_text()
    )
    assert schema["properties"]["preserved"] == {"const": True}
    assert schema["properties"]["warm_state"] == {
        "enum": ["standalone_diagnostic_only", "uncontrolled_not_paired"]
    }


def test_binary_quality_uses_non_degenerate_small_sample_wilson_intervals() -> None:
    all_success = _binary_quality_interval([1.0, 1.0, 1.0])
    all_failure = _binary_quality_interval([0.0, 0.0, 0.0])
    assert all_success.estimate == 1
    assert all_success.lower_95 is not None and all_success.lower_95 < 1
    assert all_success.upper_95 == 1
    assert all_failure.estimate == 0
    assert all_failure.lower_95 == 0
    assert all_failure.upper_95 is not None and all_failure.upper_95 > 0
    assert all_success.method == all_failure.method == "Wilson-95"


def test_load_summary_does_not_trust_event_token_totals_without_request_ledger() -> None:
    events = []
    for index in range(4):
        payload = {
            "epoch_id": f"epoch-{index}",
            "route_id": "r",
            "shape": "short_short",
            "phase": "soak_block",
            "offered_rps": 1.0,
            "duration_seconds": 30,
            "scheduled": 30,
            "completed": 30,
            "successful": 30,
            "healthy": True,
            "successful_input_tokens": 3_000,
            "successful_output_tokens": 600,
        }
        events.append({"kind": "load_epoch", "payload_json": json.dumps(payload)})
    row = summarize_load_events(events)[0]
    assert summarize_load_events(events) == summarize_load_events(list(reversed(events)))
    assert row["successful_rpm"] == 60
    assert row["successful_input_tpm"] is None
    assert row["successful_output_tpm"] is None
    assert row["successful_output_tpm_n"] == 0
    assert row["tpm_reporting_state"] == "censored_no_complete_usage_block"
    assert row["ttft_p95_across_blocks_ci_method"] == "epoch/block-bootstrap-percentile"


def test_load_summary_excludes_campaign_censored_blocks_from_capacity_estimands() -> None:
    base = {
        "route_id": "r",
        "shape": "short_short",
        "phase": "soak_block",
        "offered_rps": 1.0,
        "duration_seconds": 30,
        "actual_elapsed_seconds": 30,
        "scheduled": 30,
        "launched_logical": 30,
        "completed": 30,
        "successful": 30,
        "physical_attempts": 30,
        "physical_successes": 30,
        "healthy": True,
        "launch_guard_triggered": False,
        "launch_guard_reason": None,
        "controller_eligible": True,
        "scientific_censor_reason": None,
    }
    events = [
        {"kind": "load_epoch", "payload_json": json.dumps({**base, "epoch_id": "good"})},
        {
            "kind": "load_epoch",
            "payload_json": json.dumps(
                {
                    **base,
                    "epoch_id": "guarded",
                    "launched_logical": 0,
                    "completed": 0,
                    "successful": 0,
                    "physical_attempts": 0,
                    "physical_successes": 0,
                    "healthy": False,
                    "launch_guard_triggered": True,
                    "launch_guard_reason": "cost_guard",
                }
            ),
        },
        {
            "kind": "load_epoch",
            "payload_json": json.dumps(
                {
                    **base,
                    "epoch_id": "interrupted",
                    "launched_logical": 1,
                    "completed": 1,
                    "successful": 1,
                    "physical_attempts": 1,
                    "physical_successes": 1,
                    "healthy": False,
                    "controller_eligible": False,
                    "scientific_censor_reason": "interrupted_epoch_incomplete_no_replay",
                }
            ),
        },
    ]
    row = summarize_load_events(events)[0]
    assert row["blocks_n"] == 3
    assert row["capacity_estimand_blocks_n"] == 1
    assert row["censored_blocks_n"] == 2
    assert row["successful_rpm"] == pytest.approx(60)
    assert row["successful_rpm_n"] == 1
    assert row["requests_successful_n"] == 30
    assert row["observed_requests_successful_n"] == 31
    assert json.loads(row["censored_block_reasons_json"]) == {
        "cost_guard": 1,
        "interrupted_epoch_incomplete_no_replay": 1,
    }


def test_plots_require_a_matched_multi_route_cell(tmp_path) -> None:
    base = {
        "suite": "latency",
        "cell_id": "short_short",
        "cache_state": "uncontrolled",
        "reasoning_token_state": "reported_zero",
        "ttft_p50": 0.2,
        "ttft_p50_ci95_low": 0.1,
        "ttft_p50_ci95_high": 0.3,
        "ttft_p50_n": 20,
        "latency_p50": None,
        "decode_proxy_tps_p50": None,
        "success_rate": None,
    }
    assert _plot_matched_cells([{**base, "route_id": "only"}], tmp_path) == []
    created = _plot_matched_cells(
        [
            {**base, "route_id": "route-a"},
            {
                **base,
                "route_id": "route-b",
                "ttft_p50": 20.0,
                "ttft_p50_ci95_low": 10.0,
                "ttft_p50_ci95_high": 30.0,
            },
        ],
        tmp_path,
    )
    assert len(created) == 1
    assert (tmp_path / created[0]).stat().st_size > 0

    load_rows = [
        {**base, "suite": "load", "route_id": "route-a"},
        {**base, "suite": "load", "route_id": "route-b"},
    ]
    assert _plot_matched_cells(load_rows, tmp_path / "load-request-plots") == []
