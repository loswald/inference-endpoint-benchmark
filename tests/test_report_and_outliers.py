from __future__ import annotations

import json

from inference_bench.ledger import Ledger
from inference_bench.models import InferenceResult, RequestSpec
from inference_bench.report import (
    _plot_matched_cells,
    build_outlier_audit,
    generate_report,
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
        cost_usd=0.00003,
        cost_basis="provider_usage",
    )


def test_audit_preserves_valid_extreme_and_report_has_contract(tmp_path, route) -> None:
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="a" * 64, config_json="{}")
    for index, seconds in enumerate([1, 1, 1, 1, 100]):
        spec = _spec(index)
        result = _result(index, seconds)
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
    summary = summarize_rows(rows)
    assert summary[0]["latency_p50_n"] == 5
    assert summary[0]["cache_read_reported_n"] == 0
    assert summary[0]["cache_read_unknown_n"] == 5
    assert summary[0]["cache_miss_n"] == 0
    ledger.close()
    report = generate_report(tmp_path)
    assert report.exists()
    contract = json.loads((tmp_path / "report" / "metric-contract.json").read_text())
    assert contract["sse_event_span"]["eligible_for_token_rate"] is False
    assert (tmp_path / "report" / "outlier-audit.jsonl").exists()
    assert (tmp_path / "report" / "reproducibility-manifest.json").exists()


def test_outlier_schema_requires_preservation() -> None:
    schema = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1]
            / "schemas"
            / "outlier-audit.schema.json"
        ).read_text()
    )
    assert schema["properties"]["preserved"] == {"const": True}


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
    assert row["successful_rpm"] == 60
    assert row["successful_input_tpm"] is None
    assert row["successful_output_tpm"] is None
    assert row["successful_output_tpm_n"] == 0
    assert row["tpm_reporting_state"] == "censored_no_complete_usage_block"


def test_plots_require_a_matched_multi_route_cell(tmp_path) -> None:
    base = {
        "suite": "latency",
        "cell_id": "short_short",
        "cache_state": "uncontrolled",
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
