from __future__ import annotations

import hashlib
import json

import pytest

from inference_bench.config import CampaignConfig
from inference_bench.ledger import Ledger
from inference_bench.models import AuthConfig, InferenceResult, RequestSpec, RouteConfig
from inference_bench.report import generate_report, summarize_load_events, summarize_rows
from inference_bench.statistics import block_rate_interval
from inference_bench.validity import assess_result


def _sensitive_campaign(default_value: str) -> CampaignConfig:
    route = RouteConfig(
        id="route-public",
        provider="test",
        adapter="openai_compatible",
        model="model-public",
        base_url="https://example.invalid/v1/chat?account=private",
        auth=AuthConfig(
            env="TEST_API_KEY",
            header="X-Private-Authentication-Header",
            prefix="PrivateScheme hidden-value ",
        ),
        context_tokens=8_192,
        max_output_tokens=2_048,
        stream_usage_mode="try",
        input_usd_per_million=1,
        output_usd_per_million=2,
        documentation_source_url="https://example.invalid/documentation",
        pricing_source_url="https://example.invalid/pricing",
        evidence_retrieved_at_utc="2026-08-24T00:00:00Z",
        evidence_bundle_sha256="a" * 64,
        capabilities={
            "streaming": True,
            "documentation_checked_utc": "2026-08-24",
            "private_operator_note": "never-publish-capability-note",
        },
        extra_headers={"X-Internal-Account": "never-publish-header-value"},
        request_defaults={
            "user": default_value,
        },
    )
    return CampaignConfig(
        name="public-test",
        seed=7,
        max_wall_seconds=600,
        max_cost_usd=10,
        launch_reserve_seconds=60,
        launch_reserve_usd=1,
        concurrency=4,
        retries=1,
        routes=(route,),
        client_location="test-client",
        suites={
            "latency": {
                "enabled": True,
                "shapes": ["short_short"],
            },
        },
    )


def test_public_config_is_allowlisted_without_collapsing_identity() -> None:
    first = _sensitive_campaign("never-publish-default-one")
    second = _sensitive_campaign("never-publish-default-two")
    public = first.public_dict()
    encoded = json.dumps(public, sort_keys=True)

    for sensitive in (
        "password",
        "account=private",
        "hidden-value",
        "never-publish",
    ):
        assert sensitive not in encoded
    route = public["routes"][0]
    assert route["base_url"] == "https://example.invalid/v1/chat"
    assert route["auth"] == {"env": "TEST_API_KEY"}
    assert route["identity_hash"] == first.routes[0].identity_hash
    assert public["suites"]["latency"] == {
        "enabled": True,
        "shapes": ["short_short"],
    }
    assert first.identity_hash != second.identity_hash


def _epoch(epoch_id: str, *, usage_successes: int = 2) -> dict[str, object]:
    return {
        "epoch_id": epoch_id,
        "route_id": "r",
        "shape": "short_short",
        "phase": "soak_block",
        "offered_rps": 1.0,
        "duration_seconds": 30.0,
        "actual_elapsed_seconds": 60.0,
        "queue_end_seconds": 30.0,
        "scheduled": 30,
        "launched_logical": 2,
        "completed": 2,
        "physical_attempts": 3,
        "physical_successes": 2,
        "successful": 2,
        "healthy": True,
        "successful_input_tokens": 200 if usage_successes == 2 else None,
        "successful_output_tokens": 40 if usage_successes == 2 else None,
        "usage_complete_successful": usage_successes,
    }


def _epoch_rows(epoch_id: str, *, complete: bool = True) -> list[dict[str, object]]:
    return [
        {
            "logical_id": f"load:r:short_short:{epoch_id}:{index}",
            "attempt_index": 1,
            "state": "terminal",
            "status": "success",
            "usage_eligible": int(complete or index == 0),
            "input_tokens": 100 if complete or index == 0 else None,
            "output_tokens": 20 if complete or index == 0 else None,
        }
        for index in range(2)
    ]


def test_load_report_uses_drain_time_block_cis_and_verified_usage() -> None:
    events: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []
    for index in range(4):
        epoch_id = f"epoch-{index}"
        events.append({"kind": "load_epoch", "payload_json": json.dumps(_epoch(epoch_id))})
        rows.extend(_epoch_rows(epoch_id))

    result = summarize_load_events(events, rows=rows)[0]
    assert result["offered_rpm"] == 60
    assert result["completed_rpm"] == 2
    assert result["successful_rpm"] == 2
    assert result["physical_attempt_rpm"] == 3
    assert result["successful_input_tpm"] == 200
    assert result["successful_output_tpm"] == 40
    assert result["arrival_window_seconds_sum"] == 120
    assert result["elapsed_wall_seconds_sum"] == 240
    assert result["success_rate_n"] == 4
    assert result["success_rate_ci_method"] == "epoch/block-bootstrap-ratio-of-sums"
    assert result["success_rate"] == pytest.approx(2 / 30)
    assert result["completion_rate"] == pytest.approx(2 / 30)
    assert result["physical_attempt_success_rate"] == pytest.approx(2 / 3)
    assert result["tpm_reporting_state"] == "complete"


def test_achieved_rate_denominator_never_undercuts_arrival_window() -> None:
    epoch = {
        **_epoch("quiet-tail"),
        "offered_rps": 0.1,
        "scheduled": 3,
        "launched_logical": 3,
        "completed": 3,
        "successful": 3,
        "physical_attempts": 3,
        "physical_successes": 3,
        "actual_elapsed_seconds": 10,
        "queue_end_seconds": 0,
    }
    row = summarize_load_events([{"kind": "load_epoch", "payload_json": json.dumps(epoch)}])[0]
    assert row["offered_rpm"] == pytest.approx(6)
    assert row["completed_rpm"] == pytest.approx(6)
    assert row["successful_rpm"] == pytest.approx(6)
    assert row["raw_runner_elapsed_seconds_sum"] == pytest.approx(10)
    assert row["elapsed_wall_seconds_sum"] == pytest.approx(30)
    assert row["early_termination_seconds_sum"] == pytest.approx(20)


def test_missing_success_usage_censors_that_block_instead_of_counting_zero() -> None:
    complete_id = "complete"
    censored_id = "censored"
    events = [
        {"kind": "load_epoch", "payload_json": json.dumps(_epoch(complete_id))},
        {
            "kind": "load_epoch",
            "payload_json": json.dumps(_epoch(censored_id, usage_successes=1)),
        },
    ]
    rows = [*_epoch_rows(complete_id), *_epoch_rows(censored_id, complete=False)]
    result = summarize_load_events(events, rows=rows)[0]
    assert result["tpm_complete_blocks_n"] == 1
    assert result["tpm_censored_blocks_n"] == 1
    assert result["successful_input_tpm"] == 200
    assert result["successful_input_tpm_n"] == 1
    assert result["tpm_reporting_state"] == "partial_complete_blocks_only"

    no_ledger_result = summarize_load_events(events)[0]
    assert no_ledger_result["successful_input_tpm"] is None
    assert no_ledger_result["tpm_reporting_state"] == "censored_no_complete_usage_block"


def test_block_rate_is_ratio_of_sums_for_unequal_elapsed_blocks() -> None:
    estimate = block_rate_interval([60, 0], [60, 120], unit_name="requests", seed=4)
    assert estimate.estimate == 20
    assert estimate.n == 2
    assert estimate.method == "epoch/block-bootstrap-ratio-of-sums"


def test_cache_report_distinguishes_unknown_explicit_miss_and_hit() -> None:
    base = {
        "state": "terminal",
        "route_id": "r",
        "suite": "cache",
        "cell_id": "same-prefix",
        "cache_state": "cached_trial",
        "status": "success",
        "total_seconds": 1.0,
        "latency_eligible": 1,
        "ttft_seconds": 0.2,
        "decode_eligible": 1,
        "output_tokens": 10,
        "quality_score": None,
        "quality_eligible": 0,
        "usage_eligible": 1,
        "input_tokens": 20,
        "validity_class": "valid",
        "settled_usd": 0.001,
        "cost_basis": "provider_usage",
        "http_status": 200,
        "finish_reason": "stop",
    }
    row = summarize_rows(
        [
            {
                **base,
                "logical_id": "cache-unknown",
                "attempt_index": 1,
                "cache_read_input_tokens": None,
            },
            {
                **base,
                "logical_id": "cache-miss",
                "attempt_index": 1,
                "cache_read_input_tokens": 0,
            },
            {
                **base,
                "logical_id": "cache-hit",
                "attempt_index": 1,
                "cache_read_input_tokens": 8,
            },
        ]
    )[0]
    assert row["cache_read_reported_n"] == 2
    assert row["cache_read_unknown_n"] == 1
    assert row["cache_miss_n"] == 1
    assert row["cache_hit_n"] == 1
    assert row["cache_read_tokens_sum"] == 8


def test_report_emits_hash_bound_manifest_without_claiming_publication(
    tmp_path, route, monkeypatch
) -> None:
    monkeypatch.setattr(
        "inference_bench.report._report_source_snapshot",
        lambda run_dir: {
            "source_revision": "a" * 40,
            "source_tree_state": "clean",
            "source_dirty_tree_sha256": hashlib.sha256(b"").hexdigest(),
            "distributions": {"inference-endpoint-benchmark": "0.1.0"},
        },
    )
    ledger = Ledger(tmp_path)
    ledger.initialize(campaign_hash="b" * 64, config_json='{"sanitized":true}')
    manifest_json = json.dumps(
        {
            "schema_version": "run-manifest/v2",
            "normalized_exact_invocation": ["inference-bench", "run", "<CONFIG_OR_PATH>"],
            "raw_invocation_sha256": "c" * 64,
            "client_location": "test-fixture",
            "connection_reuse_by_route": {route.id: True},
            "http2_by_route": {route.id: False},
            "transport_max_connections_by_route": {route.id: 256},
            "transport_header_profile_by_route": {
                route.id: "openai-json-accept-encoding-identity/v1"
            },
            "request_timeout_seconds_by_route": {route.id: route.request_timeout_seconds},
            "provider_documentation_declarations": [
                {
                    "route_id": route.id,
                    "documentation_source_url": route.documentation_source_url,
                    "pricing_source_url": route.pricing_source_url,
                    "evidence_retrieved_at_utc": route.evidence_retrieved_at_utc,
                    "declared_evidence_bundle_sha256": route.evidence_bundle_sha256,
                    "verification_status": "declared_unverified_by_harness",
                }
            ],
            "transport_trust_env": False,
            "source_commit": "d" * 40,
            "source_dirty": False,
            "source_dirty_tree_sha256": "e" * 64,
            "dependency_lock_sha256": "f" * 64,
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
    spec = RequestSpec(
        logical_id="manifest-request",
        route_id=route.id,
        suite="latency",
        cell_id="short_short",
        messages=({"role": "user", "content": "not persisted"},),
        planned_input_tokens=10,
        max_output_tokens=10,
    )
    result = InferenceResult(
        logical_id=spec.logical_id,
        status="success",
        http_status=200,
        started_at_utc="2026-01-01T00:00:00Z",
        ended_at_utc="2026-01-01T00:00:01Z",
        total_seconds=1,
        time_to_headers_seconds=0.1,
        ttft_seconds=0.2,
        output_event_offsets_seconds=(0.2, 0.9),
        input_tokens=10,
        output_tokens=10,
        cost_usd=0.00003,
        cost_basis="provider_usage",
        reasoning_tokens=0,
        arrival_to_completion_seconds=1,
    )
    assert ledger.claim(
        request_id="manifest-request-1",
        attempt_index=1,
        spec=spec,
        route=route,
        reserved_usd=0.001,
        max_cost_usd=10,
        cost_reserve_usd=1,
        scheduled_at_utc=None,
    )
    ledger.finish(
        request_id="manifest-request-1",
        result=result,
        validity=assess_result(result),
        quality_score=None,
        quality_diagnostics={},
    )
    ledger.record_event_once("campaign_terminal", "campaign_terminal", {"reason": "completed"})
    ledger.close()
    (tmp_path / "campaign.public.json").write_text('{"sanitized":true}\n', encoding="utf-8")

    report = generate_report(tmp_path)
    manifest_path = tmp_path / "report" / "reproducibility-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    ledger_bytes = (tmp_path / "ledger.sqlite3").read_bytes()
    assert artifacts["ledger.sqlite3"]["sha256"] == hashlib.sha256(ledger_bytes).hexdigest()
    assert manifest["campaign"]["identity_hash"] == "b" * 64
    assert manifest["software"]["run_environment"]["python"] == "3.12.0"
    assert (
        manifest["provider_documentation_declarations"][0]["declared_evidence_bundle_sha256"]
        == "a" * 64
    )
    assert manifest["campaign"]["ended_at_utc"].endswith("Z")
    assert (
        "external evidence bundle verification"
        in manifest["release_status"]["documentation_evidence_gate"]
    )
    assert "report_generator" in manifest["software"]
    assert manifest["release_status"]["publication_gate"] == "not_implemented"
    assert manifest["release_status"]["pdf_generated"] is False
    assert "not a publication approval or PDF" in report.read_text(encoding="utf-8")
