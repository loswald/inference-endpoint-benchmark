from __future__ import annotations

from inference_bench.models import InferenceResult
from inference_bench.validity import assess_result, decode_proxy_tokens_per_second


def _result(**updates):
    values = dict(
        logical_id="x",
        status="success",
        http_status=200,
        started_at_utc="2026-01-01T00:00:00Z",
        ended_at_utc="2026-01-01T00:00:02Z",
        total_seconds=2.0,
        time_to_headers_seconds=0.1,
        ttft_seconds=0.5,
        decode_seconds=1.5,
        output_event_offsets_seconds=(0.5,),
        input_tokens=100,
        output_tokens=30,
    )
    values.update(updates)
    return InferenceResult(**values)


def test_single_sse_event_is_retained_but_quarantined_from_decode_headline() -> None:
    result = _result()
    assessment = assess_result(result)
    assert assessment.classification == "censored"
    assert result.content_event_count == 1
    assert "decode_proxy_insufficient_content_events" in assessment.reasons
    assert decode_proxy_tokens_per_second(result) is None


def test_near_zero_proxy_with_multiple_tokens_is_invalid() -> None:
    assessment = assess_result(
        _result(
            total_seconds=0.50001,
            ttft_seconds=0.5,
            output_tokens=30,
            output_event_offsets_seconds=(0.5, 0.500005),
        )
    )
    assert assessment.classification == "invalid"
    assert "decode_proxy_near_zero_with_multiple_tokens" in assessment.reasons
    assert not assessment.decode_eligible


def test_missing_usage_censors_tpm_but_keeps_latency() -> None:
    assessment = assess_result(_result(input_tokens=None, output_tokens=None))
    assert assessment.classification == "censored"
    assert assessment.latency_eligible
    assert not assessment.usage_eligible


def test_crossed_timing_is_invalid() -> None:
    assessment = assess_result(_result(time_to_headers_seconds=0.8, ttft_seconds=0.5))
    assert assessment.classification == "invalid"
    assert "headers_after_first_token" in assessment.reasons


def test_extreme_billed_token_proxy_is_preserved_but_not_headline_eligible() -> None:
    result = _result(
        total_seconds=0.502,
        ttft_seconds=0.5,
        output_tokens=30,
        output_event_offsets_seconds=(0.5, 0.501),
    )
    assessment = assess_result(result)
    # The duration gate catches the physically unresolved interval before it can become a rate.
    assert assessment.classification == "invalid"
    assert not assessment.decode_eligible


def test_resolved_but_suspicious_proxy_is_anomalous_and_quarantined() -> None:
    result = _result(
        total_seconds=0.52,
        ttft_seconds=0.5,
        decode_seconds=0.02,
        output_tokens=250,
        output_event_offsets_seconds=(0.5, 0.51),
    )
    assessment = assess_result(result)
    assert assessment.classification == "anomalous"
    assert "decode_proxy_extreme_tokens_per_second" in assessment.reasons
    assert not assessment.decode_eligible


def test_expected_validation_4xx_is_classified_as_observed_rejection() -> None:
    result = _result(
        status="client_error",
        http_status=400,
        ttft_seconds=None,
        decode_seconds=None,
        output_event_offsets_seconds=(),
        input_tokens=None,
        output_tokens=None,
    )
    assessment = assess_result(result, expected_rejection=True)
    assert assessment.classification == "valid"
    assert assessment.latency_eligible
    assert not assessment.usage_eligible
