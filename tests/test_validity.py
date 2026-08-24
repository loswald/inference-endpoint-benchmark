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


def test_single_sse_event_never_becomes_event_span_tps() -> None:
    result = _result()
    assessment = assess_result(result)
    assert assessment.classification == "valid"
    assert result.content_event_count == 1
    assert decode_proxy_tokens_per_second(result) == 20.0


def test_near_zero_proxy_with_multiple_tokens_is_invalid() -> None:
    assessment = assess_result(_result(total_seconds=0.50001, ttft_seconds=0.5, output_tokens=3))
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
