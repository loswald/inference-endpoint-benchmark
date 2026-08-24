from __future__ import annotations

import math

from .models import InferenceResult, ValidityAssessment

MIN_DECODE_PROXY_SECONDS = 1e-2
MIN_DECODE_PROXY_TOKENS = 8
MIN_DECODE_PROXY_CONTENT_EVENTS = 2
EXTREME_DECODE_PROXY_TOKENS_PER_SECOND = 10_000.0


def assess_result(
    result: InferenceResult, *, expected_rejection: bool = False
) -> ValidityAssessment:
    """Classify one result without deleting or mutating the observation.

    Eligibility is metric-specific. For example, a success with missing usage can still support
    end-to-end latency but cannot support TPM, decode rate, or cost-per-token.
    """

    invalid: list[str] = []
    censored: list[str] = []
    anomalous: list[str] = []
    informational: list[str] = []

    numeric = {
        "total_seconds": result.total_seconds,
        "time_to_headers_seconds": result.time_to_headers_seconds,
        "ttft_seconds": result.ttft_seconds,
        "decode_seconds": result.decode_seconds,
        "queue_delay_seconds": result.queue_delay_seconds,
    }
    for name, value in numeric.items():
        if value is not None and (not math.isfinite(value) or value < 0):
            invalid.append(f"{name}_invalid_seconds")
    if result.total_seconds <= 0:
        invalid.append("total_seconds_nonpositive")
    for name in ("time_to_headers_seconds", "ttft_seconds", "decode_seconds"):
        value = getattr(result, name)
        if value is not None and value > result.total_seconds + 1e-6:
            invalid.append(f"{name}_exceeds_total")
    if (
        result.time_to_headers_seconds is not None
        and result.ttft_seconds is not None
        and result.time_to_headers_seconds > result.ttft_seconds + 1e-6
    ):
        invalid.append("headers_after_first_token")

    offsets = result.output_event_offsets_seconds
    if any(not math.isfinite(value) or value < 0 for value in offsets):
        invalid.append("stream_event_offset_invalid_seconds")
    if any(right < left for left, right in zip(offsets, offsets[1:], strict=False)):
        invalid.append("stream_event_offsets_nonmonotonic")
    if offsets and offsets[-1] > result.total_seconds + 1e-6:
        invalid.append("stream_event_after_request_end")

    for name in ("input_tokens", "output_tokens"):
        value = getattr(result, name)
        if value is not None and (isinstance(value, bool) or value < 0):
            invalid.append(f"{name}_invalid_count")
    if result.cache_read_input_tokens is not None:
        if result.cache_read_input_tokens < 0:
            invalid.append("cache_read_input_tokens_invalid_count")
        elif (
            result.input_tokens is not None and result.cache_read_input_tokens > result.input_tokens
        ):
            invalid.append("cache_read_input_tokens_exceeds_input_tokens")

    success = result.status == "success"
    expected_4xx = bool(
        expected_rejection
        and result.http_status is not None
        and 400 <= result.http_status < 500
        and result.http_status not in {402, 429}
    )
    if expected_4xx:
        informational.append("expected_validation_rejection_observed")
    if not success and not expected_4xx:
        censored.append(f"request_{result.status}")
    if success and not result.usage_complete:
        censored.append("provider_usage_missing")
    if success and result.ttft_seconds is None:
        censored.append("first_output_event_missing")

    proxy_duration: float | None = None
    if result.ttft_seconds is not None:
        proxy_duration = result.total_seconds - result.ttft_seconds
    if success and (result.output_tokens or 0) >= MIN_DECODE_PROXY_TOKENS:
        if proxy_duration is None:
            censored.append("decode_proxy_missing_ttft")
        elif result.content_event_count < MIN_DECODE_PROXY_CONTENT_EVENTS:
            censored.append("decode_proxy_insufficient_content_events")
        elif proxy_duration < MIN_DECODE_PROXY_SECONDS:
            invalid.append("decode_proxy_near_zero_with_multiple_tokens")
        elif result.output_tokens is not None:
            proxy_tps = result.output_tokens / proxy_duration
            if proxy_tps > EXTREME_DECODE_PROXY_TOKENS_PER_SECOND:
                anomalous.append("decode_proxy_extreme_tokens_per_second")
    elif success:
        censored.append("decode_proxy_requires_meaningful_output_tokens")

    # SSE event spans are retained for transport diagnostics only. Event count does not affect the
    # request-minus-TTFT proxy: fewer than two events simply makes the unused event-span undefined.

    if invalid:
        classification = "invalid"
    elif censored:
        classification = "censored"
    elif anomalous:
        classification = "anomalous"
    else:
        classification = "valid"

    latency_eligible = (success or expected_4xx) and not any(
        reason
        for reason in invalid
        if reason.startswith(("total_", "ttft_", "headers_", "time_to_headers_"))
    )
    usage_eligible = (
        success
        and result.usage_complete
        and not any(
            "tokens_invalid" in reason or "token" in reason and "extreme" not in reason
            for reason in invalid
        )
    )
    decode_eligible = (
        latency_eligible
        and usage_eligible
        and (result.output_tokens or 0) >= MIN_DECODE_PROXY_TOKENS
        and result.content_event_count >= MIN_DECODE_PROXY_CONTENT_EVENTS
        and proxy_duration is not None
        and proxy_duration >= MIN_DECODE_PROXY_SECONDS
        and "decode_proxy_near_zero_with_multiple_tokens" not in invalid
        and "decode_proxy_extreme_tokens_per_second" not in anomalous
    )
    quality_eligible = success and not invalid
    return ValidityAssessment(
        classification=classification,  # type: ignore[arg-type]
        reasons=tuple(dict.fromkeys([*invalid, *censored, *anomalous, *informational])),
        latency_eligible=latency_eligible,
        usage_eligible=usage_eligible,
        decode_eligible=decode_eligible,
        quality_eligible=quality_eligible,
    )


def decode_proxy_tokens_per_second(result: InferenceResult) -> float | None:
    """Comparable billed-token proxy: completion_tokens / (request_seconds - TTFT).

    This is client-observed and includes transport/drain overhead. It is not direct server decode
    compute. Event-span timing is deliberately not used because SSE events can batch many tokens.
    """

    assessment = assess_result(result)
    if (
        not assessment.decode_eligible
        or result.ttft_seconds is None
        or result.output_tokens is None
    ):
        return None
    return result.output_tokens / (result.total_seconds - result.ttft_seconds)
