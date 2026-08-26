from __future__ import annotations

import math

from .models import (
    InferenceResult,
    ValidityAssessment,
    normalize_arrival_latency_censor_reason,
    normalize_result_status,
    normalize_usage_parse_errors,
)

# A sub-second post-TTFT interval is dominated by buffering, batching, and clock-denominator
# instability. Preserve its raw timings, but do not turn it into a public tokens/second claim.
MIN_DECODE_PROXY_SECONDS = 1.0
MIN_PHYSICALLY_RESOLVED_DECODE_PROXY_SECONDS = 1e-2
MIN_DECODE_PROXY_TOKENS = 8
MIN_DECODE_PROXY_CONTENT_EVENTS = 2
EXTREME_DECODE_PROXY_TOKENS_PER_SECOND = 10_000.0
VALIDATION_REJECTION_HTTP_STATUSES = frozenset({400, 413, 422})


def assess_result(
    result: InferenceResult,
    *,
    expected_rejection: bool = False,
    parameter_acceptance_only: bool = False,
    quality_scored: bool = False,
) -> ValidityAssessment:
    """Classify one result without deleting or mutating the observation.

    Eligibility is metric-specific. For example, a success with missing usage can still support
    end-to-end latency but cannot support TPM, decode rate, or cost-per-token.
    """

    invalid: list[str] = list(result.adapter_contract_errors)
    censored: list[str] = []
    anomalous: list[str] = []
    informational: list[str] = []

    numeric = {
        "total_seconds": result.total_seconds,
        "time_to_headers_seconds": result.time_to_headers_seconds,
        "ttft_seconds": result.ttft_seconds,
        "decode_seconds": result.decode_seconds,
        "queue_delay_seconds": result.queue_delay_seconds,
        "arrival_to_completion_seconds": result.arrival_to_completion_seconds,
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
    if result.arrival_to_completion_seconds is not None and (
        result.arrival_to_completion_seconds + 1e-6 < result.total_seconds
        or result.arrival_to_completion_seconds + 1e-6 < result.queue_delay_seconds
    ):
        invalid.append("arrival_to_completion_precedes_component_duration")

    offsets = result.output_event_offsets_seconds
    if any(not math.isfinite(value) or value < 0 for value in offsets):
        invalid.append("stream_event_offset_invalid_seconds")
    if any(right < left for left, right in zip(offsets, offsets[1:], strict=False)):
        invalid.append("stream_event_offsets_nonmonotonic")
    if offsets and offsets[-1] > result.total_seconds + 1e-6:
        invalid.append("stream_event_after_request_end")

    for name in ("input_tokens", "output_tokens"):
        value = getattr(result, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            invalid.append(f"{name}_invalid_count")
    usage_parse_errors = normalize_usage_parse_errors(result.usage_parse_errors)
    if usage_parse_errors:
        invalid.extend(f"usage_parse_error:{reason}" for reason in usage_parse_errors)
    if result.reasoning_tokens is not None:
        if (
            isinstance(result.reasoning_tokens, bool)
            or not isinstance(result.reasoning_tokens, int)
            or result.reasoning_tokens < 0
        ):
            invalid.append("reasoning_tokens_invalid_count")
        elif result.output_tokens is not None and result.reasoning_tokens > result.output_tokens:
            invalid.append("reasoning_tokens_exceed_output_tokens")
    if result.cache_read_input_tokens is not None:
        if (
            isinstance(result.cache_read_input_tokens, bool)
            or not isinstance(result.cache_read_input_tokens, int)
            or result.cache_read_input_tokens < 0
        ):
            invalid.append("cache_read_input_tokens_invalid_count")
        elif (
            result.input_tokens is not None and result.cache_read_input_tokens > result.input_tokens
        ):
            invalid.append("cache_read_input_tokens_exceeds_input_tokens")

    status = normalize_result_status(result.status)
    success = status == "success"
    observed_validation_status = bool(
        expected_rejection
        and result.http_status in VALIDATION_REJECTION_HTTP_STATUSES
        and status == "client_error"
    )
    if observed_validation_status:
        # Status alone cannot prove the intended boundary was enforced: the response body is not
        # retained and the same 4xx may describe an unrelated malformed field or model state.
        informational.append("expected_probe_observed_validation_http_status")
    elif expected_rejection and success:
        informational.append("expected_validation_rejection_not_enforced_observed_acceptance")
    elif expected_rejection and status == "client_error":
        informational.append("expected_probe_failed_for_nonvalidation_client_reason")
    elif parameter_acceptance_only and status == "client_error":
        informational.append("parameter_acceptance_probe_observed_client_error")
    elif status == "client_error":
        informational.append("unexpected_client_error")
    if not success:
        censored.append(f"request_{status}")
    if success and not result.usage_complete:
        censored.append("provider_usage_missing")
    if success and result.ttft_seconds is None:
        censored.append("first_output_event_missing")
    arrival_censor_reason = normalize_arrival_latency_censor_reason(
        result.arrival_latency_censor_reason
    )
    if arrival_censor_reason:
        censored.append(arrival_censor_reason)

    proxy_duration: float | None = None
    if result.ttft_seconds is not None:
        proxy_duration = result.total_seconds - result.ttft_seconds
    if success and (result.output_tokens or 0) >= MIN_DECODE_PROXY_TOKENS:
        if proxy_duration is None:
            censored.append("decode_proxy_missing_ttft")
        elif result.content_event_count < MIN_DECODE_PROXY_CONTENT_EVENTS:
            censored.append("decode_proxy_insufficient_content_events")
        elif proxy_duration < MIN_PHYSICALLY_RESOLVED_DECODE_PROXY_SECONDS:
            invalid.append("decode_proxy_near_zero_with_multiple_tokens")
        elif proxy_duration < MIN_DECODE_PROXY_SECONDS:
            censored.append("decode_proxy_observation_window_below_one_second")
        elif result.reasoning_tokens is None:
            censored.append("decode_proxy_reasoning_token_state_unknown")
        elif result.reasoning_tokens > 0:
            censored.append("decode_proxy_hidden_reasoning_tokens_present")
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

    latency_eligible = success and not any(
        reason
        for reason in invalid
        if reason.startswith(
            (
                "total_",
                "ttft_",
                "headers_",
                "time_to_headers_",
                "arrival_to_completion_",
            )
        )
    )
    usage_eligible = (
        success
        and result.usage_complete
        and not any(
            reason
            in {
                "input_tokens_invalid_count",
                "output_tokens_invalid_count",
                "reasoning_tokens_invalid_count",
                "reasoning_tokens_exceed_output_tokens",
                "cache_read_input_tokens_invalid_count",
                "cache_read_input_tokens_exceeds_input_tokens",
            }
            or reason.startswith("usage_parse_error:")
            for reason in invalid
        )
    )
    decode_eligible = (
        latency_eligible
        and usage_eligible
        and (result.output_tokens or 0) >= MIN_DECODE_PROXY_TOKENS
        and result.content_event_count >= MIN_DECODE_PROXY_CONTENT_EVENTS
        and result.reasoning_tokens == 0
        and proxy_duration is not None
        and proxy_duration >= MIN_DECODE_PROXY_SECONDS
        and "decode_proxy_near_zero_with_multiple_tokens" not in invalid
        and "decode_proxy_extreme_tokens_per_second" not in anomalous
    )
    # Quality is an end-to-end predeclared-task estimand. Deterministic scorers assign zero to
    # non-success outcomes, and unrelated timing/usage defects do not erase otherwise scoreable
    # task output. The engine explicitly tells this metric contract whether a score exists.
    quality_eligible = quality_scored
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
