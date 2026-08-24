from __future__ import annotations

import base64
import hashlib
import itertools
import struct
import zlib
from dataclasses import replace

import pytest

from inference_bench.workloads import (
    VISION_BLUE_PNG_DATA_URI,
    _pairwise_cover,
    materialize_messages,
    plan_cache,
    plan_capability,
    plan_context,
    plan_interactions,
    plan_latency,
    shape_spec,
)


def test_programmatic_vision_fixture_is_exactly_solid_blue() -> None:
    encoded = VISION_BLUE_PNG_DATA_URI.removeprefix("data:image/png;base64,")
    png = base64.b64decode(encoded)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    idat = bytearray()
    width = height = 0
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        payload = png[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    pixels = zlib.decompress(idat)
    assert (width, height) == (64, 64)
    assert pixels == (b"\x00" + b"\x00\x00\xff" * width) * height


def test_covering_array_covers_every_pair() -> None:
    factors = {"a": [0, 1, 2], "b": ["x", "y"], "c": [False, True]}
    rows = _pairwise_cover(factors)
    for left_index, left in enumerate(factors):
        for right in list(factors)[left_index + 1 :]:
            observed = {(row[left], row[right]) for row in rows}
            assert observed == set(itertools.product(factors[left], factors[right]))


def test_interaction_cover_is_built_over_realized_route_levels(route) -> None:
    limited_route = replace(route, max_output_tokens=128)
    config = {
        "temperatures": [0.0, 1.0],
        "top_ps": [0.25, 1.0],
        "stream": [False, True],
        "output_tokens": [64, 256, 1_024],
    }
    specs = plan_interactions(limited_route, config, seed=1)
    factors = {
        "temperature": [0.0, 1.0],
        "top_p": [0.25, 1.0],
        "stream": [False, True],
        "max_output_tokens": [64, 128],
    }
    for spec in specs:
        observed = spec.metadata["factors"]
        assert observed == {
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "stream": spec.stream,
            "max_output_tokens": spec.max_output_tokens,
        }
        assert spec.metadata["realized_factor_levels"] == factors
    for left_index, left in enumerate(factors):
        for right in list(factors)[left_index + 1 :]:
            observed_pairs = {
                (spec.metadata["factors"][left], spec.metadata["factors"][right]) for spec in specs
            }
            assert observed_pairs == set(itertools.product(factors[left], factors[right]))
    assert len({spec.cell_id for spec in specs}) == len(specs)


def test_interaction_outputs_are_not_arbitrarily_clipped_without_route_limit(route) -> None:
    unlimited_route = replace(route, max_output_tokens=None)
    specs = plan_interactions(
        unlimited_route,
        {
            "temperatures": [0.0],
            "top_ps": [1.0],
            "stream": [True],
            "output_tokens": [64, 2_048],
        },
        seed=1,
    )
    assert {spec.max_output_tokens for spec in specs} == {64, 2_048}


def test_context_plan_is_lazy_and_retrieval_aware(route) -> None:
    specs = plan_context(route, {"percentages": [99]}, seed=1)
    fixed = specs[0]
    assert fixed.messages == ()
    assert fixed.metadata["prompt_kind"] == "long_context"
    message = materialize_messages(fixed)[0]["content"]
    assert "BEGIN_MARKER" in message and "MIDDLE_MARKER" in message and "END_MARKER" in message


def test_unverified_capability_parameters_are_acceptance_only(route) -> None:
    specs = plan_capability(route, {}, seed=1)
    acceptance = [spec for spec in specs if spec.cell_id.startswith("parameter_acceptance_only_")]
    assert len(acceptance) == 16
    assert {spec.metadata["capability_evidence_scope"] for spec in acceptance} == {
        "parameter_acceptance_only"
    }
    assert all("feature_behavior_unverified_reason" in spec.metadata for spec in acceptance)


def test_nominal_above_context_target_never_claims_expected_rejection(route) -> None:
    spec = plan_context(route, {"percentages": [99]}, seed=1)[-1]
    assert spec.cell_id.startswith("context_nominal_100pct_plus_1_target:")
    assert "expected_rejection" not in spec.metadata
    assert spec.metadata["boundary_evidence_scope"] == (
        "nominal_target_requires_provider_reported_usage"
    )


def test_cache_trials_are_stratified(route) -> None:
    specs = plan_cache(route, {"repeats": 2, "prefix_tokens": 20}, seed=1)
    assert {spec.cell_id for spec in specs} == {
        "cached_trial:in36:out32",
        "uncached_trial:in36:out32",
    }
    cached = [spec for spec in specs if spec.cell_id.startswith("cached_trial:")]
    assert cached[0].messages[0]["content"] == cached[1].messages[0]["content"]
    uncached = [spec for spec in specs if spec.cell_id.startswith("uncached_trial:")]
    assert uncached[0].messages[0]["content"] != uncached[1].messages[0]["content"]


def test_long_shape_targets_are_explicit_and_fail_or_clip_against_route_limits(route) -> None:
    large = replace(route, context_tokens=131_072, max_output_tokens=65_536)
    long_input = shape_spec(
        large,
        "long_short",
        "long-input",
        suite="load",
        shape_config={"long_input_tokens": 100_000, "long_input_overflow": "fail"},
    )
    assert long_input.planned_input_tokens == 100_000
    assert long_input.cell_id.startswith("long_short:in100000:out128")
    assert long_input.metadata["workload_input_was_clipped"] is False

    long_output = shape_spec(
        large,
        "short_long",
        "long-output",
        suite="load",
        shape_config={"long_output_tokens": 32_768, "long_output_overflow": "fail"},
    )
    assert long_output.max_output_tokens == 32_768
    assert long_output.cell_id.startswith("short_long:in256:out32768")
    assert long_output.metadata["workload_output_was_clipped"] is False

    with pytest.raises(ValueError, match="long_input_tokens=100000 exceeds"):
        shape_spec(
            route,
            "long_short",
            "too-long-input",
            suite="load",
            shape_config={"long_input_tokens": 100_000, "long_input_overflow": "fail"},
        )
    clipped_input = shape_spec(
        route,
        "long_short",
        "clipped-input",
        suite="load",
        shape_config={"long_input_tokens": 100_000, "long_input_overflow": "clip"},
    )
    assert clipped_input.planned_input_tokens == 8_192 - 128
    assert clipped_input.metadata["workload_input_was_clipped"] is True

    with pytest.raises(ValueError, match="long_output_tokens=32768 exceeds"):
        shape_spec(
            route,
            "short_long",
            "too-long-output",
            suite="load",
            shape_config={"long_output_tokens": 32_768, "long_output_overflow": "fail"},
        )
    clipped_output = shape_spec(
        route,
        "short_long",
        "clipped-output",
        suite="load",
        shape_config={"long_output_tokens": 32_768, "long_output_overflow": "clip"},
    )
    assert clipped_output.max_output_tokens == 2_048
    assert clipped_output.metadata["workload_output_was_clipped"] is True

    with pytest.raises(ValueError, match="cannot fit the short_long 256-token prompt"):
        shape_spec(
            replace(route, context_tokens=256),
            "short_long",
            "no-output-room",
            suite="load",
            shape_config={"long_output_tokens": 1, "long_output_overflow": "fail"},
        )
    with pytest.raises(ValueError, match="cannot fit the short_short 256-token prompt"):
        shape_spec(
            replace(route, context_tokens=256),
            "short_short",
            "no-short-room",
            suite="load",
        )

    tiny = replace(route, context_tokens=1_024)
    with pytest.raises(ValueError, match="cannot fit the mixed structured 1024-token prompt"):
        next(
            shape_spec(tiny, "mixed", f"tiny-mixed-{index}", suite="load")
            for index in range(64)
            if int(hashlib.sha256(f"tiny-mixed-{index}".encode()).hexdigest(), 16) % 4 == 3
        )

    static = plan_latency(
        large,
        {
            "repeats": 1,
            "shapes": ["long_short", "short_long"],
            "long_input_tokens": 100_000,
            "long_input_overflow": "fail",
            "long_output_tokens": 32_768,
            "long_output_overflow": "fail",
        },
        seed=7,
    )
    assert [(spec.planned_input_tokens, spec.max_output_tokens) for spec in static] == [
        (100_000, 128),
        (256, 32_768),
    ]
