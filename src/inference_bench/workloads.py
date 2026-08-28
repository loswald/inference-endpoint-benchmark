from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .config import suite_applies_to_route
from .models import RequestSpec, RouteConfig


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def solid_rgb_png_data_uri(
    *, width: int = 64, height: int = 64, rgb: tuple[int, int, int] = (0, 0, 255)
) -> str:
    """Create the exact vision stimulus in code so pixels, prompt, and scorer cannot diverge."""
    if width <= 0 or height <= 0 or any(channel not in range(256) for channel in rgb):
        raise ValueError("invalid RGB PNG dimensions or channel")
    scanline = b"\x00" + bytes(rgb) * width
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanline * height, level=9))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


VISION_BLUE_PNG_DATA_URI = solid_rgb_png_data_uri()
VISION_RED_PNG_DATA_URI = solid_rgb_png_data_uri(rgb=(255, 0, 0))


def _matched_cell(label: str, input_tokens: int, output_tokens: int) -> str:
    return f"{label}:in{input_tokens}:out{output_tokens}"


def _words(target_tokens: int, seed: str) -> str:
    """Deterministic token-like text; provider usage remains authoritative."""
    if target_tokens <= 0:
        return ""
    digest = hashlib.sha256(seed.encode()).hexdigest()[:12]
    # Leading-space common words are single tokens in many BPE/SentencePiece vocabularies. Exact
    # provider usage remains authoritative; the nonce prevents automatic prefix-cache reuse.
    return f" nonce{digest}" + " the" * max(0, target_tokens - 1)


def context_marker_values(key: str, seed: int) -> tuple[str, str, str]:
    """Independent opaque values; seeing one marker cannot reveal either of the others."""

    return tuple(
        hashlib.sha256(f"context-marker/v2:{seed}:{key}:{position}".encode()).hexdigest()[:24]
        for position in range(3)
    )  # type: ignore[return-value]


def long_context_prompt(target_tokens: int, markers: tuple[str, str, str]) -> str:
    # Three independent separated retrieval anchors make acceptance distinguishable from use.
    each = max(1, (target_tokens - 120) // 3)
    return (
        f"BEGIN_MARKER={markers[0]}\n{_words(each, markers[0])}\n"
        f"MIDDLE_MARKER={markers[1]}\n{_words(each, markers[1])}\n"
        f"END_MARKER={markers[2]}\n{_words(each, markers[2])}\n"
        "Return BEGIN_MARKER, MIDDLE_MARKER, and END_MARKER values in that order, separated "
        "by |, and nothing else."
    )


def shape_spec(
    route: RouteConfig,
    shape: str,
    logical_id: str,
    *,
    suite: str,
    cell_suffix: str = "",
    seed: int = 1,
    workload_key: str | None = None,
    matched_cell_suffix: str | None = None,
    shape_config: dict[str, Any] | None = None,
) -> RequestSpec:
    context = route.context_tokens or 32_768
    output_max = route.max_output_tokens or 4_096
    shape_config = shape_config or {}
    # Planners pass an explicit route-neutral key. Never infer one with substring replacement:
    # short/adversarial route IDs such as "a" or "do" can occur incidentally in other labels.
    comparison_id = workload_key or logical_id
    comparison_suffix = cell_suffix if matched_cell_suffix is None else matched_cell_suffix

    def route_target(axis: str) -> Any:
        by_route = shape_config.get(f"long_{axis}_tokens_by_route") or {}
        if by_route and not isinstance(by_route, dict):
            raise ValueError(f"long_{axis}_tokens_by_route must be a mapping")
        return by_route.get(route.id, shape_config.get(f"long_{axis}_tokens"))

    if shape == "short_short":
        input_tokens = 256
        if context - input_tokens <= 0:
            raise ValueError(
                f"route {route.id} context window cannot fit the short_short 256-token prompt "
                "and any positive output"
            )
        output_tokens = min(128, output_max, context - input_tokens)
        prompt = _words(input_tokens, comparison_id) + "\nReply exactly SQWISH_OK."
    elif shape == "long_short":
        configured_target = route_target("input")
        if configured_target is not None and route.context_tokens is None:
            # An explicit fixed anchor is a legitimate boundary probe when the provider does not
            # publish a window. It is not clipped against the generic 32K planning fallback.
            context = int(configured_target) + min(128, output_max)
        output_tokens = min(128, output_max, max(1, context - 1))
        available_input_tokens = context - output_tokens
        if available_input_tokens <= 0:
            raise ValueError(
                f"route {route.id} context window cannot fit the long_short prompt and output"
            )
        if configured_target is None:
            requested_input_tokens = min(32_768, max(2_048, context // 4))
            overflow_policy = "route_relative_default_clip"
        else:
            if (
                isinstance(configured_target, bool)
                or not isinstance(configured_target, int)
                or configured_target <= 0
            ):
                raise ValueError("long_input_tokens must be a positive integer")
            requested_input_tokens = configured_target
            overflow_policy = shape_config.get("long_input_overflow", "fail")
            if overflow_policy not in {"fail", "clip"}:
                raise ValueError("long_input_overflow must be fail or clip")
        if requested_input_tokens > available_input_tokens:
            if overflow_policy == "fail":
                raise ValueError(
                    f"route {route.id} long_input_tokens={requested_input_tokens} exceeds the "
                    f"nominal combined-context allowance {available_input_tokens} after "
                    f"reserving {output_tokens} output tokens"
                )
            input_tokens = available_input_tokens
        else:
            input_tokens = requested_input_tokens
        prompt = ""
        markers = context_marker_values(comparison_id, seed)
    elif shape == "short_long":
        input_tokens = 256
        combined_context_output_allowance = context - input_tokens
        if combined_context_output_allowance <= 0:
            raise ValueError(
                f"route {route.id} context window cannot fit the short_long 256-token prompt "
                "and any positive output"
            )
        available_output_tokens = min(output_max, combined_context_output_allowance)
        configured_target = route_target("output")
        if configured_target is None:
            requested_output_tokens = 4_096
            output_overflow_policy = "route_relative_default_clip"
        else:
            if (
                isinstance(configured_target, bool)
                or not isinstance(configured_target, int)
                or configured_target <= 0
            ):
                raise ValueError("long_output_tokens must be a positive integer")
            requested_output_tokens = configured_target
            output_overflow_policy = shape_config.get("long_output_overflow", "fail")
            if output_overflow_policy not in {"fail", "clip"}:
                raise ValueError("long_output_overflow must be fail or clip")
        if route.max_output_tokens is None and route.context_tokens is None:
            output_tokens = requested_output_tokens
        elif requested_output_tokens > available_output_tokens:
            if output_overflow_policy == "fail":
                raise ValueError(
                    f"route {route.id} long_output_tokens={requested_output_tokens} exceeds the "
                    f"documented/combined-context allowance {available_output_tokens}"
                )
            output_tokens = available_output_tokens
        else:
            output_tokens = requested_output_tokens
        prompt = (
            _words(input_tokens, comparison_id)
            + "\nWrite a coherent numbered technical explanation of approximately "
            + f"{output_tokens} tokens. Continue until the requested length."
        )
    elif shape == "mixed":
        choice = int(hashlib.sha256(comparison_id.encode()).hexdigest(), 16) % 4
        if choice in {0, 1, 2}:
            subtype = ("short_short", "long_short", "short_long")[choice]
            selected = shape_spec(
                route,
                subtype,
                logical_id,
                suite=suite,
                cell_suffix=cell_suffix,
                seed=seed,
                workload_key=comparison_id,
                matched_cell_suffix=comparison_suffix,
                shape_config=shape_config,
            )
            return replace(
                selected,
                cell_id=(
                    f"mixed:{subtype}:in{selected.planned_input_tokens}:"
                    f"out{selected.max_output_tokens}{comparison_suffix}"
                ),
                metadata={**selected.metadata, "shape": "mixed", "mixed_subtype": subtype},
            )
        input_tokens = 1_024
        if context - input_tokens <= 0:
            raise ValueError(
                f"route {route.id} context window cannot fit the mixed structured 1024-token "
                "prompt and any positive output"
            )
        output_tokens = min(512, output_max, context - input_tokens)
        prompt = (
            _words(input_tokens, comparison_id) + "\nReturn valid JSON with keys summary and risks."
        )
    else:
        raise ValueError(f"unknown workload shape: {shape}")
    return RequestSpec(
        logical_id=logical_id,
        route_id=route.id,
        suite=suite,
        cell_id=(
            f"{shape}:in{input_tokens}:out{output_tokens}{comparison_suffix}"
            if shape != "mixed"
            else f"mixed:structured:in{input_tokens}:out{output_tokens}{comparison_suffix}"
        ),
        messages=() if shape == "long_short" else ({"role": "user", "content": prompt},),
        planned_input_tokens=input_tokens,
        max_output_tokens=output_tokens,
        stream=True,
        timeout_seconds=route.request_timeout_seconds,
        metadata={
            "shape": shape,
            "workload_input_target_tokens": input_tokens,
            "workload_output_limit_tokens": output_tokens,
            **(
                {
                    "workload_input_requested_tokens": requested_input_tokens,
                    "workload_input_overflow_policy": overflow_policy,
                    "workload_input_was_clipped": input_tokens != requested_input_tokens,
                    "documented_context_tokens": route.context_tokens,
                    "route_limit_documented": route.context_tokens is not None,
                }
                if shape == "long_short"
                else {}
            ),
            **(
                {
                    "workload_output_requested_tokens": requested_output_tokens,
                    "workload_output_overflow_policy": output_overflow_policy,
                    "workload_output_was_clipped": output_tokens != requested_output_tokens,
                    "documented_context_tokens": route.context_tokens,
                    "documented_max_output_tokens": route.max_output_tokens,
                    "route_limit_documented": (
                        route.context_tokens is not None and route.max_output_tokens is not None
                    ),
                }
                if shape == "short_long"
                else {}
            ),
            **({"mixed_subtype": "structured"} if shape == "mixed" else {}),
            **({"quality": "exact", "expected": "SQWISH_OK"} if shape == "short_short" else {}),
            **(
                {
                    "quality": "longform_completion",
                    "minimum_output_tokens": max(1, round(output_tokens * 0.75)),
                }
                if shape == "short_long"
                else {}
            ),
            **(
                {
                    "quality": "json_keys",
                    "expected_keys": ["summary", "risks"],
                }
                if shape == "mixed"
                else {}
            ),
            **(
                {
                    "prompt_kind": "long_context",
                    "target_tokens": input_tokens,
                    "context_markers": list(markers),
                    "quality": "context_markers",
                }
                if shape == "long_short"
                else {}
            ),
        },
    )


def plan_latency(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    repeats = int(config.get("repeats", 10))
    result: list[RequestSpec] = []
    for shape in config.get("shapes", ["short_short", "long_short", "short_long", "mixed"]):
        for index in range(repeats):
            logical = f"latency:{route.id}:{shape}:{index}"
            result.append(
                shape_spec(
                    route,
                    shape,
                    logical,
                    suite="latency",
                    seed=seed,
                    workload_key=f"latency:{{route}}:{shape}:{index}",
                    shape_config=config,
                )
            )
    return result


def plan_warmup(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    result: list[RequestSpec] = []
    for shape in config.get("shapes", ["short_short"]):
        for repeat in range(int(config.get("repeats", 2))):
            logical = f"warmup:{route.id}:{shape}:{repeat}"
            result.append(
                shape_spec(
                    route,
                    shape,
                    logical,
                    suite="warmup",
                    seed=seed,
                    workload_key=f"warmup:{{route}}:{shape}:{repeat}",
                    shape_config=config,
                )
            )
    return result


def plan_capability(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    base = shape_spec(
        route,
        "short_short",
        f"capability:{route.id}:baseline",
        suite="capability",
        workload_key="capability:{route}:baseline",
    )
    probes: list[RequestSpec] = [
        replace(
            base,
            cell_id=_matched_cell(
                "baseline_stream", base.planned_input_tokens, base.max_output_tokens
            ),
        )
    ]
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:nonstream",
            cell_id=_matched_cell(
                "baseline_nonstream", base.planned_input_tokens, base.max_output_tokens
            ),
            stream=False,
        )
    )
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:json",
            cell_id=_matched_cell(
                "structured_json", base.planned_input_tokens, base.max_output_tokens
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Return exactly one JSON object containing only the integer field answer "
                        "equal to 7."
                    ),
                },
            ),
            response_format={"type": "json_object"},
            metadata={"quality": "json_answer_7"},
        )
    )
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:json-schema",
            cell_id=_matched_cell(
                "structured_json_schema", base.planned_input_tokens, base.max_output_tokens
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Return exactly one JSON object containing only the integer field "
                        "answer equal to 7."
                    ),
                },
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_object",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer", "const": 7}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
            metadata={"quality": "json_answer_7"},
        )
    )
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:tool",
            cell_id=_matched_cell("tool_call", base.planned_input_tokens, base.max_output_tokens),
            messages=({"role": "user", "content": "Use lookup_weather for Reykjavík."},),
            tools=(tool,),
            tool_choice="required",
            metadata={"quality": "tool_city_reykjavik"},
        )
    )
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:parallel-tools",
            cell_id=_matched_cell(
                "parallel_tool_calls", base.planned_input_tokens, base.max_output_tokens
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use lookup_weather once for Reykjavík and once for Tokyo. "
                        "Return exactly two tool calls and no prose."
                    ),
                },
            ),
            tools=(tool,),
            tool_choice="required",
            parallel_tool_calls=True,
            metadata={"quality": "tool_cities_reykjavik_tokyo"},
        )
    )
    nested_tool = {
        "type": "function",
        "function": {
            "name": "book_trip",
            "description": "Book a trip for one passenger",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "passenger": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "contact": {
                                "type": "object",
                                "properties": {"email": {"type": "string"}},
                                "required": ["email"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["name", "contact"],
                        "additionalProperties": False,
                    },
                },
                "required": ["destination", "passenger"],
                "additionalProperties": False,
            },
        },
    }
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:nested-tool-schema",
            cell_id=_matched_cell(
                "nested_tool_schema", base.planned_input_tokens, base.max_output_tokens
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use book_trip for destination Lisbon, passenger Ada, email "
                        "ada@example.com. Return only the tool call."
                    ),
                },
            ),
            tools=(nested_tool,),
            tool_choice="required",
            metadata={"quality": "tool_nested_trip"},
        )
    )
    for tool_count in config.get("tool_counts", [1, 8, 32, 64]):
        count = int(tool_count)
        tools = tuple(
            {
                "type": "function",
                "function": {
                    "name": f"select_target_{index}",
                    "description": f"Select deterministic target {index}",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
            for index in range(count)
        )
        target_name = f"select_target_{count - 1}"
        probes.append(
            replace(
                base,
                logical_id=f"capability:{route.id}:tool-count:{count}",
                cell_id=_matched_cell(
                    f"tool_count_{count}", base.planned_input_tokens, base.max_output_tokens
                ),
                messages=(
                    {
                        "role": "user",
                        "content": f"Call {target_name} with integer value 7 and no prose.",
                    },
                ),
                tools=tools,
                tool_choice={"type": "function", "function": {"name": target_name}},
                metadata={"quality": "tool_named_target", "expected": target_name},
            )
        )
    image_content = [
        {
            "type": "text",
            "text": "This image is one solid color. Reply with only its basic color name.",
        },
        {"type": "image_url", "image_url": {"url": VISION_BLUE_PNG_DATA_URI}},
    ]
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:vision",
            cell_id=_matched_cell(
                "vision_small_png", base.planned_input_tokens, base.max_output_tokens
            ),
            messages=({"role": "user", "content": image_content},),
            metadata={"quality": "exact_blue"},
        )
    )
    for label, width, height in (
        ("wide", 512, 64),
        ("tall", 64, 512),
        ("large", 1024, 1024),
    ):
        probes.append(
            replace(
                base,
                logical_id=f"capability:{route.id}:vision-{label}",
                cell_id=_matched_cell(
                    f"vision_{label}_png_{width}x{height}",
                    base.planned_input_tokens,
                    base.max_output_tokens,
                ),
                messages=(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This image is one solid color. Reply with only its basic "
                                    "color name."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": solid_rgb_png_data_uri(width=width, height=height)
                                },
                            },
                        ],
                    },
                ),
                metadata={"quality": "exact_blue", "image_width": width, "image_height": height},
            )
        )
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:vision-two-images",
            cell_id=_matched_cell(
                "vision_two_png_images", base.planned_input_tokens, base.max_output_tokens
            ),
            messages=(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Reply with the basic color of the first image, then |, then the "
                                "basic color of the second image, and nothing else."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": VISION_RED_PNG_DATA_URI}},
                        {"type": "image_url", "image_url": {"url": VISION_BLUE_PNG_DATA_URI}},
                    ],
                },
            ),
            metadata={"quality": "exact_red_blue"},
        )
    )
    probes.extend(
        [
            replace(
                base,
                logical_id=f"capability:{route.id}:stop",
                cell_id=_matched_cell(
                    "stop_sequence_triggered",
                    base.planned_input_tokens,
                    base.max_output_tokens,
                ),
                messages=(
                    {
                        "role": "user",
                        "content": "Output exactly the text ALPHA STOP OMEGA and nothing else.",
                    },
                ),
                stop=("STOP",),
                metadata={"scorer": "exact", "expected": "ALPHA"},
            ),
            replace(
                base,
                logical_id=f"capability:{route.id}:logprobs",
                cell_id=_matched_cell(
                    "logprobs_presence",
                    base.planned_input_tokens,
                    base.max_output_tokens,
                ),
                logprobs=True,
                metadata={
                    "capability_evidence_scope": "response_logprobs_presence_enforced_by_adapter",
                },
            ),
        ]
    )
    for label, configured_seed in (("same-a", 17), ("same-b", 17), ("different", 23)):
        probes.append(
            replace(
                base,
                logical_id=f"capability:{route.id}:seed:{label}",
                cell_id=_matched_cell(
                    f"seed_pair_{label}", base.planned_input_tokens, base.max_output_tokens
                ),
                messages=(
                    {
                        "role": "user",
                        "content": (
                            "Choose one item from alpha, beta, gamma, delta and reply with only "
                            "that item."
                        ),
                    },
                ),
                seed=configured_seed,
                metadata={
                    "capability_evidence_scope": "paired_seed_output_hash",
                    "seed_pair_group": "seed-determinism-v1",
                    "seed_pair_role": label,
                },
            )
        )
    temperatures = config.get("temperatures", [-0.01, 0.0, 0.5, 1.0, 2.0, 2.01])
    top_ps = config.get("top_ps", [-0.01, 0.0, 0.25, 0.5, 0.75, 1.0, 1.01])
    for index, value in enumerate(temperatures):
        probes.append(
            replace(
                base,
                logical_id=f"capability:{route.id}:temperature:{index}",
                cell_id=_matched_cell(
                    f"parameter_acceptance_only_nominal_temperature_{value}",
                    base.planned_input_tokens,
                    base.max_output_tokens,
                ),
                temperature=float(value),
                metadata={
                    **base.metadata,
                    "capability_evidence_scope": "parameter_acceptance_only",
                    "feature_behavior_unverified_reason": (
                        "route_specific_documented_temperature_range_not_configured"
                    ),
                },
            )
        )
    for index, value in enumerate(top_ps):
        probes.append(
            replace(
                base,
                logical_id=f"capability:{route.id}:top_p:{index}",
                cell_id=_matched_cell(
                    f"parameter_acceptance_only_nominal_top_p_{value}",
                    base.planned_input_tokens,
                    base.max_output_tokens,
                ),
                top_p=float(value),
                metadata={
                    **base.metadata,
                    "capability_evidence_scope": "parameter_acceptance_only",
                    "feature_behavior_unverified_reason": (
                        "route_specific_documented_top_p_range_not_configured"
                    ),
                },
            )
        )
    configured_groups = (config.get("probe_groups_by_route") or {}).get(
        route.id, config.get("probe_groups")
    )
    if configured_groups is None:
        return probes
    selected_groups = set(configured_groups)

    def probe_group(spec: RequestSpec) -> str:
        suffix = spec.logical_id.removeprefix(f"capability:{route.id}:")
        if suffix in {"baseline", "nonstream"}:
            return "transport_baseline"
        if suffix in {"json", "json-schema"}:
            return "structured_output"
        if suffix.startswith(("tool", "parallel-tools", "nested-tool-schema")):
            return "tool_calling"
        if suffix.startswith("vision"):
            return "vision"
        return "parameter_validation"

    return [probe for probe in probes if probe_group(probe) in selected_groups]


def _pairwise_cover(factors: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Small deterministic greedy strength-two covering array."""
    import itertools

    names = list(factors)
    candidates = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(factors[name] for name in names))
    ]
    uncovered: set[tuple[str, Any, str, Any]] = set()
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            uncovered.update((left, a, right, b) for a in factors[left] for b in factors[right])
    selected: list[dict[str, Any]] = []
    while uncovered:

        def covered(candidate: dict[str, Any]) -> set[tuple[str, Any, str, Any]]:
            pairs: set[tuple[str, Any, str, Any]] = set()
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    pair = (left, candidate[left], right, candidate[right])
                    if pair in uncovered:
                        pairs.add(pair)
            return pairs

        best = max(candidates, key=lambda item: (len(covered(item)), repr(item)))
        newly = covered(best)
        if not newly:
            raise AssertionError("covering-array construction stalled")
        selected.append(best)
        uncovered.difference_update(newly)
        candidates.remove(best)
    return selected


def plan_interactions(
    route: RouteConfig, config: dict[str, Any], *, seed: int
) -> list[RequestSpec]:
    configured_outputs = [int(value) for value in config.get("output_tokens", [64, 256, 1_024])]
    realized_outputs: list[int] = []
    for configured in configured_outputs:
        realized = (
            min(configured, route.max_output_tokens) if route.max_output_tokens else configured
        )
        if realized not in realized_outputs:
            realized_outputs.append(realized)
    factors = {
        "temperature": list(config.get("temperatures", [0.0, 1.0, 2.0])),
        "top_p": list(config.get("top_ps", [0.1, 0.5, 1.0])),
        "stream": list(config.get("stream", [False, True])),
        # The covering array is built over the levels that will actually be sent. Clipping after
        # construction can collapse levels and silently destroy strength-two coverage.
        "max_output_tokens": realized_outputs,
    }
    result: list[RequestSpec] = []
    base = shape_spec(
        route,
        "short_short",
        f"interaction:{route.id}:base",
        suite="interactions",
        workload_key="interaction:{route}:base",
    )
    for index, row in enumerate(_pairwise_cover(factors)):
        realized_output = int(row["max_output_tokens"])
        factor_label = (
            f"interaction_t={float(row['temperature']):g}_p={float(row['top_p']):g}_"
            f"stream={int(bool(row['stream']))}"
        )
        result.append(
            replace(
                base,
                logical_id=f"interaction:{route.id}:{index}",
                cell_id=_matched_cell(factor_label, base.planned_input_tokens, realized_output),
                temperature=float(row["temperature"]),
                top_p=float(row["top_p"]),
                stream=bool(row["stream"]),
                max_output_tokens=realized_output,
                metadata={
                    "covering_array_strength": 2,
                    "factors": row,
                    "realized_factor_levels": factors,
                },
            )
        )
    return result


def plan_context(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    percentages = config.get("percentages", [1, 10, 25, 50, 75, 90, 95, 99])
    result: list[RequestSpec] = []
    anchors: list[tuple[str, int, float | None, str]] = []
    if route.context_tokens is not None:
        for percentage in percentages:
            anchors.append(
                (
                    f"{percentage:g}pct",
                    max(64, round(route.context_tokens * float(percentage) / 100)),
                    float(percentage),
                    "documented_context_percentage",
                )
            )
    for target in config.get("fixed_tokens", []):
        anchors.append((f"fixed-{int(target)}", int(target), None, "fixed_token_anchor"))
    seen_targets: set[int] = set()
    for label, target, percentage, anchor_basis in anchors:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        output_target = min(64, route.max_output_tokens or 64)
        logical = f"context:{route.id}:{label}"
        markers = context_marker_values(f"context:{{route}}:{label}", seed)
        result.append(
            RequestSpec(
                logical_id=logical,
                route_id=route.id,
                suite="context",
                cell_id=f"context_{label}:in{target}:out{output_target}",
                messages=(),
                planned_input_tokens=target,
                max_output_tokens=output_target,
                stream=True,
                timeout_seconds=route.request_timeout_seconds,
                metadata={
                    "quality": "context_markers",
                    "context_markers": list(markers),
                    "percentage": percentage,
                    "context_anchor_basis": anchor_basis,
                    "prompt_kind": "long_context",
                    "target_tokens": target,
                },
            )
        )
    # Synthetic token-like counts are only planning estimates. This nominal target therefore does
    # not claim to exceed the provider-tokenized boundary unless reported usage proves that fact.
    if route.context_tokens is not None:
        above = route.context_tokens + 1
        above_logical = f"context:{route.id}:above"
        markers = context_marker_values("context:{route}:above", seed)
        result.append(
            RequestSpec(
                logical_id=above_logical,
                route_id=route.id,
                suite="context",
                cell_id=f"context_nominal_100pct_plus_1_target:in{above}:out1",
                messages=(),
                planned_input_tokens=above,
                max_output_tokens=1,
                stream=True,
                timeout_seconds=route.request_timeout_seconds,
                metadata={
                    "quality": "context_markers",
                    "boundary_evidence_scope": "nominal_target_requires_provider_reported_usage",
                    "percentage": 100.0001,
                    "prompt_kind": "long_context",
                    "target_tokens": above,
                    "context_markers": list(markers),
                },
            )
        )
    return result


def plan_output(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    documented_maximum = route.max_output_tokens
    maximum = documented_maximum or int(config.get("fallback_max_output_tokens", 4_096))
    realized_ceiling = min(int(config.get("realized_generation_ceiling", 16_384)), maximum)
    anchors = [
        *(value for value in (32, 256, 1_024, 4_096) if value <= maximum),
        realized_ceiling,
        max(1, round(maximum * 0.9)),
        maximum,
    ]
    if documented_maximum is not None:
        anchors.append(documented_maximum + 1)
    anchors = sorted(set(anchors))
    result: list[RequestSpec] = []
    for tokens in anchors:
        logical = f"output:{route.id}:{tokens}"
        expected_rejection = documented_maximum is not None and tokens > documented_maximum
        realized_generation = tokens <= realized_ceiling and not expected_rejection
        prompt = (
            (
                f"Generate {tokens} tokens of numbered synthetic prose. Continue until the "
                "requested length unless the API limit stops you."
            )
            if realized_generation
            else "Reply exactly SQWISH_OK."
        )
        result.append(
            RequestSpec(
                logical_id=logical,
                route_id=route.id,
                suite="output",
                cell_id=f"requested_output_{tokens}",
                messages=(
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ),
                planned_input_tokens=64,
                max_output_tokens=tokens,
                stream=True,
                timeout_seconds=route.request_timeout_seconds,
                metadata={
                    "requested_output_tokens": tokens,
                    "output_limit_basis": (
                        "documented_route_limit"
                        if documented_maximum is not None
                        else "local_screen_ceiling_not_a_provider_limit"
                    ),
                    "expected_rejection": expected_rejection,
                    "expected_rejection_kind": (
                        "requested_output_limit" if expected_rejection else None
                    ),
                    "output_evidence_scope": (
                        "realized_generation"
                        if realized_generation
                        else "requested_limit_acceptance_only"
                    ),
                    **(
                        {
                            "quality": "longform_completion",
                            "minimum_output_tokens": max(1, round(tokens * 0.75)),
                        }
                        if realized_generation
                        else {"quality": "exact", "expected": "SQWISH_OK"}
                    ),
                },
            )
        )
    return result


QUALITY_TASKS = (
    ("exact_arithmetic", "Return only the integer result of 137 + 286.", "exact", "423"),
    ("instruction", "Reply with exactly SQWISH_OK and nothing else.", "exact", "SQWISH_OK"),
    (
        "json",
        "Return exactly one JSON object containing only integer keys a=2 and b=3.",
        "json_fields",
        {"a": 2, "b": 3},
    ),
    ("retrieval", "Remember code ORCHID-713. What is the code?", "contains", "ORCHID-713"),
)


def plan_quality(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    repeats = int(config.get("repeats", 3))
    result: list[RequestSpec] = []
    for task_id, prompt, scorer, expected in QUALITY_TASKS:
        for repeat in range(repeats):
            result.append(
                RequestSpec(
                    logical_id=f"quality:{route.id}:{task_id}:{repeat}",
                    route_id=route.id,
                    suite="quality",
                    cell_id=_matched_cell(task_id, 64, min(256, route.max_output_tokens or 256)),
                    messages=({"role": "user", "content": prompt},),
                    planned_input_tokens=64,
                    max_output_tokens=min(256, route.max_output_tokens or 256),
                    stream=True,
                    timeout_seconds=route.request_timeout_seconds,
                    temperature=0,
                    metadata={"scorer": scorer, "expected": expected},
                )
            )
    return result


def plan_cache(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    """Matched cached/uncached prefix trials.

    Some providers cache automatically and expose no disable switch. Uncached cells therefore use
    a deterministic unique prefix per request; cached cells reuse an identical long prefix. The
    declared cache state is a trial assignment, while provider-reported cached tokens are retained
    as the observed cache state.
    """
    repeats = int(config.get("repeats", 5))
    prefix_tokens = int(config.get("prefix_tokens", 4_096))
    shared = _words(prefix_tokens, f"cache-shared:{{route}}:{seed}")
    result: list[RequestSpec] = []
    for cache_state in ("cached_trial", "uncached_trial"):
        for repeat in range(repeats):
            prefix = (
                shared
                if cache_state == "cached_trial"
                else _words(prefix_tokens, f"cache-unique:{{route}}:{seed}:{repeat}")
            )
            logical = f"cache:{route.id}:{cache_state}:{repeat}"
            result.append(
                RequestSpec(
                    logical_id=logical,
                    route_id=route.id,
                    suite="cache",
                    cell_id=_matched_cell(
                        cache_state,
                        prefix_tokens + 16,
                        min(32, route.max_output_tokens or 32),
                    ),
                    messages=(
                        {
                            "role": "user",
                            "content": prefix
                            + "\nReturn only the final word appearing before this instruction.",
                        },
                    ),
                    planned_input_tokens=prefix_tokens + 16,
                    max_output_tokens=min(32, route.max_output_tokens or 32),
                    stream=True,
                    timeout_seconds=route.request_timeout_seconds,
                    temperature=0,
                    metadata={"cache_state": cache_state},
                )
            )
    return result


def plan_time_variation(
    route: RouteConfig, config: dict[str, Any], *, seed: int
) -> list[RequestSpec]:
    """Matched low-load measurements sampled at fixed offsets in a bounded study.

    This is intentionally a dedicated campaign, not traffic overlapped with capacity tests. Each
    panel repeats the same route-neutral workloads, allowing within-route comparisons across the
    registered study window without changing task content or offered load.
    """

    panels = int(config.get("panels", 12))
    interval_seconds = float(config.get("interval_minutes", 120)) * 60
    repeats = int(config.get("samples_per_route_shape", 3))
    stable_repeats = int(config.get("stable_exact_prompt_repeats", repeats))
    shapes = config.get("shapes", ["short_short", "long_short"])
    result: list[RequestSpec] = []

    def panel_unique_key_with_matched_mixed_subtype(
        shape: str, stable_key: str, panel: int, repeat_index: int
    ) -> str:
        """Change prompt bytes without changing the registered mixed-workload subtype."""

        prefix = (
            f"time-variation:{{route}}:{shape}:panel-{panel:03d}:"
            f"cache-cold-repeat-{repeat_index:03d}"
        )
        if shape != "mixed":
            return prefix
        stable_choice = int(hashlib.sha256(stable_key.encode()).hexdigest(), 16) % 4
        for nonce in range(256):
            candidate = f"{prefix}:nonce-{nonce:03d}"
            if int(hashlib.sha256(candidate.encode()).hexdigest(), 16) % 4 == stable_choice:
                return candidate
        raise AssertionError("unable to construct a matched cache-cold mixed-workload key")

    for panel in range(panels):
        for shape in shapes:
            for repeat in range(repeats):
                logical = f"time-variation:{route.id}:panel-{panel:03d}:{shape}:{repeat:03d}"
                if repeat < stable_repeats:
                    repeat_index = repeat
                    cache_condition = "stable_exact_prompt_across_panels"
                    prompt_identity = f"stable-{repeat_index:03d}"
                    workload_key = (
                        f"time-variation:{{route}}:{shape}:stable-repeat-{repeat_index:03d}"
                    )
                else:
                    repeat_index = repeat - stable_repeats
                    cache_condition = "panel_unique_cache_cold"
                    prompt_identity = f"panel-{panel:03d}-cold-{repeat_index:03d}"
                    stable_key = (
                        f"time-variation:{{route}}:{shape}:stable-repeat-{repeat_index:03d}"
                    )
                    workload_key = panel_unique_key_with_matched_mixed_subtype(
                        shape,
                        stable_key,
                        panel,
                        repeat_index,
                    )
                spec = shape_spec(
                    route,
                    shape,
                    logical,
                    suite="time_variation",
                    seed=seed,
                    workload_key=workload_key,
                    matched_cell_suffix=f":panel={panel:03d}",
                    shape_config=config,
                )
                result.append(
                    replace(
                        spec,
                        metadata={
                            **spec.metadata,
                            "time_variation_panel": panel,
                            "time_variation_offset_seconds": panel * interval_seconds,
                            "time_variation_repeat": repeat,
                            "time_variation_prompt_identity": prompt_identity,
                            "cache_condition": cache_condition,
                        },
                    )
                )
    return result


PLANNERS = {
    "warmup": plan_warmup,
    "latency": plan_latency,
    "capability": plan_capability,
    "context": plan_context,
    "output": plan_output,
    "quality": plan_quality,
    "cache": plan_cache,
    "interactions": plan_interactions,
    "time_variation": plan_time_variation,
}


def plan_static_suites(
    routes: Iterable[RouteConfig], suites: dict[str, dict[str, Any]], *, seed: int
) -> list[RequestSpec]:
    result: list[RequestSpec] = []
    for route in routes:
        for name, planner in PLANNERS.items():
            config = suites.get(name)
            if (
                config is not None
                and config.get("enabled", True)
                and suite_applies_to_route(config, route.id)
            ):
                result.extend(planner(route, config, seed=seed))
    return result


def materialize_messages(spec: RequestSpec) -> list[dict[str, Any]]:
    """Build a synthetic payload at send time so million-token plans remain small."""
    if spec.messages:
        return list(spec.messages)
    if spec.metadata.get("prompt_kind") == "long_context":
        values = spec.metadata.get("context_markers")
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise ValueError(f"invalid independent context markers: {spec.logical_id}")
        return [
            {
                "role": "user",
                "content": long_context_prompt(
                    int(spec.metadata["target_tokens"]),
                    tuple(values),  # type: ignore[arg-type]
                ),
            }
        ]
    raise ValueError(f"request has no messages or recognized prompt descriptor: {spec.logical_id}")
