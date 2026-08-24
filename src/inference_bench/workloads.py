from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .models import RequestSpec, RouteConfig

# 1x1 RGB PNG. It is intentionally tiny: capability discovery should not conflate image support
# with payload-size limits. Image-envelope tests belong in a separate configured cell.
SMALL_PNG_DATA_URI = (
    "data:image/png;base64,"
    + base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000c4944415408d763f8cfc000000301010018dd8db10000000049454e44ae426082"
        )
    ).decode()
)


def _words(target_tokens: int, seed: str) -> str:
    """Deterministic token-like text; provider usage remains authoritative."""
    if target_tokens <= 0:
        return ""
    digest = hashlib.sha256(seed.encode()).hexdigest()[:12]
    # Leading-space common words are single tokens in many BPE/SentencePiece vocabularies. Exact
    # provider usage remains authoritative; the nonce prevents automatic prefix-cache reuse.
    return f" nonce{digest}" + " the" * max(0, target_tokens - 1)


def long_context_prompt(target_tokens: int, nonce: str) -> str:
    # Three separated retrieval anchors make acceptance distinguishable from successful use.
    each = max(1, (target_tokens - 120) // 3)
    return (
        f"BEGIN_MARKER={nonce}-A\n{_words(each, nonce + 'a')}\n"
        f"MIDDLE_MARKER={nonce}-B\n{_words(each, nonce + 'b')}\n"
        f"END_MARKER={nonce}-C\n{_words(each, nonce + 'c')}\n"
        "Return the three marker values exactly, separated by |, and nothing else."
    )


def shape_spec(
    route: RouteConfig,
    shape: str,
    logical_id: str,
    *,
    suite: str,
    cell_suffix: str = "",
    seed: int = 1,
) -> RequestSpec:
    context = route.context_tokens or 32_768
    output_max = route.max_output_tokens or 4_096
    if shape == "short_short":
        input_tokens, output_tokens = 256, min(128, output_max)
        prompt = _words(input_tokens, logical_id) + "\nReply with a concise checksum."
    elif shape == "long_short":
        input_tokens, output_tokens = min(32_768, max(2_048, context // 4)), min(128, output_max)
        prompt = ""
    elif shape == "short_long":
        input_tokens, output_tokens = 256, min(4_096, output_max)
        prompt = (
            _words(input_tokens, logical_id)
            + "\nWrite a coherent numbered technical explanation of approximately "
            + f"{output_tokens} tokens."
        )
    elif shape == "mixed":
        choice = int(hashlib.sha256(logical_id.encode()).hexdigest(), 16) % 4
        if choice in {0, 1, 2}:
            subtype = ("short_short", "long_short", "short_long")[choice]
            selected = shape_spec(route, subtype, logical_id, suite=suite, cell_suffix=cell_suffix)
            return replace(
                selected,
                cell_id=f"mixed{cell_suffix}",
                metadata={**selected.metadata, "shape": "mixed", "mixed_subtype": subtype},
            )
        input_tokens, output_tokens = 1_024, min(512, output_max)
        prompt = (
            _words(input_tokens, logical_id) + "\nReturn valid JSON with keys summary and risks."
        )
    else:
        raise ValueError(f"unknown workload shape: {shape}")
    return RequestSpec(
        logical_id=logical_id,
        route_id=route.id,
        suite=suite,
        cell_id=f"{shape}{cell_suffix}",
        messages=() if shape == "long_short" else ({"role": "user", "content": prompt},),
        planned_input_tokens=input_tokens,
        max_output_tokens=output_tokens,
        stream=True,
        metadata={
            "shape": shape,
            **(
                {
                    "prompt_kind": "long_context",
                    "target_tokens": input_tokens,
                    "nonce": logical_id[-16:],
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
            result.append(shape_spec(route, shape, logical, suite="latency", seed=seed))
    return result


def plan_capability(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    base = shape_spec(route, "short_short", f"capability:{route.id}:baseline", suite="capability")
    probes: list[RequestSpec] = [replace(base, cell_id="baseline_stream")]
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:nonstream",
            cell_id="baseline_nonstream",
            stream=False,
        )
    )
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:json",
            cell_id="structured_json",
            messages=(
                {"role": "user", "content": "Return JSON with integer field answer equal to 7."},
            ),
            response_format={"type": "json_object"},
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
            cell_id="tool_call",
            messages=({"role": "user", "content": "Use lookup_weather for Reykjavík."},),
            tools=(tool,),
            tool_choice="required",
            metadata={"quality": "tool_city_reykjavik"},
        )
    )
    image_content = [
        {"type": "text", "text": "What is the dominant color? Reply white."},
        {"type": "image_url", "image_url": {"url": SMALL_PNG_DATA_URI}},
    ]
    probes.append(
        replace(
            base,
            logical_id=f"capability:{route.id}:vision",
            cell_id="vision_small_png",
            messages=({"role": "user", "content": image_content},),
            metadata={"quality": "contains_white"},
        )
    )
    probes.extend(
        [
            replace(base, logical_id=f"capability:{route.id}:seed", cell_id="seed", seed=17),
            replace(base, logical_id=f"capability:{route.id}:stop", cell_id="stop", stop=("STOP",)),
            replace(
                base,
                logical_id=f"capability:{route.id}:logprobs",
                cell_id="logprobs",
                logprobs=True,
            ),
        ]
    )
    temperatures = config.get("temperatures", [-0.01, 0.0, 0.5, 1.0, 2.0, 2.01])
    top_ps = config.get("top_ps", [-0.01, 0.0, 0.25, 0.5, 0.75, 1.0, 1.01])
    for index, value in enumerate(temperatures):
        probes.append(
            replace(
                base,
                logical_id=f"capability:{route.id}:temperature:{index}",
                cell_id=f"temperature_{value}",
                temperature=float(value),
            )
        )
    for index, value in enumerate(top_ps):
        probes.append(
            replace(
                base,
                logical_id=f"capability:{route.id}:top_p:{index}",
                cell_id=f"top_p_{value}",
                top_p=float(value),
            )
        )
    return probes


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
    factors = {
        "temperature": list(config.get("temperatures", [0.0, 1.0, 2.0])),
        "top_p": list(config.get("top_ps", [0.1, 0.5, 1.0])),
        "stream": list(config.get("stream", [False, True])),
        "max_output_tokens": list(config.get("output_tokens", [64, 256, 1_024])),
    }
    result: list[RequestSpec] = []
    base = shape_spec(route, "short_short", f"interaction:{route.id}:base", suite="interactions")
    for index, row in enumerate(_pairwise_cover(factors)):
        result.append(
            replace(
                base,
                logical_id=f"interaction:{route.id}:{index}",
                cell_id=f"pairwise_{index:03d}",
                temperature=float(row["temperature"]),
                top_p=float(row["top_p"]),
                stream=bool(row["stream"]),
                max_output_tokens=min(
                    int(row["max_output_tokens"]), route.max_output_tokens or 1_024
                ),
                metadata={"covering_array_strength": 2, "factors": row},
            )
        )
    return result


def plan_context(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    if route.context_tokens is None:
        return []
    percentages = config.get("percentages", [1, 10, 25, 50, 75, 90, 95, 99])
    result: list[RequestSpec] = []
    for percentage in percentages:
        target = max(64, round(route.context_tokens * float(percentage) / 100))
        nonce = hashlib.sha256(f"{route.id}:{percentage}:{seed}".encode()).hexdigest()[:16]
        logical = f"context:{route.id}:{percentage:g}pct"
        result.append(
            RequestSpec(
                logical_id=logical,
                route_id=route.id,
                suite="context",
                cell_id=f"context_{percentage:g}pct",
                messages=(),
                planned_input_tokens=target,
                max_output_tokens=min(64, route.max_output_tokens or 64),
                stream=True,
                metadata={
                    "quality": "context_markers",
                    "nonce": nonce,
                    "percentage": percentage,
                    "prompt_kind": "long_context",
                    "target_tokens": target,
                },
            )
        )
    # Safe rejection probes test request validation without demanding realized output.
    above = route.context_tokens + 1
    nonce = hashlib.sha256(f"{route.id}:above:{seed}".encode()).hexdigest()[:16]
    result.append(
        RequestSpec(
            logical_id=f"context:{route.id}:above",
            route_id=route.id,
            suite="context",
            cell_id="context_above_documented",
            messages=(),
            planned_input_tokens=above,
            max_output_tokens=1,
            stream=True,
            metadata={
                "expected_rejection": True,
                "percentage": 100.0001,
                "prompt_kind": "long_context",
                "target_tokens": above,
                "nonce": nonce,
            },
        )
    )
    return result


def plan_output(route: RouteConfig, config: dict[str, Any], *, seed: int) -> list[RequestSpec]:
    maximum = route.max_output_tokens or int(config.get("fallback_max_output_tokens", 4_096))
    anchors = [32, 256, 1_024, 4_096, max(1, round(maximum * 0.9)), maximum, maximum + 1]
    anchors = sorted(set(anchors))
    result: list[RequestSpec] = []
    for tokens in anchors:
        logical = f"output:{route.id}:{tokens}"
        expected_rejection = tokens > maximum
        result.append(
            RequestSpec(
                logical_id=logical,
                route_id=route.id,
                suite="output",
                cell_id=f"requested_output_{tokens}",
                messages=(
                    {
                        "role": "user",
                        "content": (
                            f"Generate {tokens} tokens of numbered synthetic prose. "
                            "Continue until the "
                            "requested length unless the API limit stops you."
                        ),
                    },
                ),
                planned_input_tokens=64,
                max_output_tokens=tokens,
                stream=True,
                metadata={
                    "requested_output_tokens": tokens,
                    "expected_rejection": expected_rejection,
                },
            )
        )
    return result


QUALITY_TASKS = (
    ("exact_arithmetic", "Return only the integer result of 137 + 286.", "exact", "423"),
    ("instruction", "Reply with exactly SQWISH_OK and nothing else.", "exact", "SQWISH_OK"),
    ("json", "Return JSON with keys a=2 and b=3.", "json_fields", {"a": 2, "b": 3}),
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
                    cell_id=task_id,
                    messages=({"role": "user", "content": prompt},),
                    planned_input_tokens=64,
                    max_output_tokens=min(256, route.max_output_tokens or 256),
                    stream=True,
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
    shared = _words(prefix_tokens, f"cache-shared:{route.id}:{seed}")
    result: list[RequestSpec] = []
    for cache_state in ("cached_trial", "uncached_trial"):
        for repeat in range(repeats):
            prefix = (
                shared
                if cache_state == "cached_trial"
                else _words(prefix_tokens, f"cache-unique:{route.id}:{seed}:{repeat}")
            )
            logical = f"cache:{route.id}:{cache_state}:{repeat}"
            result.append(
                RequestSpec(
                    logical_id=logical,
                    route_id=route.id,
                    suite="cache",
                    cell_id=cache_state,
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
                    temperature=0,
                    metadata={"cache_state": cache_state},
                )
            )
    return result


PLANNERS = {
    "latency": plan_latency,
    "capability": plan_capability,
    "context": plan_context,
    "output": plan_output,
    "quality": plan_quality,
    "cache": plan_cache,
    "interactions": plan_interactions,
}


def plan_static_suites(
    routes: Iterable[RouteConfig], suites: dict[str, dict[str, Any]], *, seed: int
) -> list[RequestSpec]:
    result: list[RequestSpec] = []
    for route in routes:
        for name, planner in PLANNERS.items():
            config = suites.get(name)
            if config is not None and config.get("enabled", True):
                result.extend(planner(route, config, seed=seed))
    return result


def materialize_messages(spec: RequestSpec) -> list[dict[str, Any]]:
    """Build a synthetic payload at send time so million-token plans remain small."""
    if spec.messages:
        return list(spec.messages)
    if spec.metadata.get("prompt_kind") == "long_context":
        return [
            {
                "role": "user",
                "content": long_context_prompt(
                    int(spec.metadata["target_tokens"]), str(spec.metadata["nonce"])
                ),
            }
        ]
    raise ValueError(f"request has no messages or recognized prompt descriptor: {spec.logical_id}")
