# ruff: noqa: E501
from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ENDPOINTS = (
    "deepseek-v4-flash-0731",
    "gemma-4-31B-it",
    "glm-5.2",
    "kimi-k3",
    "minimax-m2.5",
    "mimo-v2.5-pro",
    "nemotron-3-ultra-550b",
    "nvidia-nemotron-3-super-120b",
    "openai-gpt-oss-120b",
    "qwen3.5-397b-a17b",
    "qwen3.8-max",
)

ENDPOINT_LABELS = {
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "gemma-4-31B-it": "Gemma 4 31B",
    "glm-5.2": "GLM 5.2",
    "kimi-k3": "Kimi K3",
    "minimax-m2.5": "MiniMax M2.5",
    "mimo-v2.5-pro": "MiMo V2.5 Pro",
    "nemotron-3-ultra-550b": "Nemotron 3 Ultra 550B",
    "nvidia-nemotron-3-super-120b": "Nemotron 3 Super 120B",
    "openai-gpt-oss-120b": "GPT-OSS 120B",
    "qwen3.5-397b-a17b": "Qwen3.5 397B A17B",
    "qwen3.8-max": "Qwen3.8 Max",
}

SHAPES = ("short_short", "input100k_short", "input32k_short", "short_long", "mixed")
SHAPE_LABELS = {
    "short_short": "Short prompt / short answer",
    "input100k_short": "100K prompt / short answer",
    "input32k_short": "32K prompt / short answer",
    "long_short": "100K prompt / short answer",
    "short_long": "Short prompt / long answer",
    "mixed": "Mixed application traffic",
}

CAPACITY_SOURCE = "do-combined-capacity-20260829"
FIXED_RATE_SOURCE = "do-direct-soak-20260823-r1"
VARIATION_SOURCE = "do-six-hour-variation-20260828-r1"

NAVY = "#13253F"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
RED = "#B91C1C"
SLATE = "#64748B"
LIGHT = "#F5F7FA"
GRID = "#CBD5E1"
INK = "#172033"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int:
    number = _number(value)
    return 0 if number is None else int(number)


def _bool(value: Any) -> bool | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _json_interval(value: Any) -> tuple[float, float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(parsed, dict):
        low = _number(parsed.get("ci95_low"))
        high = _number(parsed.get("ci95_high"))
    elif isinstance(parsed, list) and len(parsed) == 2:
        low, high = _number(parsed[0]), _number(parsed[1])
    else:
        return None
    if low is None or high is None or low > high:
        return None
    return low, high


def _first_number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _first_text(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _label(endpoint: str) -> str:
    return ENDPOINT_LABELS.get(endpoint, endpoint)


def _state_text(value: str) -> str:
    mapping = {
        "confirmed_right_censored_lower_bound": "Repeatedly passed through tested ceiling",
        "confirmed_bracketed_interval": "Repeatedly passed lower rate; failed higher rate",
        "censored_no_valid_healthy_epoch": "No healthy baseline established within tested range",
        "unconfirmed_healthy_observation_only": "Healthy observation; not repeat-confirmed",
        "measured_capacity_state_without_numeric_bound": "Measured; no numeric bound",
        "censored_nonmonotonic_overload": "Non-monotonic response; no numeric bound",
    }
    return mapping.get(value, value.replace("_", " ") if value else "Not measured")


def _capacity_result(row: dict[str, str]) -> str:
    claim = str(row.get("capacity_claim") or "")
    low = _number(row.get("capacity_lower_bound_rps"))
    high = _number(row.get("capacity_upper_bound_rps"))
    if claim == "confirmed_bracketed_interval" and low is not None and high is not None:
        return f"{_format_rate(low)}-{_format_rate(high)} req/s bracket"
    if claim == "confirmed_right_censored_lower_bound" and low is not None:
        return f"at least {_format_rate(low)} req/s (tested lower bound)"
    if claim == "censored_no_valid_healthy_epoch" and high is not None:
        return (
            f"no healthy baseline at tested rates; lowest {_format_rate(high)} req/s "
            "(lower rates untested)"
        )
    return _state_text(claim)


def _format_rate(value: float) -> str:
    if value < 0.1:
        rendered = f"{value:.3f}"
    elif value < 1:
        rendered = f"{value:.2f}"
    else:
        rendered = f"{value:.3g}"
    return rendered.rstrip("0").rstrip(".")


def _fixed_rate_state(row: dict[str, str]) -> str:
    if not row:
        return "not measured"
    if _bool(row.get("soak_acceptance_pass")) is True:
        return "passed"
    status = str(row.get("status") or "")
    if "transport_gate" in status or "baseline" in status and "gate" in status:
        return "transport-gated"
    if _bool(row.get("execution_complete")) is not True:
        return "could not establish baseline"
    if _bool(row.get("scientifically_complete")) is True:
        return "tested-rate non-pass"
    if status:
        return "measured unresolved"
    return "could not establish baseline"


def _capability_state(
    capability_index: dict[tuple[str | None, str | None], dict[str, str]],
    endpoint: str,
    dimension: str,
    panel_rows: list[dict[str, str]] | None = None,
) -> str:
    if dimension == "automatic_prompt_cache":
        if panel_rows is None:
            return "automatic, best effort (documented)"
        stable = sum(
            _number(row.get("cache_read_tokens_sum")) or 0.0
            for row in panel_rows
            if row.get("route_id") == endpoint
            and row.get("cache_stratum") == "stable_exact_prompt"
        )
        fresh = sum(
            _number(row.get("cache_read_tokens_sum")) or 0.0
            for row in panel_rows
            if row.get("route_id") == endpoint
            and row.get("cache_stratum") == "panel_unique_cold"
        )
        if stable > max(fresh * 2, fresh + 512):
            return "automatic (docs); exact-prefix reuse observed"
        if stable > 0 or fresh > 0:
            return "automatic (docs); cache reads not specific to exact repeats"
        return "automatic (docs); no cache reads observed in this panel"
    if dimension == "batch_open_models":
        return "unsupported (documented)"
    evidence = capability_index.get((endpoint, dimension), {})
    state = _first_text(evidence, "functional_status", "transport_status")
    scored = _integer(evidence.get("functional_scored_count")) or 0
    passed = _integer(evidence.get("functional_pass_count")) or 0
    if dimension == "vision":
        documented = {
            "kimi-k3": "image documented",
            "qwen3.8-max": "image/video documented",
            "glm-5.2": "text-only documented",
            "mimo-v2.5-pro": "text-only documented",
        }.get(endpoint, "not documented as vision-capable")
        if scored:
            return f"{documented}; live probe {state.replace('_', ' ')} ({passed}/{scored})"
        return f"{documented}; live probe inconclusive"
    if not state:
        return "not measured"
    if scored:
        return f"{state.replace('_', ' ')} ({passed}/{scored})"
    return state.replace("_", " ")


def _compact_capability_state(
    capability_index: dict[tuple[str | None, str | None], dict[str, str]],
    endpoint: str,
    dimension: str,
    panel_rows: list[dict[str, str]],
) -> str:
    """Fit the cross-endpoint matrix without discarding denominators or provenance."""

    if dimension == "automatic_prompt_cache":
        full = _capability_state(capability_index, endpoint, dimension, panel_rows)
        if "reuse observed" in full:
            return "automatic; repeat-prefix reads seen"
        if "not specific" in full:
            return "automatic; reads not repeat-specific"
        return "automatic; no reads seen"
    if dimension == "batch_open_models":
        return "unsupported by product contract"
    evidence = capability_index.get((endpoint, dimension), {})
    state = _first_text(evidence, "functional_status", "transport_status")
    scored = _integer(evidence.get("functional_scored_count")) or 0
    passed = _integer(evidence.get("functional_pass_count")) or 0
    if dimension == "vision":
        documented = {
            "kimi-k3": "docs: image",
            "qwen3.8-max": "docs: image/video",
            "glm-5.2": "docs: text only",
            "mimo-v2.5-pro": "docs: text only",
        }.get(endpoint, "not listed for vision")
        if scored:
            return f"{documented}; probe {passed}/{scored}"
        return f"{documented}; probe inconclusive"
    if not state:
        return "not measured"
    short_state = {
        "observed_functional": "pass",
        "observed_not_functional": "fail",
        "observed_degraded": "degraded",
        "observed_rejected_or_unsupported": "rejected",
    }.get(state, state.replace("_", " "))
    return f"{short_state} {passed}/{scored}" if scored else short_state


def _format_integer(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "not documented"
    return f"{int(round(number)):,}"


def _endpoint_status_counts(
    panel_rows: list[dict[str, str]], endpoint: str
) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in panel_rows:
        if row.get("route_id") != endpoint:
            continue
        try:
            counts = json.loads(row.get("status_counts") or "{}")
        except json.JSONDecodeError:
            counts = {}
        for status, count in counts.items():
            result[str(status)] += int(count)
    return result


def _endpoint_decision(
    stats: dict[str, float], counts: Counter[str]
) -> tuple[str, str, str]:
    if stats["rate"] < 0.8:
        label = "PRODUCTION CAUTION"
        color = RED
        reading = (
            "Do not use as the only route yet. Start with conservative concurrency, adaptive backoff, "
            "a circuit breaker, and a tested fallback."
        )
    elif stats["rate"] < 0.98:
        label = "PILOT WITH FALLBACK"
        color = ORANGE
        reading = (
            "Usable for a monitored pilot at the measured low load. Keep retry/backoff and a fallback; "
            "do not infer high-load capacity from this result."
        )
    else:
        label = "LOW-LOAD RELIABILITY PASS"
        color = TEAL
        reading = (
            "A strong candidate for a monitored pilot at the measured low load. Size production traffic "
            "from the exact adaptive-load recipe below."
        )
    failures = int(stats["n"] - stats["success"])
    if failures:
        reading += (
            f" Failure mix: {counts.get('rate_limited', 0)} rate-limited, "
            f"{counts.get('timeout', 0)} timed out, {counts.get('server_error', 0)} server errors."
        )
    return label, color, reading


def _profile_identities(endpoint: str) -> tuple[tuple[str, str, str], ...]:
    long_mixed = (
        "long_short:in50000:out128"
        if endpoint == "minimax-m2.5"
        else "long_short:in100000:out128"
    )
    long_label = "Mixed: 50K input" if endpoint == "minimax-m2.5" else "Mixed: 100K input"
    return (
        ("short_short", "not_applicable", "Short / short"),
        ("long_short", "not_applicable", "100K / short"),
        ("short_long", "not_applicable", "Short / long"),
        ("mixed", "structured:in1024:out512", "Mixed: structured"),
        ("mixed", long_mixed, long_label),
    )


def _interval_value(row: dict[str, str], stem: str, digits: int = 1) -> str:
    estimate = _number(row.get(stem))
    low = _number(row.get(f"{stem}_ci95_low"))
    high = _number(row.get(f"{stem}_ci95_high"))
    if estimate is None:
        return "not estimable"
    if low is None or high is None:
        return f"{estimate:.{digits}f} (no interval)"
    return f"{estimate:.{digits}f} [{low:.{digits}f}-{high:.{digits}f}]"


def _endpoint_profile_rows(
    endpoint: str,
    panel_rows: list[dict[str, str]],
    across_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for shape, subtype, label in _profile_identities(endpoint):
        panel_subset = [
            row
            for row in panel_rows
            if row.get("route_id") == endpoint
            and row.get("shape") == shape
            and row.get("mixed_subtype") == subtype
        ]
        attempted = sum(_integer(row.get("attempted_n")) or 0 for row in panel_subset)
        successes = sum(_integer(row.get("success_n")) or 0 for row in panel_subset)
        low, high = _wilson(successes, attempted)
        by_stratum = {
            str(row.get("cache_stratum")): row
            for row in across_rows
            if row.get("route_id") == endpoint
            and row.get("shape") == shape
            and row.get("mixed_subtype") == subtype
        }
        stable = by_stratum.get("stable_exact_prompt", {})
        fresh = by_stratum.get("panel_unique_cold", {})
        result.append(
            {
                "label": label,
                "success": (
                    f"{successes}/{attempted} ({100 * successes / attempted:.1f}%; "
                    f"95% {100 * low:.1f}-{100 * high:.1f}%)"
                    if attempted
                    else "not measured"
                ),
                "latency": (
                    f"{_interval_value(stable, 'across_panel_request_latency_median_median')} / "
                    f"{_interval_value(fresh, 'across_panel_request_latency_median_median')} s"
                ),
                "throughput": (
                    f"{_interval_value(stable, 'across_panel_eligible_output_rate_median_median')} / "
                    f"{_interval_value(fresh, 'across_panel_eligible_output_rate_median_median')} tok/s"
                ),
            }
        )
    return result


def _boundary_note(rows: list[dict[str, str]]) -> str:
    prompt = next((row for row in rows if row.get("dimension") == "prompt context window"), {})
    combined = next(
        (row for row in rows if row.get("dimension") == "combined prompt + requested output"),
        {},
    )
    parts: list[str] = []
    prompt_value = _first_number(
        prompt,
        "maximum_functionally_valid_input_tokens",
        "maximum_accepted_input_tokens",
        "observed_value",
    )
    if prompt_value is not None:
        parts.append(
            f"retrieval-valid prompt accepted through {_format_integer(prompt_value)} estimated tokens "
            f"({_state_text(str(prompt.get('boundary_censoring') or 'measured'))})"
        )
    combined_value = _first_number(
        combined,
        "maximum_accepted_combined_target_tokens",
        "observed_value",
    )
    if combined_value is not None:
        parts.append(
            f"combined prompt-plus-output target transport-accepted through {_format_integer(combined_value)} "
            "estimated tokens; retrieval was not verified"
        )
    if not parts:
        return "No conclusive context boundary was established."
    return "; ".join(parts) + ". Estimated tokens can differ from the provider tokenizer."


def _limit_finding(item: dict[str, str]) -> str:
    """Turn the machine boundary record into a compact, auditable statement."""

    dimension = str(item.get("dimension") or "")
    finding = str(item.get("finding") or "")
    if "nonmonotonic" in finding:
        return "Mixed accept/reject order; no numeric boundary"
    if dimension == "prompt context window":
        value = _first_number(
            item,
            "maximum_functionally_valid_input_tokens",
            "maximum_accepted_input_tokens",
            "observed_value",
        )
        if value is not None:
            return f"Retrieval-valid through {_format_integer(value)} estimated tokens"
    if dimension == "combined prompt + requested output":
        value = _first_number(
            item,
            "maximum_accepted_combined_target_tokens",
            "observed_value",
        )
        if value is not None:
            return f"Transport accepted through {_format_integer(value)} estimated tokens"
    if dimension == "output limit":
        value = _first_number(item, "maximum_realized_output_tokens", "observed_value")
        if value is not None:
            return f"Observed generation through {_format_integer(value)} tokens"
    return _state_text(finding or "measured_no_exact_boundary")


def _current_capacity(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("source_id") == CAPACITY_SOURCE]


def _endpoint_workload_evidence(
    endpoint: str,
    capacity_rows: list[dict[str, str]],
    fixed_index: dict[tuple[str | None, str | None], dict[str, str]],
) -> list[dict[str, str]]:
    """Keep the 100K adaptive and historical 32K fixed-rate recipes distinct."""

    capacity_index = {
        str(row.get("shape")): row for row in capacity_rows if row.get("endpoint_id") == endpoint
    }
    result: list[dict[str, str]] = []
    for shape in SHAPES:
        capacity_row = capacity_index.get(shape)
        fixed_row = fixed_index.get((endpoint, shape))
        capacity_text = (
            _capacity_result(capacity_row)
            if capacity_row is not None
            else "adaptive search not run for this exact recipe"
        )
        fixed_text = (
            _fixed_rate_state(fixed_row)
            if fixed_row is not None
            else "fixed-rate test not run for this exact recipe"
        )
        result.append(
            {
                "shape": shape,
                "capacity_text": capacity_text,
                "fixed_rate_text": fixed_text,
            }
        )
    return result


def _validate_variation_tables(
    panel_rows: list[dict[str, str]],
    across_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
) -> None:
    """Require every registered variation estimand exactly once, including mixed subtype."""

    mixed_subtype = {
        endpoint: (
            "long_short:in50000:out128"
            if endpoint == "minimax-m2.5"
            else "long_short:in100000:out128"
        )
        for endpoint in ENDPOINTS
    }
    subtype_by_endpoint_shape = {
        (endpoint, shape): (
            ("structured:in1024:out512", mixed_subtype[endpoint])
            if shape == "mixed"
            else ("not_applicable",)
        )
        for endpoint in ENDPOINTS
        for shape in ("short_short", "long_short", "short_long", "mixed")
    }
    strata = ("stable_exact_prompt", "panel_unique_cold")
    expected_panel = {
        (endpoint, shape, subtype, panel, stratum)
        for (endpoint, shape), subtypes in subtype_by_endpoint_shape.items()
        for subtype in subtypes
        for panel in range(7)
        for stratum in strata
    }
    expected_across = {
        (endpoint, shape, subtype, stratum)
        for (endpoint, shape), subtypes in subtype_by_endpoint_shape.items()
        for subtype in subtypes
        for stratum in strata
    }
    expected_paired = {
        (endpoint, shape, subtype)
        for (endpoint, shape), subtypes in subtype_by_endpoint_shape.items()
        for subtype in subtypes
    }

    def exact(
        rows: list[dict[str, str]], keys: tuple[str, ...], expected: set[tuple[Any, ...]], name: str
    ) -> None:
        observed = [tuple(row.get(key) for key in keys) for row in rows]
        if len(observed) != len(set(observed)):
            raise ValueError(f"{name} contains duplicate full identities")
        observed_set = set(observed)
        if observed_set != expected:
            missing = len(expected - observed_set)
            extra = len(observed_set - expected)
            raise ValueError(
                f"{name} identity mismatch: expected {len(expected)}, found {len(observed)}; "
                f"missing {missing}, extra {extra}"
            )

    normalized_panel = [{**row, "panel": str(_integer(row.get("panel")))} for row in panel_rows]
    expected_panel_text = {
        (endpoint, shape, subtype, str(panel), stratum)
        for endpoint, shape, subtype, panel, stratum in expected_panel
    }
    exact(
        normalized_panel,
        ("route_id", "shape", "mixed_subtype", "panel", "cache_stratum"),
        expected_panel_text,
        "variation panel table",
    )
    exact(
        across_rows,
        ("route_id", "shape", "mixed_subtype", "cache_stratum"),
        expected_across,
        "variation across-panel table",
    )
    exact(
        paired_rows,
        ("route_id", "shape", "mixed_subtype"),
        expected_paired,
        "variation paired-cache table",
    )


def _variation_endpoint_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for endpoint in ENDPOINTS:
        endpoint_rows = [row for row in rows if row.get("route_id") == endpoint]
        total = sum(_integer(row.get("attempted_n")) for row in endpoint_rows)
        success = sum(_integer(row.get("success_n")) for row in endpoint_rows)
        low, high = _wilson(success, total)
        result[endpoint] = {
            "n": float(total),
            "success": float(success),
            "rate": success / total if total else 0.0,
            "low": low,
            "high": high,
        }
    return result


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save_figure(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=210, bbox_inches="tight", pad_inches=0.16)
    return path


def _plot_reliability_forest(panel_rows: list[dict[str, str]], destination: Path) -> Path:
    plt = _configure_matplotlib()
    stats = _variation_endpoint_stats(panel_rows)
    endpoints = list(reversed(ENDPOINTS))
    estimates = [stats[endpoint]["rate"] * 100 for endpoint in endpoints]
    lows = [stats[endpoint]["low"] * 100 for endpoint in endpoints]
    highs = [stats[endpoint]["high"] * 100 for endpoint in endpoints]
    colors = [
        TEAL if estimate >= 95 else ORANGE if estimate >= 80 else RED for estimate in estimates
    ]
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    for index, (estimate, low, high, color, endpoint) in enumerate(
        zip(estimates, lows, highs, colors, endpoints, strict=True)
    ):
        ax.plot([low, high], [index, index], color=color, linewidth=2.4, solid_capstyle="round")
        ax.scatter(
            [estimate], [index], color=color, s=48, zorder=3, edgecolor="white", linewidth=0.8
        )
        n = int(stats[endpoint]["n"])
        ax.text(
            min(101.6, high + 0.9),
            index,
            f"{estimate:.1f}%  (n={n})",
            va="center",
            fontsize=8.8,
            color=INK,
        )
    ax.axvline(95, color=SLATE, linestyle="--", linewidth=1.0)
    ax.set_yticks(range(len(endpoints)), [_label(endpoint) for endpoint in endpoints])
    ax.set_xlim(35, 104)
    ax.set_xlabel("Successful requests (%) with Wilson 95% interval")
    ax.set_title("Six-hour low-load reliability by endpoint", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save_figure(fig, destination)


def _plot_panel_outcomes(panel_rows: list[dict[str, str]], destination: Path) -> Path:
    plt = _configure_matplotlib()
    by_panel: dict[int, Counter[str]] = defaultdict(Counter)
    for row in panel_rows:
        panel = _integer(row.get("panel"))
        total = _integer(row.get("attempted_n"))
        success = _integer(row.get("success_n"))
        try:
            statuses = json.loads(str(row.get("status_counts") or "{}"))
        except json.JSONDecodeError:
            statuses = {}
        rate_limited = _integer(statuses.get("rate_limited"))
        timeout = _integer(statuses.get("timeout"))
        server = _integer(statuses.get("server_error"))
        client = max(0, total - success - rate_limited - timeout - server)
        by_panel[panel].update(
            success=success,
            rate_limited=rate_limited,
            timeout=timeout,
            other=server + client,
        )
    panels = sorted(by_panel)
    categories = (
        ("success", "Success", TEAL),
        ("rate_limited", "Rate limited", ORANGE),
        ("timeout", "Timeout", RED),
        ("other", "Other failure", SLATE),
    )
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    bottom = [0.0] * len(panels)
    for key, label, color in categories:
        values = []
        for panel in panels:
            total = sum(by_panel[panel].values())
            values.append(100 * by_panel[panel][key] / total if total else 0.0)
        ax.bar(panels, values, bottom=bottom, color=color, width=0.67, label=label)
        bottom = [left + value for left, value in zip(bottom, values, strict=True)]
    for panel in panels:
        total = sum(by_panel[panel].values())
        success = by_panel[panel]["success"]
        ax.text(panel, 102.0, f"{100 * success / total:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(panels, [f"{panel}h" for panel in panels])
    ax.set_ylim(0, 107)
    ax.set_ylabel("Share of 176 scheduled requests (%)")
    ax.set_xlabel("Hours since the study began")
    ax.set_title(
        "Every hourly panel completed; failures rose late in the run", loc="left", fontweight="bold"
    )
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, destination)


def _plot_capacity_shape(
    capacity_rows: list[dict[str, str]], shape: str, destination: Path
) -> Path:
    plt = _configure_matplotlib()
    rows = [row for row in capacity_rows if row.get("shape") == shape]
    index = {row.get("endpoint_id"): row for row in rows}
    endpoints = list(reversed(ENDPOINTS))
    fig, ax = plt.subplots(figsize=(10.4, 6.1))
    positive_values: list[float] = []
    for position, endpoint in enumerate(endpoints):
        row = index.get(endpoint)
        if not row:
            axis_y = ax.get_yaxis_transform()
            ax.scatter(
                [0.025],
                [position],
                marker="s",
                s=38,
                color=SLATE,
                transform=axis_y,
                zorder=3,
            )
            ax.text(
                0.055,
                position,
                "100K search not run; separate 32K evidence on endpoint page",
                va="center",
                fontsize=8.0,
                color=SLATE,
                transform=axis_y,
            )
            continue
        claim = str(row.get("capacity_claim") or "")
        low = _number(row.get("capacity_lower_bound_rps"))
        high = _number(row.get("capacity_upper_bound_rps"))
        if low is not None:
            positive_values.append(low)
        if high is not None:
            positive_values.append(high)
        if claim == "confirmed_bracketed_interval" and low is not None and high is not None:
            ax.plot([low, high], [position, position], color=BLUE, linewidth=2.5)
            ax.scatter([low], [position], color=TEAL, s=45, zorder=3)
            ax.scatter([high], [position], marker="x", color=RED, s=55, zorder=3)
            ax.text(
                high * 1.08,
                position,
                f"{_format_rate(low)}-{_format_rate(high)} req/s",
                va="center",
                fontsize=8.4,
            )
        elif claim == "confirmed_right_censored_lower_bound" and low is not None:
            ax.scatter([low], [position], color=TEAL, s=48, zorder=3)
            ax.annotate(
                "",
                xy=(low * 1.5, position),
                xytext=(low, position),
                arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.8},
            )
            ax.text(
                low * 1.62,
                position,
                f"at least {_format_rate(low)} req/s",
                va="center",
                fontsize=8.4,
            )
        elif claim == "censored_no_valid_healthy_epoch" and high is not None:
            ax.scatter([high], [position], marker="|", color=ORANGE, s=80, zorder=3)
            ax.text(
                high * 1.08,
                position,
                f"search stopped at {_format_rate(high)} req/s; lower untested",
                va="center",
                fontsize=8.4,
            )
        else:
            marker_value = (
                _first_number(row, "highest_observed_healthy_rps", "tested_min_offered_rps") or 0.03
            )
            positive_values.append(marker_value)
            ax.scatter([marker_value], [position], marker="D", color=ORANGE, s=40, zorder=3)
            ax.text(marker_value * 1.08, position, _state_text(claim), va="center", fontsize=8.2)
    ax.set_xscale("log")
    if positive_values:
        ax.set_xlim(max(0.02, min(positive_values) / 2.3), max(positive_values) * 3.2)
    ax.set_yticks(range(len(endpoints)), [_label(endpoint) for endpoint in endpoints])
    ax.set_xlabel("Offered requests per second (log scale)")
    ax.set_title(SHAPE_LABELS.get(shape, shape), loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save_figure(fig, destination)


def _plot_fixed_rate(fixed_rows: list[dict[str, str]], destination: Path) -> Path:
    plt = _configure_matplotlib()
    usable = [row for row in fixed_rows if row.get("source_id") == FIXED_RATE_SOURCE]
    index = {(row.get("endpoint_id"), row.get("shape")): row for row in usable}
    shapes = ("short_short", "input32k_short", "short_long", "mixed")
    values = {
        "passed": 3,
        "tested-rate non-pass": 2,
        "transport-gated": 1,
        "measured unresolved": 0,
        "could not establish baseline": 0,
        "not measured": -1,
    }
    matrix = []
    for endpoint in ENDPOINTS:
        matrix.append(
            [values[_fixed_rate_state(index.get((endpoint, shape), {}))] for shape in shapes]
        )
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#E2E8F0", "#94A3B8", "#FBBF24", "#FCA5A5", "#5EEAD4"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10.3, 5.7))
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(shapes)), [SHAPE_LABELS[shape] for shape in shapes])
    ax.set_yticks(range(len(ENDPOINTS)), [_label(endpoint) for endpoint in ENDPOINTS])
    for y, endpoint in enumerate(ENDPOINTS):
        for x, shape in enumerate(shapes):
            state = _fixed_rate_state(index.get((endpoint, shape), {}))
            headline = {
                "passed": "PASS",
                "tested-rate non-pass": "NON-PASS",
                "transport-gated": "NO VALID TEST",
                "measured unresolved": "UNRESOLVED",
                "could not establish baseline": "NO VALID TEST",
                "not measured": "NOT RUN",
            }[state]
            row = index.get((endpoint, shape), {})
            tested_rate = _number(row.get("candidate_rate_rps"))
            label = (
                f"{headline}\n{_format_rate(tested_rate)} req/s"
                if tested_rate is not None
                else headline
            )
            ax.text(
                x, y, label, ha="center", va="center", fontsize=8.0, color=INK, fontweight="bold"
            )
    ax.tick_params(axis="x", rotation=0, length=0, pad=9)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        "Two-minute fixed-rate test: did the registered rules pass?",
        loc="left",
        fontweight="bold",
    )
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.text(
        0.01,
        0.01,
        "NON-PASS means the tested rate did not meet every registered condition; it is not an "
        "endpoint failure. Exact condition reasons are retained in the block table.",
        fontsize=8.5,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save_figure(fig, destination)


def _plot_cache_effect(cache_rows: list[dict[str, str]], destination: Path) -> Path:
    plt = _configure_matplotlib()
    panels = (
        ("short_short", "not_applicable", "Short / short"),
        ("long_short", "not_applicable", "Long input / short"),
        ("short_long", "not_applicable", "Short / long"),
        ("mixed", "structured:in1024:out512", "Mixed: structured"),
        ("mixed", "long_short:", "Mixed: long input"),
    )
    endpoints = list(reversed(ENDPOINTS))
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.0), sharey=True)
    axes_flat = list(axes.flat)
    values_seen: list[float] = []
    for axis, (shape, subtype, title) in zip(axes_flat, panels, strict=False):
        index = {
            row.get("route_id"): row
            for row in cache_rows
            if row.get("shape") == shape
            and (
                str(row.get("mixed_subtype") or "").startswith(subtype)
                if subtype.endswith(":")
                else row.get("mixed_subtype") == subtype
            )
        }
        for position, endpoint in enumerate(endpoints):
            row = index.get(endpoint, {})
            # Source rows are cold minus stable; reverse the sign so negative means the stable
            # exact prefix was faster, matching the axis label.
            cold_minus_stable = _number(row.get("paired_request_latency_median_difference_median"))
            cold_low = _number(row.get("paired_request_latency_median_difference_median_ci95_low"))
            cold_high = _number(
                row.get("paired_request_latency_median_difference_median_ci95_high")
            )
            estimate = None if cold_minus_stable is None else -cold_minus_stable
            low = None if cold_high is None else -cold_high
            high = None if cold_low is None else -cold_low
            if estimate is None:
                axis.scatter([0], [position], marker="s", color=SLATE, s=18)
                continue
            values_seen.append(estimate)
            color = TEAL if estimate < 0 else ORANGE
            if low is not None and high is not None:
                values_seen.extend([low, high])
                axis.plot([low, high], [position, position], color=color, linewidth=1.5)
            axis.scatter([estimate], [position], color=color, s=24, zorder=3)
        axis.axvline(0, color=INK, linewidth=0.8)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="y", length=0, labelsize=7.4)
    axes_flat[-1].axis("off")
    for axis in axes[:, 0]:
        axis.set_yticks(range(len(endpoints)), [_label(endpoint) for endpoint in endpoints])
    for axis in axes[1, :2]:
        axis.set_xlabel("Stable - fresh latency (s)", fontsize=8.7)
    if values_seen:
        lower, upper = min(values_seen), max(values_seen)
        span = max(1.0, upper - lower)
        for axis in axes_flat[:-1]:
            axis.set_xlim(lower - span * 0.08, upper + span * 0.08)
    fig.suptitle(
        "Stable exact-prefix effect by matched workload",
        x=0.07,
        y=0.995,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout()
    return _save_figure(fig, destination)


def _pdf_styles() -> dict[str, Any]:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FinalTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=30,
            textColor=colors.HexColor(NAVY),
            alignment=TA_LEFT,
            spaceAfter=9,
        ),
        "subtitle": ParagraphStyle(
            "FinalSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor(SLATE),
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "FinalH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(NAVY),
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "FinalH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor(BLUE),
            spaceBefore=5,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "FinalBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor(INK),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "FinalSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.4,
            textColor=colors.HexColor(SLATE),
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "FinalTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=9.2,
            textColor=colors.HexColor(INK),
        ),
        "table_header": ParagraphStyle(
            "FinalTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9.0,
            textColor=colors.white,
        ),
        "kpi": ParagraphStyle(
            "FinalKpi",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=23,
            textColor=colors.HexColor(NAVY),
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "FinalKpiLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor(SLATE),
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "FinalCallout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor(NAVY),
            spaceAfter=5,
        ),
    }


def _page_footer(canvas: Any, document: Any) -> None:
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.4)
    canvas.line(16 * mm, 13 * mm, document.pagesize[0] - 16 * mm, 13 * mm)
    canvas.setFillColor(colors.HexColor(SLATE))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(
        16 * mm, 8 * mm, "DigitalOcean hosted open-model inference | measured 28-29 Aug 2026"
    )
    canvas.drawRightString(
        document.pagesize[0] - 16 * mm,
        8 * mm,
        f"{canvas.getPageNumber()}",
    )
    canvas.restoreState()


def _table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    header: bool = True,
    font_size: float = 7.3,
) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    commands: list[tuple[Any, ...]] = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
            ]
        )
    else:
        commands.append(
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor(LIGHT)])
        )
    return Table(rows, colWidths=widths, repeatRows=1 if header else 0, style=TableStyle(commands))


def _caption(styles: dict[str, Any], tested: str, shows: str, does_not: str) -> Any:
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows = [
        [Paragraph("Tested", styles["table"]), Paragraph(tested, styles["small"])],
        [Paragraph("Shows", styles["table"]), Paragraph(shows, styles["small"])],
        [Paragraph("Does not show", styles["table"]), Paragraph(does_not, styles["small"])],
    ]
    return Table(
        rows,
        colWidths=[30 * mm, 145 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(LIGHT)),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor(GRID)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        ),
    )


def _image(path: Path, max_width: float, max_height: float) -> Any:
    from PIL import Image as PILImage
    from reportlab.platypus import Image

    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    result = Image(str(path), width=width * scale, height=height * scale)
    result.hAlign = "CENTER"
    return result


def _safe_text_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )


_PRIVATE_IDENTIFIER_FIELDS = frozenset({"request_id", "logical_id", "reservation_id"})
_PRIVATE_IDENTIFIER_PATTERN = re.compile(
    r"(?i)\b(request_id|logical_id|reservation_id)\b"
)


def _scrub_public_value(value: Any) -> Any:
    """Remove internal identifier keys from nested public audit values."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _PRIVATE_IDENTIFIER_FIELDS:
                continue
            if (
                normalized_key == "sampling_unit"
                or normalized_key.endswith("_sampling_unit")
            ) and str(item).strip().lower() == "request_id":
                result[key] = "request"
            else:
                result[key] = _scrub_public_value(item)
        return result
    if isinstance(value, list):
        return [_scrub_public_value(item) for item in value]
    return value


def _sanitize_public_csv(source: Path, destination: Path) -> None:
    """Copy a result CSV while omitting private journal identifiers.

    Scientific values and row counts are preserved. Identifier columns and identifier
    keys embedded in JSON audit cells are publication-internal metadata, so they are
    removed from the public copy.
    """
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        fieldnames = [
            name
            for name in reader.fieldnames
            if name.strip().lower() not in _PRIVATE_IDENTIFIER_FIELDS
        ]
        rows: list[dict[str, str]] = []
        for source_row in reader:
            row: dict[str, str] = {}
            for name in fieldnames:
                value = source_row.get(name, "") or ""
                normalized_name = name.strip().lower()
                if (
                    normalized_name == "sampling_unit"
                    or normalized_name.endswith("_sampling_unit")
                ) and value.strip().lower() == "request_id":
                    value = "request"
                stripped = value.strip()
                if stripped and stripped[0] in "[{":
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        pass
                    else:
                        value = json.dumps(
                            _scrub_public_value(parsed),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                if _PRIVATE_IDENTIFIER_PATTERN.search(value):
                    raise ValueError(
                        "internal identifier token remains after public CSV sanitization: "
                        f"{source.name}:{name}"
                    )
                row[name] = value
            rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_pdf(
    output_pdf: Path,
    *,
    summary_dir: Path,
    variation_dir: Path,
    figures: dict[str, Path],
    inventory: list[dict[str, str]],
    capacity: list[dict[str, str]],
    fixed_rate: list[dict[str, str]],
    capabilities: list[dict[str, str]],
    limits: list[dict[str, str]],
    panel_rows: list[dict[str, str]],
    across_rows: list[dict[str, str]],
    cache_effect_rows: list[dict[str, str]],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    styles = _pdf_styles()
    document = BaseDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title="DigitalOcean hosted inference: technical benchmark and engineering guide",
        author="Sqwish Labs",
        subject="Reproducible dated benchmark snapshot of 11 DigitalOcean-hosted open-model endpoints",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates([PageTemplate(id="body", frames=[frame], onPage=_page_footer)])

    endpoint_stats = _variation_endpoint_stats(panel_rows)
    total_n = int(sum(item["n"] for item in endpoint_stats.values()))
    total_success = int(sum(item["success"] for item in endpoint_stats.values()))
    total_low, total_high = _wilson(total_success, total_n)
    current_capacity = _current_capacity(capacity)
    fixed_current = [row for row in fixed_rate if row.get("source_id") == FIXED_RATE_SOURCE]
    fixed_passes = sum(_fixed_rate_state(row) == "passed" for row in fixed_current)
    fixed_index = {(row.get("endpoint_id"), row.get("shape")): row for row in fixed_current}
    inventory_index = {row.get("endpoint_id"): row for row in inventory}
    capability_index = {
        (row.get("endpoint_id"), row.get("capability_dimension")): row for row in capabilities
    }
    limits_by_endpoint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in limits:
        limits_by_endpoint[str(row.get("endpoint_id"))].append(row)

    story: list[Any] = [
        Spacer(1, 16 * mm),
        Paragraph("DigitalOcean hosted inference", styles["title"]),
        Paragraph(
            "Technical benchmark and engineering guide for the 11 DigitalOcean-hosted open-model routes frozen in the 27-29 August 2026 study",
            styles["subtitle"],
        ),
        Spacer(1, 4 * mm),
        Table(
            [
                [
                    Paragraph("11", styles["kpi"]),
                    Paragraph("1,232", styles["kpi"]),
                    Paragraph("7", styles["kpi"]),
                    Paragraph(f"{100 * total_success / total_n:.1f}%", styles["kpi"]),
                ],
                [
                    Paragraph("hosted open-model endpoints", styles["kpi_label"]),
                    Paragraph("matched low-load requests", styles["kpi_label"]),
                    Paragraph("hourly panels over six hours", styles["kpi_label"]),
                    Paragraph("overall request success", styles["kpi_label"]),
                ],
            ],
            colWidths=[44 * mm] * 4,
            rowHeights=[13 * mm, 14 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(LIGHT)),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            ),
        ),
        Spacer(1, 9 * mm),
        Paragraph(
            "Bottom line: eight endpoints completed every request in the registered six-hour low-load panel. "
            "Qwen3.5 397B A17B and Nemotron 3 Super 120B became materially rate-limited late in the run; "
            "Gemma 4 31B had smaller reliability losses. Capacity and fixed-rate results remain workload-specific.",
            styles["callout"],
        ),
        Paragraph(
            "This report separates observed behavior from product documentation, reports uncertainty and denominators, "
            "and never converts an unmeasured or failed condition into a blank success. Commercial pass-through "
            "routes are outside this DigitalOcean-hosted study.",
            styles["body"],
        ),
        Spacer(1, 5 * mm),
        Paragraph(
            "Measurement window: 28-29 August 2026 (UTC) | Publication build: 29 August 2026",
            styles["small"],
        ),
        Paragraph(
            "Claim boundary: results characterize this measured configuration and six-hour within-run window. "
            "The catalog can change after this dated snapshot. These results are not a 24-hour, daily, diurnal, "
            "or indefinite production guarantee.",
            styles["small"],
        ),
        PageBreak(),
        Paragraph("Engineering decision map", styles["h1"]),
        Paragraph(
            "Start with reliability at low offered load, then consult the exact workload page for capacity. "
            "A green reliability result does not imply that every feature or high-load workload is suitable.",
            styles["body"],
        ),
    ]

    decision_rows: list[list[Any]] = [
        [
            Paragraph("Endpoint", styles["table_header"]),
            Paragraph("6 h low-load reliability", styles["table_header"]),
            Paragraph("Adaptive-load cells with repeat-confirmed result", styles["table_header"]),
            Paragraph("2 min fixed-rate passes", styles["table_header"]),
            Paragraph("Engineering reading", styles["table_header"]),
        ]
    ]
    for endpoint in ENDPOINTS:
        stats = endpoint_stats[endpoint]
        endpoint_capacity = [row for row in current_capacity if row.get("endpoint_id") == endpoint]
        confirmed = sum(
            str(row.get("capacity_claim") or "").startswith("confirmed_")
            for row in endpoint_capacity
        )
        fixed_pass = sum(
            _fixed_rate_state(row) == "passed"
            for row in fixed_current
            if row.get("endpoint_id") == endpoint
        )
        if stats["rate"] == 1.0:
            reading = (
                "Reliable at the registered low-load panel; size with the exact workload evidence."
            )
        elif stats["rate"] >= 0.9:
            reading = "Mostly reliable here; retain retry/backoff and watch late-run errors."
        else:
            reading = "Material rate limiting in this run; use conservative concurrency and live telemetry."
        decision_rows.append(
            [
                Paragraph(_label(endpoint), styles["table"]),
                Paragraph(
                    f"{int(stats['success'])}/{int(stats['n'])} = {100 * stats['rate']:.1f}%<br/>"
                    f"95% CI {100 * stats['low']:.1f}-{100 * stats['high']:.1f}%",
                    styles["table"],
                ),
                Paragraph(f"{confirmed}/{len(endpoint_capacity)}", styles["table"]),
                Paragraph(f"{fixed_pass}/4", styles["table"]),
                Paragraph(reading, styles["table"]),
            ]
        )
    story.extend(
        [
            _table(decision_rows, [35 * mm, 35 * mm, 34 * mm, 24 * mm, 48 * mm]),
            Spacer(1, 4 * mm),
            Paragraph(
                f"Across all endpoints: {total_success}/{total_n} successes "
                f"(Wilson 95% CI {100 * total_low:.2f}-{100 * total_high:.2f}%). "
                "Do not average endpoint latency or throughput into a provider ranking; models and recipes differ.",
                styles["small"],
            ),
            PageBreak(),
            Paragraph("Exactly what was tested", styles["h1"]),
            Paragraph(
                "Four workload recipes exercise different bottlenecks. Every result in this report stays attached "
                "to its endpoint, recipe, source campaign, and sampling unit.",
                styles["body"],
            ),
        ]
    )
    recipe_rows = [
        [
            Paragraph("Recipe", styles["table_header"]),
            Paragraph("Registered request", styles["table_header"]),
            Paragraph("What it stresses", styles["table_header"]),
            Paragraph("How to use the result", styles["table_header"]),
        ],
        [
            Paragraph("Short / short", styles["table"]),
            Paragraph("256-token prompt -> 128-token answer target", styles["table"]),
            Paragraph("Request scheduling, queueing, and first-token response", styles["table"]),
            Paragraph("Interactive assistants and small structured calls", styles["table"]),
        ],
        [
            Paragraph("Long input / short answer", styles["table"]),
            Paragraph(
                "100,000-token prompt -> 128 tokens; MiniMax uses 50,000 tokens", styles["table"]
            ),
            Paragraph("Prompt ingestion, context handling, and cache reuse", styles["table"]),
            Paragraph("Document analysis and retrieval-heavy requests", styles["table"]),
        ],
        [
            Paragraph("Short / long", styles["table"]),
            Paragraph("256-token prompt -> 4,096-token answer target", styles["table"]),
            Paragraph("Long generation, request occupancy, timeouts", styles["table"]),
            Paragraph("Drafting, synthesis, and extended reasoning outputs", styles["table"]),
        ],
        [
            Paragraph("Mixed traffic", styles["table"]),
            Paragraph(
                "Fixed seeded blend of short, long-input, long-output, JSON, and tool-like requests",
                styles["table"],
            ),
            Paragraph("Scheduler fairness across a realistic heterogeneous queue", styles["table"]),
            Paragraph(
                "Shared application traffic; do not compare its latency to one homogeneous recipe",
                styles["table"],
            ),
        ],
    ]
    story.extend(
        [
            _table(recipe_rows, [31 * mm, 51 * mm, 45 * mm, 49 * mm]),
            Spacer(1, 6 * mm),
            Paragraph("Experiment sequence", styles["h2"]),
            Paragraph(
                "1. Functional and boundary probes establish what the API accepts. 2. Adaptive load search raises "
                "offered traffic, backs off on degradation, and requires three separated healthy confirmations. "
                "3. A two-minute fixed-rate test holds one selected rate for four adjacent 30-second blocks and "
                "checks recovery. 4. Seven matched hourly panels measure low-load variation over exactly six hours.",
                styles["body"],
            ),
            Paragraph(
                "Internal code name note: the adaptive load search is AIMD (additive increase, multiplicative decrease). "
                "The two-minute fixed-rate stability test was historically called a soak; this report uses the plain name.",
                styles["small"],
            ),
            PageBreak(),
            Paragraph("Evidence and uncertainty", styles["h1"]),
        ]
    )
    evidence_rows = [
        [
            Paragraph("Question", styles["table_header"]),
            Paragraph("Evidence", styles["table_header"]),
            Paragraph("Uncertainty unit", styles["table_header"]),
            Paragraph("Valid claim", styles["table_header"]),
        ],
        [
            Paragraph("Does a request work?", styles["table"]),
            Paragraph("Functional/capability probes", styles["table"]),
            Paragraph("Request; Wilson interval for binary outcomes", styles["table"]),
            Paragraph(
                "Observed support or measured failure for the exact request", styles["table"]
            ),
        ],
        [
            Paragraph("What rate repeatedly passes?", styles["table"]),
            Paragraph("Adaptive load search + three separated confirmations", styles["table"]),
            Paragraph("Load epoch/block", styles["table"]),
            Paragraph(
                "Measured lower bound or bracket, never a theoretical maximum", styles["table"]
            ),
        ],
        [
            Paragraph("Does one rate hold briefly?", styles["table"]),
            Paragraph("Four contiguous 30-second fixed-rate blocks", styles["table"]),
            Paragraph("Block; exploratory Student-t interval", styles["table"]),
            Paragraph("Two-minute measured pass/failure at that rate", styles["table"]),
        ],
        [
            Paragraph("Did behavior change over six hours?", styles["table"]),
            Paragraph("Seven matched hourly panels", styles["table"]),
            Paragraph(
                "Request-level Wilson/bootstrap; panel-cluster intervals for repeated-panel effects",
                styles["table"],
            ),
            Paragraph("Within-run six-hour variation only", styles["table"]),
        ],
    ]
    story.extend(
        [
            _table(evidence_rows, [37 * mm, 50 * mm, 45 * mm, 44 * mm]),
            Spacer(1, 6 * mm),
            Paragraph("Metric eligibility", styles["h2"]),
            Paragraph(
                "End-to-end latency is reported for successful requests. Time to first token is reported only when the "
                "stream exposed a valid first-output timestamp. Decode throughput additionally requires complete usage, "
                "a decode window of at least one second, and at least 16 output tokens. In the six-hour panel only 198 "
                "requests met that stricter throughput rule, across three endpoints; the other eight endpoints are "
                "labelled not estimable rather than assigned a number.",
                styles["body"],
            ),
            Paragraph(
                "The eligible six-hour decode observations ranged up to 87.3 tokens/s. Impossible historical outliers "
                "are excluded by the explicit timestamp, duration, output-length, and usage checks and retained in the audit.",
                styles["body"],
            ),
        ]
    )

    for shape, figure_key in (
        ("short_short", "capacity-short_short"),
        ("input100k_short", "capacity-input100k_short"),
        ("short_long", "capacity-short_long"),
        ("mixed", "capacity-mixed"),
    ):
        story.extend(
            [
                PageBreak(),
                Paragraph(f"Adaptive load search: {SHAPE_LABELS[shape]}", styles["h1"]),
                _image(figures[figure_key], 176 * mm, 118 * mm),
                Spacer(1, 3 * mm),
                _caption(
                    styles,
                    "Open-loop offered traffic for this exact endpoint and recipe; a numeric result requires three separated healthy confirmations.",
                    "A teal point is a repeatedly passing tested rate. An orange marker means the search did not establish a bound; read its direct label for the tested floor.",
                    "Theoretical maximum, recommended production rate, or performance for a different prompt/output recipe.",
                ),
            ]
        )

    story.extend(
        [
            PageBreak(),
            Paragraph("Two-minute fixed-rate stability", styles["h1"]),
            _image(figures["fixed-rate"], 176 * mm, 112 * mm),
            Spacer(1, 3 * mm),
            _caption(
                styles,
                "One candidate rate for four adjacent 30-second analysis blocks, with registered reliability, latency, queueing, usage, quality, and recovery checks.",
                f"{fixed_passes} of 44 endpoint-by-workload cells passed every acceptance condition. Non-passes mean the tested rate missed at least one condition; transport-gated cells have no scientific pass/fail result.",
                "Six-hour stability or safe operation above the tested rate. Adjacent-block intervals do not model serial correlation.",
            ),
            PageBreak(),
            Paragraph("Six-hour matched panel: reliability", styles["h1"]),
            _image(figures["reliability-forest"], 176 * mm, 112 * mm),
            Spacer(1, 3 * mm),
            _caption(
                styles,
                "112 scheduled low-load requests per endpoint: four recipes x four repeats x seven hourly panels; no retries.",
                "Endpoint-specific success rate with Wilson 95% interval. Eight endpoints completed all 112 requests.",
                "Behavior at high concurrency, outside this six-hour window, or under a different region/account/quota state.",
            ),
            PageBreak(),
            Paragraph("Six-hour matched panel: when failures appeared", styles["h1"]),
            _image(figures["panel-outcomes"], 176 * mm, 104 * mm),
            Spacer(1, 3 * mm),
            _caption(
                styles,
                "Seven globally scheduled panels at hours 0-6; each panel contains 176 matched requests at one request per second.",
                "All panels completed. Success fell in hours 5 and 6, driven mainly by HTTP 429 responses on three endpoints.",
                "A daily or diurnal pattern. Seven points from one contiguous run cannot separate time-of-day from transient service or quota conditions.",
            ),
            PageBreak(),
            Paragraph("Prompt reuse and automatic exact-prefix caching", styles["h1"]),
            _image(figures["cache-effect"], 176 * mm, 111 * mm),
            Spacer(1, 3 * mm),
            _caption(
                styles,
                "Within each endpoint, recipe, and panel, two stable exact-prefix requests were paired with two panel-unique fresh-prefix requests.",
                "Effects are reported separately for each endpoint and recipe. Some cells were faster with a stable prefix; many uncertainty intervals crossed zero.",
                "A cache hit guarantee or universal latency reduction. DigitalOcean caching is automatic and best effort.",
            ),
            Paragraph(
                "Stable and fresh-prefix reliability was similar overall, but combining unlike models and recipes into one effect would hide important differences. "
                "The chart and paired-cache table therefore keep every endpoint and recipe separate. Negative values mean the stable prefix was faster; an interval "
                "crossing zero means this six-hour run could not establish the direction. Cache-read counters confirm reuse only where the provider exposed it; a "
                "cache read does not guarantee lower time-to-first-token, total latency, or settled cost in every cell.",
                styles["body"],
            ),
            PageBreak(),
            Paragraph("Capability evidence", styles["h1"]),
            Paragraph(
                "Transport success and functional success are separate. A 2xx response that does not satisfy the requested schema or tool contract is not a functional pass.",
                styles["body"],
            ),
            Paragraph(
                "Platform contracts: hosted open models use automatic best-effort exact-prefix prompt caching. DigitalOcean batch inference does not support these open-source hosted models. "
                "Those documented product states are not inferred from malformed-parameter probes.",
                styles["small"],
            ),
            Paragraph(
                "Official product sources (verified 29 August 2026): "
                '<link href="https://docs.digitalocean.com/products/inference/how-to/use-prompt-caching/" color="#2457A7">prompt caching</link>; '
                '<link href="https://docs.digitalocean.com/products/inference/details/limits/" color="#2457A7">batch limits</link>; '
                '<link href="https://docs.digitalocean.com/products/inference/details/models/" color="#2457A7">model catalog and feature matrix</link>.'
                " Product documentation can change after the measurement date.",
                styles["small"],
            ),
        ]
    )
    capability_dims = (
        "response_format",
        "tools",
        "parallel_tool_calls",
        "vision",
        "automatic_prompt_cache",
        "batch_open_models",
    )
    capability_rows: list[list[Any]] = [
        [Paragraph("Endpoint", styles["table_header"])]
        + [
            Paragraph(dim.replace("_", " ").title(), styles["table_header"])
            for dim in capability_dims
        ]
    ]
    for endpoint in ENDPOINTS:
        row: list[Any] = [Paragraph(_label(endpoint), styles["table"])]
        for dim in capability_dims:
            state = _compact_capability_state(capability_index, endpoint, dim, panel_rows)
            row.append(Paragraph(state.replace("_", " "), styles["table"]))
        capability_rows.append(row)
    story.extend(
        [
            _table(
                capability_rows,
                [36 * mm, 23 * mm, 22 * mm, 25 * mm, 20 * mm, 27 * mm, 23 * mm],
                font_size=5.9,
            ),
            PageBreak(),
            Paragraph("Observed context boundaries", styles["h1"]),
            Paragraph(
                "Accepted probes are not silently upgraded to exact maxima. Right-censored means the endpoint accepted through the largest tested value; "
                "interval-censored means the boundary lies between a measured accept and reject. Timeouts, 429s, and 5xx responses are inconclusive for capability.",
                styles["body"],
            ),
        ]
    )
    limit_rows: list[list[Any]] = [
        [
            Paragraph("Endpoint", styles["table_header"]),
            Paragraph("Dimension", styles["table_header"]),
            Paragraph("Documented", styles["table_header"]),
            Paragraph("Measured finding", styles["table_header"]),
            Paragraph("Boundary state", styles["table_header"]),
        ]
    ]
    for endpoint in ENDPOINTS:
        selected = [
            row
            for row in limits_by_endpoint[endpoint]
            if row.get("dimension")
            in {"prompt context window", "maximum output", "context", "output"}
        ]
        if not selected:
            selected = limits_by_endpoint[endpoint][:2]
        for item in selected[:2]:
            limit_rows.append(
                [
                    Paragraph(_label(endpoint), styles["table"]),
                    Paragraph(str(item.get("dimension") or "").replace("_", " "), styles["table"]),
                    Paragraph(
                        _format_integer(item.get("documented_value")), styles["table"]
                    ),
                    Paragraph(
                        _limit_finding(item),
                        styles["table"],
                    ),
                    Paragraph(
                        _state_text(str(item.get("boundary_censoring") or "not established")),
                        styles["table"],
                    ),
                ]
            )
    story.append(_table(limit_rows, [35 * mm, 30 * mm, 27 * mm, 58 * mm, 26 * mm], font_size=6.6))

    output_rows: list[list[Any]] = [
        [
            Paragraph("Endpoint", styles["table_header"]),
            Paragraph("Frozen catalog value", styles["table_header"]),
            Paragraph("Largest response observed", styles["table_header"]),
            Paragraph("What this establishes", styles["table_header"]),
        ]
    ]
    for endpoint in ENDPOINTS:
        item = next(
            (
                row
                for row in limits_by_endpoint[endpoint]
                if row.get("dimension") == "output limit"
            ),
            {},
        )
        observed = _first_number(item, "maximum_realized_output_tokens", "observed_value")
        output_rows.append(
            [
                Paragraph(_label(endpoint), styles["table"]),
                Paragraph(_format_integer(item.get("documented_value")), styles["table"]),
                Paragraph(
                    f"{_format_integer(observed)} generated tokens"
                    if observed is not None
                    else "not measured",
                    styles["table"],
                ),
                Paragraph(
                    "At least this response length; not an exact maximum"
                    if observed is not None
                    else "No usable long-output observation",
                    styles["table"],
                ),
            ]
        )
    story.extend(
        [
            PageBreak(),
            Paragraph("Observed long-output responses", styles["h1"]),
            Paragraph(
                "These probes answer a practical question: how long a response did the endpoint actually complete in this campaign? They do not silently turn "
                "that response length into a maximum. A model can stop because its answer is complete or it emits an end token; establishing an exact limit "
                "requires a measured accept/reject boundary.",
                styles["body"],
            ),
            _table(output_rows, [42 * mm, 35 * mm, 46 * mm, 53 * mm], font_size=6.7),
            Spacer(1, 4 * mm),
            _caption(
                styles,
                "Registered long-output probes from the static/capability campaign; one largest completed response retained per endpoint.",
                "A measured lower bound on completed response length for the tested request, next to the frozen catalog value used by the runner.",
                "The exact maximum output, a guarantee that every request reaches this length, or current catalog metadata after 29 August 2026.",
            ),
        ]
    )

    story.extend(
        [
            PageBreak(),
            Paragraph("Operational implications", styles["h1"]),
            Paragraph(
                "Use retry with randomized exponential backoff, per-endpoint concurrency controls, and circuit breaking. Treat 429 as a capacity signal, not an instruction to replay immediately. "
                "Do not retry malformed 400-class requests blindly. Preserve idempotency and observe both request count and tokens per minute because the binding quota can vary with workload.",
                styles["body"],
            ),
            Paragraph("Endpoint-specific cautions from this run", styles["h2"]),
            Paragraph(
                "<b>Qwen3.5 397B A17B:</b> 72/112 successes (64.3%); 30 rate limits and 10 timeouts. Its short/long cell completed only 11/28 requests. "
                "Operate with conservative concurrency and alert on both 429 and long-tail latency.",
                styles["body"],
            ),
            Paragraph(
                "<b>Nemotron 3 Super 120B:</b> 62/112 successes (55.4%); 47 rate limits and 3 timeouts. Failures were concentrated late in the six-hour run. "
                "Production use needs strict adaptive throttling and a fallback path.",
                styles["body"],
            ),
            Paragraph(
                "<b>Gemma 4 31B:</b> 105/112 successes (93.8%). Reliability was substantially better than the two endpoints above, but not perfect; retain backoff and monitor protocol-level failures in addition to HTTP status.",
                styles["body"],
            ),
            Paragraph(
                "For the eight 112/112 endpoints, the correct conclusion is narrower: they were reliable at this registered low offered load during this run. "
                "Use the workload-specific adaptive-load and fixed-rate pages before setting production concurrency.",
                styles["body"],
            ),
            PageBreak(),
            Paragraph("Reproducibility and provenance", styles["h1"]),
            Paragraph(
                "The benchmark runner is open-loop, parallel, idempotent, and receipt-backed. Requests remain attached to an exact endpoint, recipe, phase, panel, and source commit. "
                "The public package contains aggregate tables and audit manifests only; credentials, prompts, generated outputs, response bodies, and raw headers are excluded.",
                styles["body"],
            ),
        ]
    )
    provenance_rows = [
        [
            Paragraph("Evidence layer", styles["table_header"]),
            Paragraph("Exact source", styles["table_header"]),
            Paragraph("What it contributes", styles["table_header"]),
        ],
        [
            Paragraph("Static / capability", styles["table"]),
            Paragraph("do-complete-20260827-r1; terminal SHA256SUMS 0b2a4772...", styles["table"]),
            Paragraph(
                "Caching, capabilities, context/output, quality, baseline latency", styles["table"]
            ),
        ],
        [
            Paragraph("Adaptive load", styles["table"]),
            Paragraph(
                "do-capacity-20260828-r2 plus completed gap cells from do-six-hour-variation-20260828-r1",
                styles["table"],
            ),
            Paragraph(
                "Endpoint-by-workload confirmed bounds and brackets; unresolved searches remain explicitly labelled",
                styles["table"],
            ),
        ],
        [
            Paragraph("Fixed-rate stability", styles["table"]),
            Paragraph("do-direct-soak-20260823-r1", styles["table"]),
            Paragraph("Four-block two-minute stability and recovery evidence", styles["table"]),
        ],
        [
            Paragraph("Six-hour variation", styles["table"]),
            Paragraph(
                "do-six-hour-variation-20260828-r1; SHA256SUMS 9d3ea8a161ce...", styles["table"]
            ),
            Paragraph(
                "7/7 hourly panels; 1,232/1,232 core requests; stable versus fresh-prefix matched design",
                styles["table"],
            ),
        ],
    ]
    story.extend(
        [
            _table(provenance_rows, [38 * mm, 69 * mm, 69 * mm]),
            Spacer(1, 5 * mm),
            Paragraph(
                "Six-hour immutable identities: source commit 603061da297c5422a7ba1750a110d80bf26b4757; "
                "campaign hash 50218753aa9404e0946bbbc04692f95986df6042ad57a54cf19ea467a11c19de; "
                "terminal ledger SHA-256 b78194ef68ba9dbdbec9f6000917a266231b6b915722da4f4429722e894669dc.",
                styles["small"],
            ),
            Paragraph(
                "Machine-readable companions include the capacity table, fixed-rate cell/block tables, capability and boundary tables, "
                "six-hour panel/across-panel/cache-effect tables, outlier audit, coverage matrix, and recursive publication-safety result.",
                styles["body"],
            ),
        ]
    )

    for endpoint in ENDPOINTS:
        inventory_row = inventory_index.get(endpoint, {})
        stats = endpoint_stats[endpoint]
        endpoint_capacity = [row for row in current_capacity if row.get("endpoint_id") == endpoint]
        counts = _endpoint_status_counts(panel_rows, endpoint)
        decision_label, decision_color, decision_text = _endpoint_decision(stats, counts)
        context_text = _format_integer(inventory_row.get("context_window"))
        output_text = _format_integer(inventory_row.get("max_output_tokens"))
        if context_text != "not documented":
            context_text += " tokens"
        if output_text != "not documented":
            output_text += " tokens"
        price_text = (
            f"${inventory_row.get('input_usd_per_million') or '?'} / "
            f"${inventory_row.get('output_usd_per_million') or '?'} per 1M input / output"
        )
        banner_background = {
            RED: "#FEF2F2",
            ORANGE: "#FFF7ED",
            TEAL: "#ECFDF5",
        }[decision_color]
        story.extend(
            [
                PageBreak(),
                Paragraph(_label(endpoint), styles["h1"]),
                Table(
                    [
                        [
                            Paragraph(
                                f'<font color="{decision_color}"><b>{decision_label}</b></font>',
                                styles["table"],
                            ),
                            Paragraph(decision_text, styles["table"]),
                        ]
                    ],
                    colWidths=[45 * mm, 131 * mm],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(banner_background)),
                            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(decision_color)),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    ),
                ),
                Spacer(1, 4 * mm),
                _table(
                    [
                        [
                            Paragraph("Provider", styles["table_header"]),
                            Paragraph("Documented context", styles["table_header"]),
                            Paragraph("Documented max output", styles["table_header"]),
                            Paragraph("Published token price", styles["table_header"]),
                        ],
                        [
                            Paragraph(str(inventory_row.get("provider") or "not recorded"), styles["table"]),
                            Paragraph(context_text, styles["table"]),
                            Paragraph(output_text, styles["table"]),
                            Paragraph(price_text, styles["table"]),
                        ],
                    ],
                    [35 * mm, 42 * mm, 42 * mm, 57 * mm],
                    font_size=7.0,
                ),
                Spacer(1, 4 * mm),
                Paragraph(
                    f"Six-hour low-load result: <b>{int(stats['success'])}/{int(stats['n'])}</b> successful "
                    f"({100 * stats['rate']:.1f}%; Wilson 95% CI {100 * stats['low']:.1f}-{100 * stats['high']:.1f}%). "
                    "Every workload below has 14 or 28 requests across seven hourly panels; 429s and timeouts count as measured failures.",
                    styles["body"],
                ),
                Paragraph("Traffic-handling evidence", styles["h2"]),
            ]
        )

        def state_paragraph(text: str) -> Any:
            lower = text.lower()
            if any(token in lower for token in ("failed", "failure", "below")):
                color = RED
            elif any(token in lower for token in ("passed", "at least", "bracket")):
                color = TEAL
            elif any(token in lower for token in ("not run", "not measured", "baseline")):
                color = SLATE
            else:
                color = ORANGE
            return Paragraph(
                f'<font color="{color}"><b>{escape(text)}</b></font>', styles["table"]
            )

        workload_rows: list[list[Any]] = [
            [
                Paragraph("Exact recipe", styles["table_header"]),
                Paragraph("Adaptive load search", styles["table_header"]),
                Paragraph("Two-minute fixed-rate test", styles["table_header"]),
            ]
        ]
        for evidence in _endpoint_workload_evidence(endpoint, endpoint_capacity, fixed_index):
            shape = evidence["shape"]
            workload_rows.append(
                [
                    Paragraph(SHAPE_LABELS[shape], styles["table"]),
                    state_paragraph(evidence["capacity_text"]),
                    state_paragraph(evidence["fixed_rate_text"]),
                ]
            )
        story.append(_table(workload_rows, [45 * mm, 72 * mm, 59 * mm], font_size=6.9))

        profile_rows: list[list[Any]] = [
            [
                Paragraph("Low-load workload", styles["table_header"]),
                Paragraph("Request success (Wilson 95%)", styles["table_header"]),
                Paragraph("Panel-median latency: stable / fresh, s [95%]", styles["table_header"]),
                Paragraph("Eligible decode rate: stable / fresh, tok/s [95%]", styles["table_header"]),
            ]
        ]
        for profile in _endpoint_profile_rows(endpoint, panel_rows, across_rows):
            profile_rows.append(
                [
                    Paragraph(profile["label"], styles["table"]),
                    Paragraph(profile["success"], styles["table"]),
                    Paragraph(profile["latency"], styles["table"]),
                    Paragraph(profile["throughput"], styles["table"]),
                ]
            )
        story.extend(
            [
                Spacer(1, 3 * mm),
                Paragraph("Six-hour behavior by matched low-load workload", styles["h2"]),
                _table(profile_rows, [31 * mm, 41 * mm, 55 * mm, 49 * mm], font_size=6.2),
                Paragraph(
                    "Stable / fresh means repeated exact token prefix / panel-unique fresh prefix. Latency and decode intervals resample the seven hourly panels; "
                    "not estimable means the strict timestamp, complete-usage, one-second decode-window, and 16-output-token rules were not met.",
                    styles["small"],
                ),
            ]
        )
        endpoint_states = [
            (dim, _capability_state(capability_index, endpoint, dim, panel_rows))
            for dim in capability_dims
        ]
        endpoint_cap_rows = [
            [
                Paragraph("Capability", styles["table_header"]),
                Paragraph("Measured / documented state", styles["table_header"]),
                Paragraph("Capability", styles["table_header"]),
                Paragraph("Measured / documented state", styles["table_header"]),
            ]
        ]
        for left, right in zip(endpoint_states[::2], endpoint_states[1::2], strict=True):
            endpoint_cap_rows.append(
                [
                    Paragraph(left[0].replace("_", " "), styles["table"]),
                    Paragraph(left[1].replace("_", " "), styles["table"]),
                    Paragraph(right[0].replace("_", " "), styles["table"]),
                    Paragraph(right[1].replace("_", " "), styles["table"]),
                ]
            )
        story.extend(
            [
                Spacer(1, 3 * mm),
                Paragraph("Capabilities, caching, and context boundary", styles["h2"]),
                _table(endpoint_cap_rows, [31 * mm, 57 * mm, 31 * mm, 57 * mm], font_size=6.2),
                Paragraph(
                    "Measured context: " + _boundary_note(limits_by_endpoint[endpoint]),
                    styles["small"],
                ),
                Paragraph(
                    'Product-contract sources: <link href="https://docs.digitalocean.com/products/inference/details/models/" color="#2457A7">model catalog</link>, '
                    '<link href="https://docs.digitalocean.com/products/inference/details/pricing/" color="#2457A7">pricing</link>, '
                    '<link href="https://docs.digitalocean.com/products/inference/how-to/use-prompt-caching/" color="#2457A7">prompt caching</link>, and '
                    '<link href="https://docs.digitalocean.com/products/inference/details/limits/" color="#2457A7">batch limits</link>; verified 29 August 2026.',
                    styles["small"],
                ),
            ]
        )

    def invariant_canvas(*args: Any, **kwargs: Any) -> Any:
        kwargs["invariant"] = 1
        # Uncompressed page streams avoid a reproducible Poppler rendering defect in which
        # selected table/band content disappeared while remaining extractable in other readers.
        kwargs["pageCompression"] = 0
        return pdf_canvas.Canvas(*args, **kwargs)

    document.build(story, canvasmaker=invariant_canvas)


def generate_digitalocean_final_report(
    summary_dir: str | Path,
    variation_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Build the sanitized DigitalOcean engineering report and its figure/data package."""

    summary = Path(summary_dir).resolve()
    variation = Path(variation_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        if output.name != "digitalocean-final-report":
            raise ValueError(
                "refusing to replace an output directory not named digitalocean-final-report"
            )
        shutil.rmtree(output)
    figures_dir = output / "figures"
    data_dir = output / "data"
    figures_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    inventory = _read_csv(summary / "endpoint-inventory.csv")
    capacity = _read_csv(summary / "capacity-summary.csv")
    fixed_rate = _read_csv(summary / "soak-cell-summary.csv")
    capabilities = _read_csv(summary / "capability-evidence.csv")
    limits = _read_csv(summary / "observed-limits.csv")
    panel_path = variation / "variation-panel-summary.csv"
    across_path = variation / "variation-across-panel-summary.csv"
    cache_effect_path = variation / "variation-paired-cache-effects.csv"
    panel_rows = _read_csv(panel_path)
    across_rows = _read_csv(across_path)
    cache_effect_rows = _read_csv(cache_effect_path)

    if {str(row.get("endpoint_id")) for row in inventory} != set(ENDPOINTS):
        raise ValueError(
            "endpoint inventory must contain exactly the 11 registered hosted endpoints"
        )
    if any(
        "arcee" in json.dumps(row, sort_keys=True).lower()
        for row in inventory + capacity + fixed_rate
    ):
        raise ValueError("Arcee must not appear in the DigitalOcean hosted-model report")
    _validate_variation_tables(panel_rows, across_rows, cache_effect_rows)

    figures = {
        "reliability-forest": _plot_reliability_forest(
            panel_rows, figures_dir / "six-hour-reliability-forest.png"
        ),
        "panel-outcomes": _plot_panel_outcomes(
            panel_rows, figures_dir / "six-hour-panel-outcomes.png"
        ),
        "fixed-rate": _plot_fixed_rate(fixed_rate, figures_dir / "fixed-rate-stability-matrix.png"),
        "cache-effect": _plot_cache_effect(
            cache_effect_rows, figures_dir / "prompt-reuse-effect.png"
        ),
    }
    current_capacity = _current_capacity(capacity)
    for shape in ("short_short", "input100k_short", "short_long", "mixed"):
        figures[f"capacity-{shape}"] = _plot_capacity_shape(
            current_capacity,
            shape,
            figures_dir / f"adaptive-load-{shape}.png",
        )

    summary_names = (
        "endpoint-inventory.csv",
        "endpoint-summary.csv",
        "capacity-summary.csv",
        "capacity-controller-summary-20260828.csv",
        "capacity-coverage-ledger-20260828.csv",
        "capacity-load-block-summary-20260828.csv",
        "soak-cell-summary.csv",
        "soak-block-summary.csv",
        "capability-evidence.csv",
        "observed-limits.csv",
        "coverage-matrix.csv",
        "cache-state-metrics.csv",
        "cache-verification-pairs.csv",
        "static-verification-summary.csv",
        "quality-pair-summary.csv",
        "recovery-summary.csv",
        "capacity-provenance-manifest.json",
        "static-verification-manifest.json",
    )
    for name in summary_names:
        source = summary / name
        if source.is_file():
            destination = data_dir / name
            if source.suffix.lower() == ".csv":
                _sanitize_public_csv(source, destination)
            else:
                _safe_text_copy(source, destination)
    variation_names = (
        "variation-panel-summary.csv",
        "variation-across-panel-summary.csv",
        "variation-paired-cache-effects.csv",
        "variation-provenance-manifest.json",
        "variation-summary.json",
    )
    for name in variation_names:
        source = variation / name
        if source.is_file():
            destination = data_dir / source.name
            if source.suffix.lower() == ".csv":
                _sanitize_public_csv(source, destination)
            else:
                _safe_text_copy(source, destination)
    readme = summary / "README.md"
    if readme.is_file():
        _safe_text_copy(readme, output / "README.md")

    output_pdf = output / "digitalocean-inference-endpoints-technical-benchmark-2026-08-29.pdf"
    _build_pdf(
        output_pdf,
        summary_dir=summary,
        variation_dir=variation,
        figures=figures,
        inventory=inventory,
        capacity=capacity,
        fixed_rate=fixed_rate,
        capabilities=capabilities,
        limits=limits,
        panel_rows=panel_rows,
        across_rows=across_rows,
        cache_effect_rows=cache_effect_rows,
    )
    return output_pdf
