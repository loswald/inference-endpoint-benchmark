from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Estimate:
    estimate: float | None
    lower_95: float | None
    upper_95: float | None
    n: int
    unit: str
    method: str


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires observations")
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1 - weight) + ordered[right] * weight


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    unit: str,
    seed: int = 1,
    draws: int = 2_000,
) -> Estimate:
    clean = [float(value) for value in values if math.isfinite(value)]
    if not clean:
        return Estimate(None, None, None, 0, unit, "request-bootstrap-percentile")
    point = statistic(clean)
    if len(clean) == 1:
        return Estimate(point, None, None, 1, unit, "single-observation-no-CI")
    rng = random.Random(seed)
    samples = [statistic([rng.choice(clean) for _ in clean]) for _ in range(draws)]
    return Estimate(
        point,
        quantile(samples, 0.025),
        quantile(samples, 0.975),
        len(clean),
        unit,
        "request-bootstrap-percentile",
    )


def median_interval(values: Sequence[float], *, unit: str, seed: int = 1) -> Estimate:
    return bootstrap_interval(values, statistics.median, unit=unit, seed=seed)


def mean_interval(values: Sequence[float], *, unit: str, seed: int = 1) -> Estimate:
    return bootstrap_interval(values, statistics.mean, unit=unit, seed=seed)


def quantile_interval(
    values: Sequence[float], probability: float, *, unit: str, seed: int = 1
) -> Estimate:
    return bootstrap_interval(
        values, lambda sample: quantile(sample, probability), unit=unit, seed=seed
    )


def wilson_interval(successes: int, total: int, *, unit: str = "proportion") -> Estimate:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return Estimate(None, None, None, 0, unit, "Wilson-95")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return Estimate(p, max(0.0, centre - half), min(1.0, centre + half), total, unit, "Wilson-95")


def qualified_p99(values: Sequence[float], *, unit: str, seed: int = 1) -> Estimate:
    if len(values) < 1_000:
        return Estimate(None, None, None, len(values), unit, "withheld-n<1000")
    return quantile_interval(values, 0.99, unit=unit, seed=seed)


def block_rate_interval(
    successful_units: Sequence[float],
    block_seconds: Sequence[float],
    *,
    unit_name: str,
    seed: int = 1,
) -> Estimate:
    """Bootstrap independent load blocks, never individual output tokens."""
    if len(successful_units) != len(block_seconds):
        raise ValueError("block vectors must have equal length")
    rates = [
        units / (seconds / 60)
        for units, seconds in zip(successful_units, block_seconds, strict=True)
        if seconds > 0
    ]
    return mean_interval(rates, unit=f"{unit_name}/minute", seed=seed)
