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


def block_median_interval(values: Sequence[float], *, unit: str, seed: int = 1) -> Estimate:
    """Percentile bootstrap whose independent sampling units are epochs/blocks."""

    estimate = bootstrap_interval(values, statistics.median, unit=unit, seed=seed)
    method = "single-block-no-CI" if estimate.n == 1 else "epoch/block-bootstrap-percentile"
    return Estimate(
        estimate.estimate,
        estimate.lower_95,
        estimate.upper_95,
        estimate.n,
        estimate.unit,
        method,
    )


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
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == total else min(1.0, centre + half)
    return Estimate(p, lower, upper, total, unit, "Wilson-95")


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
    """Bootstrap paired independent blocks using a ratio-of-sums estimand.

    Unequal block durations commonly arise from post-arrival response drain. Averaging block rates
    would give a short block the same weight as a long block; the ratio of summed units to summed
    wall time retains the intended aggregate-throughput denominator.
    """
    if len(successful_units) != len(block_seconds):
        raise ValueError("block vectors must have equal length")
    pairs = [
        (float(units), float(seconds))
        for units, seconds in zip(successful_units, block_seconds, strict=True)
        if math.isfinite(units) and units >= 0 and math.isfinite(seconds) and seconds > 0
    ]
    return _paired_block_ratio_interval(
        pairs,
        numerator_scale=60.0,
        unit=f"{unit_name}/minute",
        seed=seed,
        method="epoch/block-bootstrap-ratio-of-sums",
    )


def block_proportion_interval(
    successes: Sequence[float], totals: Sequence[float], *, seed: int = 1
) -> Estimate:
    """Estimate a load-block success proportion with blocks as sampling units."""
    if len(successes) != len(totals):
        raise ValueError("block vectors must have equal length")
    pairs = [
        (float(success), float(total))
        for success, total in zip(successes, totals, strict=True)
        if math.isfinite(success) and math.isfinite(total) and total > 0 and 0 <= success <= total
    ]
    return _paired_block_ratio_interval(
        pairs,
        numerator_scale=1.0,
        unit="proportion",
        seed=seed,
        method="epoch/block-bootstrap-ratio-of-sums",
    )


def _paired_block_ratio_interval(
    pairs: Sequence[tuple[float, float]],
    *,
    numerator_scale: float,
    unit: str,
    seed: int,
    method: str,
    draws: int = 2_000,
) -> Estimate:
    if not pairs:
        return Estimate(None, None, None, 0, unit, method)

    def ratio(sample: Sequence[tuple[float, float]]) -> float:
        return numerator_scale * sum(pair[0] for pair in sample) / sum(pair[1] for pair in sample)

    point = ratio(pairs)
    if len(pairs) == 1:
        return Estimate(point, None, None, 1, unit, "single-block-no-CI")
    rng = random.Random(seed)
    samples = [ratio([rng.choice(pairs) for _ in pairs]) for _ in range(draws)]
    return Estimate(
        point,
        quantile(samples, 0.025),
        quantile(samples, 0.975),
        len(pairs),
        unit,
        method,
    )
