from __future__ import annotations

from inference_bench.statistics import block_rate_interval, qualified_p99, wilson_interval


def test_wilson_interval_has_units_and_n() -> None:
    estimate = wilson_interval(90, 100)
    assert estimate.n == 100
    assert estimate.unit == "proportion"
    assert estimate.lower_95 < estimate.estimate < estimate.upper_95


def test_p99_is_withheld_for_sparse_sample() -> None:
    estimate = qualified_p99([float(index) for index in range(999)], unit="seconds")
    assert estimate.estimate is None
    assert estimate.method == "withheld-n<1000"


def test_block_rate_uses_blocks_not_tokens_as_samples() -> None:
    estimate = block_rate_interval([100, 120, 80, 100], [30, 30, 30, 30], unit_name="tokens")
    assert estimate.n == 4
    assert estimate.unit == "tokens/minute"
