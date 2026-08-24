from __future__ import annotations

from inference_bench.load import poisson_offsets


def test_poisson_schedule_is_deterministic_and_open_loop() -> None:
    first = poisson_offsets(2.0, 10.0, seed="same")
    second = poisson_offsets(2.0, 10.0, seed="same")
    assert first == second
    assert first == sorted(first)
    assert all(0 < offset < 10 for offset in first)
    assert len(first) > 1
