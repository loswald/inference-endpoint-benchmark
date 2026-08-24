from __future__ import annotations

import itertools

from inference_bench.workloads import (
    _pairwise_cover,
    materialize_messages,
    plan_cache,
    plan_context,
)


def test_covering_array_covers_every_pair() -> None:
    factors = {"a": [0, 1, 2], "b": ["x", "y"], "c": [False, True]}
    rows = _pairwise_cover(factors)
    for left_index, left in enumerate(factors):
        for right in list(factors)[left_index + 1 :]:
            observed = {(row[left], row[right]) for row in rows}
            assert observed == set(itertools.product(factors[left], factors[right]))


def test_context_plan_is_lazy_and_retrieval_aware(route) -> None:
    specs = plan_context(route, {"percentages": [99]}, seed=1)
    fixed = specs[0]
    assert fixed.messages == ()
    assert fixed.metadata["prompt_kind"] == "long_context"
    message = materialize_messages(fixed)[0]["content"]
    assert "BEGIN_MARKER" in message and "MIDDLE_MARKER" in message and "END_MARKER" in message


def test_cache_trials_are_stratified(route) -> None:
    specs = plan_cache(route, {"repeats": 2, "prefix_tokens": 20}, seed=1)
    assert {spec.cell_id for spec in specs} == {"cached_trial", "uncached_trial"}
    cached = [spec for spec in specs if spec.cell_id == "cached_trial"]
    assert cached[0].messages[0]["content"] == cached[1].messages[0]["content"]
    uncached = [spec for spec in specs if spec.cell_id == "uncached_trial"]
    assert uncached[0].messages[0]["content"] != uncached[1].messages[0]["content"]
