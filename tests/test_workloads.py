from __future__ import annotations

import base64
import itertools
import struct
import zlib

from inference_bench.workloads import (
    VISION_BLUE_PNG_DATA_URI,
    _pairwise_cover,
    materialize_messages,
    plan_cache,
    plan_context,
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
