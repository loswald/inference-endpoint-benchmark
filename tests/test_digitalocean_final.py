from __future__ import annotations

import pytest

from inference_bench.digitalocean_final import (
    CAPACITY_SOURCE,
    ENDPOINTS,
    SHAPES,
    _endpoint_workload_evidence,
    _sanitize_public_csv,
    _validate_variation_tables,
)


def test_endpoint_workload_rows_keep_100k_and_32k_evidence_separate() -> None:
    endpoint = ENDPOINTS[0]
    capacity = [
        {
            "endpoint_id": endpoint,
            "shape": shape,
            "source_id": CAPACITY_SOURCE,
            "capacity_claim": "confirmed_right_censored_lower_bound",
            "capacity_lower_bound_rps": "1",
        }
        for shape in ("short_short", "input100k_short", "short_long", "mixed")
    ]
    fixed = {
        (endpoint, shape): {
            "endpoint_id": endpoint,
            "shape": shape,
            "source_id": "do-direct-soak-20260823-r1",
            "execution_complete": "true",
            "scientifically_complete": "true",
            "soak_acceptance_pass": "true",
        }
        for shape in ("short_short", "input32k_short", "short_long", "mixed")
    }
    rows = _endpoint_workload_evidence(endpoint, capacity, fixed)
    assert [row["shape"] for row in rows] == list(SHAPES)
    assert rows[1]["capacity_text"] == "at least 1 req/s (tested lower bound)"
    assert rows[1]["fixed_rate_text"] == "fixed-rate test not run for this exact recipe"
    assert rows[2]["capacity_text"] == "adaptive search not run for this exact recipe"
    assert rows[2]["fixed_rate_text"] == "passed"


def _variation_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    panel: list[dict[str, str]] = []
    across: list[dict[str, str]] = []
    paired: list[dict[str, str]] = []
    for endpoint in ENDPOINTS:
        long_subtype = (
            "long_short:in50000:out128"
            if endpoint == "minimax-m2.5"
            else "long_short:in100000:out128"
        )
        for shape in ("short_short", "long_short", "short_long", "mixed"):
            subtypes = (
                ("structured:in1024:out512", long_subtype)
                if shape == "mixed"
                else ("not_applicable",)
            )
            for subtype in subtypes:
                paired.append({"route_id": endpoint, "shape": shape, "mixed_subtype": subtype})
                for stratum in ("stable_exact_prompt", "panel_unique_cold"):
                    across.append(
                        {
                            "route_id": endpoint,
                            "shape": shape,
                            "mixed_subtype": subtype,
                            "cache_stratum": stratum,
                        }
                    )
                    for panel_index in range(7):
                        panel.append(
                            {
                                "route_id": endpoint,
                                "shape": shape,
                                "mixed_subtype": subtype,
                                "panel": str(panel_index),
                                "cache_stratum": stratum,
                            }
                        )
    return panel, across, paired


def test_variation_gate_requires_full_exact_identities_and_totals() -> None:
    panel, across, paired = _variation_rows()
    assert (len(panel), len(across), len(paired)) == (770, 110, 55)
    _validate_variation_tables(panel, across, paired)

    with pytest.raises(ValueError, match="duplicate full identities"):
        _validate_variation_tables([*panel, panel[0].copy()], across, paired)
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_variation_tables(panel[:-1], across, paired)
    bad_across = [row.copy() for row in across]
    bad_across[0]["mixed_subtype"] = "wrong-subtype"
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_variation_tables(panel, bad_across, paired)
    with pytest.raises(ValueError, match="duplicate full identities"):
        _validate_variation_tables(panel, across, [*paired, paired[0].copy()])


def test_public_csv_sanitizer_removes_internal_identifiers(tmp_path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "public.csv"
    source.write_text(
        'logical_id,result,sampling_unit,audit\n'
        'private-123,pass,request_id,'
        '"{""request_id"":""private-456"",""kept"":2,'
        '""sampling_unit"":""request_id""}"\n',
        encoding="utf-8",
    )

    _sanitize_public_csv(source, destination)

    text = destination.read_text(encoding="utf-8")
    assert "logical_id" not in text
    assert "request_id" not in text
    assert "private-123" not in text
    assert "private-456" not in text
    assert '""kept"":2' in text
    assert "internal_identifier" not in text
    assert "request" in text
    assert "pass" in text


def test_public_csv_sanitizer_fails_on_unclassified_identifier_token(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("result,notes\npass,request_id\n", encoding="utf-8")

    with pytest.raises(ValueError, match="internal identifier token remains"):
        _sanitize_public_csv(source, tmp_path / "public.csv")
