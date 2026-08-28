import csv
from pathlib import Path

import pytest

from inference_bench import digitalocean_atlas


def _write_endpoint_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("endpoint_id",))
        writer.writeheader()
        writer.writerows(
            (
                {"endpoint_id": "keep-endpoint"},
                {"endpoint_id": "remove-endpoint"},
            )
        )


def test_digitalocean_atlas_excludes_endpoint_from_every_input_table(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "summary"
    source.mkdir()
    for name in (
        "endpoint-inventory.csv",
        "capacity-summary.csv",
        "soak-cell-summary.csv",
        "capability-evidence.csv",
        "cache-state-metrics.csv",
        "observed-limits.csv",
    ):
        _write_endpoint_csv(source / name)

    captured: dict[str, list[dict[str, str]]] = {}
    monkeypatch.setattr(
        digitalocean_atlas,
        "_plot_capacity",
        lambda rows, _source, _destination: captured.setdefault("capacity", rows) and [],
    )
    monkeypatch.setattr(
        digitalocean_atlas,
        "_plot_soak",
        lambda rows, _source, _destination: captured.setdefault("soak", rows) and [],
    )
    monkeypatch.setattr(
        digitalocean_atlas,
        "_plot_capabilities",
        lambda rows, _destination: captured.setdefault("capabilities", rows) and Path("figure"),
    )

    def capture_pdf(
        _report,
        inventory,
        capacity,
        soak,
        capabilities,
        limits,
        _figures,
        **_kwargs,
    ) -> None:
        captured.update(
            inventory=inventory,
            capacity_pdf=capacity,
            soak_pdf=soak,
            capabilities_pdf=capabilities,
            limits=limits,
        )

    monkeypatch.setattr(digitalocean_atlas, "_build_pdf", capture_pdf)
    digitalocean_atlas.generate_digitalocean_atlas(
        source,
        tmp_path / "digitalocean-atlas",
        capacity_source="capacity-source",
        soak_source="soak-source",
        exclude_endpoints=("remove-endpoint",),
    )

    for rows in captured.values():
        assert {row["endpoint_id"] for row in rows} == {"keep-endpoint"}


def test_platform_capabilities_replace_obsolete_cache_option() -> None:
    rows = [
        {
            "endpoint_id": "model-a",
            "capability_dimension": "caching_option",
            "transport_status": "documented_unavailable",
            "functional_status": "not_scored",
        }
    ]
    cache_rows = [
        {"endpoint_id": "model-a", "cache_state": "cache_hit_observed", "request_count": "3"}
    ]
    merged = digitalocean_atlas._merge_platform_capabilities(rows, cache_rows, ["model-a"])
    by_dimension = {row["capability_dimension"]: row for row in merged}
    assert "caching_option" not in by_dimension
    assert by_dimension["automatic_prompt_cache"]["functional_status"] == "passed"
    assert by_dimension["batch_open_models"]["transport_status"] == "documented_unavailable"


@pytest.mark.parametrize(
    ("row", "expected_result", "expected_label", "expected_color"),
    (
        (
            {"status": "complete", "soak_acceptance_pass": "True"},
            "passed",
            "passed",
            "#0F766E",
        ),
        (
            {"status": "complete", "soak_acceptance_pass": "False"},
            "failed",
            "failed",
            "#B91C1C",
        ),
        (
            {"status": "baseline_transport_gate_failed", "soak_acceptance_pass": ""},
            "could_not_start",
            "could not start",
            "#64748B",
        ),
        (
            {"status": "complete", "soak_acceptance_pass": ""},
            "not_measured",
            "not measured",
            "#64748B",
        ),
    ),
)
def test_fixed_rate_result_uses_acceptance_not_execution_status(
    row: dict[str, str],
    expected_result: str,
    expected_label: str,
    expected_color: str,
) -> None:
    assert digitalocean_atlas._fixed_rate_result(row) == expected_result
    label, color, _ = digitalocean_atlas._fixed_rate_presentation(row)
    assert label == expected_label
    assert color == expected_color


def test_accepted_fixed_rate_count_excludes_completed_failures_and_transport_gates() -> None:
    rows = [
        {"status": "complete", "soak_acceptance_pass": "True"},
        {"status": "complete", "soak_acceptance_pass": "False"},
        {"status": "baseline_transport_gate_failed", "soak_acceptance_pass": ""},
        {"status": "complete", "soak_acceptance_pass": ""},
    ]
    assert digitalocean_atlas._accepted_fixed_rate_test_count(rows) == 1


def test_fixed_rate_interval_wording_is_honest_about_contiguous_blocks() -> None:
    note = digitalocean_atlas.FIXED_RATE_INTERVAL_NOTE.lower()
    assert "contiguous" in note
    assert "serial correlation is not modeled" in note
    assert "independent" not in note
