import csv
from pathlib import Path

import pytest
from pypdf import PdfReader

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
        "soak-block-summary.csv",
        "recovery-summary.csv",
        "coverage-matrix.csv",
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
        "_plot_fixed_rate_tests",
        lambda rows, _source, _destination, **_kwargs: captured.setdefault("soak", rows) and [],
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
        coverage,
        soak_blocks,
        recovery,
        _figures,
        **_kwargs,
    ) -> None:
        captured.update(
            inventory=inventory,
            capacity_pdf=capacity,
            soak_pdf=soak,
            capabilities_pdf=capabilities,
            limits=limits,
            coverage=coverage,
            soak_blocks=soak_blocks,
            recovery=recovery,
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


def test_capacity_workload_keeps_32k_and_100k_evidence_separate() -> None:
    assert (
        digitalocean_atlas._capacity_workload(
            {
                "shape": "input32k_short",
                "provenance_source_id": "do-sixhour-aimd-20260824-r1",
            }
        )
        == "input32k_short"
    )
    assert (
        digitalocean_atlas._capacity_workload(
            {
                "shape": "input32k_short",
                "provenance_source_id": "do-capacity-20260828-r2",
            }
        )
        == "input100k_short"
    )
    assert (
        digitalocean_atlas._capacity_workload(
            {"shape": "input100k_short", "provenance_source_id": "do-capacity-20260828-r2"}
        )
        == "input100k_short"
    )


def test_evidence_snapshot_counts_scientific_results_not_finished_schedules() -> None:
    capacity = [
        {
            "shape": "short_short",
            "capacity_claim": "confirmed_right_censored_lower_bound",
            "capacity_lower_bound_rps": "1",
        },
        {"shape": "mixed", "capacity_claim": "unconfirmed_healthy_observation_only"},
        {"shape": "short_long", "capacity_claim": "censored_no_valid_healthy_epoch"},
        {
            "shape": "input32k_short",
            "capacity_claim": "measured_capacity_state_without_numeric_bound",
        },
    ]
    fixed_rate = [
        {"status": "complete", "soak_acceptance_pass": "True"},
        {"status": "complete", "soak_acceptance_pass": "False"},
        {"status": "baseline_transport_gate_failed", "soak_acceptance_pass": ""},
    ]
    coverage = [
        {"status": "completed"},
        {"status": "inconclusive"},
        {"status": "unsupported"},
    ]
    snapshot = digitalocean_atlas._evidence_snapshot(capacity, fixed_rate, coverage)
    assert snapshot == {
        "capacity_total": 4,
        "capacity_confirmed": 1,
        "capacity_exploratory": 1,
        "capacity_no_healthy_epoch": 1,
        "capacity_no_numeric_bound": 1,
        "fixed_rate_total": 3,
        "fixed_rate_passed": 1,
        "fixed_rate_failed": 1,
        "fixed_rate_could_not_start": 1,
        "coverage_total": 3,
        "coverage_completed": 1,
        "coverage_inconclusive": 1,
        "coverage_unsupported": 1,
    }


def test_fixed_rate_failure_reasons_are_plain_language_and_deduplicated() -> None:
    row = {
        "cell_id": "cell-a",
        "status": "complete",
        "soak_acceptance_pass": "False",
    }
    blocks = [
        {
            "cell_id": "cell-a",
            "acceptance_reasons": (
                '["success_rate_below_0.99", "success_rate_below_0.99", '
                '"arrival_queue_growth"]'
            ),
        }
    ]
    recovery = [
        {
            "cell_id": "cell-a",
            "recovery_acceptance_reasons": '["recovery_latency_p95_above_2x_low_load"]',
        }
    ]
    reasons = digitalocean_atlas._fixed_rate_failure_reasons(row, blocks, recovery)
    assert reasons == [
        "success rate fell below 99%",
        "the request queue kept growing",
        "post-load p95 latency remained above 2x the low-load reference",
    ]
    assert all("_" not in reason for reason in reasons)


def test_interim_markdown_leads_with_truth_and_reserves_six_hour_panel() -> None:
    inventory = [{"endpoint_id": "model-a"}]
    capacity = [
        {
            "endpoint_id": "model-a",
            "source_id": "combined",
            "provenance_source_id": "do-capacity-20260828-r2",
            "shape": "input100k_short",
            "capacity_claim": "confirmed_right_censored_lower_bound",
            "capacity_lower_bound_rps": "0.5",
        }
    ]
    fixed_rate = [
        {
            "endpoint_id": "model-a",
            "source_id": "fixed",
            "shape": "input32k_short",
            "cell_id": "cell-a",
            "status": "complete",
            "soak_acceptance_pass": "False",
            "candidate_rate_rps": "0.25",
        }
    ]
    coverage = [{"endpoint_id": "model-a", "status": "inconclusive"}]
    report = digitalocean_atlas._build_interim_markdown(
        inventory,
        capacity,
        fixed_rate,
        coverage,
        [
            {
                "cell_id": "cell-a",
                "acceptance_reasons": '["success_rate_below_0.99"]',
            }
        ],
        [],
        capacity_source="combined",
        fixed_rate_source="fixed",
    )
    assert "Not complete. Not a production qualification." in report
    assert "**1/1** endpoint-workload cells" in report
    assert "**0/1** 120-second fixed-rate tests passed" in report
    assert "100,000-token prompt" in report
    assert "32,000-token prompt" in report
    assert "Six-hour matched variation panel — pending" in report
    assert "no full-day or diurnal claim" in report


def test_endpoint_pdf_sheet_keeps_facts_header_and_all_workload_rows_together(
    tmp_path: Path,
) -> None:
    endpoint_id = "glm-5.2"
    inventory = [
        {
            "endpoint_id": endpoint_id,
            "api_surface": "chat_completions",
            "server_region": "not reported",
            "api_version": "v1",
            "context_window": "262144",
            "max_output_tokens": "262144",
            "input_usd_per_million": "0.70",
            "output_usd_per_million": "2.20",
        }
    ]
    capacity = [
        {
            "endpoint_id": endpoint_id,
            "source_id": "combined",
            "provenance_source_id": "do-capacity-20260828-r2",
            "shape": shape,
            "capacity_claim": "censored_no_valid_healthy_epoch",
            "tested_min_offered_rps": "0.25",
        }
        for shape in ("short_short", "input100k_short", "short_long", "mixed")
    ]
    soak = [
        {
            "endpoint_id": endpoint_id,
            "source_id": "fixed",
            "shape": shape,
            "status": "complete",
            "soak_acceptance_pass": "False",
            "candidate_rate_rps": "0.25",
        }
        for shape in ("short_short", "input32k_short", "short_long", "mixed")
    ]
    capabilities = [
        {
            "endpoint_id": endpoint_id,
            "capability_dimension": dimension,
            "transport_status": "observed_supported",
            "functional_status": "passed",
        }
        for dimension in (
            "response_format",
            "tools",
            "parallel_tool_calls",
            "vision",
            "automatic_prompt_cache",
            "batch_open_models",
        )
    ]
    pdf = tmp_path / "endpoint-sheet.pdf"
    digitalocean_atlas._build_pdf(
        pdf,
        inventory,
        capacity,
        soak,
        capabilities,
        [],
        [],
        [],
        [],
        [],
        static_verification=[],
        cache_verification=[],
        verification_manifest={},
        capacity_manifest={},
        capacity_source="combined",
        soak_source="fixed",
    )

    endpoint_pages = [
        page.extract_text() or ""
        for page in PdfReader(str(pdf)).pages
        if "Model / API" in (page.extract_text() or "")
    ]
    assert len(endpoint_pages) == 1
    page_text = " ".join(endpoint_pages[0].split())
    for phrase in (
        "glm-5.2",
        "Model / API",
        "Region / API version",
        "Context / max output",
        "Input / output price",
        "Workload",
        "Exact recipe and source",
        "Adaptive-load result",
        "120 s fixed-rate result",
        "short prompt / short answer",
        "100K-token prompt / short answer",
        "short prompt / long answer",
        "seeded multi-workload mix",
    ):
        assert phrase in page_text
