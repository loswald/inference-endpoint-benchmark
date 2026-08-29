from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from inference_bench.reporting import (
    EvidenceState,
    EvidenceValidationError,
    assert_publishable,
    evidence_matrix,
    evidence_table,
    interval_forest,
    load_evidence_cells,
    load_report_profile,
    state_message,
    validate_evidence_cells,
)

ROOT = Path(__file__).parents[1]
DO_PROFILE = ROOT / "examples" / "report-profiles" / "digitalocean.yaml"


def test_provider_identity_workloads_and_sources_live_in_profile() -> None:
    profile = load_report_profile(DO_PROFILE)

    assert profile.provider_id == "digitalocean"
    assert len(profile.endpoints) == 11
    assert {item.id for item in profile.workloads} == {
        "short_short",
        "input32k_short",
        "input100k_short",
        "short_long",
        "mixed",
    }
    assert profile.dataset("adaptive_capacity").filters == {
        "source_id": "do-combined-capacity-20260829"
    }


def test_new_provider_and_columns_require_only_yaml_and_csv(tmp_path: Path) -> None:
    profile_path = tmp_path / "provider.yaml"
    profile_path.write_text(
        """
schema_version: 1
provider_id: example-cloud
display_name: Example Cloud
report_title: Example report
endpoints:
  route-a: {label: Route A}
workloads:
  interactive:
    label: Interactive
    description: Small request.
experiments:
  capacity:
    label: Adaptive capacity
    question: What load works?
    method: Adaptive search with repeated confirmation.
    interpretation: Bound applies to the tested recipe.
sources:
  run-1: {label: Run 1}
datasets:
  capacity:
    filename: capacity.csv
    experiment: capacity
    endpoint_column: model_name
    workload_column: recipe_name
    state_column: result_state
    source_column: run_name
    lowest_tested_column: floor_rps
    endpoint_aliases: {upstream-model-a: route-a}
    workload_aliases: {tiny: interactive}
    metrics:
      offered_rps:
        label: Offered rate
        unit: requests/second
        estimate_column: lower_rps
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "capacity.csv").write_text(
        "model_name,recipe_name,result_state,run_name,floor_rps,lower_rps\n"
        "upstream-model-a,tiny,confirmed_right_censored_lower_bound,run-1,0.01,2.5\n",
        encoding="utf-8",
    )

    profile = load_report_profile(profile_path)
    cells = load_evidence_cells(profile, tmp_path, "capacity")

    assert len(cells) == 1
    assert cells[0].endpoint_id == "route-a"
    assert cells[0].workload_id == "interactive"
    assert cells[0].state is EvidenceState.CONFIRMED_LOWER_BOUND
    assert cells[0].metrics["offered_rps"].estimate == 2.5


def test_legacy_capacity_floor_state_is_unresolved_not_endpoint_failure(tmp_path: Path) -> None:
    profile = load_report_profile(DO_PROFILE)
    (tmp_path / "capacity-summary.csv").write_text(
        "endpoint_id,shape,capacity_claim,source_id,tested_min_offered_rps,"
        "capacity_lower_bound_rps,achieved_rpm,achieved_rpm_ci95,effective_input_tpm,"
        "effective_input_tpm_ci95,effective_output_tpm,effective_output_tpm_ci95,"
        "latency_p95_seconds,latency_p95_seconds_ci95\n"
        "glm-5.2,short_short,censored_no_valid_healthy_epoch,"
        "do-combined-capacity-20260829,0.25,,,,,,,,,\n",
        encoding="utf-8",
    )

    cell = load_evidence_cells(profile, tmp_path, "adaptive_capacity")[0]
    message = state_message(cell)

    assert cell.state is EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR
    assert cell.lowest_tested_value == 0.25
    assert "lower loads were not tested" in message
    assert "usable capacity was not determined" in message
    assert "failed" not in message.lower()


def test_evidence_table_never_uses_an_empty_result_for_unresolved_cell(tmp_path: Path) -> None:
    profile = load_report_profile(DO_PROFILE)
    (tmp_path / "capacity-summary.csv").write_text(
        "endpoint_id,shape,capacity_claim,source_id,tested_min_offered_rps,"
        "capacity_lower_bound_rps,achieved_rpm,achieved_rpm_ci95,effective_input_tpm,"
        "effective_input_tpm_ci95,effective_output_tpm,effective_output_tpm_ci95,"
        "latency_p95_seconds,latency_p95_seconds_ci95\n"
        "glm-5.2,short_short,censored_no_valid_healthy_epoch,"
        "do-combined-capacity-20260829,0.25,,,,,,,,,\n",
        encoding="utf-8",
    )
    cell = load_evidence_cells(profile, tmp_path, "adaptive_capacity")[0]

    row = evidence_table([cell], profile, metric_id="confirmed_offered_rps")[0]

    assert row["result"]
    assert row["uncertainty / bounds"] == "Not applicable to this state"
    assert row["state"] == "Search stopped before finding a healthy baseline"


def test_matrix_renders_missing_exact_cells_as_not_run() -> None:
    profile = load_report_profile(DO_PROFILE)
    cells = load_evidence_cells(profile, ROOT / "reports" / "digitalocean", "adaptive_capacity")

    figure = evidence_matrix(cells, profile, experiment_id="adaptive_capacity")
    labels = {text.get_text().replace("\n", " ") for text in figure.axes[0].texts}

    assert "not run" in labels
    assert "search incomplete" in labels
    plt.close(figure)


def test_interval_chart_keeps_non_numeric_states_visible() -> None:
    profile = load_report_profile(DO_PROFILE)
    cells = load_evidence_cells(profile, ROOT / "reports" / "digitalocean", "adaptive_capacity")

    figure = interval_forest(
        cells,
        profile,
        experiment_id="adaptive_capacity",
        metric_id="confirmed_offered_rps",
        workload_id="short_short",
    )
    labels = {text.get_text() for text in figure.axes[0].texts}

    assert "search incomplete" in labels
    plt.close(figure)


def test_variation_strata_are_data_dimensions_not_provider_code() -> None:
    profile = load_report_profile(DO_PROFILE)
    cells = load_evidence_cells(profile, ROOT / "reports" / "digitalocean", "within_run_variation")

    assert len(cells) == 110
    assert {cell.dimensions["cache_stratum"] for cell in cells} == {
        "panel_unique_cold",
        "stable_exact_prompt",
    }
    assert all(cell.state is EvidenceState.MEASURED_WITH_INTERVAL for cell in cells)
    assert {cell.source_id for cell in cells} == {"do-six-hour-variation-20260828-r1"}


def test_completeness_gate_detects_exact_recipe_cells_that_were_not_run() -> None:
    profile = load_report_profile(DO_PROFILE)
    cells = load_evidence_cells(
        profile,
        ROOT / "reports" / "digitalocean",
        "adaptive_capacity",
    )

    issues = validate_evidence_cells(cells, profile, "adaptive_capacity")
    missing = [issue for issue in issues if issue.code == "missing_required_cell"]

    assert len(missing) == 11
    try:
        assert_publishable(cells, profile, "adaptive_capacity")
    except EvidenceValidationError as exc:
        assert len([issue for issue in exc.issues if issue.code == "missing_required_cell"]) == 11
    else:
        raise AssertionError("incomplete exact-recipe grid unexpectedly passed")


def test_malformed_interval_data_fails_instead_of_becoming_a_blank_metric(
    tmp_path: Path,
) -> None:
    profile = load_report_profile(DO_PROFILE)
    (tmp_path / "capacity-summary.csv").write_text(
        "endpoint_id,shape,capacity_claim,source_id,tested_min_offered_rps,"
        "capacity_lower_bound_rps,achieved_rpm,achieved_rpm_ci95,effective_input_tpm,"
        "effective_input_tpm_ci95,effective_output_tpm,effective_output_tpm_ci95,"
        "latency_p95_seconds,latency_p95_seconds_ci95\n"
        "glm-5.2,short_short,confirmed_right_censored_lower_bound,"
        "do-combined-capacity-20260829,0.25,1.0,60,not-json,,,,,,\n",
        encoding="utf-8",
    )

    try:
        load_evidence_cells(profile, tmp_path, "adaptive_capacity")
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("malformed interval unexpectedly became missing data")
