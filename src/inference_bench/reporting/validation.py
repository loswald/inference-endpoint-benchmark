"""Integrity and completeness gates for provider-neutral report evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .models import DatasetProfile, EvidenceCell, EvidenceState, ProviderReportProfile


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    endpoint_id: str = ""
    workload_id: str = ""


class EvidenceValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        detail = "; ".join(issue.message for issue in issues)
        super().__init__(detail)


def _expected_ids(
    profile: ProviderReportProfile,
    dataset: DatasetProfile,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    endpoints = dataset.expected_endpoint_ids or tuple(item.id for item in profile.endpoints)
    workloads = dataset.expected_workload_ids or tuple(item.id for item in profile.workloads)
    return endpoints, workloads


def validate_evidence_cells(
    cells: list[EvidenceCell],
    profile: ProviderReportProfile,
    dataset_id: str,
) -> list[ValidationIssue]:
    """Return publication-relevant integrity issues without mutating evidence."""

    dataset = profile.dataset(dataset_id)
    endpoint_ids = {item.id for item in profile.endpoints}
    workload_ids = {item.id for item in profile.workloads}
    source_ids = {item.id for item in profile.sources}
    issues: list[ValidationIssue] = []
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()

    for cell in cells:
        identity = (
            cell.endpoint_id,
            cell.workload_id,
            tuple(sorted(cell.dimensions.items())),
        )
        if identity in seen:
            issues.append(
                ValidationIssue(
                    "duplicate_cell",
                    f"duplicate cell for {cell.endpoint_id} × {cell.workload_id} "
                    f"with dimensions {dict(cell.dimensions)!r}",
                    cell.endpoint_id,
                    cell.workload_id,
                )
            )
        seen.add(identity)
        if cell.endpoint_id not in endpoint_ids:
            issues.append(
                ValidationIssue(
                    "unlisted_endpoint",
                    f"endpoint {cell.endpoint_id!r} is not declared in the report profile",
                    cell.endpoint_id,
                    cell.workload_id,
                )
            )
        if cell.workload_id not in workload_ids:
            issues.append(
                ValidationIssue(
                    "unlisted_workload",
                    f"workload {cell.workload_id!r} is not declared in the report profile",
                    cell.endpoint_id,
                    cell.workload_id,
                )
            )
        if cell.source_id and cell.source_id not in source_ids:
            issues.append(
                ValidationIssue(
                    "unlisted_source",
                    f"source {cell.source_id!r} is not declared in the report profile",
                    cell.endpoint_id,
                    cell.workload_id,
                )
            )
        if cell.state is EvidenceState.UNKNOWN:
            issues.append(
                ValidationIssue(
                    "unknown_state",
                    f"source state {cell.raw_state!r} has no canonical meaning",
                    cell.endpoint_id,
                    cell.workload_id,
                )
            )
        if (
            cell.state is EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR
            and cell.lowest_tested_value is None
        ):
            issues.append(
                ValidationIssue(
                    "missing_search_floor",
                    "an incomplete adaptive search must report its lowest tested load",
                    cell.endpoint_id,
                    cell.workload_id,
                )
            )

    if dataset.require_complete_grid:
        if dataset.dimension_columns:
            issues.append(
                ValidationIssue(
                    "ambiguous_dimension_grid",
                    "complete-grid validation needs explicit expected dimension strata",
                )
            )
        else:
            expected_endpoints, expected_workloads = _expected_ids(profile, dataset)
            observed = {(cell.endpoint_id, cell.workload_id) for cell in cells}
            for endpoint_id in expected_endpoints:
                for workload_id in expected_workloads:
                    if (endpoint_id, workload_id) not in observed:
                        issues.append(
                            ValidationIssue(
                                "missing_required_cell",
                                f"required cell {endpoint_id} × {workload_id} was not run",
                                endpoint_id,
                                workload_id,
                            )
                        )
    return issues


def assert_publishable(
    cells: list[EvidenceCell],
    profile: ProviderReportProfile,
    dataset_id: str,
) -> None:
    issues = validate_evidence_cells(cells, profile, dataset_id)
    if issues:
        raise EvidenceValidationError(issues)
