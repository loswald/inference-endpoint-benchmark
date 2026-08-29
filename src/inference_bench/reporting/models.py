"""Typed, provider-neutral evidence and report-profile contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceDisposition(StrEnum):
    """High-level meaning of an evidence state.

    This is intentionally separate from presentation text.  In particular, an
    adaptive search that reached its configured floor without finding a healthy
    baseline is unresolved, not a measured endpoint failure.
    """

    ESTABLISHED = "established"
    EXPLORATORY = "exploratory"
    MEASURED_NEGATIVE = "measured_negative"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class EvidenceState(StrEnum):
    """Canonical state vocabulary shared by all provider reports."""

    CONFIRMED_LOWER_BOUND = "confirmed_lower_bound"
    CONFIRMED_INTERVAL = "confirmed_interval"
    MEASURED = "measured"
    MEASURED_WITH_INTERVAL = "measured_with_interval"
    HEALTHY_EXPLORATORY = "healthy_exploratory"
    SEARCH_INCOMPLETE_BELOW_FLOOR = "search_incomplete_below_floor"
    MEASURED_NEGATIVE = "measured_negative"
    TRANSPORT_GATED = "transport_gated"
    UNSUPPORTED = "unsupported"
    CENSORED = "censored"
    NOT_RUN = "not_run"
    UNKNOWN = "unknown"


def _finite_optional(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be finite when present")


@dataclass(frozen=True)
class MetricInterval:
    """One estimate and its uncertainty, with an explicit sampling unit."""

    estimate: float | None = None
    low: float | None = None
    high: float | None = None
    sample_size: int | None = None
    unit: str = ""
    sampling_unit: str = "request"
    interval_kind: str = "none"

    def __post_init__(self) -> None:
        _finite_optional(self.estimate, "estimate")
        _finite_optional(self.low, "low")
        _finite_optional(self.high, "high")
        allowed_kinds = {"none", "confidence", "range", "evidence_bounds"}
        if self.interval_kind not in allowed_kinds:
            raise ValueError(f"unknown interval_kind: {self.interval_kind}")
        if self.interval_kind in {"confidence", "range"} and (
            (self.low is None) != (self.high is None)
        ):
            raise ValueError("confidence interval requires both low and high")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("confidence interval low cannot exceed high")
        if self.sample_size is not None and self.sample_size < 0:
            raise ValueError("sample_size cannot be negative")


@dataclass(frozen=True)
class EndpointProfile:
    id: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class WorkloadProfile:
    id: str
    label: str
    description: str
    recipe: str = ""


@dataclass(frozen=True)
class ExperimentProfile:
    id: str
    label: str
    question: str
    method: str
    interpretation: str


@dataclass(frozen=True)
class SourceProfile:
    id: str
    label: str
    provenance: str = ""


@dataclass(frozen=True)
class MetricColumns:
    """Declarative mapping from a CSV row to one normalized metric."""

    id: str
    label: str
    unit: str
    estimate_column: str | None = None
    low_column: str | None = None
    high_column: str | None = None
    interval_json_column: str | None = None
    sample_size_column: str | None = None
    sampling_unit: str = "request"
    interval_kind: str = "none"


@dataclass(frozen=True)
class DatasetProfile:
    """Declarative mapping from one input table to evidence cells."""

    id: str
    filename: str
    experiment_id: str
    endpoint_column: str
    workload_column: str
    state_column: str | None
    metrics: tuple[MetricColumns, ...]
    default_state: str | None = None
    state_rules: tuple[StateRule, ...] = ()
    source_column: str | None = None
    source_value: str | None = None
    lowest_tested_column: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    endpoint_aliases: dict[str, str] = field(default_factory=dict)
    workload_aliases: dict[str, str] = field(default_factory=dict)
    state_aliases: dict[str, str] = field(default_factory=dict)
    dimension_columns: dict[str, str] = field(default_factory=dict)
    expected_endpoint_ids: tuple[str, ...] = ()
    expected_workload_ids: tuple[str, ...] = ()
    require_complete_grid: bool = False


@dataclass(frozen=True)
class StateRule:
    """A small declarative row-to-state rule.

    Rules use exact string equality and are evaluated in profile order.  This is
    intentionally much smaller and safer than executing provider-specific Python or
    arbitrary expressions from configuration.
    """

    state: str
    equals: dict[str, str]


@dataclass(frozen=True)
class ProviderReportProfile:
    """Everything provider-specific that a generic report renderer needs."""

    schema_version: int
    provider_id: str
    display_name: str
    report_title: str
    endpoints: tuple[EndpointProfile, ...]
    workloads: tuple[WorkloadProfile, ...]
    experiments: tuple[ExperimentProfile, ...]
    sources: tuple[SourceProfile, ...]
    datasets: tuple[DatasetProfile, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported report-profile schema version: {self.schema_version}")
        if not self.provider_id.strip() or not self.display_name.strip():
            raise ValueError("provider_id and display_name are required")
        self._validate_unique("endpoint", [item.id for item in self.endpoints])
        self._validate_unique("workload", [item.id for item in self.workloads])
        self._validate_unique("experiment", [item.id for item in self.experiments])
        self._validate_unique("source", [item.id for item in self.sources])
        self._validate_unique("dataset", [item.id for item in self.datasets])
        experiment_ids = {item.id for item in self.experiments}
        for dataset in self.datasets:
            if dataset.experiment_id not in experiment_ids:
                raise ValueError(
                    f"dataset {dataset.id!r} references unknown experiment "
                    f"{dataset.experiment_id!r}"
                )
            self._validate_unique(
                f"metric in dataset {dataset.id!r}", [metric.id for metric in dataset.metrics]
            )

    @staticmethod
    def _validate_unique(kind: str, values: list[str]) -> None:
        if any(not value.strip() for value in values):
            raise ValueError(f"{kind} ids cannot be empty")
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate {kind} ids: {', '.join(duplicates)}")

    def endpoint(self, endpoint_id: str) -> EndpointProfile:
        return next(
            (endpoint for endpoint in self.endpoints if endpoint.id == endpoint_id),
            EndpointProfile(endpoint_id, endpoint_id),
        )

    def workload(self, workload_id: str) -> WorkloadProfile:
        return next(
            (workload for workload in self.workloads if workload.id == workload_id),
            WorkloadProfile(workload_id, workload_id, "No workload description supplied."),
        )

    def experiment(self, experiment_id: str) -> ExperimentProfile:
        try:
            return next(item for item in self.experiments if item.id == experiment_id)
        except StopIteration as exc:
            raise KeyError(experiment_id) from exc

    def dataset(self, dataset_id: str) -> DatasetProfile:
        try:
            return next(item for item in self.datasets if item.id == dataset_id)
        except StopIteration as exc:
            raise KeyError(dataset_id) from exc

    def metric(self, experiment_id: str, metric_id: str) -> MetricColumns:
        for dataset in self.datasets:
            if dataset.experiment_id != experiment_id:
                continue
            for metric in dataset.metrics:
                if metric.id == metric_id:
                    return metric
        raise KeyError(f"{experiment_id}:{metric_id}")


@dataclass(frozen=True)
class EvidenceCell:
    """One endpoint × workload × experiment result.

    A cell always has a state, even when it has no numeric metric.  This prevents
    report tables and charts from turning unresolved, unsupported, or unrun work
    into unexplained blank space.
    """

    provider_id: str
    endpoint_id: str
    workload_id: str
    experiment_id: str
    state: EvidenceState
    metrics: dict[str, MetricInterval] = field(default_factory=dict)
    source_id: str = ""
    raw_state: str = ""
    lowest_tested_value: float | None = None
    note: str = ""
    dimensions: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite_optional(self.lowest_tested_value, "lowest_tested_value")
        if self.lowest_tested_value is not None and self.lowest_tested_value < 0:
            raise ValueError("lowest_tested_value cannot be negative")
