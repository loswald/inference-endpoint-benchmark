"""Load declarative report profiles and normalize evidence tables."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .models import (
    DatasetProfile,
    EndpointProfile,
    EvidenceCell,
    ExperimentProfile,
    MetricColumns,
    MetricInterval,
    ProviderReportProfile,
    SourceProfile,
    StateRule,
    WorkloadProfile,
)
from .states import canonical_state


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _named_items(value: Any, name: str) -> list[tuple[str, Mapping[str, Any]]]:
    mapping = _mapping(value or {}, name)
    return [
        (str(item_id), _mapping(item, f"{name}.{item_id}")) for item_id, item in mapping.items()
    ]


def load_report_profile(path: str | Path) -> ProviderReportProfile:
    """Load one provider report profile from YAML.

    Adding a provider, endpoint, workload, or data source is a configuration change;
    no provider branch is required in the reporting code.
    """

    profile_path = Path(path)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    root = _mapping(payload, "report profile")

    endpoints = tuple(
        EndpointProfile(
            id=item_id,
            label=_text(item.get("label", item_id), f"endpoints.{item_id}.label"),
            description=str(item.get("description") or ""),
        )
        for item_id, item in _named_items(root.get("endpoints"), "endpoints")
    )
    workloads = tuple(
        WorkloadProfile(
            id=item_id,
            label=_text(item.get("label", item_id), f"workloads.{item_id}.label"),
            description=_text(item.get("description"), f"workloads.{item_id}.description"),
            recipe=str(item.get("recipe") or ""),
        )
        for item_id, item in _named_items(root.get("workloads"), "workloads")
    )
    experiments = tuple(
        ExperimentProfile(
            id=item_id,
            label=_text(item.get("label", item_id), f"experiments.{item_id}.label"),
            question=_text(item.get("question"), f"experiments.{item_id}.question"),
            method=_text(item.get("method"), f"experiments.{item_id}.method"),
            interpretation=_text(
                item.get("interpretation"), f"experiments.{item_id}.interpretation"
            ),
        )
        for item_id, item in _named_items(root.get("experiments"), "experiments")
    )
    sources = tuple(
        SourceProfile(
            id=item_id,
            label=_text(item.get("label", item_id), f"sources.{item_id}.label"),
            provenance=str(item.get("provenance") or ""),
        )
        for item_id, item in _named_items(root.get("sources"), "sources")
    )

    datasets: list[DatasetProfile] = []
    for dataset_id, item in _named_items(root.get("datasets"), "datasets"):
        metrics = tuple(
            MetricColumns(
                id=metric_id,
                label=_text(metric.get("label", metric_id), f"metrics.{metric_id}.label"),
                unit=str(metric.get("unit") or ""),
                estimate_column=_optional_text(metric.get("estimate_column")),
                low_column=_optional_text(metric.get("low_column")),
                high_column=_optional_text(metric.get("high_column")),
                interval_json_column=_optional_text(metric.get("interval_json_column")),
                sample_size_column=_optional_text(metric.get("sample_size_column")),
                sampling_unit=str(metric.get("sampling_unit") or "request"),
                interval_kind=str(metric.get("interval_kind") or "none"),
            )
            for metric_id, metric in _named_items(
                item.get("metrics"), f"datasets.{dataset_id}.metrics"
            )
        )
        datasets.append(
            DatasetProfile(
                id=dataset_id,
                filename=_text(item.get("filename"), f"datasets.{dataset_id}.filename"),
                experiment_id=_text(item.get("experiment"), f"datasets.{dataset_id}.experiment"),
                endpoint_column=_text(
                    item.get("endpoint_column"), f"datasets.{dataset_id}.endpoint_column"
                ),
                workload_column=_text(
                    item.get("workload_column"), f"datasets.{dataset_id}.workload_column"
                ),
                state_column=_optional_text(item.get("state_column")),
                default_state=_optional_text(item.get("default_state")),
                state_rules=tuple(
                    StateRule(
                        state=_text(rule.get("state"), f"datasets.{dataset_id}.state_rules.state"),
                        equals={
                            str(key): str(value)
                            for key, value in _mapping(
                                rule.get("equals"),
                                f"datasets.{dataset_id}.state_rules.equals",
                            ).items()
                        },
                    )
                    for rule in (item.get("state_rules") or [])
                    if isinstance(rule, Mapping)
                ),
                source_column=_optional_text(item.get("source_column")),
                source_value=_optional_text(item.get("source_value")),
                lowest_tested_column=_optional_text(item.get("lowest_tested_column")),
                filters={
                    str(key): str(value)
                    for key, value in _mapping(item.get("filters") or {}, "filters").items()
                },
                endpoint_aliases={
                    str(key): str(value)
                    for key, value in _mapping(
                        item.get("endpoint_aliases") or {}, "endpoint_aliases"
                    ).items()
                },
                workload_aliases={
                    str(key): str(value)
                    for key, value in _mapping(
                        item.get("workload_aliases") or {}, "workload_aliases"
                    ).items()
                },
                state_aliases={
                    str(key).lower(): str(value)
                    for key, value in _mapping(
                        item.get("state_aliases") or {}, "state_aliases"
                    ).items()
                },
                dimension_columns={
                    str(key): str(value)
                    for key, value in _mapping(
                        item.get("dimension_columns") or {}, "dimension_columns"
                    ).items()
                },
                expected_endpoint_ids=tuple(
                    str(value) for value in (item.get("expected_endpoints") or [])
                ),
                expected_workload_ids=tuple(
                    str(value) for value in (item.get("expected_workloads") or [])
                ),
                require_complete_grid=bool(item.get("require_complete_grid", False)),
                metrics=metrics,
            )
        )

    return ProviderReportProfile(
        schema_version=int(root.get("schema_version", 0)),
        provider_id=_text(root.get("provider_id"), "provider_id"),
        display_name=_text(root.get("display_name"), "display_name"),
        report_title=_text(root.get("report_title"), "report_title"),
        endpoints=endpoints,
        workloads=workloads,
        experiments=experiments,
        sources=sources,
        datasets=tuple(datasets),
    )


def _optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None


def _number(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    return None if result is None else int(result)


def _matches_filters(row: Mapping[str, str], filters: Mapping[str, str]) -> bool:
    return all(str(row.get(column) or "") == expected for column, expected in filters.items())


def _interval_from_json(value: Any) -> tuple[float | None, float | None, float | None, int | None]:
    if not value:
        return None, None, None, None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("configured interval column contains invalid JSON") from exc
    if isinstance(parsed, list):
        if len(parsed) != 2:
            raise ValueError("configured interval list must contain exactly two bounds")
        return None, _number(parsed[0]), _number(parsed[1]), None
    if not isinstance(parsed, Mapping):
        raise ValueError("configured interval JSON must be a two-item list or an object")
    return (
        _number(parsed.get("estimate")),
        _number(parsed.get("ci95_low")),
        _number(parsed.get("ci95_high")),
        _integer(parsed.get("n_units")),
    )


def _raw_state(row: Mapping[str, str], dataset: DatasetProfile) -> str:
    for rule in dataset.state_rules:
        if _matches_filters(row, rule.equals):
            return rule.state
    if dataset.state_column:
        return str(row.get(dataset.state_column) or dataset.default_state or "")
    return dataset.default_state or ""


def load_evidence_cells(
    profile: ProviderReportProfile,
    data_directory: str | Path,
    dataset_id: str,
) -> list[EvidenceCell]:
    """Normalize a configured CSV into provider-neutral evidence cells."""

    dataset = profile.dataset(dataset_id)
    path = Path(data_directory) / dataset.filename
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    cells: list[EvidenceCell] = []
    for row in rows:
        if not _matches_filters(row, dataset.filters):
            continue
        raw_endpoint = str(row.get(dataset.endpoint_column) or "")
        raw_workload = str(row.get(dataset.workload_column) or "")
        endpoint_id = dataset.endpoint_aliases.get(raw_endpoint, raw_endpoint)
        workload_id = dataset.workload_aliases.get(raw_workload, raw_workload)
        raw_state = _raw_state(row, dataset)
        metrics: dict[str, MetricInterval] = {}
        for metric in dataset.metrics:
            json_estimate, json_low, json_high, json_n = _interval_from_json(
                row.get(metric.interval_json_column) if metric.interval_json_column else None
            )
            explicit_estimate = (
                _number(row.get(metric.estimate_column)) if metric.estimate_column else None
            )
            metrics[metric.id] = MetricInterval(
                estimate=explicit_estimate if explicit_estimate is not None else json_estimate,
                low=_number(row.get(metric.low_column)) if metric.low_column else json_low,
                high=_number(row.get(metric.high_column)) if metric.high_column else json_high,
                sample_size=(
                    _integer(row.get(metric.sample_size_column))
                    if metric.sample_size_column
                    else json_n
                ),
                unit=metric.unit,
                sampling_unit=metric.sampling_unit,
                interval_kind=metric.interval_kind,
            )
        cells.append(
            EvidenceCell(
                provider_id=profile.provider_id,
                endpoint_id=endpoint_id,
                workload_id=workload_id,
                experiment_id=dataset.experiment_id,
                state=canonical_state(raw_state, dataset.state_aliases),
                metrics=metrics,
                source_id=(
                    str(row.get(dataset.source_column) or "")
                    if dataset.source_column
                    else dataset.source_value or ""
                ),
                raw_state=raw_state,
                lowest_tested_value=(
                    _number(row.get(dataset.lowest_tested_column))
                    if dataset.lowest_tested_column
                    else None
                ),
                dimensions={
                    dimension: str(row.get(column) or "")
                    for dimension, column in dataset.dimension_columns.items()
                },
                attributes=dict(row),
            )
        )
    return cells
