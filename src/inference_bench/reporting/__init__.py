"""Provider-neutral reporting primitives for inference benchmark evidence.

The modules in this package deliberately know nothing about DigitalOcean, Bedrock,
Azure, Vertex, OpenRouter, or any individual model.  Provider names, endpoint labels,
workload recipes, input files, column mappings, and source filters live in a report
profile.  The same normalized evidence cells can therefore drive tables, charts, and
future document renderers for any provider.
"""

from .charts import ChartStyle, evidence_matrix, interval_forest, save_figure
from .models import (
    DatasetProfile,
    EndpointProfile,
    EvidenceCell,
    EvidenceDisposition,
    EvidenceState,
    ExperimentProfile,
    MetricColumns,
    MetricInterval,
    ProviderReportProfile,
    SourceProfile,
    StateRule,
    WorkloadProfile,
)
from .profile import load_evidence_cells, load_report_profile
from .states import StatePresentation, canonical_state, state_message, state_presentation
from .tables import evidence_table, render_markdown_table
from .validation import (
    EvidenceValidationError,
    ValidationIssue,
    assert_publishable,
    validate_evidence_cells,
)

__all__ = [
    "ChartStyle",
    "DatasetProfile",
    "EndpointProfile",
    "EvidenceCell",
    "EvidenceDisposition",
    "EvidenceState",
    "EvidenceValidationError",
    "ExperimentProfile",
    "MetricColumns",
    "MetricInterval",
    "ProviderReportProfile",
    "SourceProfile",
    "StatePresentation",
    "StateRule",
    "ValidationIssue",
    "WorkloadProfile",
    "canonical_state",
    "assert_publishable",
    "evidence_matrix",
    "evidence_table",
    "interval_forest",
    "load_evidence_cells",
    "load_report_profile",
    "render_markdown_table",
    "save_figure",
    "state_message",
    "state_presentation",
    "validate_evidence_cells",
]
