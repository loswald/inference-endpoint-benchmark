"""Canonical evidence-state semantics and plain-language presentation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .models import EvidenceCell, EvidenceDisposition, EvidenceState


@dataclass(frozen=True)
class StatePresentation:
    label: str
    short_label: str
    description: str
    disposition: EvidenceDisposition
    color: str
    marker: str


STATE_PRESENTATIONS: dict[EvidenceState, StatePresentation] = {
    EvidenceState.CONFIRMED_LOWER_BOUND: StatePresentation(
        "Confirmed through the highest tested load",
        "confirmed ≥",
        "Repeated, separated healthy observations establish a lower bound; "
        "the true limit may be higher.",
        EvidenceDisposition.ESTABLISHED,
        "#0F766E",
        "o",
    ),
    EvidenceState.CONFIRMED_INTERVAL: StatePresentation(
        "Confirmed operating interval",
        "confirmed range",
        "Repeated healthy observations establish the lower edge and measured overload "
        "establishes the upper edge.",
        EvidenceDisposition.ESTABLISHED,
        "#2563EB",
        "o",
    ),
    EvidenceState.MEASURED: StatePresentation(
        "Measured",
        "measured",
        "The registered experiment completed and produced an eligible descriptive result.",
        EvidenceDisposition.ESTABLISHED,
        "#0F766E",
        "o",
    ),
    EvidenceState.MEASURED_WITH_INTERVAL: StatePresentation(
        "Measured with uncertainty interval",
        "measured + CI",
        "The registered experiment completed and reports an estimate with its stated "
        "uncertainty interval.",
        EvidenceDisposition.ESTABLISHED,
        "#2563EB",
        "o",
    ),
    EvidenceState.HEALTHY_EXPLORATORY: StatePresentation(
        "Healthy observation, not repeat-confirmed",
        "exploratory",
        "The tested load worked, but the registered repeat requirement was not completed.",
        EvidenceDisposition.EXPLORATORY,
        "#7C3AED",
        "D",
    ),
    EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR: StatePresentation(
        "Search stopped before finding a healthy baseline",
        "search incomplete",
        "No healthy baseline was established within the tested range. Lower loads remain "
        "untested, so this is not an endpoint-capacity failure.",
        EvidenceDisposition.UNRESOLVED,
        "#D97706",
        "v",
    ),
    EvidenceState.MEASURED_NEGATIVE: StatePresentation(
        "Measured negative result",
        "measured negative",
        "The experiment completed and the registered success criterion was not met.",
        EvidenceDisposition.MEASURED_NEGATIVE,
        "#B91C1C",
        "X",
    ),
    EvidenceState.TRANSPORT_GATED: StatePresentation(
        "Could not establish a valid transport baseline",
        "transport-gated",
        "Transport behavior prevented the scientific comparison from starting; it is not "
        "a capability or capacity verdict.",
        EvidenceDisposition.UNRESOLVED,
        "#C2410C",
        "s",
    ),
    EvidenceState.UNSUPPORTED: StatePresentation(
        "Unsupported by the product contract",
        "unsupported",
        "The provider documentation or API contract says this operation is unavailable.",
        EvidenceDisposition.NOT_APPLICABLE,
        "#64748B",
        "s",
    ),
    EvidenceState.CENSORED: StatePresentation(
        "Measured, but no valid estimate was established",
        "censored",
        "The observation is retained, but a guard, timeout, non-monotonic response, or "
        "eligibility rule prevents the requested estimate.",
        EvidenceDisposition.UNRESOLVED,
        "#D97706",
        "P",
    ),
    EvidenceState.NOT_RUN: StatePresentation(
        "Experiment not run for this exact cell",
        "not run",
        "No observation exists for this exact endpoint, workload recipe, and experiment contract.",
        EvidenceDisposition.UNRESOLVED,
        "#94A3B8",
        "s",
    ),
    EvidenceState.UNKNOWN: StatePresentation(
        "Unrecognized evidence state",
        "unknown",
        "The input state was not recognized; inspect the source row before making a claim.",
        EvidenceDisposition.UNRESOLVED,
        "#475569",
        "$?$",
    ),
}


LEGACY_STATE_ALIASES: dict[str, EvidenceState] = {
    "confirmed_right_censored_lower_bound": EvidenceState.CONFIRMED_LOWER_BOUND,
    "confirmed_bracketed_interval": EvidenceState.CONFIRMED_INTERVAL,
    "complete": EvidenceState.MEASURED,
    "measured": EvidenceState.MEASURED,
    "measured_with_interval": EvidenceState.MEASURED_WITH_INTERVAL,
    "unconfirmed_healthy_observation_only": EvidenceState.HEALTHY_EXPLORATORY,
    "censored_no_valid_healthy_epoch": EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR,
    "left_censored_no_healthy_at_lowest_tested_rate": EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR,
    "completed_no_healthy_at_lowest_tested_rate": EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR,
    "measured_failure": EvidenceState.MEASURED_NEGATIVE,
    "failed": EvidenceState.MEASURED_NEGATIVE,
    "transport_gated": EvidenceState.TRANSPORT_GATED,
    "baseline_transport_gated": EvidenceState.TRANSPORT_GATED,
    "documented_unavailable": EvidenceState.UNSUPPORTED,
    "documented_unsupported": EvidenceState.UNSUPPORTED,
    "unsupported": EvidenceState.UNSUPPORTED,
    "measured_capacity_state_without_numeric_bound": EvidenceState.CENSORED,
    "censored_nonmonotonic_overload": EvidenceState.CENSORED,
    "campaign_censored_before_start": EvidenceState.NOT_RUN,
    "not_run": EvidenceState.NOT_RUN,
    "not_measured": EvidenceState.NOT_RUN,
}


def canonical_state(
    value: str | EvidenceState | None,
    aliases: Mapping[str, str] | None = None,
) -> EvidenceState:
    """Normalize a provider/source-specific state without silently inventing meaning."""

    if isinstance(value, EvidenceState):
        return value
    normalized = str(value or "").strip().lower()
    if aliases and normalized in aliases:
        normalized = str(aliases[normalized]).strip().lower()
    try:
        return EvidenceState(normalized)
    except ValueError:
        return LEGACY_STATE_ALIASES.get(normalized, EvidenceState.UNKNOWN)


def state_presentation(state: EvidenceState | str) -> StatePresentation:
    return STATE_PRESENTATIONS[canonical_state(state)]


def _format_number(value: float) -> str:
    if value < 0.1:
        rendered = f"{value:.3f}"
    elif value < 1:
        rendered = f"{value:.2f}"
    else:
        rendered = f"{value:.3g}"
    return rendered.rstrip("0").rstrip(".")


def state_message(cell: EvidenceCell, *, load_unit: str = "requests/second") -> str:
    """Return a complete, claim-safe sentence for a report cell."""

    presentation = state_presentation(cell.state)
    if (
        cell.state is EvidenceState.SEARCH_INCOMPLETE_BELOW_FLOOR
        and cell.lowest_tested_value is not None
    ):
        floor = _format_number(cell.lowest_tested_value)
        return (
            f"No healthy baseline was established at or above {floor} {load_unit}; "
            "lower loads were not tested, so usable capacity was not determined."
        )
    if cell.raw_state and cell.state is EvidenceState.UNKNOWN:
        return f"Unrecognized source state {cell.raw_state!r}; no result is claimed."
    return presentation.description
