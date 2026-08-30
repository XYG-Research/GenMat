from __future__ import annotations

from typing import Any

import numpy as np

from .base import GeneratedCandidate, ObservableRole


# Stable scientific provenance ID; the legacy prefix is part of the record schema.
RANKING_METHOD = "genim_scientific_triage_v1"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _geometry_component(candidate: GeneratedCandidate) -> float:
    """Score cheap geometry diagnostics without pretending to predict stability."""

    metrics = candidate.validation.metrics
    ratio = _number(metrics.get("min_covalent_distance_ratio"))
    if ratio is None:
        distance_score = 0.45
    elif ratio < 0.55:
        distance_score = 0.0
    elif ratio <= 0.9:
        distance_score = (ratio - 0.55) / 0.35
    else:
        # A large minimum ratio is not automatically better; it can indicate a
        # dilute cell.  Coordination and connectivity below carry that signal.
        distance_score = 1.0

    min_coordination = _number(metrics.get("min_coordination"))
    coordination_score = 0.55 if min_coordination is None else (1.0 if min_coordination >= 1 else 0.0)

    connected = metrics.get("connected")
    connectivity_score = 0.6 if connected is None else (1.0 if bool(connected) else 0.2)

    packing = _number(metrics.get("covalent_packing_fraction"))
    if packing is None:
        packing_score = 0.55
    elif 0.08 <= packing <= 0.78:
        packing_score = 1.0
    elif packing < 0.08:
        packing_score = max(0.0, packing / 0.08)
    else:
        packing_score = max(0.0, 1.0 - (packing - 0.78) / 0.4)

    return float(
        np.clip(
            0.45 * distance_score
            + 0.25 * coordination_score
            + 0.20 * connectivity_score
            + 0.10 * packing_score,
            0.0,
            1.0,
        )
    )


def _backend_component(candidate: GeneratedCandidate) -> float:
    keys = (
        "parse_valid",
        "bond_consistent",
        "symmetry_consistent",
        "spacegroup_consistent",
    )
    values = [candidate.backend_metrics[key] for key in keys if key in candidate.backend_metrics]
    if not values:
        return 0.55
    return float(sum(bool(value) for value in values) / len(values))


def _constraint_component(candidate: GeneratedCandidate) -> float:
    if candidate.condition_compliant is True:
        return 1.0
    if candidate.condition_compliant is False:
        return 0.0
    return 0.55


def _observable_component(candidate: GeneratedCandidate) -> float:
    """Reward evidence completeness, not favorable-looking arbitrary energies."""

    if not candidate.scientific_observables:
        return 0.0
    role_score = {
        ObservableRole.CONDITIONING_TARGET: 0.1,
        ObservableRole.MODEL_EMISSION: 0.35,
        ObservableRole.POSTPROCESSED_ESTIMATE: 0.65,
        ObservableRole.CALCULATED: 1.0,
    }
    best = max(role_score[observable.role] for observable in candidate.scientific_observables)
    if any(observable.independently_validated for observable in candidate.scientific_observables):
        best = max(best, 0.9)
    return float(best)


def score_candidate(candidate: GeneratedCandidate) -> tuple[float, dict[str, float]]:
    """Return a transparent triage score for ordering, never a stability claim."""

    components = {
        "software_validity": 1.0 if candidate.valid else 0.0,
        "geometry_quality": _geometry_component(candidate),
        "backend_consistency": _backend_component(candidate),
        "constraint_evidence": _constraint_component(candidate),
        "observable_evidence": _observable_component(candidate),
    }
    if not candidate.valid:
        return 0.0, components
    score = (
        0.30 * components["software_validity"]
        + 0.30 * components["geometry_quality"]
        + 0.17 * components["backend_consistency"]
        + 0.15 * components["constraint_evidence"]
        + 0.08 * components["observable_evidence"]
    )
    if candidate.condition_compliant is False:
        score *= 0.25
    if candidate.duplicate_of is not None:
        score *= 0.5
    return float(round(max(0.0, min(1.0, score)), 8)), components


def rank_candidates(candidates: list[GeneratedCandidate]) -> None:
    """Attach deterministic ranks to selected candidates in place."""

    for candidate in candidates:
        score, components = score_candidate(candidate)
        candidate.ranking_score = score
        candidate.ranking_components = {
            name: float(round(value, 8)) for name, value in components.items()
        }
        candidate.ranking_method = RANKING_METHOD
        candidate.rank = None

    ordered = sorted(
        (candidate for candidate in candidates if candidate.selected),
        key=lambda candidate: (
            -(candidate.ranking_score or 0.0),
            candidate.candidate_id,
        ),
    )
    for rank, candidate in enumerate(ordered, start=1):
        candidate.rank = rank


__all__ = ["RANKING_METHOD", "rank_candidates", "score_candidate"]
