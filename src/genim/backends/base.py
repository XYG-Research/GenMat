from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from ..validate import ValidationReport


DEFAULT_VALIDATION_OPTIONS: dict[str, Any] = {
    "min_dist": 0.5,
    "symprec": 1e-2,
    "min_dist_factor": 0.55,
    "max_dist_factor": None,
    "max_nn_factor": 1.5,
    "min_coordination": 1,
    "require_connected": False,
    "max_atoms": 500,
    "vol_per_atom_min": 1.0,
    "vol_per_atom_max": 100.0,
}


def _normalise_elements(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    elements: list[str] = []
    for value in values:
        symbol = str(value).strip()
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown chemical element: {symbol!r}")
        if symbol in elements:
            raise ValueError(f"Duplicate chemical element: {symbol!r}")
        elements.append(symbol)
    return tuple(elements)


def _format_matra_number(value: float) -> str:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError("Matra numeric conditions must be finite and > 0")
    rounded = round(number)
    if abs(number - rounded) < 1e-10 and 1 <= rounded <= 399:
        return str(int(rounded))
    return f"C*{number:.8g}"


class ConstraintApplication(str, Enum):
    """How a backend applies a requested generation constraint.

    This describes model behavior, not scientific verification.  A conditioned
    model can still emit a structure that violates the requested constraint.
    """

    UNSUPPORTED = "unsupported"
    CONSTRUCTION = "construction"
    CONDITIONING = "conditioning"
    SAMPLING_FILTER = "sampling_filter"
    POST_FILTER = "post_filter"


class ConstraintStatus(str, Enum):
    """Outcome of evaluating one requested constraint against evidence."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class ConstraintAssessment:
    """Auditable evidence for one constraint on one decoded candidate."""

    constraint: str
    status: ConstraintStatus
    method: str
    evidence_level: str
    detail: str | None = None

    def __post_init__(self) -> None:
        constraint = str(self.constraint).strip()
        method = str(self.method).strip()
        evidence_level = str(self.evidence_level).strip()
        if not constraint or not method or not evidence_level:
            raise ValueError("Constraint assessments require constraint, method, and evidence_level")
        status = self.status if isinstance(self.status, ConstraintStatus) else ConstraintStatus(str(self.status))
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "evidence_level", evidence_level)

    @property
    def value(self) -> bool | None:
        if self.status is ConstraintStatus.SATISFIED:
            return True
        if self.status is ConstraintStatus.VIOLATED:
            return False
        return None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "method": self.method,
            "evidence_level": self.evidence_level,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BackendCapabilities:
    """Machine-readable declaration of constraint handling by a backend."""

    constraint_handling: Mapping[str, ConstraintApplication | str] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        normalised: dict[str, ConstraintApplication] = {}
        for name, mode in self.constraint_handling.items():
            field_name = str(name).strip()
            if not field_name:
                raise ValueError("Capability constraint names must not be empty")
            normalised[field_name] = (
                mode if isinstance(mode, ConstraintApplication) else ConstraintApplication(str(mode))
            )
        object.__setattr__(self, "constraint_handling", normalised)

    @property
    def supported_constraints(self) -> frozenset[str]:
        return frozenset(
            name
            for name, mode in self.constraint_handling.items()
            if mode is not ConstraintApplication.UNSUPPORTED
        )

    def handling_for(self, name: str) -> ConstraintApplication:
        return self.constraint_handling.get(str(name), ConstraintApplication.UNSUPPORTED)

    def applied_fields(self, requested: set[str] | tuple[str, ...]) -> list[str]:
        return sorted(name for name in requested if name in self.supported_constraints)

    def unsupported_fields(self, requested: set[str] | tuple[str, ...]) -> list[str]:
        return sorted(name for name in requested if name not in self.supported_constraints)

    def to_record(self) -> dict[str, Any]:
        return {
            "constraint_handling": {
                name: mode.value for name, mode in sorted(self.constraint_handling.items())
            },
            "supported_constraints": sorted(self.supported_constraints),
            "notes": self.notes,
        }


def normalise_backend_capabilities(
    capabilities: BackendCapabilities | Mapping[str, ConstraintApplication | str] | set[str] | frozenset[str],
) -> BackendCapabilities:
    """Accept legacy sets while exposing the new capability schema."""

    if isinstance(capabilities, BackendCapabilities):
        return capabilities
    if isinstance(capabilities, Mapping):
        return BackendCapabilities(capabilities)
    return BackendCapabilities(
        {str(name): ConstraintApplication.CONDITIONING for name in capabilities},
        notes="Legacy capability set; application mode assumed to be conditioning.",
    )


@dataclass(frozen=True)
class GenerationConstraints:
    """Backend-neutral generation intent with explicit portability boundaries.

    ``elements``, ``stoichiometry`` and ``spacegroup_number`` can be evaluated
    on any decoded structure. ``matra_wyckoff_indices`` uses Matra's checkpoint
    vocabulary indices and is intentionally named as a backend-specific field
    rather than pretending those indices are portable Wyckoff letters.
    """

    elements: tuple[str, ...] | list[str] = ()
    stoichiometry: tuple[float, ...] | list[float] = ()
    spacegroup_number: int | None = None
    matra_wyckoff_indices: tuple[int, ...] | list[int] = ()
    stability: str = "any"  # any | stable | unstable
    target_e_hull: float | None = None
    exact_elements: bool = True

    def __post_init__(self) -> None:
        elements = _normalise_elements(tuple(self.elements))
        stoichiometry = tuple(float(value) for value in self.stoichiometry)
        if stoichiometry and not elements:
            raise ValueError("stoichiometry requires elements")
        if stoichiometry and len(stoichiometry) != len(elements):
            raise ValueError("stoichiometry must have one value per element")
        if any(not np.isfinite(value) or value <= 0 for value in stoichiometry):
            raise ValueError("stoichiometry values must be finite and > 0")

        spacegroup = self.spacegroup_number
        if spacegroup is not None and not (1 <= int(spacegroup) <= 230):
            raise ValueError("spacegroup_number must be in 1..230")

        wyckoff = tuple(int(value) for value in self.matra_wyckoff_indices)
        if any(value < 1 or value > 990 for value in wyckoff):
            raise ValueError("matra_wyckoff_indices must be in 1..990")

        stability = str(self.stability or "any").strip().lower()
        if stability not in {"any", "stable", "unstable"}:
            raise ValueError("stability must be one of: any, stable, unstable")
        if self.target_e_hull is not None:
            target_e_hull = float(self.target_e_hull)
            if not np.isfinite(target_e_hull) or target_e_hull < 0:
                raise ValueError("target_e_hull must be finite and >= 0")
            object.__setattr__(self, "target_e_hull", target_e_hull)

        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "stoichiometry", stoichiometry)
        object.__setattr__(self, "spacegroup_number", None if spacegroup is None else int(spacegroup))
        object.__setattr__(self, "matra_wyckoff_indices", wyckoff)
        object.__setattr__(self, "stability", stability)

    def requested_fields(self) -> tuple[str, ...]:
        fields: list[str] = []
        if self.elements:
            fields.append("elements")
        if self.stoichiometry:
            fields.append("stoichiometry")
        if self.spacegroup_number is not None:
            fields.append("spacegroup_number")
        if self.matra_wyckoff_indices:
            fields.append("matra_wyckoff_indices")
        if self.stability != "any":
            fields.append("stability")
        if self.target_e_hull is not None:
            fields.append("target_e_hull")
        return tuple(fields)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elements": list(self.elements),
            "stoichiometry": list(self.stoichiometry),
            "spacegroup_number": self.spacegroup_number,
            "matra_wyckoff_indices": list(self.matra_wyckoff_indices),
            "stability": self.stability,
            "target_e_hull": self.target_e_hull,
            "exact_elements": bool(self.exact_elements),
        }

    def to_matra_prompt(self, *, supported_constraints: set[str] | frozenset[str] | None = None) -> str | None:
        """Translate supported constraints into Matra's documented block syntax."""

        supported = set(self.requested_fields()) if supported_constraints is None else set(supported_constraints)
        if self.elements and "elements" in supported and not self.exact_elements:
            raise ValueError("Matra element prompting supports exact element sets, not required subsets")
        blocks: list[str] = []
        if "stability" in supported and self.stability == "stable":
            blocks.append("EHULL_DISC EH0")
        elif "stability" in supported and self.stability == "unstable":
            blocks.append("EHULL_DISC EH1")
        if self.elements and "elements" in supported:
            blocks.append(f"AMT {len(self.elements)}")
            blocks.append("ELMS " + " ".join(self.elements))
        if self.stoichiometry and "stoichiometry" in supported:
            blocks.append("STOICH " + " ".join(_format_matra_number(value) for value in self.stoichiometry))
        if self.spacegroup_number is not None and "spacegroup_number" in supported:
            blocks.append(f"SPACEGROUP S{self.spacegroup_number}")
        if self.matra_wyckoff_indices and "matra_wyckoff_indices" in supported:
            blocks.append("WYCKOFF " + " ".join(f"W{value}" for value in self.matra_wyckoff_indices))
        if self.target_e_hull is not None and "target_e_hull" in supported:
            blocks.append(f"EHULL C*{float(self.target_e_hull):.8g}")
        if not blocks:
            return None
        return " STOP ".join(blocks) + " STOP"


@dataclass(frozen=True)
class GenerationSettings:
    n: int = 1
    batch_size: int = 16
    max_sites: int = 25
    min_sites: int = 1
    temperature: float = 0.8
    top_k: int = 0
    seed: int | None = None
    forward: int = 150
    decode_jobs: int = 0
    require_backend_consistency: bool = True
    validation_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.n) < 0:
            raise ValueError("GenerationSettings.n must be >= 0")
        if int(self.batch_size) < 1:
            raise ValueError("GenerationSettings.batch_size must be >= 1")
        if int(self.max_sites) < 1 or int(self.min_sites) < 1:
            raise ValueError("GenerationSettings site limits must be >= 1")
        if int(self.min_sites) > int(self.max_sites):
            raise ValueError("GenerationSettings.min_sites must be <= max_sites")
        if not np.isfinite(float(self.temperature)) or float(self.temperature) <= 0:
            raise ValueError("GenerationSettings.temperature must be finite and > 0")
        if int(self.top_k) < 0:
            raise ValueError("GenerationSettings.top_k must be >= 0")
        if int(self.forward) < 1:
            raise ValueError("GenerationSettings.forward must be >= 1")
        if int(self.decode_jobs) < 0:
            raise ValueError("GenerationSettings.decode_jobs must be >= 0")

    def validation_kwargs(self) -> dict[str, Any]:
        options = dict(DEFAULT_VALIDATION_OPTIONS)
        options.update(self.validation_options)
        return options

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["validation_options"] = self.validation_kwargs()
        return record


def structure_payload(atoms: Atoms | None) -> dict[str, Any] | None:
    if atoms is None:
        return None
    return {
        "lattice": np.asarray(atoms.cell.array, dtype=float).tolist(),
        "frac_coords": np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float).tolist(),
        "species": list(atoms.get_chemical_symbols()),
    }


def evaluate_constraints(
    atoms: Atoms | None,
    validation: ValidationReport,
    condition: GenerationConstraints,
) -> dict[str, ConstraintAssessment]:
    """Evaluate every requested constraint without conflating unknown with pass."""

    assessments: dict[str, ConstraintAssessment] = {}
    if atoms is None:
        return {
            name: ConstraintAssessment(
                constraint=name,
                status=ConstraintStatus.NOT_EVALUATED,
                method="decoded_structure_unavailable",
                evidence_level="L0",
                detail="No decoded structure was available for independent evaluation.",
            )
            for name in condition.requested_fields()
        }

    if condition.elements:
        actual_elements = set(atoms.get_chemical_symbols())
        requested = set(condition.elements)
        passed = (
            actual_elements == requested if condition.exact_elements else requested.issubset(actual_elements)
        )
        assessments["elements"] = ConstraintAssessment(
            constraint="elements",
            status=ConstraintStatus.SATISFIED if passed else ConstraintStatus.VIOLATED,
            method="decoded_structure.element_set",
            evidence_level="L1",
        )
    if condition.stoichiometry:
        actual_elements = set(atoms.get_chemical_symbols())
        if actual_elements != set(condition.elements):
            passed = False
        else:
            symbols = atoms.get_chemical_symbols()
            counts = np.asarray([symbols.count(symbol) for symbol in condition.elements], dtype=float)
            target = np.asarray(condition.stoichiometry, dtype=float)
            passed = bool(
                counts.sum() > 0
                and np.allclose(counts / counts.sum(), target / target.sum(), rtol=0.0, atol=1e-8)
            )
        assessments["stoichiometry"] = ConstraintAssessment(
            constraint="stoichiometry",
            status=ConstraintStatus.SATISFIED if passed else ConstraintStatus.VIOLATED,
            method="decoded_structure.reduced_composition",
            evidence_level="L1",
        )
    if condition.spacegroup_number is not None:
        actual = validation.metrics.get("spacegroup_number")
        if actual is None:
            assessments["spacegroup_number"] = ConstraintAssessment(
                constraint="spacegroup_number",
                status=ConstraintStatus.NOT_EVALUATED,
                method="spglib.unavailable",
                evidence_level="L1",
                detail="The validation report did not contain a space-group number.",
            )
        else:
            passed = int(actual) == condition.spacegroup_number
            assessments["spacegroup_number"] = ConstraintAssessment(
                constraint="spacegroup_number",
                status=ConstraintStatus.SATISFIED if passed else ConstraintStatus.VIOLATED,
                method="spglib.spacegroup_number",
                evidence_level="L1",
            )
    if condition.matra_wyckoff_indices:
        assessments["matra_wyckoff_indices"] = ConstraintAssessment(
            constraint="matra_wyckoff_indices",
            status=ConstraintStatus.NOT_EVALUATED,
            method="matra_token_indices_not_portable",
            evidence_level="L0",
            detail="Checkpoint-specific Matra indices cannot be reconstructed from ASE alone.",
        )
    if condition.stability != "any":
        assessments["stability"] = ConstraintAssessment(
            constraint="stability",
            status=ConstraintStatus.NOT_EVALUATED,
            method="energy_model_not_run",
            evidence_level="L1",
            detail="A stability prompt is not evidence of thermodynamic stability.",
        )
    if condition.target_e_hull is not None:
        assessments["target_e_hull"] = ConstraintAssessment(
            constraint="target_e_hull",
            status=ConstraintStatus.NOT_EVALUATED,
            method="convex_hull_not_run",
            evidence_level="L1",
            detail="Energy-above-hull requires a named energy model and compatible reference set.",
        )
    return assessments


def evaluate_condition(
    atoms: Atoms | None,
    validation: ValidationReport,
    condition: GenerationConstraints,
) -> dict[str, bool | None]:
    """Compatibility view of :func:`evaluate_constraints`."""

    return {name: assessment.value for name, assessment in evaluate_constraints(atoms, validation, condition).items()}


def condition_compliance(checks: dict[str, bool | None]) -> bool | None:
    if not checks:
        return True
    values = list(checks.values())
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def overall_constraint_status(checks: Mapping[str, bool | None]) -> ConstraintStatus:
    compliance = condition_compliance(dict(checks))
    if compliance is True:
        return ConstraintStatus.SATISFIED
    if compliance is False:
        return ConstraintStatus.VIOLATED
    return ConstraintStatus.NOT_EVALUATED


def _assessments_from_checks(
    checks: Mapping[str, bool | None],
) -> dict[str, ConstraintAssessment]:
    return {
        name: ConstraintAssessment(
            constraint=name,
            status=(
                ConstraintStatus.SATISFIED
                if value is True
                else ConstraintStatus.VIOLATED
                if value is False
                else ConstraintStatus.NOT_EVALUATED
            ),
            method="legacy_boolean_check",
            evidence_level="unreported",
        )
        for name, value in checks.items()
    }


@dataclass
class GeneratedCandidate:
    candidate_id: str
    backend: str
    model_name: str
    checkpoint_sha256: str | None
    atoms: Atoms | None
    validation: ValidationReport
    condition: dict[str, Any]
    condition_checks: dict[str, bool | None]
    applied_constraints: list[str]
    unsupported_constraints: list[str]
    sampling: dict[str, Any]
    constraint_assessments: dict[str, ConstraintAssessment] = field(default_factory=dict)
    backend_metrics: dict[str, Any] = field(default_factory=dict)
    raw_sequence: str | None = None
    error: str | None = None
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        if not self.constraint_assessments:
            self.constraint_assessments = _assessments_from_checks(self.condition_checks)
            return
        assessment_checks = {
            name: assessment.value for name, assessment in self.constraint_assessments.items()
        }
        if not self.condition_checks:
            self.condition_checks = assessment_checks
        elif dict(self.condition_checks) != assessment_checks:
            raise ValueError("constraint_assessments and condition_checks disagree")

    @property
    def valid(self) -> bool:
        return bool(self.atoms is not None and self.validation.valid and self.error is None)

    @property
    def condition_compliant(self) -> bool | None:
        return condition_compliance(self.condition_checks)

    @property
    def constraint_status(self) -> ConstraintStatus:
        return overall_constraint_status(self.condition_checks)

    @property
    def selected(self) -> bool:
        """Whether this candidate survived software validation and deduplication."""

        return bool(self.valid and self.condition_compliant is not False and self.duplicate_of is None)

    @property
    def accepted(self) -> bool:
        """Backward-compatible alias for :attr:`selected`."""

        return self.selected

    @property
    def selection_reason(self) -> str:
        if not self.valid:
            return self.error or self.validation.reason or "invalid"
        if self.condition_compliant is False:
            return "constraint_violation"
        if self.duplicate_of is not None:
            return "duplicate"
        if self.condition_compliant is None:
            return "selected_with_unverified_constraints"
        return "selected"

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "backend": self.backend,
            "model_name": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "valid": self.valid,
            "selected": self.selected,
            "accepted": self.accepted,
            "selection_reason": self.selection_reason,
            "error": self.error,
            "duplicate_of": self.duplicate_of,
            "formula": None if self.atoms is None else self.atoms.get_chemical_formula(mode="hill"),
            "validation": {
                "valid": self.validation.valid,
                "reason": self.validation.reason,
                "metrics": dict(self.validation.metrics),
            },
            "condition": dict(self.condition),
            "condition_checks": dict(self.condition_checks),
            "condition_compliant": self.condition_compliant,
            "constraint_status": self.constraint_status.value,
            "constraint_assessments": {
                name: assessment.to_record()
                for name, assessment in sorted(self.constraint_assessments.items())
            },
            "applied_constraints": list(self.applied_constraints),
            "unsupported_constraints": list(self.unsupported_constraints),
            "sampling": dict(self.sampling),
            "backend_metrics": dict(self.backend_metrics),
            "raw_sequence": self.raw_sequence,
            "structure": structure_payload(self.atoms),
        }


@runtime_checkable
class GenerationBackend(Protocol):
    backend_name: str
    model_name: str
    checkpoint_sha256: str | None
    capabilities: BackendCapabilities | Mapping[str, ConstraintApplication | str] | frozenset[str]

    def propose(
        self,
        *,
        condition: GenerationConstraints,
        config: GenerationSettings,
    ) -> list[GeneratedCandidate]:
        ...


# Compatibility aliases retained for GenIM 0.3 callers.  New code should use
# the generation/constraint terminology exported above.
GenerationCondition = GenerationConstraints
ProposalConfig = GenerationSettings
ProposedStructure = GeneratedCandidate
ProposalBackend = GenerationBackend


__all__ = [
    "BackendCapabilities",
    "ConstraintApplication",
    "ConstraintAssessment",
    "ConstraintStatus",
    "DEFAULT_VALIDATION_OPTIONS",
    "GeneratedCandidate",
    "GenerationBackend",
    "GenerationConstraints",
    "GenerationSettings",
    "GenerationCondition",
    "ProposalBackend",
    "ProposalConfig",
    "ProposedStructure",
    "condition_compliance",
    "evaluate_condition",
    "evaluate_constraints",
    "normalise_backend_capabilities",
    "overall_constraint_status",
    "structure_payload",
]
