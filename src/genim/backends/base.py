from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class GenerationCondition:
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

    def to_matra_prompt(self) -> str | None:
        """Translate supported constraints into Matra's documented block syntax."""

        if self.elements and not self.exact_elements:
            raise ValueError("Matra element prompting supports exact element sets, not required subsets")
        blocks: list[str] = []
        if self.stability == "stable":
            blocks.append("EHULL_DISC EH0")
        elif self.stability == "unstable":
            blocks.append("EHULL_DISC EH1")
        if self.elements:
            blocks.append(f"AMT {len(self.elements)}")
            blocks.append("ELMS " + " ".join(self.elements))
        if self.stoichiometry:
            blocks.append("STOICH " + " ".join(_format_matra_number(value) for value in self.stoichiometry))
        if self.spacegroup_number is not None:
            blocks.append(f"SPACEGROUP S{self.spacegroup_number}")
        if self.matra_wyckoff_indices:
            blocks.append("WYCKOFF " + " ".join(f"W{value}" for value in self.matra_wyckoff_indices))
        if self.target_e_hull is not None:
            blocks.append(f"EHULL C*{float(self.target_e_hull):.8g}")
        if not blocks:
            return None
        return " STOP ".join(blocks) + " STOP"


@dataclass(frozen=True)
class ProposalConfig:
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
            raise ValueError("ProposalConfig.n must be >= 0")
        if int(self.batch_size) < 1:
            raise ValueError("ProposalConfig.batch_size must be >= 1")
        if int(self.max_sites) < 1 or int(self.min_sites) < 1:
            raise ValueError("ProposalConfig site limits must be >= 1")
        if int(self.min_sites) > int(self.max_sites):
            raise ValueError("ProposalConfig.min_sites must be <= max_sites")
        if float(self.temperature) <= 0:
            raise ValueError("ProposalConfig.temperature must be > 0")
        if int(self.forward) < 1:
            raise ValueError("ProposalConfig.forward must be >= 1")
        if int(self.decode_jobs) < 0:
            raise ValueError("ProposalConfig.decode_jobs must be >= 0")

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


def evaluate_condition(
    atoms: Atoms | None,
    validation: ValidationReport,
    condition: GenerationCondition,
) -> dict[str, bool | None]:
    checks: dict[str, bool | None] = {}
    if atoms is None:
        return {field: None for field in condition.requested_fields()}

    if condition.elements:
        actual_elements = set(atoms.get_chemical_symbols())
        requested = set(condition.elements)
        checks["elements"] = (
            actual_elements == requested if condition.exact_elements else requested.issubset(actual_elements)
        )
    if condition.stoichiometry:
        actual_elements = set(atoms.get_chemical_symbols())
        if actual_elements != set(condition.elements):
            checks["stoichiometry"] = False
        else:
            symbols = atoms.get_chemical_symbols()
            counts = np.asarray([symbols.count(symbol) for symbol in condition.elements], dtype=float)
            target = np.asarray(condition.stoichiometry, dtype=float)
            checks["stoichiometry"] = bool(
                counts.sum() > 0
                and np.allclose(counts / counts.sum(), target / target.sum(), rtol=0.0, atol=1e-8)
            )
    if condition.spacegroup_number is not None:
        actual = validation.metrics.get("spacegroup_number")
        checks["spacegroup_number"] = None if actual is None else int(actual) == condition.spacegroup_number
    if condition.matra_wyckoff_indices:
        checks["matra_wyckoff_indices"] = None
    if condition.stability != "any":
        checks["stability"] = None
    if condition.target_e_hull is not None:
        checks["target_e_hull"] = None
    return checks


def condition_compliance(checks: dict[str, bool | None]) -> bool | None:
    if not checks:
        return True
    values = list(checks.values())
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


@dataclass
class ProposedStructure:
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
    backend_metrics: dict[str, Any] = field(default_factory=dict)
    raw_sequence: str | None = None
    error: str | None = None
    duplicate_of: str | None = None

    @property
    def valid(self) -> bool:
        return bool(self.atoms is not None and self.validation.valid and self.error is None)

    @property
    def condition_compliant(self) -> bool | None:
        return condition_compliance(self.condition_checks)

    @property
    def accepted(self) -> bool:
        return bool(self.valid and self.condition_compliant is not False and self.duplicate_of is None)

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "backend": self.backend,
            "model_name": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "valid": self.valid,
            "accepted": self.accepted,
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
            "applied_constraints": list(self.applied_constraints),
            "unsupported_constraints": list(self.unsupported_constraints),
            "sampling": dict(self.sampling),
            "backend_metrics": dict(self.backend_metrics),
            "raw_sequence": self.raw_sequence,
            "structure": structure_payload(self.atoms),
        }


@runtime_checkable
class ProposalBackend(Protocol):
    backend_name: str
    model_name: str
    checkpoint_sha256: str | None
    capabilities: frozenset[str]

    def propose(
        self,
        *,
        condition: GenerationCondition,
        config: ProposalConfig,
    ) -> list[ProposedStructure]:
        ...


__all__ = [
    "DEFAULT_VALIDATION_OPTIONS",
    "GenerationCondition",
    "ProposalBackend",
    "ProposalConfig",
    "ProposedStructure",
    "condition_compliance",
    "evaluate_condition",
    "structure_payload",
]
