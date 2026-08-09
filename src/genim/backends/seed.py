from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import cellpar_to_cell

from ..composition import ratios_to_integer_counts
from ..validate import validate_atoms_report
from .base import (
    BackendCapabilities,
    ConstraintApplication,
    GeneratedCandidate,
    GenerationConstraints,
    GenerationSettings,
    evaluate_constraints,
)


class AlgorithmicSeedBackend:
    """Construct reproducible, composition-exact periodic starting geometries.

    This backend is deliberately not presented as a learned predictor.  It is
    the always-available baseline used when no compatible checkpoint is loaded,
    and its candidates still pass through the same validation and constraint
    evidence contract as checkpoint-generated structures.
    """

    backend_name = "algorithmic_seed"
    model_name = "genim-algorithmic-seed-v1"
    checkpoint_sha256 = None
    capabilities = BackendCapabilities(
        {
            "elements": ConstraintApplication.CONSTRUCTION,
            "stoichiometry": ConstraintApplication.CONSTRUCTION,
        },
        notes=(
            "Creates composition-exact periodic starting geometries. It does not "
            "claim checkpoint inference, target-space-group enforcement, or "
            "thermodynamic stability."
        ),
    )

    @staticmethod
    def _integer_counts(condition: GenerationConstraints) -> dict[str, int]:
        if not condition.elements:
            raise ValueError("Algorithmic seed generation requires at least one element")
        ratios = condition.stoichiometry or tuple(1.0 for _ in condition.elements)
        return ratios_to_integer_counts(list(condition.elements), list(ratios))

    @staticmethod
    def _cell_and_positions(
        species: list[str],
        *,
        rng: np.random.Generator,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        radii = [
            float(covalent_radii[Atoms(symbol).numbers[0]])
            for symbol in species
        ]
        mean_radius_cubed = float(np.mean(np.power(radii, 3)))
        target_volume_per_atom = float(
            np.clip(1.5 * (4.0 / 3.0) * math.pi * mean_radius_cubed, 12.0, 95.0)
        )
        base = (len(species) * target_volume_per_atom) ** (1.0 / 3.0)
        variation = min(0.12, 0.025 + 0.035 * float(temperature))
        lengths = base * (1.0 + rng.uniform(-variation, variation, size=3))
        angles = 90.0 + rng.uniform(-4.5, 4.5, size=3)
        cell = np.asarray(cellpar_to_cell([*lengths.tolist(), *angles.tolist()]), dtype=float)

        dimension = int(math.ceil(len(species) ** (1.0 / 3.0)))
        jitter_scale = min(0.11, 0.025 + 0.025 * float(temperature)) / dimension
        positions: list[list[float]] = []
        for x in range(dimension):
            for y in range(dimension):
                for z in range(dimension):
                    if len(positions) >= len(species):
                        break
                    point = (
                        (np.asarray([x, y, z], dtype=float) + 0.5) / dimension
                        + rng.uniform(-jitter_scale, jitter_scale, size=3)
                    ) % 1.0
                    positions.append(point.tolist())
        return cell, np.asarray(positions, dtype=float), target_volume_per_atom

    def propose(
        self,
        *,
        condition: GenerationConstraints,
        config: GenerationSettings,
    ) -> list[GeneratedCandidate]:
        counts = self._integer_counts(condition)
        species_template = [
            symbol
            for symbol, count in counts.items()
            for _ in range(int(count))
        ]
        if len(species_template) > int(config.max_sites):
            raise ValueError(
                f"Reduced composition requires {len(species_template)} sites, "
                f"above max_sites={int(config.max_sites)}"
            )

        requested = set(condition.requested_fields())
        applied = self.capabilities.applied_fields(requested)
        unsupported = self.capabilities.unsupported_fields(requested)
        proposals: list[GeneratedCandidate] = []
        base_seed = 7 if config.seed is None else int(config.seed)
        validation_options = config.validation_kwargs()
        # Connectivity is chemistry-model dependent and is not a meaningful
        # rejection criterion for a generic pre-relaxation seed.
        if "max_nn_factor" not in config.validation_options:
            validation_options["max_nn_factor"] = None
        if "min_coordination" not in config.validation_options:
            validation_options["min_coordination"] = None

        for index in range(int(config.n)):
            payload = json.dumps(
                {
                    "condition": condition.as_dict(),
                    "seed": base_seed,
                    "index": index,
                    "model": self.model_name,
                },
                sort_keys=True,
            )
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            local_seed = int(digest[:16], 16)
            rng = np.random.default_rng(local_seed)
            species = list(species_template)
            rng.shuffle(species)
            cell, positions, target_vpa = self._cell_and_positions(
                species,
                rng=rng,
                temperature=float(config.temperature),
            )
            atoms = Atoms(symbols=species, cell=cell, scaled_positions=positions, pbc=True)
            validation = validate_atoms_report(atoms, **validation_options)
            assessments = evaluate_constraints(atoms, validation, condition)
            proposals.append(
                GeneratedCandidate(
                    candidate_id=f"seed-{digest[:16]}-{index + 1:03d}",
                    backend=self.backend_name,
                    model_name=self.model_name,
                    checkpoint_sha256=None,
                    atoms=atoms,
                    validation=validation,
                    condition=condition.as_dict(),
                    condition_checks={
                        name: assessment.value for name, assessment in assessments.items()
                    },
                    applied_constraints=applied,
                    unsupported_constraints=unsupported,
                    sampling=config.as_dict(),
                    constraint_assessments=assessments,
                    backend_metrics={
                        "generation_mode": "algorithmic_seed",
                        "algorithm": "stratified_fractional_grid_with_seeded_jitter",
                        "target_volume_per_atom": target_vpa,
                        "validation_profile": "periodic_geometry_seed",
                        "learned_model": False,
                    },
                )
            )
        return proposals


__all__ = ["AlgorithmicSeedBackend"]
