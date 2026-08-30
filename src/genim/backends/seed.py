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
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
        radii = np.asarray([
            float(covalent_radii[Atoms(symbol).numbers[0]])
            for symbol in species
        ], dtype=float)
        sphere_volume = float(np.sum((4.0 / 3.0) * math.pi * np.power(radii, 3)))
        target_packing = float(np.clip(0.30 - 0.025 * (float(temperature) - 0.8), 0.22, 0.36))
        target_volume = float(
            np.clip(
                sphere_volume / target_packing,
                8.0 * len(species),
                95.0 * len(species),
            )
        )

        def farthest_points(cell_shape: np.ndarray) -> np.ndarray:
            if len(species) == 1:
                return np.asarray([[0.0, 0.0, 0.0]], dtype=float)
            pool_size = max(192, 48 * len(species))
            pool = rng.random((pool_size, 3))
            chosen = [pool[int(rng.integers(0, pool_size))]]
            for _ in range(1, len(species)):
                delta = pool[:, None, :] - np.asarray(chosen, dtype=float)[None, :, :]
                delta -= np.round(delta)
                distances = np.linalg.norm(delta @ cell_shape, axis=2)
                score = np.min(distances, axis=1)
                chosen.append(pool[int(np.argmax(score))])
            return np.asarray(chosen, dtype=float)

        def ratio_metrics(atoms: Atoms) -> tuple[float, float]:
            if len(atoms) < 2:
                return 1.0, 1.0
            distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
            denominator = radii[:, None] + radii[None, :]
            ratios = np.divide(
                distances,
                denominator,
                out=np.full_like(distances, np.inf),
                where=denominator > 0,
            )
            np.fill_diagonal(ratios, np.inf)
            return float(np.min(ratios)), float(np.max(np.min(ratios, axis=1)))

        best: tuple[float, np.ndarray, np.ndarray, float, float] | None = None
        anisotropy = min(0.30, 0.06 + 0.08 * float(temperature))
        angle_span = min(12.0, 2.5 + 4.0 * float(temperature))
        for _ in range(64):
            length_factors = np.exp(rng.normal(0.0, anisotropy, size=3))
            angles = 90.0 + rng.uniform(-angle_span, angle_span, size=3)
            shape = np.asarray(
                cellpar_to_cell([*length_factors.tolist(), *angles.tolist()]),
                dtype=float,
            )
            shape /= abs(float(np.linalg.det(shape))) ** (1.0 / 3.0)
            positions = farthest_points(shape)
            cell = shape * target_volume ** (1.0 / 3.0)
            atoms = Atoms(symbols=species, cell=cell, scaled_positions=positions, pbc=True)
            min_ratio, max_nearest_ratio = ratio_metrics(atoms)

            if min_ratio < 0.78 and max_nearest_ratio <= 1.45:
                factor = 0.78 / max(min_ratio, 1e-8)
                cell = cell * factor
            elif max_nearest_ratio > 1.45 and min_ratio >= 0.78:
                factor = 1.45 / max_nearest_ratio
                cell = cell * factor
            atoms.set_cell(cell, scale_atoms=True)
            min_ratio, max_nearest_ratio = ratio_metrics(atoms)
            penalty = (
                80.0 * max(0.0, 0.72 - min_ratio) ** 2
                + 30.0 * max(0.0, max_nearest_ratio - 1.50) ** 2
                + 0.1 * abs(min_ratio - 0.9)
            )
            record = (penalty, cell.copy(), positions.copy(), min_ratio, max_nearest_ratio)
            if best is None or record[0] < best[0]:
                best = record
                if penalty < 1e-5:
                    break

        assert best is not None
        _, cell, positions, min_ratio, max_nearest_ratio = best
        diagnostics: dict[str, float | int | str] = {
            "algorithm": "covalent_radius_farthest_point_seed_v2",
            "target_packing_fraction": target_packing,
            "target_volume_per_atom": target_volume / len(species),
            "constructed_min_covalent_ratio": min_ratio,
            "constructed_max_nearest_covalent_ratio": max_nearest_ratio,
            "placement_trials": 64,
        }
        return cell, positions, diagnostics

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
            cell, positions, construction_metrics = self._cell_and_positions(
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
                        **construction_metrics,
                        "validation_profile": "periodic_geometry_seed",
                        "learned_model": False,
                    },
                )
            )
        return proposals


__all__ = ["AlgorithmicSeedBackend"]
