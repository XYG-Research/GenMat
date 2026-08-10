from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable

import numpy as np
from ase import Atoms

from ..dedup import atoms_hash
from ..validate import validate_atoms_report
from .base import (
    GeneratedCandidate,
    GenerationConstraints,
    GenerationSettings,
    evaluate_constraints,
)


MUTATION_METHOD = "genim_composition_preserving_mutation_v2"


def population_plan(n: int, mutation_fraction: float) -> tuple[int, int]:
    """Return direct-parent and mutant counts for a fixed-size population."""

    total = max(0, int(n))
    if total <= 1:
        return total, 0
    requested = int(math.floor(total * float(mutation_fraction) + 0.5))
    mutant_count = min(total - 1, max(0, requested))
    return total - mutant_count, mutant_count


def _crystal_system(spacegroup_number: int | None) -> str:
    if spacegroup_number is None:
        return "unconstrained"
    number = int(spacegroup_number)
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    return "cubic"


def _mutate_cell(
    atoms: Atoms,
    *,
    rng: np.random.Generator,
    strain: float,
    spacegroup_number: int | None,
) -> list[float]:
    """Apply metric-compatible row scaling for the requested crystal system."""

    amplitude = float(strain)
    if amplitude <= 0:
        return [1.0, 1.0, 1.0]

    def factor() -> float:
        return float(np.exp(rng.normal(0.0, amplitude)))

    system = _crystal_system(spacegroup_number)
    if system in {"cubic", "trigonal"}:
        # Isotropic scaling is safe for both hexagonal-axis and primitive
        # rhombohedral representations of trigonal space groups.
        shared = factor()
        scales = np.asarray([shared, shared, shared], dtype=float)
    elif system in {"tetragonal", "hexagonal"}:
        basal = factor()
        scales = np.asarray([basal, basal, factor()], dtype=float)
    else:
        scales = np.asarray([factor(), factor(), factor()], dtype=float)

    atoms.set_cell(np.asarray(atoms.cell.array, dtype=float) * scales[:, None], scale_atoms=True)
    return [float(value) for value in scales]


def _mutate_coordinates_and_occupancy(
    atoms: Atoms,
    *,
    rng: np.random.Generator,
    displacement: float,
) -> tuple[float, bool]:
    """Perturb coordinates and occasionally swap unlike species, preserving composition."""

    rms = 0.0
    if float(displacement) > 0 and len(atoms):
        delta = rng.normal(0.0, float(displacement), size=(len(atoms), 3))
        delta -= np.mean(delta, axis=0, keepdims=True)
        atoms.set_positions(np.asarray(atoms.positions, dtype=float) + delta)
        atoms.wrap()
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

    symbols = atoms.get_chemical_symbols()
    unlike_pairs = [
        (first, second)
        for first in range(len(symbols))
        for second in range(first + 1, len(symbols))
        if symbols[first] != symbols[second]
    ]
    swapped = bool(unlike_pairs and rng.random() < 0.35)
    if swapped:
        first, second = unlike_pairs[int(rng.integers(0, len(unlike_pairs)))]
        symbols[first], symbols[second] = symbols[second], symbols[first]
        atoms.set_chemical_symbols(symbols)
    return rms, swapped


def _mutate_symmetry_orbits(
    atoms: Atoms,
    *,
    rng: np.random.Generator,
    displacement: float,
    symprec: float,
) -> tuple[float, bool]:
    """Move complete Wyckoff orbits and swap compatible orbits.

    A random displacement for each representative site is projected onto its
    site-stabilizer invariant subspace, then propagated to every symmetry-
    equivalent atom. This permits internal-coordinate mutation without treating
    independently jittered atoms as symmetry preserving.
    """

    import spglib

    cell = np.asarray(atoms.cell.array, dtype=float)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    numbers = np.asarray(atoms.numbers, dtype=int)
    dataset = spglib.get_symmetry_dataset(
        (cell, scaled, numbers),
        symprec=float(symprec),
    )
    if dataset is None or not len(atoms):
        return 0.0, False

    rotations = np.asarray(dataset.rotations, dtype=int)
    translations = np.asarray(dataset.translations, dtype=float)
    equivalents = np.asarray(dataset.equivalent_atoms, dtype=int)
    orbit_indices = [
        np.flatnonzero(equivalents == representative)
        for representative in sorted(set(int(value) for value in equivalents))
    ]
    inverse_cell = np.linalg.inv(cell)
    min_length = max(float(np.min(np.linalg.norm(cell, axis=1))), 1e-8)
    fractional_tolerance = max(5e-5, 2.0 * float(symprec) / min_length)

    def periodic_error(left: np.ndarray, right: np.ndarray) -> float:
        delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        delta -= np.rint(delta)
        return float(np.linalg.norm(delta))

    displacements = np.zeros_like(scaled)
    if float(displacement) > 0:
        for indices in orbit_indices:
            representative = int(indices[0])
            origin = scaled[representative]
            stabilizer = [
                rotation
                for rotation, translation in zip(rotations, translations)
                if periodic_error(rotation @ origin + translation, origin)
                <= fractional_tolerance
            ]
            raw_cartesian = rng.normal(0.0, float(displacement), size=3)
            raw_fractional = raw_cartesian @ inverse_cell
            representative_delta = (
                np.mean(
                    [rotation @ raw_fractional for rotation in stabilizer],
                    axis=0,
                )
                if stabilizer
                else raw_fractional
            )

            for target_value in indices:
                target = int(target_value)
                mapped_delta = None
                for rotation, translation in zip(rotations, translations):
                    if periodic_error(
                        rotation @ origin + translation,
                        scaled[target],
                    ) <= fractional_tolerance:
                        mapped_delta = rotation @ representative_delta
                        break
                if mapped_delta is not None:
                    displacements[target] = mapped_delta

        scaled = (scaled + displacements) % 1.0
        atoms.set_scaled_positions(scaled)
        atoms.wrap()

    symbols = atoms.get_chemical_symbols()
    compatible_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for first_index, first in enumerate(orbit_indices):
        first_symbols = {symbols[int(index)] for index in first}
        if len(first_symbols) != 1:
            continue
        for second in orbit_indices[first_index + 1 :]:
            second_symbols = {symbols[int(index)] for index in second}
            if (
                len(first) == len(second)
                and len(second_symbols) == 1
                and first_symbols != second_symbols
            ):
                compatible_pairs.append((first, second))

    swapped = bool(compatible_pairs and rng.random() < 0.35)
    if swapped:
        first, second = compatible_pairs[int(rng.integers(0, len(compatible_pairs)))]
        first_symbol = symbols[int(first[0])]
        second_symbol = symbols[int(second[0])]
        for index in first:
            symbols[int(index)] = second_symbol
        for index in second:
            symbols[int(index)] = first_symbol
        atoms.set_chemical_symbols(symbols)

    cartesian_displacements = displacements @ cell
    rms = float(np.sqrt(np.mean(np.sum(cartesian_displacements**2, axis=1))))
    return rms, swapped


def _fingerprint(atoms: Atoms) -> str:
    return atoms_hash(
        atoms,
        symprec=1e-2,
        frac_tol=2e-3,
        cell_tol=2e-3,
        mode="full",
    )


def mutate_candidate(
    parent: GeneratedCandidate,
    *,
    condition: GenerationConstraints,
    config: GenerationSettings,
    mutation_index: int,
    occupied_fingerprints: set[str],
) -> GeneratedCandidate | None:
    """Create one validated child without changing atom counts or elements.

    Exact requested space groups use cell-metric mutations only. Unconstrained
    or P1 populations may also perturb coordinates and swap unlike occupancies.
    Every child is independently revalidated and re-audited.
    """

    if parent.atoms is None or not parent.valid:
        return None
    payload = json.dumps(
        {
            "parent": parent.candidate_id,
            "index": int(mutation_index),
            "seed": 7 if config.seed is None else int(config.seed),
            "method": MUTATION_METHOD,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    exact_spacegroup = condition.spacegroup_number

    for attempt in range(1, int(config.mutation_attempts) + 1):
        atoms = parent.atoms.copy()
        cell_scales = _mutate_cell(
            atoms,
            rng=rng,
            strain=float(config.mutation_strain),
            spacegroup_number=exact_spacegroup,
        )
        displacement_rms = 0.0
        occupancy_swapped = False
        if config.preserve_spacegroup and exact_spacegroup not in {None, 1}:
            displacement_rms, occupancy_swapped = _mutate_symmetry_orbits(
                atoms,
                rng=rng,
                displacement=float(config.mutation_displacement),
                symprec=float(config.validation_kwargs()["symprec"]),
            )
        else:
            displacement_rms, occupancy_swapped = _mutate_coordinates_and_occupancy(
                atoms,
                rng=rng,
                displacement=float(config.mutation_displacement),
            )

        validation = validate_atoms_report(atoms, **config.validation_kwargs())
        assessments = evaluate_constraints(atoms, validation, condition)
        checks = {name: assessment.value for name, assessment in assessments.items()}
        if not validation.valid or any(value is False for value in checks.values()):
            continue
        fingerprint = _fingerprint(atoms)
        if fingerprint in occupied_fingerprints:
            continue
        occupied_fingerprints.add(fingerprint)

        return GeneratedCandidate(
            candidate_id=f"{parent.candidate_id}-mut-{mutation_index + 1:03d}",
            backend=parent.backend,
            model_name=parent.model_name,
            checkpoint_sha256=parent.checkpoint_sha256,
            atoms=atoms,
            validation=validation,
            condition=condition.as_dict(),
            condition_checks=checks,
            applied_constraints=list(parent.applied_constraints),
            unsupported_constraints=list(parent.unsupported_constraints),
            sampling=config.as_dict(),
            constraint_assessments=assessments,
            # Parent energies and property emissions belong to the parent
            # geometry and are intentionally invalidated by mutation.
            scientific_observables=[],
            backend_metrics={
                "population_role": "mutant",
                "mutation_method": MUTATION_METHOD,
                "parent_candidate_id": parent.candidate_id,
                "composition_preserved": True,
                "spacegroup_preservation_requested": bool(
                    config.preserve_spacegroup and exact_spacegroup is not None
                ),
                "crystal_system": _crystal_system(exact_spacegroup),
                "cell_scales": cell_scales,
                "coordinate_displacement_rms_angstrom": displacement_rms,
                "occupancy_swapped": occupancy_swapped,
                "symmetry_orbit_mutation": bool(
                    config.preserve_spacegroup and exact_spacegroup not in {None, 1}
                ),
                "mutation_attempt": attempt,
                "invalidated_parent_observables": [
                    observable.name for observable in parent.scientific_observables
                ],
            },
        )
    return None


def augment_population(
    parents: Iterable[GeneratedCandidate],
    *,
    target_size: int,
    mutant_count: int,
    condition: GenerationConstraints,
    config: GenerationSettings,
) -> list[GeneratedCandidate]:
    """Append validated mutants while keeping the requested population size."""

    population = list(parents)[: max(0, int(target_size) - int(mutant_count))]
    viable = [
        candidate
        for candidate in population
        if candidate.valid and candidate.condition_compliant is not False and candidate.atoms is not None
    ]
    if not viable or mutant_count <= 0:
        return population

    occupied = {
        _fingerprint(candidate.atoms)
        for candidate in viable
        if candidate.atoms is not None
    }
    for mutation_index in range(int(mutant_count)):
        parent = viable[mutation_index % len(viable)]
        child = mutate_candidate(
            parent,
            condition=condition,
            config=config,
            mutation_index=mutation_index,
            occupied_fingerprints=occupied,
        )
        if child is not None:
            population.append(child)
    return population


__all__ = [
    "MUTATION_METHOD",
    "augment_population",
    "mutate_candidate",
    "population_plan",
]
