from __future__ import annotations

from collections import Counter
import math

from ase import Atoms

from genmat.synth import _RatioConstraint, _apply_supercell_plan, _plan_ratio_supercell, _ratio_constraint
from genmat.validate import validate_atoms


def _l12_parent() -> Atoms:
    return Atoms(
        symbols=["Al", "Ni", "Ni", "Ni"],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ],
        cell=[3.57, 3.57, 3.57],
        pbc=True,
    )


def test_ratio_supercell_realizes_l12_multicomponent_target() -> None:
    atoms = _l12_parent()

    planned = _plan_ratio_supercell(
        atoms,
        target=["Ni", "Co", "Fe", "Al"],
        weights=[1, 1, 1, 1],
        symprec=1e-2,
        max_atoms=200,
        max_total_copies=8,
    )

    assert planned is not None
    orbits, plan = planned
    assert math.prod(plan.repeats) == 3

    realized = _apply_supercell_plan(atoms, orbits=orbits, plan=plan)
    counts = Counter(realized.get_chemical_symbols())
    assert counts == Counter({"Ni": 3, "Co": 3, "Fe": 3, "Al": 3})

    ok, reason = validate_atoms(realized, min_dist=1.5, symprec=1e-2)
    assert ok, reason


def test_ratio_constraint_supports_x_wildcards_for_partial_ratio() -> None:
    constraint = _ratio_constraint(
        ratios=["1", "1", "X", "X"],
        ratio_mode=None,
        percent_tol=0.1,
        required_n=4,
        nelements_total=4,
    )

    assert isinstance(constraint, _RatioConstraint)
    assert constraint.mode == "ratio"
    assert constraint.specified_indices == (0, 1)
    assert constraint.specified_values == (1, 1)
    assert constraint.remainder is None


def test_ratio_constraint_supports_x_wildcards_for_partial_percent() -> None:
    constraint = _ratio_constraint(
        ratios=["25", "25", "X", "X"],
        ratio_mode="percent",
        percent_tol=0.1,
        required_n=4,
        nelements_total=4,
    )

    assert isinstance(constraint, _RatioConstraint)
    assert constraint.mode == "percent_tol"
    assert constraint.specified_indices == (0, 1)
    assert constraint.specified_values == (0.25, 0.25)
    assert constraint.remainder == (0.5, 0.1)
