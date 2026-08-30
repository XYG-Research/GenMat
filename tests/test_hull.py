from __future__ import annotations

import math

import pytest

from genmat.hull import compute_e_above_hull_eV_per_atom


def test_compute_e_above_hull_binary_simple() -> None:
    pytest.importorskip("pymatgen")

    refs = [
        {"composition": {"A": 1}, "energy_per_atom_eV": 0.0},
        {"composition": {"B": 1}, "energy_per_atom_eV": 0.0},
        {"composition": {"A": 1, "B": 1}, "energy_per_atom_eV": -1.0},
    ]

    cand = {"composition": {"A": 1, "B": 1}, "energy_per_atom_eV": -0.8}
    eah = compute_e_above_hull_eV_per_atom(candidate=cand, references=refs)
    assert eah is not None
    assert math.isclose(float(eah), 0.2, rel_tol=0, abs_tol=1e-6)
