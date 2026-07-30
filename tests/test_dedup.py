from __future__ import annotations

import numpy as np
from ase import Atoms

from genim.dedup import atoms_hash


def test_atoms_hash_is_stable_under_small_perturbations() -> None:
    atoms1 = Atoms(
        symbols=["Fe", "Si"],
        cell=np.eye(3) * 3.0,
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    atoms2 = atoms1.copy()
    atoms2.cell[0, 0] += 0.004  # within cell_tol=0.01
    atoms2.set_scaled_positions([[0.0004, 0.0, 0.0], [0.5004, 0.5, 0.5]])  # within frac_tol=0.001

    h1 = atoms_hash(atoms1, symprec=1e-2, frac_tol=1e-3, cell_tol=1e-2)
    h2 = atoms_hash(atoms2, symprec=1e-2, frac_tol=1e-3, cell_tol=1e-2)
    assert h1 == h2
