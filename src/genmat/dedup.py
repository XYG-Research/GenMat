from __future__ import annotations

import hashlib

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import cell_to_cellpar


def _atoms_to_spglib_cell(atoms: Atoms):
    lattice = np.asarray(atoms.cell.array, dtype=float)
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    return (lattice, frac, numbers)


def _standardize_cell(cell, *, symprec: float):
    return spglib.standardize_cell(
        cell,
        to_primitive=False,
        no_idealize=False,
        symprec=float(symprec),
    )


def _quantize(arr: np.ndarray, *, tol: float) -> np.ndarray:
    if tol <= 0:
        raise ValueError("tol must be > 0")
    return np.round(np.asarray(arr, dtype=float) / float(tol)).astype(np.int64)


def atoms_hash(
    atoms: Atoms,
    *,
    symprec: float,
    frac_tol: float,
    cell_tol: float,
    mode: str = "full",
) -> str:
    """
    Compute a stable, approximate hash for de-duplication.

    Strategy:
    - spglib standardize_cell (conventional, no_idealize=False)
    - quantize cell parameters and fractional coordinates
    - sort atoms deterministically by (Z, x, y, z)
    """
    cell = _atoms_to_spglib_cell(atoms)
    std = _standardize_cell(cell, symprec=symprec)
    if std is None:
        lattice, frac, numbers = cell
    else:
        lattice, frac, numbers = std

    lattice = np.asarray(lattice, dtype=float)
    frac = np.asarray(frac, dtype=float) % 1.0
    numbers = np.asarray(numbers, dtype=np.int64)

    mode = str(mode or "full").strip().lower()
    if mode not in ("full", "prototype"):
        raise ValueError("mode must be one of: full, prototype")

    if mode == "prototype":
        q_cellpar = np.zeros(6, dtype=np.int64)
    else:
        cellpar = np.asarray(cell_to_cellpar(lattice), dtype=float)  # (6,)
        q_cellpar = _quantize(cellpar, tol=cell_tol)  # (6,)

    q_frac = _quantize(frac, tol=frac_tol)  # (N,3)

    # Canonicalize under global fractional translation (periodic wrap), on the quantized grid.
    # This stabilizes hashes across equivalent origin choices and tiny numeric perturbations.
    grid = int(round(1.0 / float(frac_tol)))
    if grid <= 0:
        raise ValueError("frac_tol must be > 0")
    q_frac = q_frac % grid

    best_numbers = None
    best_frac = None

    # Any canonical shift can be chosen to bring some atom onto the origin on the quantized grid.
    shifts = np.unique(q_frac, axis=0)
    for shift in shifts:
        shifted = (q_frac - shift) % grid
        order = np.lexsort((shifted[:, 2], shifted[:, 1], shifted[:, 0], numbers))
        n_ord = numbers[order]
        f_ord = shifted[order]

        if best_numbers is None:
            best_numbers = n_ord
            best_frac = f_ord
            continue

        # Compare lexicographically (numbers then coords) for a deterministic minimum.
        cur = np.concatenate([n_ord.reshape(-1), f_ord.reshape(-1)])
        best = np.concatenate([best_numbers.reshape(-1), best_frac.reshape(-1)])
        if tuple(cur.tolist()) < tuple(best.tolist()):
            best_numbers = n_ord
            best_frac = f_ord

    assert best_numbers is not None and best_frac is not None
    numbers = best_numbers
    q_frac = best_frac

    # Serialize as bytes (avoid Python repr nondeterminism across versions).
    payload = np.concatenate(
        [
            q_cellpar.reshape(-1),
            np.asarray([len(numbers)], dtype=np.int64),
            numbers.reshape(-1),
            q_frac.reshape(-1),
        ]
    ).astype(np.int64)

    h = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
    return str(h)
