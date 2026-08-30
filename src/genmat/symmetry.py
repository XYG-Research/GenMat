from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms
from ase.data import chemical_symbols
from ase.geometry import cell_to_cellpar


@dataclass(frozen=True)
class WyckoffSite:
    element: str
    wyckoff: str
    frac: np.ndarray  # (3,)


@dataclass(frozen=True)
class WyckoffStructure:
    hall_number: int
    spacegroup_number: int
    cellpar: np.ndarray  # (6,) a,b,c,alpha,beta,gamma
    sites: list[WyckoffSite]


def _atoms_to_spglib_cell(atoms: Atoms):
    lattice = np.asarray(atoms.cell.array, dtype=float)
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    return (lattice, frac, numbers)


def _standardize_cell(cell, *, symprec: float):
    std = spglib.standardize_cell(
        cell,
        to_primitive=False,
        no_idealize=False,
        symprec=symprec,
    )
    return std


def extract_wyckoff_structure(atoms: Atoms, *, symprec: float) -> WyckoffStructure:
    """
    Standardize the input cell (conventional) and extract Wyckoff sites via spglib.
    """
    cell = _atoms_to_spglib_cell(atoms)
    std = _standardize_cell(cell, symprec=symprec)
    if std is None:
        raise ValueError("spglib.standardize_cell failed")

    dataset = spglib.get_symmetry_dataset(std, symprec=symprec)
    if dataset is None:
        raise ValueError("spglib.get_symmetry_dataset failed")

    def _ds(ds, name: str):
        return getattr(ds, name) if hasattr(ds, name) else ds[name]

    hall_number = int(_ds(dataset, "hall_number"))
    spg_number = int(_ds(dataset, "number"))
    wyckoffs = list(_ds(dataset, "wyckoffs"))
    eq = np.asarray(_ds(dataset, "equivalent_atoms"), dtype=int)

    lattice, frac, numbers = std
    frac = np.asarray(frac, dtype=float) % 1.0
    numbers = np.asarray(numbers, dtype=int)

    groups: dict[int, list[int]] = {}
    for i, gid in enumerate(eq.tolist()):
        groups.setdefault(gid, []).append(i)

    sites: list[WyckoffSite] = []
    for gid, idxs in groups.items():
        rep = min(idxs)
        elems = {int(numbers[i]) for i in idxs}
        if len(elems) != 1:
            raise ValueError("Mixed-species equivalent_atoms group; likely disordered structure")
        z = int(numbers[rep])
        element = chemical_symbols[z]
        wy = str(wyckoffs[rep]) if wyckoffs[rep] is not None else "?"
        sites.append(WyckoffSite(element=element, wyckoff=wy, frac=frac[rep].copy()))

    # Deterministic order helps training stability.
    sites.sort(key=lambda s: (s.wyckoff, s.element, float(s.frac[0]), float(s.frac[1]), float(s.frac[2])))

    cellpar = np.asarray(cell_to_cellpar(np.asarray(lattice, dtype=float)), dtype=float)
    return WyckoffStructure(
        hall_number=hall_number,
        spacegroup_number=spg_number,
        cellpar=cellpar,
        sites=sites,
    )
