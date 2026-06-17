from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


@dataclass(frozen=True)
class StructureRecord:
    lattice: np.ndarray  # (3,3) in Angstrom
    frac_coords: np.ndarray  # (N,3) in [0,1)
    species: list[str]  # length N (element symbols)

    def to_ase(self) -> Atoms:
        return Atoms(
            symbols=self.species,
            cell=self.lattice,
            scaled_positions=self.frac_coords,
            pbc=True,
        )

    @staticmethod
    def from_ase(atoms: Atoms) -> "StructureRecord":
        lattice = np.asarray(atoms.cell.array, dtype=float)
        frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
        species = list(atoms.get_chemical_symbols())
        return StructureRecord(lattice=lattice, frac_coords=frac, species=species)


def parse_pymatgen_mson_structure(obj: dict) -> StructureRecord:
    """
    Parse a pymatgen MSONable Structure dict (as returned by MP API) into a minimal record.
    We avoid depending on pymatgen at runtime.
    """
    lattice = obj.get("lattice", {})
    matrix = lattice.get("matrix", None)
    if matrix is None:
        raise ValueError("Missing lattice.matrix in structure")

    sites = obj.get("sites", None)
    if not isinstance(sites, list) or not sites:
        raise ValueError("Missing sites in structure")

    frac_coords: list[list[float]] = []
    species: list[str] = []

    for s in sites:
        abc = s.get("abc", None)
        if abc is None:
            raise ValueError("Site missing abc fractional coords")
        sp = s.get("species", None)
        if not isinstance(sp, list) or not sp:
            raise ValueError("Site missing species list")

        # Choose the highest-occupancy element if needed.
        best = max(sp, key=lambda x: float(x.get("occu", 0.0)))
        el = best.get("element", None) or best.get("name", None)
        if not isinstance(el, str) or not el:
            raise ValueError("Invalid species element")
        if el not in atomic_numbers:
            raise ValueError(f"Unknown element symbol: {el}")

        frac_coords.append([float(abc[0]), float(abc[1]), float(abc[2])])
        species.append(el)

    lat = np.asarray(matrix, dtype=float)
    frac = np.asarray(frac_coords, dtype=float) % 1.0
    return StructureRecord(lattice=lat, frac_coords=frac, species=species)

