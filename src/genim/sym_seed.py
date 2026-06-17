from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import cellpar_to_cell

from .structure_format import StructureRecord
from .symmetry import extract_wyckoff_structure
from .validate import validate_atoms


@dataclass(frozen=True)
class SymSeedConfig:
    n_sites: int = 3
    n_per_sg: int = 1
    max_atoms: int = 200
    min_dist: float = 1.5
    symprec: float = 1e-2
    seed: int = 7


def _representative_hall_by_sg() -> dict[int, int]:
    rep: dict[int, int] = {}
    for hall in range(1, 531):
        t = spglib.get_spacegroup_type(hall)
        if t is None:
            continue
        sg = int(getattr(t, "number") if hasattr(t, "number") else t["number"])
        rep.setdefault(sg, hall)
    return rep


def _crystal_system(spacegroup_number: int) -> str:
    sg = int(spacegroup_number)
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    if 195 <= sg <= 230:
        return "cubic"
    raise ValueError("spacegroup_number must be in 1..230")


def _random_cellpar(*, system: str, rng: random.Random) -> tuple[float, float, float, float, float, float]:
    # Use a conservative scale to make overlap rejection feasible even for high-multiplicity orbits.
    if system == "triclinic":
        a = rng.uniform(4.0, 9.0)
        b = rng.uniform(4.0, 9.0)
        c = rng.uniform(4.0, 9.0)
        alpha = rng.uniform(60.0, 120.0)
        beta = rng.uniform(60.0, 120.0)
        gamma = rng.uniform(60.0, 120.0)
        return a, b, c, alpha, beta, gamma

    if system == "monoclinic":
        a = rng.uniform(4.0, 10.0)
        b = rng.uniform(4.0, 10.0)
        c = rng.uniform(4.0, 10.0)
        beta = rng.uniform(95.0, 125.0)
        return a, b, c, 90.0, beta, 90.0

    if system == "orthorhombic":
        a = rng.uniform(4.0, 10.0)
        b = rng.uniform(4.0, 10.0)
        c = rng.uniform(4.0, 12.0)
        return a, b, c, 90.0, 90.0, 90.0

    if system == "tetragonal":
        a = rng.uniform(4.0, 10.0)
        c = rng.uniform(4.0, 14.0)
        return a, a, c, 90.0, 90.0, 90.0

    if system in ("trigonal", "hexagonal"):
        a = rng.uniform(4.0, 10.0)
        c = rng.uniform(4.0, 16.0)
        return a, a, c, 90.0, 90.0, 120.0

    if system == "cubic":
        a = rng.uniform(4.0, 12.0)
        return a, a, a, 90.0, 90.0, 90.0

    raise ValueError(f"Unknown crystal system: {system!r}")


def _unique_frac(fracs: np.ndarray, *, tol: float = 1e-5) -> np.ndarray:
    fracs = np.asarray(fracs, dtype=float) % 1.0
    q = np.round(fracs / float(tol)).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return fracs[np.sort(idx)]


def _orbit_from_rep(rep: np.ndarray, *, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    rep = np.asarray(rep, dtype=float)
    pts = []
    for r, t in zip(rotations, translations):
        pts.append((r @ rep + t) % 1.0)
    return _unique_frac(np.asarray(pts, dtype=float))


def _min_dist_between(fracs_a: np.ndarray, fracs_b: np.ndarray, cell: np.ndarray) -> float:
    fracs_a = np.asarray(fracs_a, dtype=float) % 1.0
    fracs_b = np.asarray(fracs_b, dtype=float) % 1.0
    cell = np.asarray(cell, dtype=float)
    if fracs_a.size == 0 or fracs_b.size == 0:
        return float("inf")

    md = float("inf")
    for p in fracs_a:
        dp = fracs_b - p[None, :]
        dp -= np.round(dp)
        dr = dp @ cell
        d = np.sqrt(np.sum(dr * dr, axis=1))
        m = float(d.min()) if d.size else float("inf")
        if m < md:
            md = m
    return float(md)


def _min_dist_within(fracs: np.ndarray, cell: np.ndarray) -> float:
    fracs = np.asarray(fracs, dtype=float) % 1.0
    cell = np.asarray(cell, dtype=float)
    n = int(fracs.shape[0])
    if n < 2:
        return float("inf")

    md = float("inf")
    for i in range(n):
        dp = fracs[i + 1 :] - fracs[i]
        dp -= np.round(dp)
        dr = dp @ cell
        d = np.sqrt(np.sum(dr * dr, axis=1))
        if d.size:
            m = float(d.min())
            if m < md:
                md = m
    return float(md)


def _build_one(
    *,
    spacegroup_number: int,
    hall_number: int,
    elements: list[str],
    cfg: SymSeedConfig,
    rng: random.Random,
) -> StructureRecord:
    system = _crystal_system(spacegroup_number)

    ops = spglib.get_symmetry_from_database(int(hall_number))
    if ops is None:
        raise ValueError(f"spglib database missing hall_number={hall_number}")
    rotations = np.asarray(ops["rotations"], dtype=int)
    translations = np.asarray(ops["translations"], dtype=float)

    for _attempt in range(200):
        a, b, c, alpha, beta, gamma = _random_cellpar(system=system, rng=rng)
        cell = np.asarray(cellpar_to_cell([a, b, c, alpha, beta, gamma]), dtype=float)

        fracs_all: list[np.ndarray] = []
        syms_all: list[str] = []

        ok = True
        for site_i in range(int(cfg.n_sites)):
            sym = elements[site_i % len(elements)]
            placed = False
            for _rep_try in range(500):
                rep = np.array([rng.random(), rng.random(), rng.random()], dtype=float)
                orbit = _orbit_from_rep(rep, rotations=rotations, translations=translations)
                if orbit.size == 0:
                    continue
                if int(len(fracs_all) + len(orbit)) > int(cfg.max_atoms):
                    continue
                if _min_dist_within(orbit, cell) < float(cfg.min_dist):
                    continue
                if fracs_all:
                    prev = np.asarray(fracs_all, dtype=float)
                    if _min_dist_between(orbit, prev, cell) < float(cfg.min_dist):
                        continue

                fracs_all.extend([p.copy() for p in orbit])
                syms_all.extend([sym] * int(len(orbit)))
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        atoms = Atoms(symbols=syms_all, cell=cell, scaled_positions=np.asarray(fracs_all, dtype=float), pbc=True)
        ok2, _reason = validate_atoms(atoms, min_dist=float(cfg.min_dist), symprec=float(cfg.symprec))
        if not ok2:
            continue

        try:
            wy = extract_wyckoff_structure(atoms, symprec=float(cfg.symprec))
        except Exception:
            continue
        if int(wy.spacegroup_number) != int(spacegroup_number):
            continue

        return StructureRecord.from_ase(atoms)

    raise RuntimeError(f"Failed to generate a valid seed structure for SG={spacegroup_number} (Hall={hall_number}).")


def write_symmetry_seeds_jsonl(
    *,
    out_path: Path,
    spacegroups: list[int],
    elements: list[str],
    cfg: SymSeedConfig,
) -> None:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rep_hall = _representative_hall_by_sg()
    rng = random.Random(int(cfg.seed))

    with out_path.open("w", encoding="utf-8") as f:
        for sg in spacegroups:
            sg = int(sg)
            if sg not in rep_hall:
                raise ValueError(f"No representative Hall number found for SG={sg}")
            hall = int(rep_hall[sg])

            for j in range(int(cfg.n_per_sg)):
                rec = _build_one(spacegroup_number=sg, hall_number=hall, elements=elements, cfg=cfg, rng=rng)
                row = {
                    "source": "sym_seed",
                    "id": f"sym_seed_sg{sg}_hall{hall}_{j}",
                    "formula": None,
                    "nsites": int(len(rec.species)),
                    "energy_above_hull": None,
                    "structure": {
                        "lattice": rec.lattice.tolist(),
                        "frac_coords": rec.frac_coords.tolist(),
                        "species": rec.species,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
