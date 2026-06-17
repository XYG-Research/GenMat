from __future__ import annotations

from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io import read
from ase.data import covalent_radii
from ase.neighborlist import NeighborList


def _min_pair_distance(atoms: Atoms) -> float:
    pos = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return float("inf")

    # O(N^2) is fine for typical generated cell sizes.
    min_d = float("inf")
    for i in range(n):
        dp = pos[i + 1 :] - pos[i]
        dp -= np.round(dp)  # MIC in fractional coords
        dr = dp @ cell
        d = np.sqrt(np.sum(dr * dr, axis=1))
        if d.size:
            m = float(d.min())
            if m < min_d:
                min_d = m
    return float(min_d)


def _min_pair_distance_ratio(atoms: Atoms) -> float:
    """
    Return min over i<j of d(i,j)/(r_cov(i)+r_cov(j)) under MIC.

    This is a chemistry-aware sanity check that helps reject unphysical overlaps
    even when an absolute min_dist is too permissive for some element pairs.
    """
    pos = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    nums = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    n = pos.shape[0]
    if n < 2:
        return float("inf")

    # Preload covalent radii (Å). Unknown radii are treated as 0 and ignored via denom<=0.
    r = np.zeros(int(nums.max()) + 1, dtype=float)
    for z in np.unique(nums):
        z = int(z)
        if 0 <= z < len(covalent_radii):
            rv = float(covalent_radii[z])
            if np.isfinite(rv) and rv > 0:
                r[z] = rv

    min_ratio = float("inf")
    for i in range(n):
        dp = pos[i + 1 :] - pos[i]
        dp -= np.round(dp)  # MIC in fractional coords
        dr = dp @ cell
        d = np.sqrt(np.sum(dr * dr, axis=1))
        if not d.size:
            continue
        ri = float(r[int(nums[i])])
        if ri <= 0:
            continue
        rj = r[nums[i + 1 :]]
        denom = ri + rj
        ok = denom > 0
        if not np.any(ok):
            continue
        ratio = d[ok] / denom[ok]
        m = float(np.min(ratio))
        if m < min_ratio:
            min_ratio = m
    return float(min_ratio)


def _atoms_to_spglib_cell(atoms: Atoms):
    lattice = np.asarray(atoms.cell.array, dtype=float)
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    return (lattice, frac, numbers)


def _min_coordination_within_factor(atoms: Atoms, *, max_nn_factor: float) -> int:
    """
    Compute the minimum coordination number across atoms, where neighbors are defined by
    a chemistry-aware cutoff:

        d(i,j) <= max_nn_factor * (r_cov(i) + r_cov(j))

    This helps reject "dilute" / disconnected structures where some atoms have no bonded neighbors.

    Notes:
      - Uses ASE NeighborList with per-atom cutoffs c_i = max_nn_factor * r_cov(i),
        so pair cutoff is c_i + c_j.
      - covalent_radii are in Å.
    """
    n = int(len(atoms))
    if n <= 0:
        return 0

    nums = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    cutoffs = np.zeros(n, dtype=float)
    for i, z in enumerate(nums.tolist()):
        if 0 <= int(z) < len(covalent_radii):
            rv = float(covalent_radii[int(z)])
            if np.isfinite(rv) and rv > 0:
                cutoffs[i] = float(max_nn_factor) * rv

    # If radii are missing (shouldn't happen for real elements), fall back to a safe small cutoff.
    if not np.any(cutoffs > 0):
        cutoffs[:] = float(max_nn_factor) * 1.0

    nl = NeighborList(cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    min_cn = 10**9
    for i in range(n):
        idx, _offsets = nl.get_neighbors(i)
        cn = int(len(idx))
        if cn < min_cn:
            min_cn = cn
    return int(min_cn if min_cn != 10**9 else 0)


def _is_connected_within_factor(atoms: Atoms, *, max_nn_factor: float) -> bool:
    """
    Return True if the neighbor graph is connected, using the same chemistry-aware cutoff
    as `_min_coordination_within_factor`.

        d(i,j) <= max_nn_factor * (r_cov(i) + r_cov(j))
    """
    n = int(len(atoms))
    if n <= 1:
        return True

    nums = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    cutoffs = np.zeros(n, dtype=float)
    for i, z in enumerate(nums.tolist()):
        if 0 <= int(z) < len(covalent_radii):
            rv = float(covalent_radii[int(z)])
            if np.isfinite(rv) and rv > 0:
                cutoffs[i] = float(max_nn_factor) * rv

    if not np.any(cutoffs > 0):
        cutoffs[:] = float(max_nn_factor) * 1.0

    nl = NeighborList(cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        i = int(stack.pop())
        idx, _offsets = nl.get_neighbors(i)
        for j in idx.tolist():
            j = int(j)
            if not seen[j]:
                seen[j] = True
                stack.append(j)

    return bool(np.all(seen))


def validate_atoms(
    atoms: Atoms,
    *,
    min_dist: float,
    symprec: float,
    min_dist_factor: float | None = None,
    max_dist_factor: float | None = None,
    max_nn_factor: float | None = None,
    min_coordination: int | None = None,
    require_connected: bool | None = None,
    max_atoms: int | None = None,
    vol_per_atom_min: float | None = None,
    vol_per_atom_max: float | None = None,
) -> tuple[bool, str]:
    vol = float(atoms.get_volume())
    if not np.isfinite(vol) or vol <= 1e-6:
        return False, "invalid_volume"

    n_atoms = int(len(atoms))
    if n_atoms <= 0:
        return False, "empty_cell"

    if max_atoms is not None and n_atoms > int(max_atoms):
        return False, f"natoms>{int(max_atoms)}"

    vpa = float(vol) / float(n_atoms)
    if vol_per_atom_min is not None and vpa < float(vol_per_atom_min):
        return False, f"vol_per_atom<{float(vol_per_atom_min)}"
    if vol_per_atom_max is not None and vpa > float(vol_per_atom_max):
        return False, f"vol_per_atom>{float(vol_per_atom_max)}"

    md = _min_pair_distance(atoms)
    if not np.isfinite(md) or md < float(min_dist):
        return False, f"min_dist<{min_dist}"

    if min_dist_factor is not None or max_dist_factor is not None:
        mdr = _min_pair_distance_ratio(atoms)
        if not np.isfinite(mdr):
            return False, "min_dist_factor_nan"
        if min_dist_factor is not None and mdr < float(min_dist_factor):
            return False, f"min_dist_factor<{float(min_dist_factor)}"
        if max_dist_factor is not None and mdr > float(max_dist_factor):
            return False, f"min_dist_factor>{float(max_dist_factor)}"

    if max_nn_factor is not None and min_coordination is not None:
        try:
            fac = float(max_nn_factor)
            if not np.isfinite(fac) or fac <= 0:
                return False, "bad_max_nn_factor"
            min_cn = _min_coordination_within_factor(atoms, max_nn_factor=fac)
            if int(min_cn) < int(min_coordination):
                return False, f"min_coord<{int(min_coordination)}"
        except Exception:
            return False, "coordination_check_failed"

    if require_connected:
        if max_nn_factor is None:
            return False, "connected_requires_max_nn_factor"
        try:
            fac = float(max_nn_factor)
            if not np.isfinite(fac) or fac <= 0:
                return False, "bad_max_nn_factor"
            if not _is_connected_within_factor(atoms, max_nn_factor=fac):
                return False, "disconnected"
        except Exception:
            return False, "connectivity_check_failed"

    dataset = spglib.get_symmetry_dataset(_atoms_to_spglib_cell(atoms), symprec=symprec)
    if dataset is None:
        return False, "no_symmetry"

    num = int(getattr(dataset, "number") if hasattr(dataset, "number") else dataset["number"])
    if not (1 <= num <= 230):
        return False, "bad_spacegroup"

    return True, "ok"


def validate_cif_dir(cif_dir: Path, *, min_dist: float, symprec: float) -> bool:
    cif_dir = cif_dir.resolve()
    paths = sorted([p for p in cif_dir.glob("*.cif") if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No .cif files found under {cif_dir}")

    ok_n = 0
    fail = {}
    for p in paths:
        try:
            atoms = read(str(p))
            ok, reason = validate_atoms(atoms, min_dist=min_dist, symprec=symprec)
            if ok:
                ok_n += 1
            else:
                fail[reason] = fail.get(reason, 0) + 1
        except Exception:
            fail["read_error"] = fail.get("read_error", 0) + 1

    total = len(paths)
    print(f"Validated {total} CIFs: ok={ok_n}, fail={total-ok_n}")
    if fail:
        for k in sorted(fail):
            print(f"  - {k}: {fail[k]}")
    return ok_n == total
