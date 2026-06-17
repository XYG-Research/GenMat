from __future__ import annotations

import math

import numpy as np
from ase import Atoms

from .validate import _min_pair_distance, _min_pair_distance_ratio


def autoscale_cell_isotropic(
    atoms: Atoms,
    *,
    min_dist: float | None = None,
    min_dist_factor: float | None = None,
    max_dist_factor: float | None = None,
    vol_per_atom_min: float | None = None,
    vol_per_atom_max: float | None = None,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> tuple[Atoms, float]:
    """
    Isotropically scale the cell (and atom positions) to bring simple geometry
    metrics into a desired range.

    Notes:
      - Distances scale ~ linearly with factor f.
      - Volume-per-atom scales ~ f^3.
      - Fractional coordinates are preserved.
    """
    n = int(len(atoms))
    if n <= 0:
        return atoms, 1.0

    vol = float(atoms.get_volume())
    if not math.isfinite(vol) or vol <= 1e-12:
        return atoms, 1.0

    vpa = vol / float(n)

    need_ratio = (min_dist_factor is not None) or (max_dist_factor is not None)
    md = float(_min_pair_distance(atoms))
    mdr = float(_min_pair_distance_ratio(atoms)) if need_ratio else float("nan")

    f_low = 1.0
    f_high = float("inf")

    if min_dist is not None and math.isfinite(md) and md > 0:
        f_low = max(f_low, float(min_dist) / md)

    if need_ratio and math.isfinite(mdr) and mdr > 0:
        if min_dist_factor is not None:
            f_low = max(f_low, float(min_dist_factor) / mdr)
        if max_dist_factor is not None:
            f_high = min(f_high, float(max_dist_factor) / mdr)

    if vol_per_atom_min is not None and math.isfinite(vpa) and vpa > 0:
        f_low = max(f_low, float(float(vol_per_atom_min) / vpa) ** (1.0 / 3.0))
    if vol_per_atom_max is not None and math.isfinite(vpa) and vpa > 0:
        f_high = min(f_high, float(float(vol_per_atom_max) / vpa) ** (1.0 / 3.0))

    # Apply scaling limits.
    f_low = max(f_low, float(scale_min))
    f_high = min(f_high, float(scale_max))

    if not math.isfinite(f_low) or not math.isfinite(f_high):
        return atoms, 1.0
    if f_low > f_high:
        return atoms, 1.0

    desired = 1.0
    if desired < f_low:
        desired = f_low
    elif desired > f_high:
        desired = f_high

    if abs(desired - 1.0) < 1e-6:
        return atoms, 1.0

    a2 = atoms.copy()
    a2.set_cell(a2.cell.array * float(desired), scale_atoms=True)

    # Always wrap for stable downstream symmetry checks and neat CIF output.
    sp = np.asarray(a2.get_scaled_positions(wrap=True), dtype=float)
    a2.set_scaled_positions(sp)
    return a2, float(desired)

