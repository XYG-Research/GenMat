from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ase import Atoms


def atoms_counts(atoms: Atoms) -> dict[str, int]:
    c = Counter(atoms.get_chemical_symbols())
    return {str(k): int(v) for k, v in c.items() if int(v) > 0}


def reduced_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    vals = []
    out: dict[str, int] = {}
    for k, v in counts.items():
        try:
            n = int(v)
        except Exception:
            continue
        if n <= 0:
            continue
        out[str(k)] = n
        vals.append(n)
    if not out:
        return {}
    g = 0
    for n in vals:
        g = math.gcd(g, abs(int(n)))
    if g <= 1:
        return dict(out)
    return {k: int(v // g) for k, v in out.items()}


def reduced_formula(counts: Mapping[str, Any]) -> str:
    """
    Return a human-friendly reduced formula.

    Uses pymatgen if available; otherwise falls back to alphabetical ordering.
    """
    red = reduced_counts(counts)
    if not red:
        return ""

    try:
        from pymatgen.core.composition import Composition  # type: ignore

        return Composition({k: float(v) for k, v in red.items()}).reduced_formula
    except Exception:
        parts = []
        for el in sorted(red):
            n = int(red[el])
            parts.append(f"{el}{'' if n == 1 else n}")
        return "".join(parts)


def composition_key(counts: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    red = reduced_counts(counts)
    return tuple(sorted(((str(k), int(v)) for k, v in red.items() if int(v) > 0)))

