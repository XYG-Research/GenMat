from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import reduce
from typing import Any, Mapping

from ase import Atoms
from ase.formula import Formula
from ase.data import atomic_numbers


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


def parse_formula_counts(formula: str) -> dict[str, int]:
    """Parse a chemical formula into reduced positive integer counts.

    ASE's formula parser supplies support for grouped formulae such as
    ``Ca3(PO4)2``.  Space-separated element lists (``"Fe Ni"``) are accepted
    as a convenience, while ratio notation is handled by
    :func:`ratios_to_integer_counts` so that decimal intent remains explicit.
    """

    text = str(formula).strip()
    if not text:
        raise ValueError("formula must not be empty")
    if ":" in text or "=" in text:
        raise ValueError("ratio notation must be supplied as elements and stoichiometry")
    compact = "".join(text.replace(",", " ").split())
    try:
        counts = Formula(compact).count()
    except Exception as exc:
        raise ValueError(f"Invalid chemical formula: {formula!r}") from exc
    if not counts:
        raise ValueError("formula contains no elements")
    unknown = sorted(symbol for symbol in counts if symbol not in atomic_numbers)
    if unknown:
        raise ValueError(f"Unknown chemical element: {unknown[0]!r}")
    return reduced_counts(counts)


def ratios_to_integer_counts(
    elements: list[str] | tuple[str, ...],
    ratios: list[float] | tuple[float, ...],
    *,
    max_denominator: int = 1000,
) -> dict[str, int]:
    """Convert positive composition ratios into their smallest integer cell."""

    if not elements or len(elements) != len(ratios):
        raise ValueError("elements and stoichiometry must have the same non-zero length")
    normalised: list[str] = []
    fractions: list[Fraction] = []
    for symbol, value in zip(elements, ratios):
        element = str(symbol).strip()
        if element not in atomic_numbers:
            raise ValueError(f"Unknown chemical element: {element!r}")
        if element in normalised:
            raise ValueError(f"Duplicate chemical element: {element!r}")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("stoichiometry values must be finite and > 0")
        normalised.append(element)
        fractions.append(Fraction(str(number)).limit_denominator(int(max_denominator)))

    def _lcm(left: int, right: int) -> int:
        return abs(left * right) // math.gcd(left, right)

    denominator = reduce(_lcm, (fraction.denominator for fraction in fractions), 1)
    integers = [fraction.numerator * (denominator // fraction.denominator) for fraction in fractions]
    divisor = reduce(math.gcd, integers)
    return {element: int(value // divisor) for element, value in zip(normalised, integers)}


__all__ = [
    "atoms_counts",
    "composition_key",
    "parse_formula_counts",
    "ratios_to_integer_counts",
    "reduced_counts",
    "reduced_formula",
]
