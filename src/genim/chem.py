from __future__ import annotations

from dataclasses import dataclass

from ase.data import chemical_symbols

NON_METALS: set[str] = {
    # Reactive / common non-metals
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "I",
    # Noble gases
    "He",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
    "Rn",
    "Og",
    # Other non-metals / halogens (including superheavy where relevant)
    "At",
    "Ts",
}

METALLOIDS: set[str] = {"B", "Si", "Ge", "As", "Sb", "Te", "Po"}


def allowed_intermetallic_elements(*, include_metalloids: bool) -> list[str]:
    out: list[str] = []
    for z in range(1, len(chemical_symbols)):
        sym = chemical_symbols[z]
        if sym in NON_METALS:
            continue
        if (not include_metalloids) and (sym in METALLOIDS):
            continue
        out.append(sym)
    return out


@dataclass(frozen=True)
class IntermetallicFilter:
    include_metalloids: bool = False

    def is_allowed_element(self, symbol: str) -> bool:
        if symbol in NON_METALS:
            return False
        if (not self.include_metalloids) and (symbol in METALLOIDS):
            return False
        return True

    def is_intermetallic(self, elements: list[str] | set[str]) -> bool:
        elems = set(elements)
        if len(elems) < 2:
            return False
        return all(self.is_allowed_element(e) for e in elems)
