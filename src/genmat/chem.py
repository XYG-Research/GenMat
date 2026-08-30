from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ase.data import chemical_symbols

# These sets are used only by the explicit ``metallic`` and ``intermetallic``
# policies.  The default ``any`` policy accepts every real IUPAC element.
NON_METALS: set[str] = {
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
    "He",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
    "Rn",
    "Og",
    "At",
    "Ts",
}

METALLOIDS: set[str] = {"B", "Si", "Ge", "As", "Sb", "Te", "Po"}
VALID_ELEMENTS: tuple[str, ...] = tuple(str(x) for x in chemical_symbols[1:] if str(x))
VALID_ELEMENT_SET: frozenset[str] = frozenset(VALID_ELEMENTS)
CHEMISTRY_MODES: tuple[str, ...] = ("any", "metallic", "intermetallic")


def _normalise_symbols(values: Iterable[str] | None, *, field: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        symbol = str(value).strip()
        if symbol not in VALID_ELEMENT_SET:
            raise ValueError(f"Unknown chemical element in {field}: {symbol!r}")
        if symbol not in out:
            out.append(symbol)
    return tuple(out)


@dataclass(frozen=True)
class ChemistryPolicy:
    """Explicit, reproducible composition-domain constraints.

    ``any`` accepts every real element and is the general-purpose default.
    ``metallic`` accepts one or more metal/metalloid species.
    ``intermetallic`` preserves GenIM's historical rule: at least two distinct
    metal/metalloid species and no non-metals.

    ``allowed_elements`` and ``excluded_elements`` provide an exact user-defined
    domain without pretending that broad labels such as "inorganic" have a
    universally unambiguous algorithmic definition.
    """

    mode: str = "any"
    include_metalloids: bool = True
    allowed_elements: tuple[str, ...] | list[str] | set[str] | None = None
    excluded_elements: tuple[str, ...] | list[str] | set[str] | None = None
    min_elements: int | None = None
    max_elements: int | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode or "any").strip().lower()
        if mode not in CHEMISTRY_MODES:
            raise ValueError(f"Unknown chemistry mode {mode!r}; expected one of {', '.join(CHEMISTRY_MODES)}")
        allowed = _normalise_symbols(self.allowed_elements, field="allowed_elements")
        excluded = _normalise_symbols(self.excluded_elements, field="excluded_elements") or ()
        if allowed is not None and set(allowed).intersection(excluded):
            overlap = sorted(set(allowed).intersection(excluded))
            raise ValueError(f"Elements cannot be both allowed and excluded: {overlap}")

        mode_minimum = 2 if mode == "intermetallic" else 1
        min_elements = mode_minimum if self.min_elements is None else int(self.min_elements)
        if min_elements < mode_minimum:
            raise ValueError(f"chemistry mode {mode!r} requires min_elements >= {mode_minimum}")
        if self.max_elements is not None and int(self.max_elements) < int(min_elements):
            raise ValueError("max_elements must be >= min_elements")
        if allowed is not None and len(allowed) < int(min_elements):
            raise ValueError("allowed_elements contains fewer species than min_elements")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "allowed_elements", allowed)
        object.__setattr__(self, "excluded_elements", tuple(excluded))
        object.__setattr__(self, "min_elements", int(min_elements))
        if self.max_elements is not None:
            object.__setattr__(self, "max_elements", int(self.max_elements))

    def is_allowed_element(self, symbol: str) -> bool:
        symbol = str(symbol).strip()
        if symbol not in VALID_ELEMENT_SET:
            return False
        if self.allowed_elements is not None and symbol not in self.allowed_elements:
            return False
        if symbol in self.excluded_elements:
            return False
        if self.mode in {"metallic", "intermetallic"}:
            if symbol in NON_METALS:
                return False
            if not self.include_metalloids and symbol in METALLOIDS:
                return False
        return True

    def available_elements(self) -> list[str]:
        return [symbol for symbol in VALID_ELEMENTS if self.is_allowed_element(symbol)]

    def accepts_elements(self, elements: Iterable[str]) -> bool:
        elems = set(str(x).strip() for x in elements)
        if any(not value for value in elems):
            return False
        if len(elems) < int(self.min_elements):
            return False
        if self.max_elements is not None and len(elems) > int(self.max_elements):
            return False
        return all(self.is_allowed_element(symbol) for symbol in elems)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "include_metalloids": self.include_metalloids,
            "allowed_elements": list(self.allowed_elements) if self.allowed_elements is not None else None,
            "excluded_elements": list(self.excluded_elements),
            "min_elements": self.min_elements,
            "max_elements": self.max_elements,
        }


def available_elements(
    *,
    mode: str = "any",
    include_metalloids: bool = True,
    allowed_elements: Iterable[str] | None = None,
    excluded_elements: Iterable[str] | None = None,
) -> list[str]:
    return ChemistryPolicy(
        mode=mode,
        include_metalloids=include_metalloids,
        allowed_elements=None if allowed_elements is None else tuple(allowed_elements),
        excluded_elements=None if excluded_elements is None else tuple(excluded_elements),
    ).available_elements()


def allowed_intermetallic_elements(*, include_metalloids: bool) -> list[str]:
    """Compatibility helper for checkpoints and callers created before GenIM 0.2."""

    return available_elements(mode="intermetallic", include_metalloids=include_metalloids)


@dataclass(frozen=True)
class IntermetallicFilter:
    """Backward-compatible facade over :class:`ChemistryPolicy`."""

    include_metalloids: bool = False

    @property
    def policy(self) -> ChemistryPolicy:
        return ChemistryPolicy(mode="intermetallic", include_metalloids=self.include_metalloids)

    def is_allowed_element(self, symbol: str) -> bool:
        return self.policy.is_allowed_element(symbol)

    def is_intermetallic(self, elements: list[str] | set[str]) -> bool:
        return self.policy.accepts_elements(elements)


__all__ = [
    "CHEMISTRY_MODES",
    "METALLOIDS",
    "NON_METALS",
    "VALID_ELEMENTS",
    "ChemistryPolicy",
    "IntermetallicFilter",
    "allowed_intermetallic_elements",
    "available_elements",
]
