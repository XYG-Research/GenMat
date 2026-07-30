from __future__ import annotations

import json
from pathlib import Path

from ase.build import bulk
from ase.spacegroup import crystal

from .structure_format import StructureRecord


def _row(source_id: str, atoms) -> dict:
    rec = StructureRecord.from_ase(atoms)
    return {
        "source": "example",
        "id": source_id,
        "formula": atoms.get_chemical_formula(),
        "nsites": len(atoms),
        "energy_above_hull": None,
        "structure": {
            "lattice": rec.lattice.tolist(),
            "frac_coords": rec.frac_coords.tolist(),
            "species": rec.species,
        },
    }


def make_examples_jsonl(out_path: Path) -> None:
    """
    Create a tiny dataset that exercises the full pipeline without MP access.
    These are *not* intended as a scientifically meaningful training set.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    # B2 / CsCl (Pm-3m, 221)
    examples.append(
        ("NiAl_B2", crystal(symbols=("Ni", "Al"), basis=[(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=221, cellpar=[2.88, 2.88, 2.88, 90, 90, 90]))
    )

    # L1_2 / Cu3Au (Pm-3m, 221)
    examples.append(
        (
            "Cu3Au_L12",
            crystal(
                symbols=("Au", "Cu", "Cu", "Cu"),
                basis=[(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
                spacegroup=221,
                cellpar=[3.75, 3.75, 3.75, 90, 90, 90],
            ),
        )
    )

    # L1_0 / CuAu (P4/mmm, 123) – tetragonal distortion of B2-like ordering
    examples.append(
        (
            "CuAu_L10",
            crystal(
                symbols=("Cu", "Au"),
                basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                spacegroup=123,
                cellpar=[3.90, 3.90, 3.60, 90, 90, 90],
            ),
        )
    )

    # B1 / NaCl-like ordering (Fm-3m, 225) – included as another symmetry pattern
    examples.append(
        (
            "TiC_like_dummy",
            crystal(
                symbols=("Ti", "Al"),
                basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                spacegroup=225,
                cellpar=[4.10, 4.10, 4.10, 90, 90, 90],
            ),
        )
    )

    # Chemically diverse non-intermetallic structures exercise the universal
    # vocabulary and prevent the offline smoke test from encoding a metallic-only assumption.
    examples.append(("NaCl_rocksalt", bulk("NaCl", "rocksalt", a=5.64)))
    examples.append(("MgO_rocksalt", bulk("MgO", "rocksalt", a=4.21)))
    examples.append(("Si_diamond", bulk("Si", "diamond", a=5.43)))
    examples.append(("GaAs_zincblende", bulk("GaAs", "zincblende", a=5.65)))

    # Simple hcp-based binary (P63/mmc, 194) with two species on 2c / 2a-like positions
    examples.append(
        (
            "MgZn_hcp_like",
            crystal(
                symbols=("Mg", "Zn"),
                basis=[(1 / 3, 2 / 3, 1 / 4), (0, 0, 0)],
                spacegroup=194,
                cellpar=[3.20, 3.20, 5.20, 90, 90, 120],
            ),
        )
    )

    with out_path.open("w", encoding="utf-8") as f:
        for name, atoms in examples:
            f.write(json.dumps(_row(name, atoms), ensure_ascii=False) + "\n")
