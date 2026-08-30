from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

from genim.ovito_tachyon_panel import _display_names, _expand_atoms_for_visualization, discover_structure_files


def _write_demo_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    atoms = Atoms(
        symbols=["Fe", "Ni"],
        scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    cif_path = tmp_path / "demo.cif"
    poscar_path = tmp_path / "POSCAR"
    contcar_path = tmp_path / "nested" / "CONTCAR"
    contcar_path.parent.mkdir()
    write(str(cif_path), atoms, format="cif")
    write(str(poscar_path), atoms, format="vasp", direct=True, vasp5=True)
    write(str(contcar_path), atoms, format="vasp", direct=True, vasp5=True)
    return cif_path, poscar_path, contcar_path


def test_discover_structure_files_supports_cif_and_vasp_names(tmp_path: Path) -> None:
    cif_path, poscar_path, contcar_path = _write_demo_files(tmp_path)
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    direct = discover_structure_files(tmp_path, recursive=False)
    recursive = discover_structure_files(tmp_path, recursive=True)

    assert direct == [cif_path, poscar_path]
    assert recursive == [cif_path, contcar_path, poscar_path]


def test_display_names_use_parent_directory_for_generic_vasp_names(tmp_path: Path) -> None:
    _cif_path, poscar_path, contcar_path = _write_demo_files(tmp_path)

    names = _display_names([poscar_path, contcar_path], tmp_path)

    assert names[poscar_path] == tmp_path.name
    assert names[contcar_path] == "nested"


def test_expand_atoms_for_visualization_adds_periodic_neighbors(tmp_path: Path) -> None:
    atoms = Atoms(
        symbols=["Cu"],
        scaled_positions=[(0.0, 0.0, 0.0)],
        cell=[3.6, 3.6, 3.6],
        pbc=True,
    )

    expanded = _expand_atoms_for_visualization(atoms, radius_scale=1.0)

    assert len(expanded) > len(atoms)
