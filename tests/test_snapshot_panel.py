from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write
from PIL import Image

from genmat.snapshot_panel import _guess_bonds, _label_display_text, _label_segments, _periodic_bonds, _render_structure_square, _select_snapshot_items, render_snapshot_panel


def _write_demo_cif(path: Path, symbols: list[str]) -> None:
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ][: len(symbols)],
        cell=[4.2, 4.2, 4.2],
        pbc=True,
    )
    write(str(path), atoms, format="cif")


def test_select_snapshot_items_uses_id_range_and_actual_counts(tmp_path: Path) -> None:
    cif_dir = tmp_path / "Fe-Co-Ni-Al_4el"
    cif_dir.mkdir()
    _write_demo_cif(cif_dir / "genmat_00009.cif", ["Fe", "Co", "Ni", "Al"])
    _write_demo_cif(cif_dir / "genim_00015.cif", ["Fe", "Fe", "Co", "Al"])

    items = _select_snapshot_items(cif_dir=cif_dir, n=5, seed=7, id_start=9, id_end=10)

    assert len(items) == 1
    assert items[0].sid == 9
    assert items[0].label == "009-Fe1Co1Ni1Al1"


def test_render_snapshot_panel_smoke(tmp_path: Path) -> None:
    cif_dir = tmp_path / "Fe-Co-Ni-Al_4el"
    cif_dir.mkdir()
    _write_demo_cif(cif_dir / "genmat_00009.cif", ["Fe", "Co", "Ni", "Al"])
    _write_demo_cif(cif_dir / "genim_00010.cif", ["Fe", "Fe", "Co", "Al"])

    out_path = render_snapshot_panel(
        cif_dir=cif_dir,
        out_path=tmp_path / "panel.png",
        n=2,
        seed=7,
        cols=6,
        size_px=120,
        label_h_px=28,
        pad_px=10,
        margin_px=12,
        supersample=1,
    )

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_guess_bonds_detects_simple_pair() -> None:
    atoms = Atoms(symbols=["Fe", "Ni"], positions=[(0.0, 0.0, -1.1), (0.0, 0.0, 1.1)])

    bonds = _guess_bonds(atoms.get_positions(), atoms.get_atomic_numbers())

    assert bonds == [(0, 1)]


def test_periodic_bonds_add_nearest_neighbor_shell_and_cross_boundary() -> None:
    atoms = Atoms(
        symbols=["Fe", "Ni"],
        positions=[(2.0, 2.0, 2.0), (2.0, 2.0, 5.6)],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )

    bonds = _periodic_bonds(atoms)

    assert len(bonds) >= 1
    assert any({int(i), int(j)} == {0, 1} for i, j, _shift, _dist in bonds)


def test_label_segments_keep_id_normal_and_counts_as_subscripts() -> None:
    segs = _label_segments("009-Fe12Co5Ni3Al10")

    assert segs == [
        ("009-", False),
        ("Fe", False),
        ("12", True),
        ("Co", False),
        ("5", True),
        ("Ni", False),
        ("3", True),
        ("Al", False),
        ("10", True),
    ]


def test_label_display_text_uses_unicode_subscripts_for_counts_only() -> None:
    assert _label_display_text("009-Fe12Co5Ni3Al10") == "009-Fe₁₂Co₅Ni₃Al₁₀"


def test_render_structure_square_draws_atom_body_without_cell() -> None:
    atoms = Atoms(symbols=["O"], positions=[(0.0, 0.0, 0.0)])

    img = _render_structure_square(
        atoms,
        size_px=120,
        supersample=1,
        atom_scale=0.78,
        cell_line_w_px=3,
    )

    assert isinstance(img, Image.Image)
    reds = 0
    for px in img.convert("RGB").getdata():
        r, g, b = (int(v) for v in px)
        if r > g + 20 and r > b + 20:
            reds += 1
    assert reds > 150
