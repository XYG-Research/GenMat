from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read, write
from PIL import Image, ImageDraw, ImageFont

from .snapshot_panel import (
    _LABEL_RGBA,
    _PAPER_BG_RGBA,
    _build_panel,
    _cell_corners,
    _font_arial_bold,
    _inside_box,
    _nature_rgb,
    _safe_filename,
    _tighten_structure_image,
)


# OVITO installation directory. Override with the GENIM_OVITO_ROOT environment
# variable or the --ovito-root CLI flag. Defaults to "ovito" on PATH/cwd so the
# package ships without any machine-specific absolute path.
DEFAULT_OVITO_ROOT = Path(os.environ.get("GENIM_OVITO_ROOT", "ovito"))
SUPPORTED_EXACT_NAMES = {"poscar", "contcar"}
SUPPORTED_SUFFIXES = {".cif"}


@dataclass(frozen=True)
class StructureEntry:
    path: Path
    display_name: str
    formula: str
    staged_path: Path
    raw_png_path: Path
    tile_name: str
    type_styles: list[dict[str, object]]
    cell_line_width: float


def _read_structure(path: Path) -> Atoms:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return read(str(path), format="cif")
    if name in SUPPORTED_EXACT_NAMES:
        return read(str(path), format="vasp")
    return read(str(path))


def discover_structure_files(input_dir: Path, *, recursive: bool = False) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    matched: dict[str, Path] = {}
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name not in SUPPORTED_EXACT_NAMES and suffix not in SUPPORTED_SUFFIXES:
            continue
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        matched.setdefault(key, path)

    files = list(matched.values())
    files.sort(key=lambda p: str(p.relative_to(input_dir) if p.is_relative_to(input_dir) else p).lower())
    return files


def _unique_symbol_order(atoms: Atoms) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sym in atoms.get_chemical_symbols():
        sym0 = str(sym)
        if sym0 not in seen:
            seen.add(sym0)
            out.append(sym0)
    return out


def _ovito_radius(symbol: str, *, radius_scale: float) -> float:
    z = int(atomic_numbers.get(str(symbol), 0))
    if 0 <= z < len(covalent_radii) and math.isfinite(float(covalent_radii[z])):
        base = float(covalent_radii[z])
    else:
        base = 1.2
    return float(min(1.85, max(0.58, float(radius_scale) * base)))


def _cell_line_width(atoms: Atoms) -> float:
    cell = np.asarray(atoms.cell.array, dtype=float).reshape(3, 3)
    lengths = [float(np.linalg.norm(v)) for v in cell]
    lengths = [v for v in lengths if math.isfinite(v) and v > 1e-8]
    if not lengths:
        return 0.22
    mean_len = float(sum(lengths) / len(lengths))
    return float(min(0.42, max(0.18, 0.035 * mean_len)))


def _expand_atoms_for_visualization(atoms: Atoms, *, radius_scale: float) -> Atoms:
    expanded = atoms.copy()
    if len(expanded) == 0:
        return expanded
    try:
        expanded.wrap()
    except Exception:
        pass

    cell = np.asarray(expanded.cell.array, dtype=float).reshape(3, 3)
    has_cell = bool(np.isfinite(cell).all()) and abs(float(np.linalg.det(cell))) > 1e-8
    if not has_cell:
        return expanded

    positions = np.asarray(expanded.get_positions(), dtype=float)
    center = 0.5 * (cell[0] + cell[1] + cell[2])
    pos0 = positions - center[None, :]
    corners0 = _cell_corners(cell) - center[None, :]
    box_mins = np.min(corners0, axis=0)
    box_maxs = np.max(corners0, axis=0)

    pbc = np.asarray(getattr(expanded, "pbc", [False, False, False]), dtype=bool).reshape(3)
    ranges = [(-1, 0, 1) if bool(pbc[k]) and bool(np.any(np.abs(cell[k]) > 1e-8)) else (0,) for k in range(3)]
    shifts = [tuple(int(v) for v in vals) for vals in np.array(np.meshgrid(*ranges, indexing="ij")).T.reshape(-1, 3)]

    out_symbols: list[str] = []
    out_positions: list[np.ndarray] = []
    for idx, sym in enumerate(expanded.get_chemical_symbols()):
        radius = _ovito_radius(str(sym), radius_scale=radius_scale)
        margin = 0.92 * float(radius) + 0.06
        p0 = np.asarray(pos0[int(idx)], dtype=float)
        for shift in shifts:
            shift_vec = np.asarray(shift, dtype=float) @ cell
            p = p0 + shift_vec
            if _inside_box(p, box_mins, box_maxs, margin):
                out_symbols.append(str(sym))
                out_positions.append(p + center)

    if not out_positions:
        return expanded

    return Atoms(symbols=out_symbols, positions=np.asarray(out_positions, dtype=float), cell=expanded.cell, pbc=expanded.pbc)


def _display_names(paths: Sequence[Path], input_dir: Path) -> dict[Path, str]:
    input_dir = input_dir.expanduser().resolve()
    provisional: dict[Path, str] = {}
    counts: dict[str, int] = {}
    for path in paths:
        if path.name.lower() in SUPPORTED_EXACT_NAMES:
            name = path.parent.name.strip() or path.name
        elif path.suffix:
            name = path.stem
        else:
            name = path.name
        name = str(name).strip() or path.name
        provisional[path] = name
        counts[name] = counts.get(name, 0) + 1

    resolved: dict[Path, str] = {}
    for path in paths:
        name = provisional[path]
        if counts.get(name, 0) > 1:
            try:
                rel = path.relative_to(input_dir)
                name = rel.as_posix()
            except Exception:
                name = path.name
        resolved[path] = name
    return resolved


def _fit_multiline_font(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    *,
    max_width: int,
    max_height: int,
    base_size: int,
    spacing: int = 2,
) -> ImageFont.ImageFont:
    text = "\n".join(lines)
    size = max(10, int(base_size))
    while size > 10:
        font = _font_arial_bold(size)
        try:
            bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
            width = int(bbox[2] - bbox[0])
            height = int(bbox[3] - bbox[1])
        except Exception:
            width = int(draw.textlength(max(lines, key=len), font=font))
            height = int(len(lines) * max(10, 0.82 * size))
        if width <= int(max_width) and height <= int(max_height):
            return font
        size -= 1
    return _font_arial_bold(10)


def _tile_image_multiline(
    structure_img: Image.Image,
    *,
    lines: Sequence[str],
    image_h_px: int,
    label_h_px: int,
    min_tile_w_px: int,
    spacing: int = 2,
) -> Image.Image:
    sq = structure_img.convert("RGBA")
    tile_w = max(int(sq.width), int(min_tile_w_px))
    tile = Image.new("RGBA", (tile_w, int(image_h_px + label_h_px)), _PAPER_BG_RGBA)
    img_x = max(0, (tile_w - int(sq.width)) // 2)
    img_y = max(0, int(image_h_px) - int(sq.height))
    tile.paste(sq, (img_x, img_y))

    draw = ImageDraw.Draw(tile)
    font = _fit_multiline_font(
        draw,
        lines,
        max_width=tile_w - 10,
        max_height=int(label_h_px) - 4,
        base_size=int(0.28 * int(label_h_px)),
        spacing=spacing,
    )
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    text_w = int(bbox[2] - bbox[0])
    text_h = int(bbox[3] - bbox[1])
    x = max(2, (tile_w - text_w) // 2)
    y = int(image_h_px) + max(1, (int(label_h_px) - text_h) // 2) - 1
    draw.multiline_text((x, y), text, fill=_LABEL_RGBA, font=font, spacing=spacing, align="center")
    return tile


def _type_styles(atoms: Atoms, *, radius_scale: float) -> list[dict[str, object]]:
    styles: list[dict[str, object]] = []
    for symbol in _unique_symbol_order(atoms):
        z = int(atomic_numbers.get(str(symbol), 0))
        color = _nature_rgb(str(symbol), z)
        styles.append(
            {
                "name": str(symbol),
                "color": [float(color[0]), float(color[1]), float(color[2])],
                "radius": float(_ovito_radius(str(symbol), radius_scale=radius_scale)),
            }
        )
    return styles


def _ovitos_executable(ovito_root: Path) -> Path:
    ovito_root = ovito_root.expanduser().resolve()
    ovitos = ovito_root / "ovitos.exe"
    if not ovitos.is_file():
        raise FileNotFoundError(ovitos)
    return ovitos


def _helper_script() -> Path:
    helper = Path(__file__).with_name("_ovito_tachyon_helper.py").resolve()
    if not helper.is_file():
        raise FileNotFoundError(helper)
    return helper


def _run_ovito_helper(*, manifest_path: Path, ovito_root: Path, input_dir: Path) -> None:
    cmd = [
        str(_ovitos_executable(ovito_root)),
        str(_helper_script()),
        "--manifest",
        str(manifest_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(input_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
        raise RuntimeError(f"OVITO Tachyon render failed with exit code {completed.returncode}.\n{details}".rstrip())


def _prepare_entries(
    *,
    files: Sequence[Path],
    input_dir: Path,
    staging_dir: Path,
    renders_dir: Path,
    radius_scale: float,
) -> list[StructureEntry]:
    display_names = _display_names(files, input_dir)
    entries: list[StructureEntry] = []
    for idx, path in enumerate(files, start=1):
        atoms = _read_structure(path)
        formula = atoms.get_chemical_formula(mode="metal", empirical=True) or atoms.get_chemical_formula()
        expanded = _expand_atoms_for_visualization(atoms, radius_scale=radius_scale)
        safe_name = f"{idx:02d}_{_safe_filename(display_names[path])}"
        staged_path = staging_dir / f"{safe_name}.vasp"
        raw_png_path = renders_dir / f"{safe_name}.png"
        write(str(staged_path), expanded, format="vasp", direct=False, vasp5=True)
        entries.append(
            StructureEntry(
                path=path,
                display_name=display_names[path],
                formula=str(formula),
                staged_path=staged_path,
                raw_png_path=raw_png_path,
                tile_name=safe_name,
                type_styles=_type_styles(atoms, radius_scale=radius_scale),
                cell_line_width=_cell_line_width(atoms),
            )
        )
    return entries


def render_ovito_tachyon_panel(
    *,
    input_dir: Path,
    out_path: Path | None = None,
    cols: int = 6,
    size_px: int = 640,
    label_h_px: int = 180,
    pad_px: int = 6,
    margin_px: int = 10,
    radius_scale: float = 1.0,
    ovito_root: Path = DEFAULT_OVITO_ROOT,
    recursive: bool = False,
    tiles_dir: Path | None = None,
    raw_dir: Path | None = None,
    show_cell: bool = True,
    min_tile_width_factor: float = 1.08,
    cell_color_rgb: tuple[float, float, float] = (0.31, 0.38, 0.46),
    elev_deg: float = 10.0,
    azim_deg: float = -78.0,
    fov_deg: float = 32.0,
    ambient_occlusion_samples: int = 64,
    antialiasing_samples: int = 64,
    direct_light_intensity: float = 0.9,
    ambient_occlusion_brightness: float = 0.8,
) -> Path:
    input_dir = input_dir.expanduser().resolve()
    files = discover_structure_files(input_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No CONTCAR/POSCAR/CIF files found under: {input_dir}")

    if out_path is None:
        out_path = (input_dir / "snapshot_panel_tachyon.png").resolve()
    else:
        out_path = out_path.expanduser().resolve()
    if tiles_dir is not None:
        tiles_dir = tiles_dir.expanduser().resolve()
        tiles_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir is not None:
        raw_dir = raw_dir.expanduser().resolve()
        raw_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="genim_ovito_tachyon_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        staging_dir = tmp_dir / "structures"
        renders_dir = tmp_dir / "renders"
        staging_dir.mkdir(parents=True, exist_ok=True)
        renders_dir.mkdir(parents=True, exist_ok=True)

        entries = _prepare_entries(
            files=files,
            input_dir=input_dir,
            staging_dir=staging_dir,
            renders_dir=renders_dir,
            radius_scale=radius_scale,
        )

        manifest = {
            "size_px": int(size_px),
            "background_rgb": [float(v) / 255.0 for v in _PAPER_BG_RGBA[:3]],
            "generate_alpha": True,
            "show_cell": bool(show_cell),
            "cell_color_rgb": [float(cell_color_rgb[0]), float(cell_color_rgb[1]), float(cell_color_rgb[2])],
            "camera": {
                "elev_deg": float(elev_deg),
                "azim_deg": float(azim_deg),
                "fov_deg": float(fov_deg),
            },
            "renderer": {
                "ambient_occlusion": True,
                "ambient_occlusion_samples": int(ambient_occlusion_samples),
                "ambient_occlusion_brightness": float(ambient_occlusion_brightness),
                "antialiasing": True,
                "antialiasing_samples": int(antialiasing_samples),
                "direct_light": True,
                "direct_light_intensity": float(direct_light_intensity),
                "shadows": True,
                "depth_of_field": False,
            },
            "jobs": [
                {
                    "input_path": str(entry.staged_path),
                    "output_path": str(entry.raw_png_path),
                    "type_styles": entry.type_styles,
                    "cell_line_width": float(entry.cell_line_width),
                }
                for entry in entries
            ],
        }
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _run_ovito_helper(manifest_path=manifest_path, ovito_root=ovito_root, input_dir=input_dir)

        rendered: list[tuple[StructureEntry, Image.Image]] = []
        for entry in entries:
            raw_img = Image.open(entry.raw_png_path).convert("RGBA")
            canvas = Image.new("RGBA", raw_img.size, _PAPER_BG_RGBA)
            canvas.alpha_composite(raw_img)
            tightened = _tighten_structure_image(canvas)
            rendered.append((entry, tightened))
            if raw_dir is not None:
                tightened.save(raw_dir / f"{entry.tile_name}.png", format="PNG")

        image_h_px = max(int(img.height) for _entry, img in rendered)
        min_tile_w_px = max(int(img.width) for _entry, img in rendered)
        min_tile_w_px = max(min_tile_w_px, int(size_px * float(min_tile_width_factor)))
        tiles: list[Image.Image] = []
        for idx, (entry, structure_img) in enumerate(rendered, start=1):
            lines = [
                f"{idx:02d}. {entry.display_name}",
                entry.formula,
            ]
            tile = _tile_image_multiline(
                structure_img,
                lines=lines,
                image_h_px=image_h_px,
                label_h_px=label_h_px,
                min_tile_w_px=min_tile_w_px,
            )
            tiles.append(tile)
            if tiles_dir is not None:
                tile.save(tiles_dir / f"{entry.tile_name}.png", format="PNG")

        panel = _build_panel(tiles, cols=cols, pad_px=pad_px, margin_px=margin_px)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(out_path, format="PNG")
        return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a crystal-structure panel using OVITO's Tachyon renderer."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing CONTCAR/POSCAR/CIF files (default: current directory).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <input_dir>/snapshot_panel_tachyon.png).")
    parser.add_argument("--cols", type=int, default=6, help="How many tiles per row (default: 6).")
    parser.add_argument("--size", type=int, default=640, help="Raw render canvas size in px (default: 640).")
    parser.add_argument("--label-h", type=int, default=180, help="Label area height in px (default: 180).")
    parser.add_argument("--pad", type=int, default=6, help="Gap between tiles in px (default: 6).")
    parser.add_argument("--margin", type=int, default=10, help="Panel outer margin in px (default: 10).")
    parser.add_argument("--radius-scale", type=float, default=1.0, help="Per-element sphere radius multiplier (default: 1.0).")
    parser.add_argument("--ovito-root", type=Path, default=DEFAULT_OVITO_ROOT, help=f"OVITO installation directory (default: {DEFAULT_OVITO_ROOT}).")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subdirectories for supported structure files.")
    parser.add_argument("--tiles-dir", type=Path, default=None, help="Optional directory to save labeled tiles.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional directory to save cropped structure renders without labels.")
    parser.add_argument("--show-cell", dest="show_cell", action="store_true", help="Render the simulation cell lines (default: on).")
    parser.add_argument("--no-cell", dest="show_cell", action="store_false", help="Hide the simulation cell lines.")
    parser.set_defaults(show_cell=True)
    parser.add_argument("--min-tile-width-factor", type=float, default=1.08, help="Minimum tile width as a multiple of render size (default: 1.08).")
    parser.add_argument("--elev", type=float, default=10.0, help="Camera elevation angle in degrees (default: 10).")
    parser.add_argument("--azim", type=float, default=-78.0, help="Camera azimuth angle in degrees (default: -78).")
    parser.add_argument("--fov", type=float, default=32.0, help="Perspective field of view in degrees (default: 32).")
    parser.add_argument("--ao-samples", type=int, default=64, help="Ambient-occlusion sample count (default: 64).")
    parser.add_argument("--aa-samples", type=int, default=64, help="Antialiasing sample count (default: 64).")
    parser.add_argument("--light-intensity", type=float, default=0.9, help="Directional-light intensity (default: 0.9).")
    parser.add_argument("--ao-brightness", type=float, default=0.8, help="Ambient-occlusion brightness (default: 0.8).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = render_ovito_tachyon_panel(
        input_dir=args.input_dir,
        out_path=args.out,
        cols=args.cols,
        size_px=args.size,
        label_h_px=args.label_h,
        pad_px=args.pad,
        margin_px=args.margin,
        radius_scale=args.radius_scale,
        ovito_root=args.ovito_root,
        recursive=bool(args.recursive),
        tiles_dir=args.tiles_dir,
        raw_dir=args.raw_dir,
        show_cell=bool(args.show_cell),
        min_tile_width_factor=float(args.min_tile_width_factor),
        elev_deg=float(args.elev),
        azim_deg=float(args.azim),
        fov_deg=float(args.fov),
        ambient_occlusion_samples=int(args.ao_samples),
        antialiasing_samples=int(args.aa_samples),
        direct_light_intensity=float(args.light_intensity),
        ambient_occlusion_brightness=float(args.ao_brightness),
    )
    print(f"Saved OVITO Tachyon panel: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
