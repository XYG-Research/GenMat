from __future__ import annotations

import argparse
import io
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from ase.io import read
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


_CELL_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)

_PAPER_BG_RGBA = (247, 249, 250, 255)
_LABEL_RGBA = (39, 47, 58, 255)
_CELL_LIGHT_RGBA = (205, 214, 224, 196)
_CELL_DARK_RGBA = (84, 100, 118, 220)
_FRESH_ELEMENT_RGB: dict[str, tuple[float, float, float]] = {
    "H": (0.93, 0.95, 0.97),
    "C": (0.49, 0.57, 0.64),
    "N": (0.47, 0.65, 0.88),
    "O": (0.95, 0.53, 0.53),
    "S": (0.96, 0.78, 0.39),
    "P": (0.96, 0.67, 0.41),
    "Al": (0.77, 0.84, 0.93),
    "Ti": (0.66, 0.79, 0.90),
    "V": (0.57, 0.77, 0.76),
    "Cr": (0.52, 0.78, 0.69),
    "Mn": (0.80, 0.64, 0.77),
    "Fe": (0.94, 0.57, 0.55),
    "Co": (0.82, 0.70, 0.90),
    "Ni": (0.50, 0.84, 0.63),
    "Cu": (0.96, 0.64, 0.45),
    "Zn": (0.72, 0.80, 0.91),
    "Zr": (0.68, 0.83, 0.88),
    "Nb": (0.57, 0.71, 0.83),
    "Mo": (0.55, 0.63, 0.80),
    "Pd": (0.83, 0.85, 0.89),
    "Ag": (0.90, 0.91, 0.94),
    "Sn": (0.76, 0.82, 0.88),
    "W": (0.60, 0.66, 0.74),
    "Pt": (0.86, 0.86, 0.89),
    "Au": (0.95, 0.78, 0.46),
}
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(frozen=True)
class SnapshotItem:
    path: Path
    sid: int | None
    sid_text: str
    label: str


def _font_arial_bold(size: int) -> ImageFont.ImageFont:
    cand = [
        r"C:\Windows\Fonts\Arial Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
    ]
    for p in cand:
        try:
            return ImageFont.truetype(p, size=max(8, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def _safe_filename(name: str) -> str:
    s = str(name).strip().replace(" ", "_")
    out: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
    return "".join(out) or "snapshot"


def _find_cifs(cif_dir: Path) -> list[Path]:
    cif_dir = Path(cif_dir).expanduser().resolve()
    uniq: dict[str, Path] = {}
    for p in list(cif_dir.glob("*.cif")) + list(cif_dir.glob("*.CIF")):
        if not p.is_file():
            continue
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        uniq.setdefault(key, p)
    files = list(uniq.values())
    files.sort(key=lambda p: p.name)
    return files


def _extract_sid(path: Path) -> int | None:
    m = re.search(r"(\d+)$", str(path.stem))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _format_sid(sid: int | None) -> str:
    if sid is None:
        return "NA"
    if sid < 0:
        sid = 0
    return f"{int(sid):03d}" if int(sid) < 1000 else str(int(sid))


def _counts_dict(atoms: Atoms) -> dict[str, int]:
    c = Counter(str(s) for s in atoms.get_chemical_symbols())
    return {k: int(v) for k, v in c.items() if int(v) > 0}


def _preferred_symbol_order(cif_dir: Path, atoms: Atoms) -> list[str]:
    counts = _counts_dict(atoms)
    out: list[str] = []
    seen: set[str] = set()

    name = str(Path(cif_dir).name)
    sys_part = re.sub(r"_\d+el$", "", name)
    for sym in [s for s in sys_part.split("-") if s]:
        if sym in counts and sym not in seen:
            out.append(sym)
            seen.add(sym)

    for sym in [str(s) for s in atoms.get_chemical_symbols()]:
        if sym in counts and sym not in seen:
            out.append(sym)
            seen.add(sym)

    for sym in sorted(counts, key=lambda s: (atomic_numbers.get(s, 999), s)):
        if sym not in seen:
            out.append(sym)
            seen.add(sym)
    return out


def _composition_counts_label(counts: Mapping[str, int], *, symbol_order: Sequence[str]) -> str:
    parts: list[str] = []
    used: set[str] = set()
    for sym in symbol_order:
        if sym in counts and int(counts[sym]) > 0 and sym not in used:
            parts.append(f"{sym}{int(counts[sym])}")
            used.add(sym)
    for sym in sorted(counts, key=lambda s: (atomic_numbers.get(s, 999), s)):
        if sym not in used and int(counts[sym]) > 0:
            parts.append(f"{sym}{int(counts[sym])}")
    return "".join(parts)


def _snapshot_label(*, sid: int | None, counts: Mapping[str, int], symbol_order: Sequence[str]) -> str:
    return f"{_format_sid(sid)}-{_composition_counts_label(counts, symbol_order=symbol_order)}"


def _select_snapshot_items(
    *,
    cif_dir: Path,
    n: int,
    seed: int,
    id_start: int | None,
    id_end: int | None,
) -> list[SnapshotItem]:
    paths = _find_cifs(cif_dir)
    if not paths:
        raise FileNotFoundError(f"No CIF files found under: {cif_dir}")

    cand: list[tuple[Path, int | None]] = []
    for p in paths:
        sid = _extract_sid(p)
        if id_start is not None and (sid is None or int(sid) < int(id_start)):
            continue
        if id_end is not None and (sid is None or int(sid) > int(id_end)):
            continue
        cand.append((p, sid))

    if not cand:
        raise FileNotFoundError("No CIF files matched the requested ID range")

    n = max(1, int(n))
    rng = random.Random(int(seed))
    if len(cand) > n:
        selected = rng.sample(cand, k=n)
    else:
        selected = list(cand)
    selected.sort(key=lambda item: (item[1] is None, -1 if item[1] is None else int(item[1]), item[0].name))

    out: list[SnapshotItem] = []
    for p, sid in selected:
        atoms = read(str(p))
        counts = _counts_dict(atoms)
        order = _preferred_symbol_order(cif_dir, atoms)
        out.append(SnapshotItem(path=p, sid=sid, sid_text=_format_sid(sid), label=_snapshot_label(sid=sid, counts=counts, symbol_order=order)))
    return out


def _rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_matrix(*, elev_deg: float, azim_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    return _rot_z(math.radians(roll_deg)) @ _rot_x(math.radians(elev_deg)) @ _rot_z(math.radians(azim_deg))


def _cell_corners(cell: np.ndarray) -> np.ndarray:
    a, b, c = cell[0], cell[1], cell[2]
    return np.array(
        [
            [0.0, 0.0, 0.0],
            a,
            b,
            a + b,
            c,
            a + c,
            b + c,
            a + b + c,
        ],
        dtype=float,
    )


def _prepare_geometry(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = atoms.copy()
    try:
        a.wrap()
    except Exception:
        pass

    pos = np.asarray(a.get_positions(), dtype=float)
    cell = np.asarray(a.cell.array, dtype=float).reshape(3, 3)
    has_cell = bool(np.isfinite(cell).all()) and abs(float(np.linalg.det(cell))) > 1e-8
    if has_cell:
        corners = _cell_corners(cell)
        center = 0.5 * (cell[0] + cell[1] + cell[2])
    else:
        corners = np.zeros((0, 3), dtype=float)
        center = np.mean(pos, axis=0) if len(pos) else np.zeros((3,), dtype=float)
    return pos - center[None, :], corners - center[None, :], np.asarray(a.get_atomic_numbers(), dtype=int)


def _side_view() -> tuple[float, float, float]:
    # Side view: camera direction lies approximately in the xy-plane,
    # so the projected vertical axis remains aligned with crystal z.
    return (10.0, -78.0, 0.0)


def _project(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(points, dtype=float) @ rot.T


def _fit_font(draw: ImageDraw.ImageDraw, text: str, *, max_width: int, base_size: int) -> ImageFont.ImageFont:
    size = max(10, int(base_size))
    while size > 10:
        font = _font_arial_bold(size)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            width = int(bbox[2] - bbox[0])
        except Exception:
            width = int(draw.textlength(text, font=font))
        if width <= int(max_width):
            return font
        size -= 1
    return _font_arial_bold(10)


def _label_segments(label: str) -> list[tuple[str, bool]]:
    if "-" not in str(label):
        return [(str(label), False)]
    prefix, comp = str(label).split("-", 1)
    segs: list[tuple[str, bool]] = [(prefix + "-", False)]
    pos = 0
    for m in re.finditer(r"([A-Z][a-z]*)(\d+)", comp):
        if m.start() > pos:
            segs.append((comp[pos : m.start()], False))
        segs.append((m.group(1), False))
        segs.append((m.group(2), True))
        pos = m.end()
    if pos < len(comp):
        segs.append((comp[pos:], False))
    return [(text, is_sub) for text, is_sub in segs if text]


def _label_display_text(label: str) -> str:
    out: list[str] = []
    for text, is_sub in _label_segments(label):
        out.append(str(text).translate(_SUBSCRIPT_DIGITS) if is_sub else str(text))
    return "".join(out)


def _draw_label(
    tile: Image.Image,
    *,
    label: str,
    tile_w_px: int,
    image_h_px: int,
    label_h_px: int,
) -> None:
    draw = ImageDraw.Draw(tile)
    main_font = _fit_font(draw, label, max_width=int(tile_w_px) - 6, base_size=int(0.78 * label_h_px))
    sub_font = _font_arial_bold(max(8, int(round(0.64 * int(getattr(main_font, "size", 14))))))
    segments = _label_segments(label)

    def _measure(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        except Exception:
            return int(draw.textlength(text, font=font)), int(max(10, 0.75 * int(getattr(font, "size", 12))))

    total_w = 0
    for text, is_sub in segments:
        w, _h = _measure(text, sub_font if is_sub else main_font)
        total_w += int(w)

    try:
        main_ascent, _main_descent = main_font.getmetrics()
    except Exception:
        main_ascent = max(10, int(0.78 * int(getattr(main_font, "size", 14))))
    try:
        sub_ascent, _sub_descent = sub_font.getmetrics()
    except Exception:
        sub_ascent = max(8, int(0.78 * int(getattr(sub_font, "size", 10))))

    x = max(2, (int(tile_w_px) - total_w) // 2)
    y_main = int(image_h_px)
    baseline_main = y_main + int(main_ascent)
    baseline_sub = baseline_main + max(1, int(round(0.08 * int(getattr(main_font, "size", 14)))))
    y_sub = baseline_sub - int(sub_ascent)

    for text, is_sub in segments:
        font = sub_font if is_sub else main_font
        w, _h = _measure(text, font)
        draw.text((x, y_sub if is_sub else y_main), text, fill=_LABEL_RGBA, font=font)
        x += int(w)


@lru_cache(maxsize=8)
def _unit_sphere_mesh(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, num=max(12, int(n_theta)), endpoint=True)
    phi = np.linspace(0.0, math.pi, num=max(10, int(n_phi)), endpoint=True)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    sin_p = np.sin(phi)[None, :]
    cos_p = np.cos(phi)[None, :]
    x = cos_t * sin_p
    y = sin_t * sin_p
    z = np.ones_like(cos_t) * cos_p
    return x.astype(float), y.astype(float), z.astype(float)


@lru_cache(maxsize=8)
def _unit_cylinder_mesh(n_theta: int, n_axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, num=max(18, int(n_theta)), endpoint=True)
    axis = np.linspace(0.0, 1.0, num=max(2, int(n_axis)), endpoint=True)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    zz = np.ones_like(cos_t) * axis[None, :]
    return cos_t.astype(float), sin_t.astype(float), zz.astype(float)


def _jmol_rgb(z: int) -> tuple[float, float, float]:
    z0 = int(z)
    col = jmol_colors[z0] if 0 <= z0 < len(jmol_colors) else (0.65, 0.65, 0.65)
    return tuple(float(max(0.0, min(1.0, c))) for c in col)


def _normalize_vec(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return (arr / n).astype(float)


def _nature_rgb(symbol: str, z: int) -> tuple[float, float, float]:
    sym = str(symbol or "").strip()
    if sym in _FRESH_ELEMENT_RGB:
        return _FRESH_ELEMENT_RGB[sym]
    base = np.asarray(_jmol_rgb(z), dtype=float)
    gray = np.full((3,), float(np.mean(base)), dtype=float)
    soft = 0.58 * base + 0.42 * gray
    soft = 0.90 * soft + 0.10 * np.array([0.82, 0.84, 0.88], dtype=float)
    return tuple(float(max(0.0, min(1.0, c))) for c in soft)


def _gloss_facecolors(
    *,
    base_rgb: Sequence[float],
    normals: np.ndarray,
    view_rot: np.ndarray,
    spec_main_power: float,
    spec_fill_power: float,
    shadow_mix: float,
    spec_main_gain: float,
    spec_fill_gain: float,
    rim_gain: float,
) -> np.ndarray:
    normals = np.asarray(normals, dtype=float)
    base = np.asarray(base_rgb, dtype=float).reshape(1, 1, 3)

    light_view = _normalize_vec((-0.48, -0.34, 1.00))
    fill_view = _normalize_vec((0.42, 0.22, 0.92))
    view_view = _normalize_vec((0.0, 0.0, 1.0))

    light_world = _normalize_vec(light_view @ view_rot)
    fill_world = _normalize_vec(fill_view @ view_rot)
    view_world = _normalize_vec(view_view @ view_rot)
    half_main = _normalize_vec(light_world + view_world)
    half_fill = _normalize_vec(fill_world + view_world)

    ndotl = np.clip(np.sum(normals * light_world[None, None, :], axis=2), 0.0, 1.0)
    ndotf = np.clip(np.sum(normals * fill_world[None, None, :], axis=2), 0.0, 1.0)
    ndotv = np.clip(np.sum(normals * view_world[None, None, :], axis=2), 0.0, 1.0)
    spec_main = np.clip(np.sum(normals * half_main[None, None, :], axis=2), 0.0, 1.0) ** float(spec_main_power)
    spec_fill = np.clip(np.sum(normals * half_fill[None, None, :], axis=2), 0.0, 1.0) ** float(spec_fill_power)
    rim = (1.0 - ndotv) ** 1.6

    cool_shadow = np.array([0.19, 0.21, 0.25], dtype=float).reshape(1, 1, 3)
    soft_white = np.array([0.98, 0.985, 0.99], dtype=float).reshape(1, 1, 3)

    face = base * (0.36 + 0.48 * ndotl[..., None] + 0.14 * ndotf[..., None])
    face = (1.0 - float(shadow_mix)) * face + float(shadow_mix) * cool_shadow
    face += float(spec_main_gain) * spec_main[..., None] * soft_white
    face += float(spec_fill_gain) * spec_fill[..., None] * base
    face += float(rim_gain) * rim[..., None] * soft_white
    face = np.clip(face, 0.0, 1.0)

    alpha = np.ones(face.shape[:2] + (1,), dtype=float)
    return np.concatenate([face, alpha], axis=2)


def _structure_bounds(
    pos: np.ndarray,
    corners: np.ndarray,
    radii_world: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    pts = pos if corners.size == 0 else np.vstack([pos, corners])
    if pts.size == 0:
        pts = np.zeros((1, 3), dtype=float)
    mins = np.min(pts, axis=0).astype(float)
    maxs = np.max(pts, axis=0).astype(float)
    r_pad = float(max(radii_world)) if radii_world else 0.0
    mins -= max(0.22, 0.85 * r_pad)
    maxs += max(0.22, 0.85 * r_pad)
    return mins, maxs


def _guess_bonds(
    pos: np.ndarray,
    atomic_nums: np.ndarray,
    *,
    scale: float = 1.18,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    n = int(len(atomic_nums))
    for i in range(n):
        zi = int(atomic_nums[i])
        ri = float(covalent_radii[zi]) if 0 <= zi < len(covalent_radii) and math.isfinite(float(covalent_radii[zi])) else 1.0
        for j in range(i + 1, n):
            zj = int(atomic_nums[j])
            rj = float(covalent_radii[zj]) if 0 <= zj < len(covalent_radii) and math.isfinite(float(covalent_radii[zj])) else 1.0
            d = float(np.linalg.norm(np.asarray(pos[i], dtype=float) - np.asarray(pos[j], dtype=float)))
            if d <= 1e-6:
                continue
            if d <= float(scale) * (ri + rj):
                out.append((i, j))
    return out


def _periodic_bonds(
    atoms: Atoms,
    *,
    scale: float = 1.24,
    nearest_tol: float = 1.06,
) -> list[tuple[int, int, tuple[int, int, int], float]]:
    a = atoms.copy()
    try:
        a.wrap()
    except Exception:
        pass

    pos = np.asarray(a.get_positions(), dtype=float)
    cell = np.asarray(a.cell.array, dtype=float).reshape(3, 3)
    atomic_nums = np.asarray(a.get_atomic_numbers(), dtype=int)
    n = int(len(atomic_nums))
    if n <= 1:
        return []

    bonds: dict[tuple[int, int, tuple[int, int, int]], tuple[int, int, tuple[int, int, int], float]] = {}

    def _canon(i: int, j: int, shift: Sequence[int]) -> tuple[int, int, tuple[int, int, int]]:
        s = tuple(int(v) for v in shift)
        if int(i) <= int(j):
            return int(i), int(j), s
        return int(j), int(i), tuple(-int(v) for v in s)

    pbc = np.asarray(getattr(a, "pbc", [False, False, False]), dtype=bool).reshape(3)
    ranges = [(-1, 0, 1) if bool(pbc[k]) and bool(np.any(np.abs(cell[k]) > 1e-8)) else (0,) for k in range(3)]
    shifts = [tuple(int(v) for v in s) for s in np.array(np.meshgrid(*ranges, indexing="ij")).T.reshape(-1, 3)]

    candidates_by_atom: dict[int, list[tuple[float, int, tuple[int, int, int]]]] = defaultdict(list)
    for i in range(n):
        zi = int(atomic_nums[i])
        ri = float(covalent_radii[zi]) if 0 <= zi < len(covalent_radii) and math.isfinite(float(covalent_radii[zi])) else 1.0
        for j in range(n):
            if i == j and len(shifts) == 1 and shifts[0] == (0, 0, 0):
                continue
            zj = int(atomic_nums[j])
            rj = float(covalent_radii[zj]) if 0 <= zj < len(covalent_radii) and math.isfinite(float(covalent_radii[zj])) else 1.0
            chem_cut = max(0.8, float(scale) * (ri + rj))
            for shift in shifts:
                if i == j and shift == (0, 0, 0):
                    continue
                dvec = np.asarray(pos[j], dtype=float) + np.asarray(shift, dtype=float) @ cell - np.asarray(pos[i], dtype=float)
                dist = float(np.linalg.norm(dvec))
                if not math.isfinite(dist) or dist <= 1e-8:
                    continue
                candidates_by_atom[int(i)].append((dist, int(j), tuple(int(v) for v in shift)))
                if dist <= chem_cut + 1e-6:
                    key = _canon(int(i), int(j), shift)
                    prev = bonds.get(key)
                    if prev is None or dist < float(prev[3]):
                        bonds[key] = (int(i), int(j), tuple(int(v) for v in shift), float(dist))

    for i in range(n):
        cand = sorted(candidates_by_atom.get(int(i), []), key=lambda item: item[0])
        if not cand:
            continue
        dmin = float(cand[0][0])
        shell_cut = float(dmin) * float(nearest_tol) + 1e-6
        for dist, j, shift in cand:
            if float(dist) > shell_cut:
                break
            key = _canon(int(i), int(j), shift)
            prev = bonds.get(key)
            if prev is None or float(dist) < float(prev[3]):
                bonds[key] = (int(i), int(j), tuple(int(v) for v in shift), float(dist))

    out = [(int(i), int(j), tuple(int(v) for v in shift), float(dist)) for i, j, shift, dist in bonds.values()]
    out.sort(key=lambda item: (item[0], item[1], item[2]))
    return out


def _orthonormal_frame(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = _normalize_vec(direction)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, ref))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _normalize_vec(np.cross(axis, ref))
    v = _normalize_vec(np.cross(axis, u))
    return axis, u, v


def _cylinder_surface(
    p0: Sequence[float],
    p1: Sequence[float],
    radius: float,
    *,
    n_theta: int = 24,
    n_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p0v = np.asarray(p0, dtype=float).reshape(3)
    p1v = np.asarray(p1, dtype=float).reshape(3)
    delta = p1v - p0v
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 1e-8:
        zz = np.zeros((2, 2), dtype=float)
        zeros = np.zeros_like(zz)
        normals = np.zeros((2, 2, 3), dtype=float)
        normals[..., 2] = 1.0
        return zeros, zeros, zz, normals

    axis, u, v = _orthonormal_frame(delta)
    cx, cy, zz = _unit_cylinder_mesh(n_theta, n_axis)
    radial = float(radius) * (cx[..., None] * u[None, None, :] + cy[..., None] * v[None, None, :])
    pts = p0v[None, None, :] + zz[..., None] * length * axis[None, None, :] + radial
    normals = cx[..., None] * u[None, None, :] + cy[..., None] * v[None, None, :]
    return pts[..., 0], pts[..., 1], pts[..., 2], normals


def _inside_box(point: Sequence[float], mins: np.ndarray, maxs: np.ndarray, margin: float) -> bool:
    p = np.asarray(point, dtype=float).reshape(3)
    return bool(np.all(p >= (mins - float(margin))) and np.all(p <= (maxs + float(margin))))


def _content_bbox(img: Image.Image, *, tol: int = 8) -> tuple[int, int, int, int]:
    rgba = np.asarray(img.convert("RGBA"), dtype=np.int16)
    bg = np.asarray(_PAPER_BG_RGBA, dtype=np.int16)
    diff = np.max(np.abs(rgba[:, :, :3] - bg[None, None, :3]), axis=2)
    mask = diff > int(tol)
    if not bool(np.any(mask)):
        return 0, 0, int(img.width), int(img.height)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _tighten_structure_image(img: Image.Image, *, pad_x: int = 4, pad_y: int = 2) -> Image.Image:
    left, top, right, bottom = _content_bbox(img)
    left = max(0, int(left) - int(pad_x))
    right = min(int(img.width), int(right) + int(pad_x))
    top = max(0, int(top) - int(pad_y))
    bottom = min(int(img.height), int(bottom) + int(pad_y))
    return img.crop((left, top, right, bottom))


def _render_structure_square(
    atoms: Atoms,
    *,
    size_px: int,
    supersample: int,
    atom_scale: float,
    cell_line_w_px: int,
) -> Image.Image:
    a = atoms.copy()
    try:
        a.wrap()
    except Exception:
        pass

    wrapped_pos = np.asarray(a.get_positions(), dtype=float)
    atomic_nums = np.asarray(a.get_atomic_numbers(), dtype=int)
    symbols = [str(s) for s in a.get_chemical_symbols()]
    cell = np.asarray(a.cell.array, dtype=float).reshape(3, 3)
    has_cell = bool(np.isfinite(cell).all()) and abs(float(np.linalg.det(cell))) > 1e-8
    if has_cell:
        center = 0.5 * (cell[0] + cell[1] + cell[2])
        corners0 = _cell_corners(cell) - center[None, :]
    else:
        center = np.mean(wrapped_pos, axis=0) if len(wrapped_pos) else np.zeros((3,), dtype=float)
        corners0 = np.zeros((0, 3), dtype=float)
    pos0 = wrapped_pos - center[None, :]
    render_px = int(size_px) * max(1, int(supersample))

    rot = _side_view()
    view_rot = _rotation_matrix(elev_deg=rot[0], azim_deg=rot[1], roll_deg=rot[2])

    radii_world: list[float] = []
    colors_rgb: list[tuple[float, float, float]] = []
    for idx, z in enumerate(atomic_nums):
        z0 = int(z)
        r = float(covalent_radii[z0]) if 0 <= z0 < len(covalent_radii) and math.isfinite(float(covalent_radii[z0])) else 1.0
        radii_world.append(max(0.28, 1.02 * float(atom_scale) * r))
        colors_rgb.append(_nature_rgb(symbols[idx] if idx < len(symbols) else "", z0))

    box_mins = np.min(corners0, axis=0) if corners0.size else np.min(pos0, axis=0)
    box_maxs = np.max(corners0, axis=0) if corners0.size else np.max(pos0, axis=0)

    display_atoms: dict[tuple[int, tuple[int, int, int]], np.ndarray] = {
        (int(idx), (0, 0, 0)): np.asarray(pos0[int(idx)], dtype=float) for idx in range(len(atomic_nums))
    }
    if has_cell and len(atomic_nums):
        pbc = np.asarray(getattr(a, "pbc", [False, False, False]), dtype=bool).reshape(3)
        ranges = [(-1, 0, 1) if bool(pbc[k]) and bool(np.any(np.abs(cell[k]) > 1e-8)) else (0,) for k in range(3)]
        shifts = [tuple(int(v) for v in s) for s in np.array(np.meshgrid(*ranges, indexing="ij")).T.reshape(-1, 3)]
        for idx in range(len(atomic_nums)):
            for shift in shifts:
                if shift == (0, 0, 0):
                    continue
                shift_vec = np.asarray(shift, dtype=float) @ cell
                p = np.asarray(pos0[int(idx)], dtype=float) + shift_vec
                if _inside_box(p, box_mins, box_maxs, 0.92 * float(radii_world[int(idx)]) + 0.06):
                    display_atoms.setdefault((int(idx), tuple(int(v) for v in shift)), p)

    atom_points = np.vstack(list(display_atoms.values())) if display_atoms else pos0
    mins, maxs = _structure_bounds(atom_points, corners0, radii_world)
    span_xyz = np.maximum(maxs - mins, 1e-6)

    dpi = 200
    fig = plt.figure(
        figsize=(render_px / dpi, render_px / dpi),
        dpi=dpi,
        facecolor=tuple(float(v) / 255.0 for v in _PAPER_BG_RGBA[:3]),
    )
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(tuple(float(v) / 255.0 for v in _PAPER_BG_RGBA[:3]))
    ax.set_proj_type("persp", focal_length=0.92)
    try:
        ax.view_init(elev=float(rot[0]), azim=float(rot[1]), roll=float(rot[2]))
    except TypeError:
        ax.view_init(elev=float(rot[0]), azim=float(rot[1]))

    ax.set_box_aspect((float(span_xyz[0]), float(span_xyz[1]), float(span_xyz[2])))
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_visible(False)
        except Exception:
            pass
        try:
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass

    if corners0.shape[0] >= 8:
        segs = [[tuple(float(v) for v in corners0[int(i)]), tuple(float(v) for v in corners0[int(j)])] for i, j in _CELL_EDGES]
        ax.add_collection3d(
            Line3DCollection(
                segs,
                colors=[tuple(float(v) / 255.0 for v in _CELL_LIGHT_RGBA)],
                linewidths=max(1.4, 1.08 * float(cell_line_w_px)),
            )
        )
        ax.add_collection3d(
            Line3DCollection(
                segs,
                colors=[tuple(float(v) / 255.0 for v in _CELL_DARK_RGBA)],
                linewidths=max(1.0, 0.62 * float(cell_line_w_px)),
            )
        )

    ux, uy, uz = _unit_sphere_mesh(30, 22)
    sphere_normals = np.stack([ux, uy, uz], axis=-1).astype(float)
    atom_entries = []
    for (idx, _shift), p in display_atoms.items():
        d = float(_project(np.asarray([p], dtype=float), view_rot)[0, 2])
        atom_entries.append((d, int(idx), np.asarray(p, dtype=float)))
    atom_entries.sort(key=lambda item: item[0])
    for _depth, idx, p in atom_entries:
        cx, cy, cz = (float(v) for v in p)
        r = float(radii_world[int(idx)])
        base = np.asarray(colors_rgb[int(idx)], dtype=float)
        col = tuple(float(v) for v in np.clip(0.80 * base + 0.20 * np.array([0.82, 0.84, 0.88], dtype=float), 0.0, 1.0))
        facecolors = _gloss_facecolors(
            base_rgb=col,
            normals=sphere_normals,
            view_rot=view_rot,
            spec_main_power=40.0,
            spec_fill_power=20.0,
            shadow_mix=0.22,
            spec_main_gain=0.54,
            spec_fill_gain=0.14,
            rim_gain=0.07,
        )
        ax.plot_surface(
            cx + r * ux,
            cy + r * uy,
            cz + r * uz,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=1.0,
        )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=tuple(float(v) / 255.0 for v in _PAPER_BG_RGBA[:3]),
        edgecolor=tuple(float(v) / 255.0 for v in _PAPER_BG_RGBA[:3]),
    )
    plt.close(fig)
    buf.seek(0)
    sq = Image.open(buf).convert("RGBA")
    if sq.size != (render_px, render_px):
        sq = sq.resize((render_px, render_px), resample=Image.LANCZOS)
    if render_px != int(size_px):
        sq = sq.resize((int(size_px), int(size_px)), resample=Image.LANCZOS)
    return sq


def _tile_image(
    structure_img: Image.Image,
    *,
    label: str,
    image_h_px: int,
    label_h_px: int,
) -> Image.Image:
    label_h = int(label_h_px)
    sq = structure_img.convert("RGBA")
    probe = Image.new("RGBA", (max(8, int(sq.width)), max(8, int(label_h))), _PAPER_BG_RGBA)
    probe_draw = ImageDraw.Draw(probe)
    probe_font = _fit_font(probe_draw, label, max_width=max(8, int(sq.width)), base_size=int(0.78 * label_h))
    try:
        bbox = probe_draw.textbbox((0, 0), label, font=probe_font)
        label_w = int(bbox[2] - bbox[0])
    except Exception:
        label_w = int(probe_draw.textlength(label, font=probe_font))
    tile_w = max(int(sq.width), int(label_w) + 6)
    tile = Image.new("RGBA", (int(tile_w), int(image_h_px + label_h)), _PAPER_BG_RGBA)
    img_x = max(0, (int(tile_w) - int(sq.width)) // 2)
    img_y = max(0, int(image_h_px) - int(sq.height))
    tile.paste(sq, (img_x, img_y))
    _draw_label(tile, label=label, tile_w_px=tile_w, image_h_px=image_h_px, label_h_px=label_h)
    return tile


def _build_panel(tiles: Sequence[Image.Image], *, cols: int, pad_px: int, margin_px: int) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles to assemble")
    cols = max(1, int(cols))
    rows = int(math.ceil(len(tiles) / float(cols)))
    col_widths = [0 for _ in range(cols)]
    row_heights = [0 for _ in range(rows)]
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        col_widths[c] = max(col_widths[c], int(tile.width))
        row_heights[r] = max(row_heights[r], int(tile.height))

    W = int(2 * margin_px + sum(col_widths) + max(0, cols - 1) * pad_px)
    H = int(2 * margin_px + sum(row_heights) + max(0, rows - 1) * pad_px)
    panel = Image.new("RGBA", (W, H), _PAPER_BG_RGBA)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = int(margin_px + sum(col_widths[:c]) + c * pad_px + max(0, (col_widths[c] - int(tile.width)) // 2))
        y = int(margin_px + sum(row_heights[:r]) + r * pad_px + max(0, (row_heights[r] - int(tile.height)) // 2))
        panel.paste(tile, (x, y))
    return panel


def render_snapshot_panel(
    *,
    cif_dir: Path,
    out_path: Path | None = None,
    n: int = 12,
    seed: int = 7,
    id_start: int | None = None,
    id_end: int | None = None,
    cols: int = 6,
    size_px: int = 320,
    label_h_px: int = 24,
    pad_px: int = 3,
    margin_px: int = 4,
    supersample: int = 3,
    atom_scale: float = 0.78,
    cell_line_w_px: int = 3,
    tiles_dir: Path | None = None,
) -> Path:
    cif_dir = Path(cif_dir).expanduser().resolve()
    if not cif_dir.is_dir():
        raise FileNotFoundError(cif_dir)
    if out_path is None:
        out_path = (cif_dir / "snapshot_panel.png").resolve()
    else:
        out_path = Path(out_path).expanduser().resolve()

    items = _select_snapshot_items(cif_dir=cif_dir, n=n, seed=seed, id_start=id_start, id_end=id_end)
    rendered: list[tuple[SnapshotItem, Image.Image]] = []
    if tiles_dir is not None:
        tiles_dir = Path(tiles_dir).expanduser().resolve()
        tiles_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        atoms = read(str(item.path))
        sq = _render_structure_square(
            atoms,
            size_px=size_px,
            supersample=supersample,
            atom_scale=atom_scale,
            cell_line_w_px=cell_line_w_px,
        )
        rendered.append((item, _tighten_structure_image(sq)))

    image_h_px = max(int(img.height) for _item, img in rendered) if rendered else int(size_px)
    tiles: list[Image.Image] = []
    for item, structure_img in rendered:
        tile = _tile_image(
            structure_img,
            label=item.label,
            image_h_px=image_h_px,
            label_h_px=label_h_px,
        )
        tiles.append(tile)
        if tiles_dir is not None:
            tile.save(tiles_dir / f"{_safe_filename(item.label)}.png", format="PNG")

    panel = _build_panel(tiles, cols=cols, pad_px=pad_px, margin_px=margin_px)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, format="PNG")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render a tiled crystal snapshot panel from generated CIFs.")
    p.add_argument("--cif-dir", required=True, type=Path, help="Directory containing CIF files.")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <cif_dir>/snapshot_panel.png).")
    p.add_argument("--n", type=int, default=12, help="How many structures to snapshot (default: 12).")
    p.add_argument("--seed", type=int, default=7, help="Random seed for default random sampling.")
    p.add_argument("--id-start", type=int, default=None, help="Inclusive lower bound of structure ID.")
    p.add_argument("--id-end", type=int, default=None, help="Inclusive upper bound of structure ID.")
    p.add_argument("--cols", type=int, default=6, help="How many tiles per row (default: 6).")
    p.add_argument("--size", type=int, default=320, help="Square snapshot size in px (default: 320).")
    p.add_argument("--label-h", type=int, default=24, help="Label area height in px (default: 24).")
    p.add_argument("--pad", type=int, default=3, help="Gap between tiles in px (default: 3).")
    p.add_argument("--margin", type=int, default=4, help="Panel outer margin in px (default: 4).")
    p.add_argument("--supersample", type=int, default=3, help="Supersampling factor for cleaner lines (default: 3).")
    p.add_argument("--atom-scale", type=float, default=0.78, help="Atom sphere radius multiplier for the perspective view (default: 0.78).")
    p.add_argument("--cell-line-width", type=int, default=3, help="Cell line width in px (default: 3).")
    p.add_argument("--tiles-dir", type=Path, default=None, help="Optional directory to also write per-structure tiles.")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_path = render_snapshot_panel(
        cif_dir=args.cif_dir,
        out_path=args.out,
        n=args.n,
        seed=args.seed,
        id_start=args.id_start,
        id_end=args.id_end,
        cols=args.cols,
        size_px=args.size,
        label_h_px=args.label_h,
        pad_px=args.pad,
        margin_px=args.margin,
        supersample=args.supersample,
        atom_scale=args.atom_scale,
        cell_line_w_px=args.cell_line_width,
        tiles_dir=args.tiles_dir,
    )
    print(f"Saved snapshot panel: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
