from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.constraints import FixAtoms
from ase.io import write
from ase.neighborlist import neighbor_list, natural_cutoffs
from ase.build import make_supercell as ase_make_supercell
from pymatgen.core import Composition, Element, Structure
from pymatgen.core.surface import Slab, SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .ml_relax import BulkRelaxSettings, relax_bulk_atoms
from .mlip import build_omat24_calculator


SUPPORTED_SUFFIXES = {".cif"}


@dataclass(frozen=True)
class SurfaceScreenConfig:
    input_dir: Path
    out_dir: Path
    recursive: bool = False
    max_index: int = 2
    min_d_hkl: float = 1.2
    min_slab_size: float = 12.0
    min_vacuum_size: float = 15.0
    max_slab_thickness_a: float = 15.0
    max_surface_area: float = 160.0
    min_surface_area: float = 1.0
    primitive: bool = False
    center_slab: bool = False
    max_normal_search: int | None = 1
    ftol: float = 0.1
    tol: float = 0.1
    ztol: float = 0.0
    layer_tol_a: float = 0.45
    neighbor_scale: float = 1.18
    max_terminations_per_facet: int = 0
    keep_classes: tuple[str, ...] = ("A", "B")
    write_all_initial_slabs: bool = False
    max_compensated_slab_thickness_a: float = 15.0
    write_modeling_slabs: bool = True
    inplane_rectangularize: bool = True
    max_inplane_area_multiplier: int = 4
    orthogonal_angle_tol_deg: float = 2.0
    run_mlip: bool = False
    ml_model: str = "eSEN-30M-MPtrj"
    ml_checkpoint: str | None = None
    ml_device: str = "auto"
    ml_optimizer: str = "LBFGS"
    ml_fmax: float = 0.08
    ml_steps: int = 120
    ml_maxstep: float = 0.04
    relax_surface_layers: int = 2
    max_displacement_a: float = 1.0
    surface_rmsd_a: float = 0.6
    interlayer_rel_change: float = 0.35
    severe_coord_loss: float = 2.0
    severe_coord_loss_ratio: float = 0.35
    stoich_a_tol: float = 0.05
    stoich_b_tol: float = 0.18
    asymmetry_a_tol: float = 0.15
    asymmetry_b_tol: float = 0.35
    roughness_a_tol: float = 0.35
    roughness_b_tol: float = 0.90
    coord_loss_a_tol: float = 1.0
    coord_loss_b_tol: float = 2.0
    coord_loss_ratio_a_tol: float = 0.18
    coord_loss_ratio_b_tol: float = 0.35
    severe_loss_fraction_a_tol: float = 0.20
    severe_loss_fraction_b_tol: float = 0.50
    en_polarity_a_tol: float = 0.15
    en_polarity_b_tol: float = 0.35


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): json_ready(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(item) for item in obj]
    return obj


def discover_cif_files(input_dir: Path, *, recursive: bool) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files)


def formula_from_symbols(symbols: Sequence[str]) -> str:
    if not symbols:
        return ""
    return Composition("".join(symbols)).reduced_formula


def composition_fraction_dict_from_symbols(symbols: Sequence[str]) -> dict[str, float]:
    if not symbols:
        return {}
    counts: dict[str, float] = {}
    total = 0.0
    for sym in symbols:
        counts[sym] = counts.get(sym, 0.0) + 1.0
        total += 1.0
    return {key: val / total for key, val in counts.items()}


def composition_fraction_dict_from_structure(structure: Structure) -> dict[str, float]:
    counts: dict[str, float] = {}
    total = 0.0
    for site in structure:
        for specie, occ in site.species.items():
            sym = str(specie.symbol)
            counts[sym] = counts.get(sym, 0.0) + float(occ)
            total += float(occ)
    if total <= 0:
        return {}
    return {key: val / total for key, val in counts.items()}


def composition_l1_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys)


def weighted_electronegativity(fracs: dict[str, float]) -> float:
    total = 0.0
    for sym, frac in fracs.items():
        try:
            x = Element(sym).X
        except Exception:
            x = None
        total += float(frac) * (float(x) if x is not None else 0.0)
    return total


def cluster_layers(z_coords: np.ndarray, *, tol_a: float) -> list[list[int]]:
    if len(z_coords) == 0:
        return []
    order = np.argsort(z_coords)
    groups: list[list[int]] = [[int(order[0])]]
    for raw_idx in order[1:]:
        idx = int(raw_idx)
        if abs(float(z_coords[idx]) - float(z_coords[groups[-1][-1]])) <= float(tol_a):
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def neighbor_counts_ase(atoms, *, scale: float) -> np.ndarray:
    cutoffs = natural_cutoffs(atoms, mult=float(scale))
    i_idx, _j_idx = neighbor_list("ij", atoms, cutoffs)
    counts = np.bincount(i_idx, minlength=len(atoms))
    return counts.astype(float)


def bulk_coordination_by_species(structure: Structure, *, scale: float) -> dict[str, float]:
    atoms = AseAtomsAdaptor.get_atoms(structure)
    counts = neighbor_counts_ase(atoms, scale=scale)
    species_values: dict[str, list[float]] = {}
    for idx, sym in enumerate(atoms.get_chemical_symbols()):
        species_values.setdefault(sym, []).append(float(counts[idx]))
    return {sym: float(np.mean(values)) for sym, values in species_values.items()}


def surface_layer_indices(layer_groups: list[list[int]], *, n_layers: int) -> list[int]:
    if not layer_groups:
        return []
    take = max(1, min(int(n_layers), len(layer_groups)))
    idx: list[int] = []
    for group in layer_groups[:take]:
        idx.extend(group)
    for group in layer_groups[-take:]:
        idx.extend(group)
    return sorted(set(idx))


def make_layer_formula(structure: Structure, indices: Sequence[int]) -> str:
    symbols = [structure[i].specie.symbol for i in indices]
    return formula_from_symbols(symbols)


def make_layer_fraction(structure: Structure, indices: Sequence[int]) -> dict[str, float]:
    symbols = [structure[i].specie.symbol for i in indices]
    return composition_fraction_dict_from_symbols(symbols)


def slab_area(structure: Structure) -> float:
    lattice = structure.lattice.matrix
    return float(np.linalg.norm(np.cross(lattice[0], lattice[1])))


def slab_thickness_a(structure: Structure) -> float:
    coords = np.asarray(structure.cart_coords, dtype=float)
    if len(coords) == 0:
        return 0.0
    z = coords[:, 2]
    return float(np.max(z) - np.min(z))


def projected_cell_height_a(structure: Structure) -> float:
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    a_vec, b_vec, c_vec = lattice
    normal = np.cross(a_vec, b_vec)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        return float(np.linalg.norm(c_vec))
    normal /= norm
    return float(abs(np.dot(c_vec, normal)))


def center_slab_fractional(structure: Structure) -> Structure:
    frac = np.asarray(structure.frac_coords, dtype=float).copy()
    if len(frac) == 0:
        return structure.copy()
    zmin = float(np.min(frac[:, 2]))
    zmax = float(np.max(frac[:, 2]))
    shift = 0.5 - 0.5 * (zmin + zmax)
    centered = structure.copy()
    centered.translate_sites(
        list(range(len(centered))),
        [0.0, 0.0, shift],
        frac_coords=True,
        to_unit_cell=True,
    )
    return centered


def canonical_miller(miller: Sequence[int]) -> str:
    return f"({int(miller[0])}{int(miller[1])}{int(miller[2])})"


def choose_stoich_repeats(
    structure: Structure,
    miller: Sequence[int],
    *,
    cfg: SurfaceScreenConfig,
) -> tuple[int, float, bool, float]:
    probe = SlabGenerator(
        structure,
        miller_index=tuple(int(x) for x in miller),
        min_slab_size=1.0,
        min_vacuum_size=float(cfg.min_vacuum_size),
        center_slab=False,
        primitive=bool(cfg.primitive),
        max_normal_search=cfg.max_normal_search,
    )
    repeat_height = float(probe._proj_height)
    three = 3.0 * repeat_height
    two = 2.0 * repeat_height

    two_layer_gen = build_generator_for_facet(structure, miller, cfg=cfg, min_slab_size=two)
    two_actual_min = math.inf
    for termination in possible_terminations(two_layer_gen, ftol=cfg.ftol):
        slab = center_slab_fractional(two_layer_gen.get_slab(shift=termination, tol=cfg.tol, energy=0.0))
        two_actual_min = min(two_actual_min, slab_thickness_a(slab))
    if not math.isfinite(two_actual_min):
        two_actual_min = two

    allow_override = bool(two_actual_min > float(cfg.max_slab_thickness_a) + 1e-8)
    if three <= float(cfg.max_slab_thickness_a):
        return 3, three, allow_override, float(two_actual_min)
    return 2, two, allow_override, float(two_actual_min)


def enumerate_unique_slabs_for_facet(
    structure: Structure,
    miller: Sequence[int],
    *,
    cfg: SurfaceScreenConfig,
    min_slab_size: float,
) -> tuple[float | None, list[tuple[float, Slab]]]:
    generator = build_generator_for_facet(structure, miller, cfg=cfg, min_slab_size=min_slab_size)
    slabs_with_shift: list[tuple[float, Slab]] = []
    for termination in possible_terminations(generator, ftol=cfg.ftol):
        slab = generator.get_slab(shift=termination, tol=cfg.tol, energy=0.0)
        slabs_with_shift.append((float(termination), center_slab_fractional(slab)))
    if not slabs_with_shift:
        return None, []

    area = float(slab_area(slabs_with_shift[0][1]))
    unique_slabs: list[tuple[float, Slab]] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    for shift, slab in slabs_with_shift:
        sig = termination_signature(slab)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        unique_slabs.append((shift, slab))
    if cfg.max_terminations_per_facet > 0:
        unique_slabs = unique_slabs[: cfg.max_terminations_per_facet]
    return area, unique_slabs


def cyclic_shift_distance(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 1.0
    return min(diff, 1.0 - diff)


def termination_similarity_score(
    *,
    reference_shift: float,
    reference_metrics: dict[str, Any],
    candidate_shift: float,
    candidate_metrics: dict[str, Any],
) -> float:
    score = 3.0 * cyclic_shift_distance(reference_shift, candidate_shift)
    score += 1.5 * float(reference_metrics.get("top_layer_formula", "") != candidate_metrics.get("top_layer_formula", ""))
    score += 1.5 * float(reference_metrics.get("bottom_layer_formula", "") != candidate_metrics.get("bottom_layer_formula", ""))
    score += 0.75 * float(reference_metrics.get("top_two_layers_formula", "") != candidate_metrics.get("top_two_layers_formula", ""))
    score += 0.75 * float(
        reference_metrics.get("bottom_two_layers_formula", "") != candidate_metrics.get("bottom_two_layers_formula", "")
    )
    score += abs(float(reference_metrics.get("top_bottom_l1", 0.0)) - float(candidate_metrics.get("top_bottom_l1", 0.0)))
    score += 0.5 * abs(float(reference_metrics.get("stoich_l1", 0.0)) - float(candidate_metrics.get("stoich_l1", 0.0)))
    score += 0.25 * abs(
        float(reference_metrics.get("top_surface_atom_density", 0.0))
        - float(candidate_metrics.get("top_surface_atom_density", 0.0))
    )
    score += 0.25 * abs(
        float(reference_metrics.get("bottom_surface_atom_density", 0.0))
        - float(candidate_metrics.get("bottom_surface_atom_density", 0.0))
    )
    return float(score)


def pick_thinner_fallback_candidate(
    *,
    reference_shift: float,
    reference_metrics: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    cfg: SurfaceScreenConfig,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for candidate in candidates:
        metrics = candidate["metrics"]
        if float(metrics.get("slab_thickness_a", math.inf)) > float(cfg.max_slab_thickness_a) + 1e-8:
            continue
        score = termination_similarity_score(
            reference_shift=reference_shift,
            reference_metrics=reference_metrics,
            candidate_shift=float(candidate["shift"]),
            candidate_metrics=metrics,
        )
        payload = dict(candidate)
        payload["fallback_match_score"] = float(score)
        if best is None or float(payload["fallback_match_score"]) < float(best["fallback_match_score"]):
            best = payload
    return best


def pick_two_layer_fallback_candidate(
    *,
    reference_shift: float,
    reference_metrics: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    cfg: SurfaceScreenConfig,
) -> dict[str, Any] | None:
    return pick_thinner_fallback_candidate(
        reference_shift=reference_shift,
        reference_metrics=reference_metrics,
        candidates=candidates,
        cfg=cfg,
    )


def choose_inplane_supercell_transform(
    structure: Structure,
    *,
    max_area_multiplier: int,
) -> tuple[np.ndarray, float]:
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    a_vec, b_vec = lattice[0], lattice[1]
    best_t = np.eye(3, dtype=int)
    best_score = abs(np.dot(a_vec, b_vec)) / max(np.linalg.norm(a_vec) * np.linalg.norm(b_vec), 1e-12)
    search = range(-max(2, max_area_multiplier), max(2, max_area_multiplier) + 1)
    for p in search:
        for q in search:
            for r in search:
                for s in search:
                    det = p * s - q * r
                    area_mult = abs(det)
                    if area_mult < 1 or area_mult > max_area_multiplier:
                        continue
                    t = np.array([[p, q, 0], [r, s, 0], [0, 0, 1]], dtype=int)
                    na = p * a_vec + q * b_vec
                    nb = r * a_vec + s * b_vec
                    if np.linalg.norm(na) < 1e-8 or np.linalg.norm(nb) < 1e-8:
                        continue
                    cosang = abs(np.dot(na, nb)) / max(np.linalg.norm(na) * np.linalg.norm(nb), 1e-12)
                    length_balance = abs(np.linalg.norm(na) - np.linalg.norm(nb)) / max(np.linalg.norm(na), np.linalg.norm(nb), 1e-12)
                    score = cosang + 0.03 * float(area_mult - 1) + 0.02 * length_balance
                    if score + 1e-12 < best_score:
                        best_score = score
                        best_t = t
    return best_t, float(best_score)


def modeling_friendly_structure(
    slab: Structure,
    *,
    cfg: SurfaceScreenConfig,
) -> tuple[Structure, dict[str, Any]]:
    working = slab.copy()
    chosen_transform = np.eye(3, dtype=int)
    orth_score = abs(np.dot(working.lattice.matrix[0], working.lattice.matrix[1])) / max(
        np.linalg.norm(working.lattice.matrix[0]) * np.linalg.norm(working.lattice.matrix[1]), 1e-12
    )
    if cfg.inplane_rectangularize:
        chosen_transform, orth_score = choose_inplane_supercell_transform(
            working,
            max_area_multiplier=int(cfg.max_inplane_area_multiplier),
        )
        if not np.array_equal(chosen_transform, np.eye(3, dtype=int)):
            ase_atoms = AseAtomsAdaptor.get_atoms(working)
            ase_super = ase_make_supercell(ase_atoms, chosen_transform)
            working = AseAtomsAdaptor.get_structure(ase_super)

    lattice = np.asarray(working.lattice.matrix, dtype=float)
    a_vec, b_vec, c_vec = lattice
    xhat = a_vec / max(np.linalg.norm(a_vec), 1e-12)
    normal = np.cross(a_vec, b_vec)
    normal /= max(np.linalg.norm(normal), 1e-12)
    yhat = np.cross(normal, xhat)
    yhat /= max(np.linalg.norm(yhat), 1e-12)

    old_cart = np.asarray(working.cart_coords, dtype=float)
    transform = np.vstack([xhat, yhat, normal])
    new_cart = old_cart @ transform.T

    a_new = np.array([np.linalg.norm(a_vec), 0.0, 0.0], dtype=float)
    b_new = np.array([float(np.dot(b_vec, xhat)), float(np.dot(b_vec, yhat)), 0.0], dtype=float)
    c_height = max(projected_cell_height_a(working), slab_thickness_a(working) + 0.1)
    z = new_cart[:, 2]
    slab_height = float(np.max(z) - np.min(z))
    z_shift = 0.5 * (c_height - slab_height) - float(np.min(z))
    new_cart[:, 2] = z + z_shift
    c_new = np.array([0.0, 0.0, c_height], dtype=float)
    new_lattice = np.vstack([a_new, b_new, c_new])

    modeled = Structure(
        lattice=new_lattice,
        species=working.species_and_occu,
        coords=new_cart,
        coords_are_cartesian=True,
        site_properties=working.site_properties,
    )
    gamma = float(modeled.lattice.gamma)
    alpha = float(modeled.lattice.alpha)
    beta = float(modeled.lattice.beta)
    meta = {
        "modeling_supercell_transform": ";".join(",".join(str(int(x)) for x in row) for row in chosen_transform),
        "modeling_inplane_orth_score": orth_score,
        "modeling_cell_is_orthogonal": abs(alpha - 90.0) < cfg.orthogonal_angle_tol_deg
        and abs(beta - 90.0) < cfg.orthogonal_angle_tol_deg
        and abs(gamma - 90.0) < cfg.orthogonal_angle_tol_deg,
        "modeling_alpha_deg": alpha,
        "modeling_beta_deg": beta,
        "modeling_inplane_gamma_deg": gamma,
        "modeling_atom_count": len(modeled),
    }
    return modeled, meta


def possible_terminations(generator: SlabGenerator, *, ftol: float) -> list[float]:
    frac_coords = np.asarray(generator.oriented_unit_cell.frac_coords, dtype=float)
    n_atoms = len(frac_coords)
    if n_atoms == 0:
        return []
    if n_atoms == 1:
        termination = frac_coords[0][2] + 0.5
        return [float(termination - math.floor(termination))]

    dist_matrix = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            z_dist = frac_coords[i][2] - frac_coords[j][2]
            z_dist = abs(z_dist - round(z_dist)) * float(generator._proj_height)
            dist_matrix[i, j] = z_dist
            dist_matrix[j, i] = z_dist

    z_matrix = linkage(squareform(dist_matrix))
    clusters = fcluster(z_matrix, ftol, criterion="distance")
    clst_loc: dict[int, float] = {}
    for idx, clst in enumerate(clusters):
        clst_i = int(clst)
        if clst_i not in clst_loc:
            clst_loc[clst_i] = float(frac_coords[idx][2])
    possible_clst = [coord - math.floor(coord) for coord in sorted(clst_loc.values())]

    n_terms = len(possible_clst)
    terminations: list[float] = []
    for idx in range(n_terms):
        if idx == n_terms - 1:
            termination = (possible_clst[0] + 1.0 + possible_clst[idx]) * 0.5
        else:
            termination = (possible_clst[idx] + possible_clst[idx + 1]) * 0.5
        terminations.append(float(termination - math.floor(termination)))
    return sorted(terminations)


def _termination_signature_rows(slab: Slab, *, mirror_z: bool) -> tuple[Any, ...]:
    frac = np.asarray(slab.frac_coords, dtype=float).copy()
    if len(frac):
        if mirror_z:
            frac[:, 2] = 1.0 - frac[:, 2]
        frac[:, 2] -= np.min(frac[:, 2])
    frac %= 1.0
    rows = []
    for site, coords in zip(slab, frac):
        rows.append(
            (
                str(site.specie.symbol),
                round(float(coords[0]), 5),
                round(float(coords[1]), 5),
                round(float(coords[2]), 5),
            )
        )
    rows.sort()
    return tuple(rows)


def termination_signature(slab: Slab) -> tuple[Any, ...]:
    rows = min(_termination_signature_rows(slab, mirror_z=False), _termination_signature_rows(slab, mirror_z=True))
    return (
        str(slab.composition.reduced_formula),
        int(len(slab)),
        round(float(slab_area(slab)), 4),
        rows,
    )


def is_polar_or_asymmetric(metrics: dict[str, Any], cfg: SurfaceScreenConfig) -> bool:
    return bool(metrics["dipole_risk"]) or float(metrics["top_bottom_l1"]) > cfg.asymmetry_b_tol or float(metrics["en_polarity_proxy"]) > cfg.en_polarity_b_tol


def is_broken_skeleton(metrics: dict[str, Any], cfg: SurfaceScreenConfig) -> bool:
    ratio_mean = float(metrics.get("coord_loss_ratio_mean", 0.0))
    ratio_severe = float(metrics.get("severe_loss_ratio_fraction", metrics.get("severe_loss_fraction", 0.0)))
    abs_mean = float(metrics.get("coord_loss_mean", 0.0))
    return (
        ratio_mean > cfg.coord_loss_ratio_b_tol and ratio_severe > cfg.severe_loss_fraction_b_tol
    ) or (abs_mean > cfg.coord_loss_b_tol + 1.5 and ratio_severe > 0.75)


def raw_termination_class(metrics: dict[str, Any], cfg: SurfaceScreenConfig) -> tuple[str, str]:
    polar_or_asymmetric = is_polar_or_asymmetric(metrics, cfg)
    broken_skeleton = is_broken_skeleton(metrics, cfg)
    high_roughness = float(metrics["surface_roughness_a"]) > cfg.roughness_b_tol

    if broken_skeleton and polar_or_asymmetric:
        return "raw_broken_skeleton_and_polar", "undercoordinated_surface_with_polar_or_asymmetric_termination"
    if broken_skeleton:
        return "raw_broken_skeleton", "strong_surface_coordination_loss"
    if polar_or_asymmetric:
        return "raw_polar_or_asymmetric", "polar_or_asymmetric_termination_descriptor"
    if high_roughness:
        return "raw_high_roughness", "geometrically_rough_surface_termination"
    return "raw_good", "geometrically_regular_raw_termination"


def final_termination_class(
    metrics: dict[str, Any],
    *,
    compensation_found: bool,
    allow_thickness_override: bool = False,
    cfg: SurfaceScreenConfig,
) -> tuple[str, str]:
    too_thick = float(metrics.get("slab_thickness_a", 0.0)) > float(cfg.max_slab_thickness_a)
    broken_skeleton = is_broken_skeleton(metrics, cfg)
    high_roughness = float(metrics["surface_roughness_a"]) > cfg.roughness_b_tol
    polar_or_asymmetric = is_polar_or_asymmetric(metrics, cfg)

    if too_thick and not allow_thickness_override:
        return "rejected_too_thick", "slab_exceeds_max_slab_thickness_limit"
    if broken_skeleton:
        return "rejected_broken_skeleton", "surface_remains_strongly_undercoordinated"
    if high_roughness:
        return "rejected_high_roughness", "surface_remains_geometrically_too_rough"
    if compensation_found and polar_or_asymmetric:
        return "compensated_polar_surface_candidate", "compensated_slab_retained_as_polar_surface_candidate"
    if compensation_found:
        return "compensated_surface_candidate", "compensation_reduces_asymmetry_and_preserves_geometry"
    if polar_or_asymmetric:
        return "polar_surface_candidate", "raw_polar_or_asymmetric_descriptor_retained_without_hard_rejection"
    return "direct_surface_candidate", "direct_low_complexity_surface_candidate"


def compatibility_alias_for_final_class(final_class: str) -> str:
    if final_class == "direct_surface_candidate":
        return "A"
    if final_class in {"compensated_surface_candidate", "compensated_polar_surface_candidate", "polar_surface_candidate"}:
        return "B"
    return "C"


def metrics_score(metrics: dict[str, Any], *, cfg: SurfaceScreenConfig) -> float:
    return (
        (3.0 if bool(metrics["dipole_risk"]) else 0.0)
        + 5.0 * float(metrics["top_bottom_l1"])
        + 2.0 * float(metrics["en_polarity_proxy"])
        + 2.0 * float(metrics.get("coord_loss_ratio_mean", 0.0)) / max(cfg.coord_loss_ratio_b_tol, 1e-8)
        + 2.0 * float(metrics.get("severe_loss_ratio_fraction", metrics["severe_loss_fraction"])) / max(cfg.severe_loss_fraction_b_tol, 1e-8)
        + 1.5 * float(metrics["surface_roughness_a"]) / max(cfg.roughness_b_tol, 1e-8)
    )


def analyze_termination(
    slab: Slab,
    *,
    bulk_frac: dict[str, float],
    bulk_species_cn: dict[str, float],
    cfg: SurfaceScreenConfig,
) -> dict[str, Any]:
    z = np.asarray(slab.cart_coords)[:, 2]
    layer_groups = cluster_layers(z, tol_a=cfg.layer_tol_a)
    if len(layer_groups) == 0:
        raise ValueError("No layers found in slab")

    top = layer_groups[-1]
    bottom = layer_groups[0]
    top2 = top if len(layer_groups) == 1 else layer_groups[-2] + layer_groups[-1]
    bottom2 = bottom if len(layer_groups) == 1 else layer_groups[0] + layer_groups[1]

    top_frac = make_layer_fraction(slab, top)
    bottom_frac = make_layer_fraction(slab, bottom)
    slab_frac = composition_fraction_dict_from_structure(slab)

    atoms = AseAtomsAdaptor.get_atoms(slab)
    atoms.set_pbc((True, True, False))
    cn = neighbor_counts_ase(atoms, scale=cfg.neighbor_scale)
    surface_idx = surface_layer_indices(layer_groups, n_layers=1)
    surface_two_idx = surface_layer_indices(layer_groups, n_layers=2)
    deficits: list[float] = []
    deficit_ratios: list[float] = []
    for idx in surface_two_idx:
        sym = atoms[idx].symbol
        bulk_cn = float(bulk_species_cn.get(sym, 0.0))
        deficit = max(0.0, bulk_cn - float(cn[idx]))
        deficits.append(deficit)
        deficit_ratios.append(deficit / max(bulk_cn, 1.0))
    coord_loss_mean = float(np.mean(deficits)) if deficits else 0.0
    severe_loss_fraction = float(np.mean([loss >= cfg.severe_coord_loss for loss in deficits])) if deficits else 0.0
    coord_loss_ratio_mean = float(np.mean(deficit_ratios)) if deficit_ratios else 0.0
    severe_loss_ratio_fraction = float(np.mean([loss >= cfg.severe_coord_loss_ratio for loss in deficit_ratios])) if deficit_ratios else 0.0

    top_z = np.asarray([z[idx] for idx in top], dtype=float)
    bottom_z = np.asarray([z[idx] for idx in bottom], dtype=float)
    area = slab_area(slab)
    top_density = float(len(top) / area) if area > 0 else 0.0
    bottom_density = float(len(bottom) / area) if area > 0 else 0.0
    top_bottom_l1 = composition_l1_distance(top_frac, bottom_frac)
    en_polarity = abs(weighted_electronegativity(top_frac) - weighted_electronegativity(bottom_frac))
    symmetric = bool(slab.is_symmetric(symprec=0.1))
    dipole_risk = (not symmetric) and (top_bottom_l1 > cfg.asymmetry_a_tol or en_polarity > cfg.en_polarity_a_tol)

    metrics: dict[str, Any] = {
        "surface_area_a2": area,
        "slab_thickness_a": slab_thickness_a(slab),
        "n_layers": len(layer_groups),
        "top_layer_formula": make_layer_formula(slab, top),
        "bottom_layer_formula": make_layer_formula(slab, bottom),
        "top_two_layers_formula": make_layer_formula(slab, top2),
        "bottom_two_layers_formula": make_layer_formula(slab, bottom2),
        "top_layer_count": len(top),
        "bottom_layer_count": len(bottom),
        "is_symmetric_slab": symmetric,
        "dipole_risk": dipole_risk,
        "stoich_l1": composition_l1_distance(slab_frac, bulk_frac),
        "top_bottom_l1": top_bottom_l1,
        "en_polarity_proxy": en_polarity,
        "coord_loss_mean": coord_loss_mean,
        "severe_loss_fraction": severe_loss_fraction,
        "coord_loss_ratio_mean": coord_loss_ratio_mean,
        "severe_loss_ratio_fraction": severe_loss_ratio_fraction,
        "surface_roughness_a": max(float(np.std(top_z)), float(np.std(bottom_z))),
        "top_surface_atom_density": top_density,
        "bottom_surface_atom_density": bottom_density,
        "top_surface_atom_indices": ",".join(str(i) for i in top),
        "bottom_surface_atom_indices": ",".join(str(i) for i in bottom),
        "surface_atom_indices": ",".join(str(i) for i in surface_idx),
        "surface_two_layer_indices": ",".join(str(i) for i in surface_two_idx),
        "slab_formula": slab.composition.reduced_formula,
        "slab_atom_count": len(slab),
    }
    return metrics


def prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def build_generator_for_facet(
    structure: Structure,
    miller: Sequence[int],
    *,
    cfg: SurfaceScreenConfig,
    min_slab_size: float | None = None,
) -> SlabGenerator:
    return SlabGenerator(
        structure,
        miller_index=tuple(int(x) for x in miller),
        min_slab_size=float(cfg.min_slab_size if min_slab_size is None else min_slab_size),
        min_vacuum_size=float(cfg.min_vacuum_size),
        center_slab=False,
        primitive=bool(cfg.primitive),
        max_normal_search=cfg.max_normal_search,
    )


def search_compensated_slab(
    base_slab: Slab,
    *,
    bulk_frac: dict[str, float],
    bulk_species_cn: dict[str, float],
    cfg: SurfaceScreenConfig,
) -> dict[str, Any]:
    candidates: list[tuple[str, Slab]] = []
    try:
        for idx, cand in enumerate(base_slab.get_tasker2_slabs(tol=0.01, same_species_only=True), start=1):
            candidates.append((f"tasker2_same_species_{idx}", center_slab_fractional(cand)))
    except Exception:
        pass
    try:
        for idx, cand in enumerate(base_slab.get_tasker2_slabs(tol=0.01, same_species_only=False), start=1):
            candidates.append((f"tasker2_any_species_{idx}", center_slab_fractional(cand)))
    except Exception:
        pass

    unique: list[tuple[str, Slab]] = []
    seen: set[tuple[Any, ...]] = set()
    for method, cand in candidates:
        if slab_thickness_a(cand) > float(cfg.max_compensated_slab_thickness_a) + 1e-8:
            continue
        sig = termination_signature(cand)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append((method, cand))

    best: dict[str, Any] | None = None
    for method, cand in unique:
        metrics = analyze_termination(cand, bulk_frac=bulk_frac, bulk_species_cn=bulk_species_cn, cfg=cfg)
        score = metrics_score(metrics, cfg=cfg)
        payload = {
            "method": method,
            "slab": cand,
            "metrics": metrics,
            "score": score,
        }
        if best is None or float(payload["score"]) < float(best["score"]):
            best = payload

    return {
        "attempted": True,
        "candidate_count": len(unique),
        "found": best is not None,
        "best": best,
    }


def freeze_middle_layers(atoms, *, layer_groups: list[list[int]], relax_surface_layers: int) -> list[int]:
    if not layer_groups:
        return []
    take = max(1, min(int(relax_surface_layers), max(1, len(layer_groups) // 2)))
    free_idx = set()
    for group in layer_groups[:take]:
        free_idx.update(group)
    for group in layer_groups[-take:]:
        free_idx.update(group)
    fixed = sorted(set(range(len(atoms))) - free_idx)
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return fixed


def displacement_metrics(
    initial_atoms,
    relaxed_atoms,
    *,
    surface_idx: Sequence[int],
    layer_groups: list[list[int]],
) -> dict[str, float | bool]:
    init_scaled = np.asarray(initial_atoms.get_scaled_positions(wrap=True), dtype=float)
    rel_scaled = np.asarray(relaxed_atoms.get_scaled_positions(wrap=True), dtype=float)
    delta_scaled = rel_scaled - init_scaled
    pbc = np.asarray(initial_atoms.get_pbc(), dtype=bool)
    for axis in (0, 1, 2):
        if axis < len(pbc) and bool(pbc[axis]):
            delta_scaled[:, axis] -= np.round(delta_scaled[:, axis])
    delta_cart = delta_scaled @ np.asarray(initial_atoms.cell)
    norms = np.linalg.norm(delta_cart, axis=1)
    surf = list(surface_idx)
    surface_rmsd = float(np.sqrt(np.mean(norms[surf] ** 2))) if surf else 0.0
    max_surface_disp = float(np.max(norms[surf])) if surf else 0.0

    init_z = np.asarray(initial_atoms.get_positions())[:, 2]
    rel_z = np.asarray(relaxed_atoms.get_positions())[:, 2]
    initial_layer_centers = [float(np.mean(init_z[group])) for group in layer_groups]
    relaxed_layer_centers = [float(np.mean(rel_z[group])) for group in layer_groups]
    initial_spacings = np.diff(initial_layer_centers)
    relaxed_spacings = np.diff(relaxed_layer_centers)
    rel_changes: list[float] = []
    for before, after in zip(initial_spacings, relaxed_spacings):
        if abs(before) > 1e-8:
            rel_changes.append(abs(after - before) / abs(before))
    interlayer_rel_change = float(max(rel_changes)) if rel_changes else 0.0

    cross_layer_reconstruction = False
    if len(initial_spacings) and surf:
        layer_gap = float(np.median(np.abs(initial_spacings)))
        if layer_gap > 1e-8:
            for idx in surf:
                if abs(float(rel_z[idx] - init_z[idx])) > 0.75 * layer_gap:
                    cross_layer_reconstruction = True
                    break

    return {
        "max_displacement_a": float(np.max(norms)) if len(norms) else 0.0,
        "surface_rmsd_a": surface_rmsd,
        "max_surface_displacement_a": max_surface_disp,
        "interlayer_rel_change": interlayer_rel_change,
        "cross_layer_reconstruction": cross_layer_reconstruction,
    }


def mlip_screen_candidate(
    slab: Slab,
    *,
    cfg: SurfaceScreenConfig,
    calculator: Any,
    out_dir: Path,
    stem: str,
    miller_label: str,
    termination_index: int,
) -> dict[str, Any]:
    ase_atoms = AseAtomsAdaptor.get_atoms(slab)
    ase_atoms.set_pbc((True, True, False))
    layer_groups = cluster_layers(np.asarray(ase_atoms.get_positions())[:, 2], tol_a=cfg.layer_tol_a)
    fixed_indices = freeze_middle_layers(ase_atoms, layer_groups=layer_groups, relax_surface_layers=cfg.relax_surface_layers)
    relax_settings = BulkRelaxSettings(
        optimizer=cfg.ml_optimizer,
        fmax=cfg.ml_fmax,
        steps=cfg.ml_steps,
        maxstep=cfg.ml_maxstep,
        relax_cell=False,
        cell_filter="none",
        logfile_name=None,
    )
    result, relaxed_atoms = relax_bulk_atoms(
        ase_atoms,
        calc=calculator,
        settings=relax_settings,
        out_dir=out_dir,
        relaxed_cif_name=f"{stem}_{miller_label}_term{termination_index:02d}_relaxed.cif",
        label=f"{stem}:{miller_label}:term{termination_index:02d}",
    )
    payload: dict[str, Any] = {
        "mlip_converged": bool(result.converged),
        "mlip_nsteps": int(result.nsteps),
        "mlip_energy_init_eV": result.energy_init_eV,
        "mlip_energy_final_eV": result.energy_final_eV,
        "mlip_fmax_final": result.fmax_final,
        "mlip_error": result.error,
        "mlip_relaxed_cif": result.relaxed_cif,
        "mlip_fixed_atom_count": len(fixed_indices),
    }
    if relaxed_atoms is None:
        payload["mlip_screen_pass"] = False
        payload["mlip_screen_reason"] = "relaxation_failed"
        return payload

    surface_idx = surface_layer_indices(layer_groups, n_layers=cfg.relax_surface_layers)
    disp = displacement_metrics(ase_atoms, relaxed_atoms, surface_idx=surface_idx, layer_groups=layer_groups)
    payload.update(disp)
    pass_flag = (
        float(disp["max_displacement_a"]) <= cfg.max_displacement_a
        and float(disp["surface_rmsd_a"]) <= cfg.surface_rmsd_a
        and float(disp["interlayer_rel_change"]) <= cfg.interlayer_rel_change
        and not bool(disp["cross_layer_reconstruction"])
    )
    payload["mlip_screen_pass"] = bool(pass_flag)
    if bool(pass_flag):
        payload["mlip_screen_reason"] = "passes_fast_geometric_relaxation_screen"
    elif bool(disp["cross_layer_reconstruction"]):
        payload["mlip_screen_reason"] = "cross_layer_reconstruction_detected"
    elif float(disp["surface_rmsd_a"]) > cfg.surface_rmsd_a:
        payload["mlip_screen_reason"] = "surface_rmsd_too_large"
    elif float(disp["max_displacement_a"]) > cfg.max_displacement_a:
        payload["mlip_screen_reason"] = "max_displacement_too_large"
    else:
        payload["mlip_screen_reason"] = "interlayer_distortion_too_large"
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def facet_and_slab_records_for_structure(
    path: Path,
    *,
    cfg: SurfaceScreenConfig,
    calculator: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    structure = Structure.from_file(str(path))
    bulk_frac = composition_fraction_dict_from_structure(structure)
    bulk_cn = bulk_coordination_by_species(structure, scale=cfg.neighbor_scale)
    stem = path.stem
    facet_rows: list[dict[str, Any]] = []
    slab_rows: list[dict[str, Any]] = []
    canonical_family_counter = 0

    raw_dir = (cfg.out_dir / "raw_slabs" / stem).resolve()
    accepted_dir = (cfg.out_dir / "accepted_slabs" / stem).resolve()
    compensated_dir = (cfg.out_dir / "compensated_slabs" / stem).resolve()
    modeling_dir = (cfg.out_dir / "modeling_slabs" / stem).resolve()
    relaxed_dir = (cfg.out_dir / "relaxed_slabs" / stem).resolve()
    if cfg.write_all_initial_slabs:
        raw_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    compensated_dir.mkdir(parents=True, exist_ok=True)
    if cfg.write_modeling_slabs:
        modeling_dir.mkdir(parents=True, exist_ok=True)
    if cfg.run_mlip:
        relaxed_dir.mkdir(parents=True, exist_ok=True)

    millers = get_symmetrically_distinct_miller_indices(structure, max_index=int(cfg.max_index))
    for miller in millers:
        d_hkl = float(structure.lattice.d_hkl(miller))
        stoich_repeats, stoich_target_thickness, allow_thickness_override, two_layer_min_actual_thickness = choose_stoich_repeats(
            structure,
            miller,
            cfg=cfg,
        )
        facet_row: dict[str, Any] = {
            "structure_file": path.name,
            "structure_stem": stem,
            "formula": structure.composition.reduced_formula,
            "miller": canonical_miller(miller),
            "miller_h": int(miller[0]),
            "miller_k": int(miller[1]),
            "miller_l": int(miller[2]),
            "d_hkl": d_hkl,
            "stoich_layer_repeats": stoich_repeats,
            "stoich_target_thickness_a": stoich_target_thickness,
            "two_layer_min_actual_thickness_a": two_layer_min_actual_thickness,
            "thickness_override_allowed": allow_thickness_override,
            "two_layer_termination_count": 0,
            "one_layer_termination_count": 0,
        }
        if d_hkl < cfg.min_d_hkl:
            facet_row.update({"facet_kept": False, "facet_reason": "d_hkl_too_small", "surface_area_a2": None, "termination_count": 0})
            facet_rows.append(facet_row)
            continue

        area, unique_slabs = enumerate_unique_slabs_for_facet(
            structure,
            miller,
            cfg=cfg,
            min_slab_size=stoich_target_thickness,
        )
        if not unique_slabs:
            facet_row.update({"facet_kept": False, "facet_reason": "no_terminations_generated", "surface_area_a2": None, "termination_count": 0})
            facet_rows.append(facet_row)
            continue

        two_layer_target_thickness = float(stoich_target_thickness) * 2.0 / float(max(stoich_repeats, 1))
        one_layer_target_thickness = float(stoich_target_thickness) / float(max(stoich_repeats, 1))
        preferred_candidates: list[dict[str, Any]] = []
        two_layer_candidates: list[dict[str, Any]] = []
        one_layer_candidates: list[dict[str, Any]] = []
        for pref_shift, pref_slab in unique_slabs:
            preferred_candidates.append(
                {
                    "shift": float(pref_shift),
                    "slab": pref_slab,
                    "metrics": analyze_termination(pref_slab, bulk_frac=bulk_frac, bulk_species_cn=bulk_cn, cfg=cfg),
                }
            )
        if stoich_repeats > 2:
            _two_area, two_layer_unique_slabs = enumerate_unique_slabs_for_facet(
                structure,
                miller,
                cfg=cfg,
                min_slab_size=two_layer_target_thickness,
            )
            for two_shift, two_slab in two_layer_unique_slabs:
                two_layer_candidates.append(
                    {
                        "shift": float(two_shift),
                        "slab": two_slab,
                        "metrics": analyze_termination(two_slab, bulk_frac=bulk_frac, bulk_species_cn=bulk_cn, cfg=cfg),
                    }
                )
            facet_row["two_layer_termination_count"] = len(two_layer_candidates)
        if stoich_repeats > 1:
            _one_area, one_layer_unique_slabs = enumerate_unique_slabs_for_facet(
                structure,
                miller,
                cfg=cfg,
                min_slab_size=one_layer_target_thickness,
            )
            for one_shift, one_slab in one_layer_unique_slabs:
                one_layer_candidates.append(
                    {
                        "shift": float(one_shift),
                        "slab": one_slab,
                        "metrics": analyze_termination(one_slab, bulk_frac=bulk_frac, bulk_species_cn=bulk_cn, cfg=cfg),
                    }
                )
            facet_row["one_layer_termination_count"] = len(one_layer_candidates)

        facet_row["surface_area_a2"] = area
        if area < cfg.min_surface_area:
            facet_row.update({"facet_kept": False, "facet_reason": "surface_area_too_small", "termination_count": 0})
            facet_rows.append(facet_row)
            continue
        if area > cfg.max_surface_area:
            facet_row.update({"facet_kept": False, "facet_reason": "surface_area_too_large", "termination_count": 0})
            facet_rows.append(facet_row)
            continue

        facet_row.update({"facet_kept": True, "facet_reason": "kept", "termination_count": len(unique_slabs)})
        facet_rows.append(facet_row)

        seen_final_signatures: dict[tuple[Any, ...], dict[str, Any]] = {}
        for term_idx, (shift, slab) in enumerate(unique_slabs, start=1):
            preferred_candidate = preferred_candidates[term_idx - 1]
            preferred_metrics = dict(preferred_candidate["metrics"])
            miller_label = f"h{int(miller[0])}_k{int(miller[1])}_l{int(miller[2])}"
            out_name = f"{stem}_{miller_label}_term{term_idx:02d}.cif"
            selected_slab = slab
            selected_shift = float(shift)
            selected_layer_repeats = int(stoich_repeats)
            selected_raw_metrics = dict(preferred_metrics)
            fallback_candidate = None
            fallback_source_repeats = ""
            fallback_target_repeats = ""
            if (
                int(stoich_repeats) > 2
                and not bool(allow_thickness_override)
                and float(selected_raw_metrics["slab_thickness_a"]) > float(cfg.max_slab_thickness_a) + 1e-8
                and two_layer_candidates
            ):
                fallback_candidate = pick_thinner_fallback_candidate(
                    reference_shift=selected_shift,
                    reference_metrics=selected_raw_metrics,
                    candidates=two_layer_candidates,
                    cfg=cfg,
                )
                if fallback_candidate is not None:
                    previous_repeats = int(selected_layer_repeats)
                    selected_slab = fallback_candidate["slab"]
                    selected_shift = float(fallback_candidate["shift"])
                    selected_layer_repeats = 2
                    selected_raw_metrics = dict(fallback_candidate["metrics"])
                    fallback_source_repeats = str(previous_repeats)
                    fallback_target_repeats = "2"

            if (
                int(selected_layer_repeats) == 2
                and float(selected_raw_metrics["slab_thickness_a"]) > float(cfg.max_slab_thickness_a) + 1e-8
            ):
                same_repeat_two_layer_candidates = two_layer_candidates if stoich_repeats > 2 else preferred_candidates
                if same_repeat_two_layer_candidates:
                    same_repeat_fallback = pick_thinner_fallback_candidate(
                        reference_shift=selected_shift,
                        reference_metrics=selected_raw_metrics,
                        candidates=same_repeat_two_layer_candidates,
                        cfg=cfg,
                    )
                    if same_repeat_fallback is not None:
                        fallback_candidate = same_repeat_fallback
                        selected_slab = same_repeat_fallback["slab"]
                        selected_shift = float(same_repeat_fallback["shift"])
                        selected_raw_metrics = dict(same_repeat_fallback["metrics"])
                        fallback_source_repeats = "2"
                        fallback_target_repeats = "2"

            if (
                int(selected_layer_repeats) > 1
                and float(selected_raw_metrics["slab_thickness_a"]) > float(cfg.max_slab_thickness_a) + 1e-8
                and one_layer_candidates
            ):
                one_layer_fallback = pick_thinner_fallback_candidate(
                    reference_shift=selected_shift,
                    reference_metrics=selected_raw_metrics,
                    candidates=one_layer_candidates,
                    cfg=cfg,
                )
                if one_layer_fallback is not None:
                    fallback_candidate = one_layer_fallback
                    selected_slab = one_layer_fallback["slab"]
                    selected_shift = float(one_layer_fallback["shift"])
                    fallback_source_repeats = str(int(selected_layer_repeats))
                    selected_layer_repeats = 1
                    fallback_target_repeats = "1"
                    selected_raw_metrics = dict(one_layer_fallback["metrics"])

            raw_class, raw_reason = raw_termination_class(selected_raw_metrics, cfg)
            row: dict[str, Any] = {
                "structure_file": path.name,
                "structure_stem": stem,
                "formula": structure.composition.reduced_formula,
                "miller": canonical_miller(miller),
                "miller_h": int(miller[0]),
                "miller_k": int(miller[1]),
                "miller_l": int(miller[2]),
                "d_hkl": d_hkl,
                "stoich_layer_repeats": stoich_repeats,
                "stoich_target_thickness_a": stoich_target_thickness,
                "two_layer_min_actual_thickness_a": two_layer_min_actual_thickness,
                "thickness_override_allowed": allow_thickness_override,
                "selected_layer_repeats": selected_layer_repeats,
                "termination_index": term_idx,
                "termination_shift": shift,
                "selected_termination_shift": selected_shift,
                "output_name": out_name,
                "preferred_slab_thickness_a": float(preferred_metrics["slab_thickness_a"]),
                "fallback_to_two_layer": bool(fallback_target_repeats == "2"),
                "fallback_to_one_layer": bool(fallback_target_repeats == "1"),
                "fallback_source_layer_repeats": fallback_source_repeats,
                "fallback_target_layer_repeats": fallback_target_repeats,
                "fallback_match_score": "" if fallback_candidate is None else float(fallback_candidate["fallback_match_score"]),
                "fallback_reason": "",
                "canonical_termination_family_id": "",
                "canonical_is_representative": "",
                "canonical_representative_output_name": "",
                "canonical_mapping_reason": "",
                "canonical_family_member_count": "",
                "canonical_representative_accepted_slab_cif": "",
                "canonical_representative_modeling_slab_cif": "",
                "representative_exported": False,
            }
            if fallback_candidate is not None:
                if fallback_target_repeats == fallback_source_repeats and fallback_target_repeats != "":
                    row["fallback_reason"] = "fallback_to_same_facet_same_repeat_termination_due_to_residual_thickness_limit"
                elif fallback_target_repeats == "2":
                    row["fallback_reason"] = "fallback_to_same_facet_two_layer_termination_due_to_three_layer_thickness_limit"
                elif fallback_target_repeats == "1":
                    row["fallback_reason"] = "fallback_to_same_facet_one_layer_termination_due_to_residual_thickness_limit"
            row.update(prefixed_metrics(selected_raw_metrics, "raw_"))
            row["raw_class"] = raw_class
            row["raw_class_reason"] = raw_reason
            row["raw_polar_or_asymmetric"] = bool(selected_raw_metrics["dipole_risk"]) or float(selected_raw_metrics["top_bottom_l1"]) > cfg.asymmetry_b_tol or float(
                selected_raw_metrics["en_polarity_proxy"]
            ) > cfg.en_polarity_b_tol
            row["raw_broken_skeleton"] = float(selected_raw_metrics["coord_loss_mean"]) > cfg.coord_loss_b_tol and float(
                selected_raw_metrics["severe_loss_fraction"]
            ) > cfg.severe_loss_fraction_b_tol
            row["raw_high_roughness"] = float(selected_raw_metrics["surface_roughness_a"]) > cfg.roughness_b_tol

            if cfg.write_all_initial_slabs:
                raw_path = raw_dir / out_name
                write(str(raw_path), AseAtomsAdaptor.get_atoms(selected_slab), format="cif")
                row["raw_slab_cif"] = str(raw_path)
            else:
                row["raw_slab_cif"] = ""

            compensation = {"attempted": False, "candidate_count": 0, "found": False, "best": None}
            if bool(row["raw_polar_or_asymmetric"]):
                compensation = search_compensated_slab(
                    selected_slab,
                    bulk_frac=bulk_frac,
                    bulk_species_cn=bulk_cn,
                    cfg=cfg,
                )
            row["compensation_attempted"] = bool(compensation["attempted"])
            row["compensation_candidate_count"] = int(compensation["candidate_count"])
            row["compensation_found"] = bool(compensation["found"])
            row["compensation_method"] = ""
            row["compensation_improved_top_bottom_l1"] = ""
            row["compensation_improved_coord_loss"] = ""
            row["compensation_slab_cif"] = ""
            row["modeling_slab_cif"] = ""
            row["modeling_supercell_transform"] = ""
            row["modeling_inplane_orth_score"] = ""
            row["modeling_cell_is_orthogonal"] = ""
            row["modeling_alpha_deg"] = ""
            row["modeling_beta_deg"] = ""
            row["modeling_inplane_gamma_deg"] = ""
            row["modeling_atom_count"] = ""
            row["duplicate_termination"] = False
            row["duplicate_of"] = ""

            final_metrics = dict(selected_raw_metrics)
            if bool(compensation["found"]):
                best = compensation["best"]
                assert best is not None
                row["compensation_method"] = str(best["method"])
                final_metrics = dict(best["metrics"])
                selected_slab = best["slab"]
                row["compensation_improved_top_bottom_l1"] = float(selected_raw_metrics["top_bottom_l1"]) - float(final_metrics["top_bottom_l1"])
                row["compensation_improved_coord_loss"] = float(selected_raw_metrics["coord_loss_mean"]) - float(final_metrics["coord_loss_mean"])
                comp_path = compensated_dir / out_name.replace(".cif", "_compensated.cif")
                write(str(comp_path), AseAtomsAdaptor.get_atoms(selected_slab), format="cif")
                row["compensation_slab_cif"] = str(comp_path)

            row.update(final_metrics)
            pre_mlip_class, pre_mlip_reason = final_termination_class(
                final_metrics,
                compensation_found=bool(compensation["found"]),
                allow_thickness_override=bool(allow_thickness_override),
                cfg=cfg,
            )
            row["pre_mlip_class"] = pre_mlip_class
            row["pre_mlip_reason"] = pre_mlip_reason
            row["termination_class"] = compatibility_alias_for_final_class(pre_mlip_class)
            row["termination_class_reason"] = pre_mlip_reason

            keep_after_step2 = not str(pre_mlip_class).startswith("rejected_")
            final_sig = None
            canonical_existing: dict[str, Any] | None = None
            if keep_after_step2:
                final_sig = termination_signature(selected_slab)
                canonical_existing = seen_final_signatures.get(final_sig)
                if canonical_existing is None:
                    canonical_family_counter += 1
                    row["canonical_termination_family_id"] = f"{stem}_cfam{canonical_family_counter:04d}"
                    row["canonical_is_representative"] = True
                    row["canonical_representative_output_name"] = out_name
                    row["canonical_mapping_reason"] = "canonical_representative_for_termination_signature"
                    row["representative_exported"] = True
                else:
                    row["canonical_termination_family_id"] = str(canonical_existing["canonical_termination_family_id"])
                    row["canonical_is_representative"] = False
                    row["canonical_representative_output_name"] = str(canonical_existing["output_name"])
                    row["canonical_mapping_reason"] = "same_termination_signature_as_canonical_representative"
                    row["canonical_representative_accepted_slab_cif"] = str(canonical_existing.get("accepted_slab_cif", ""))
                    row["canonical_representative_modeling_slab_cif"] = str(canonical_existing.get("modeling_slab_cif", ""))
                    row["representative_exported"] = False
            row["step2_keep"] = bool(keep_after_step2)
            row["step2_reason"] = str(row["pre_mlip_reason"])

            if keep_after_step2 and canonical_existing is None:
                accepted_path = accepted_dir / out_name
                write(str(accepted_path), AseAtomsAdaptor.get_atoms(selected_slab), format="cif")
                row["accepted_slab_cif"] = str(accepted_path)
                row["canonical_representative_accepted_slab_cif"] = str(accepted_path)
                if cfg.write_modeling_slabs:
                    modeled, model_meta = modeling_friendly_structure(selected_slab, cfg=cfg)
                    row.update(model_meta)
                    model_path = modeling_dir / out_name.replace(".cif", "_modeling.cif")
                    write(str(model_path), AseAtomsAdaptor.get_atoms(modeled), format="cif")
                    row["modeling_slab_cif"] = str(model_path)
                    row["canonical_representative_modeling_slab_cif"] = str(model_path)
            else:
                row["accepted_slab_cif"] = ""

            if keep_after_step2 and canonical_existing is not None:
                row["accepted_slab_cif"] = ""
                row["modeling_slab_cif"] = ""
                row["mlip_converged"] = ""
                row["mlip_nsteps"] = ""
                row["mlip_energy_init_eV"] = ""
                row["mlip_energy_final_eV"] = ""
                row["mlip_fmax_final"] = ""
                row["mlip_error"] = ""
                row["mlip_relaxed_cif"] = ""
                row["mlip_fixed_atom_count"] = ""
                row["max_displacement_a"] = ""
                row["surface_rmsd_a"] = ""
                row["max_surface_displacement_a"] = ""
                row["interlayer_rel_change"] = ""
                row["cross_layer_reconstruction"] = ""
                row["mlip_screen_pass"] = ""
                row["mlip_screen_reason"] = ""
                row["final_keep"] = bool(canonical_existing["final_keep"])
                row["final_class"] = str(canonical_existing["final_class"])
                row["final_reason"] = f"canonical_member_of:{canonical_existing['output_name']}"
                row["termination_class"] = str(canonical_existing["termination_class"])
                row["termination_class_reason"] = "canonical_member_inherits_representative_termination_class"
            elif cfg.run_mlip and keep_after_step2:
                assert calculator is not None
                row.update(
                    mlip_screen_candidate(
                        selected_slab,
                        cfg=cfg,
                        calculator=calculator,
                        out_dir=relaxed_dir,
                        stem=stem,
                        miller_label=miller_label,
                        termination_index=term_idx,
                    )
                )
                row["final_keep"] = bool(row.get("mlip_screen_pass", False))
                if bool(row["final_keep"]):
                    row["final_class"] = str(row["pre_mlip_class"])
                    row["final_reason"] = str(row.get("mlip_screen_reason", row["pre_mlip_reason"]))
                else:
                    row["final_class"] = "rejected_after_mlip"
                    row["final_reason"] = str(row.get("mlip_screen_reason", "failed_mlip_screen"))
            else:
                row["mlip_converged"] = ""
                row["mlip_nsteps"] = ""
                row["mlip_energy_init_eV"] = ""
                row["mlip_energy_final_eV"] = ""
                row["mlip_fmax_final"] = ""
                row["mlip_error"] = ""
                row["mlip_relaxed_cif"] = ""
                row["mlip_fixed_atom_count"] = ""
                row["max_displacement_a"] = ""
                row["surface_rmsd_a"] = ""
                row["max_surface_displacement_a"] = ""
                row["interlayer_rel_change"] = ""
                row["cross_layer_reconstruction"] = ""
                row["mlip_screen_pass"] = ""
                row["mlip_screen_reason"] = ""
                row["final_keep"] = bool(keep_after_step2)
                row["final_class"] = str(row["pre_mlip_class"])
                row["final_reason"] = str(row["pre_mlip_reason"])
            if keep_after_step2 and canonical_existing is None and final_sig is not None:
                seen_final_signatures[final_sig] = {
                    "canonical_termination_family_id": row["canonical_termination_family_id"],
                    "output_name": out_name,
                    "accepted_slab_cif": row.get("accepted_slab_cif", ""),
                    "modeling_slab_cif": row.get("canonical_representative_modeling_slab_cif", ""),
                    "final_keep": row["final_keep"],
                    "final_class": row["final_class"],
                    "termination_class": row["termination_class"],
                }
            slab_rows.append(row)

    canonical_sizes = Counter(
        str(item["canonical_termination_family_id"])
        for item in slab_rows
        if str(item.get("canonical_termination_family_id", ""))
    )
    for item in slab_rows:
        family_id = str(item.get("canonical_termination_family_id", ""))
        if family_id:
            item["canonical_family_member_count"] = int(canonical_sizes[family_id])

    raw_counter = Counter(item["raw_class"] for item in slab_rows)
    final_counter = Counter(item["final_class"] for item in slab_rows)
    canonical_family_ids = {
        str(item["canonical_termination_family_id"])
        for item in slab_rows
        if str(item.get("canonical_termination_family_id", ""))
    }
    structure_summary = {
        "structure_file": path.name,
        "structure_stem": stem,
        "formula": structure.composition.reduced_formula,
        "distinct_facets_considered": len(millers),
        "kept_facets": sum(1 for item in facet_rows if item["facet_kept"]),
        "termination_candidates": len(slab_rows),
        "raw_good_count": int(raw_counter.get("raw_good", 0)),
        "raw_polar_or_asymmetric_count": int(raw_counter.get("raw_polar_or_asymmetric", 0)),
        "raw_broken_skeleton_count": int(raw_counter.get("raw_broken_skeleton", 0)),
        "raw_broken_skeleton_and_polar_count": int(raw_counter.get("raw_broken_skeleton_and_polar", 0)),
        "compensated_surface_candidate_count": int(final_counter.get("compensated_surface_candidate", 0)),
        "compensated_polar_surface_candidate_count": int(final_counter.get("compensated_polar_surface_candidate", 0)),
        "direct_surface_candidate_count": int(final_counter.get("direct_surface_candidate", 0)),
        "polar_surface_candidate_count": int(final_counter.get("polar_surface_candidate", 0)),
        "rejected_broken_skeleton_count": int(final_counter.get("rejected_broken_skeleton", 0)),
        "rejected_too_thick_count": int(final_counter.get("rejected_too_thick", 0)),
        "rejected_high_roughness_count": int(final_counter.get("rejected_high_roughness", 0)),
        "rejected_after_mlip_count": int(final_counter.get("rejected_after_mlip", 0)),
        "step2_kept": sum(1 for item in slab_rows if item["step2_keep"]),
        "final_kept": sum(1 for item in slab_rows if item["final_keep"]),
        "canonical_family_count": len(canonical_family_ids),
        "canonical_representative_count": sum(1 for item in slab_rows if item.get("canonical_is_representative") is True),
        "canonical_member_count": sum(1 for item in slab_rows if item.get("canonical_is_representative") is False),
    }
    return facet_rows, slab_rows, [structure_summary]


def run_surface_screen(cfg: SurfaceScreenConfig) -> dict[str, Any]:
    input_files = discover_cif_files(cfg.input_dir, recursive=cfg.recursive)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    calculator = None
    if cfg.run_mlip:
        calculator = build_omat24_calculator(
            model=cfg.ml_model,
            checkpoint=cfg.ml_checkpoint,
            device=cfg.ml_device,
            quiet=True,
        )

    facet_rows: list[dict[str, Any]] = []
    slab_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in input_files:
        try:
            f_rows, s_rows, st_rows = facet_and_slab_records_for_structure(path, cfg=cfg, calculator=calculator)
            facet_rows.extend(f_rows)
            slab_rows.extend(s_rows)
            structure_rows.extend(st_rows)
        except Exception as exc:
            failures.append(
                {
                    "structure_file": path.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    write_csv(cfg.out_dir / "facet_screening.csv", facet_rows)
    write_csv(cfg.out_dir / "termination_screening.csv", slab_rows)
    write_csv(cfg.out_dir / "structure_summary.csv", structure_rows)
    write_csv(cfg.out_dir / "failures.csv", failures)

    raw_counter = Counter(str(row.get("raw_class", "")) for row in slab_rows)
    final_counter = Counter(str(row.get("final_class", "")) for row in slab_rows)
    canonical_family_ids = {
        str(row["canonical_termination_family_id"])
        for row in slab_rows
        if str(row.get("canonical_termination_family_id", ""))
    }
    summary = {
        "input_dir": str(cfg.input_dir),
        "out_dir": str(cfg.out_dir),
        "n_input_cifs": len(input_files),
        "n_failed_structures": len(failures),
        "n_kept_facets": sum(1 for row in facet_rows if bool(row.get("facet_kept"))),
        "n_termination_candidates": len(slab_rows),
        "raw_class_counts": dict(sorted(raw_counter.items())),
        "final_class_counts": dict(sorted(final_counter.items())),
        "n_compat_class_a": sum(1 for row in slab_rows if row.get("termination_class") == "A"),
        "n_compat_class_b": sum(1 for row in slab_rows if row.get("termination_class") == "B"),
        "n_compat_class_c": sum(1 for row in slab_rows if row.get("termination_class") == "C"),
        "n_after_step2": sum(1 for row in slab_rows if bool(row.get("step2_keep"))),
        "n_final_kept": sum(1 for row in slab_rows if bool(row.get("final_keep"))),
        "n_canonical_families": len(canonical_family_ids),
        "n_canonical_representatives": sum(1 for row in slab_rows if row.get("canonical_is_representative") is True),
        "n_canonical_members": sum(1 for row in slab_rows if row.get("canonical_is_representative") is False),
        "n_compensation_attempted": sum(1 for row in slab_rows if bool(row.get("compensation_attempted"))),
        "n_compensation_found": sum(1 for row in slab_rows if bool(row.get("compensation_found"))),
        "config": json_ready(asdict(cfg)),
    }
    (cfg.out_dir / "surface_screen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layered surface/facet screening for CIF files in the current directory."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing CIF files (default: current directory).")
    parser.add_argument("--out-dir", type=Path, default=Path("output/surface_screen"), help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for CIF files.")
    parser.add_argument("--max-index", type=int, default=2, help="Maximum absolute Miller index to enumerate.")
    parser.add_argument("--min-d-hkl", type=float, default=1.2, help="Minimum allowed interplanar spacing d_hkl in Angstrom.")
    parser.add_argument("--min-slab-size", type=float, default=12.0, help="Minimum slab thickness in Angstrom.")
    parser.add_argument("--min-vacuum-size", type=float, default=15.0, help="Minimum vacuum thickness in Angstrom.")
    parser.add_argument("--max-slab-thickness-a", type=float, default=15.0, help="Maximum allowed physical slab thickness in Angstrom for any retained termination.")
    parser.add_argument("--max-surface-area", type=float, default=160.0, help="Maximum surface unit-cell area in Angstrom^2.")
    parser.add_argument("--min-surface-area", type=float, default=1.0, help="Minimum surface unit-cell area in Angstrom^2.")
    parser.add_argument("--layer-tol-a", type=float, default=0.45, help="Layer clustering tolerance along the surface normal in Angstrom.")
    parser.add_argument("--max-terminations-per-facet", type=int, default=0, help="Optional hard cap per facet; 0 keeps all inequivalent terminations.")
    parser.add_argument("--write-all-initial-slabs", action="store_true", help="Write every enumerated termination CIF before screening.")
    parser.add_argument("--max-compensated-slab-thickness-a", type=float, default=15.0, help="Maximum slab thickness allowed for automatically compensated slabs in Angstrom.")
    parser.add_argument("--no-modeling-slabs", action="store_true", help="Do not export rotated/modeling-friendly slab CIFs.")
    parser.add_argument("--no-inplane-rectangularize", action="store_true", help="Disable integer in-plane supercell search for more orthogonal modeling cells.")
    parser.add_argument("--max-inplane-area-multiplier", type=int, default=4, help="Maximum area multiplier when searching in-plane rectangular supercells.")
    parser.add_argument("--orthogonal-angle-tol-deg", type=float, default=2.0, help="Tolerance used to label a modeling cell as orthogonal.")
    parser.add_argument("--run-mlip", action="store_true", help="Run optional MLIP pre-relaxation screen on Class A/B terminations.")
    parser.add_argument("--ml-model", type=str, default="eSEN-30M-MPtrj", help="MLIP model alias.")
    parser.add_argument("--ml-checkpoint", type=str, default=None, help="Optional explicit MLIP checkpoint path.")
    parser.add_argument("--ml-device", type=str, default="auto", help="MLIP device: auto, cpu, cuda.")
    parser.add_argument("--ml-optimizer", type=str, default="LBFGS", help="ASE optimizer for MLIP pre-relaxation.")
    parser.add_argument("--ml-fmax", type=float, default=0.08, help="MLIP relaxation force threshold.")
    parser.add_argument("--ml-steps", type=int, default=120, help="Maximum MLIP relaxation steps.")
    parser.add_argument("--ml-maxstep", type=float, default=0.04, help="ASE optimizer maxstep in Angstrom.")
    parser.add_argument("--relax-surface-layers", type=int, default=2, help="How many outer layers per side remain mobile during MLIP relaxation.")
    parser.add_argument("--max-displacement-a", type=float, default=1.0, help="Reject relaxed slabs exceeding this maximum atomic displacement.")
    parser.add_argument("--surface-rmsd-a", type=float, default=0.6, help="Reject relaxed slabs exceeding this surface-layer RMSD.")
    parser.add_argument("--interlayer-rel-change", type=float, default=0.35, help="Reject relaxed slabs exceeding this relative interlayer-spacing change.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = SurfaceScreenConfig(
        input_dir=args.input_dir.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        recursive=bool(args.recursive),
        max_index=int(args.max_index),
        min_d_hkl=float(args.min_d_hkl),
        min_slab_size=float(args.min_slab_size),
        min_vacuum_size=float(args.min_vacuum_size),
        max_slab_thickness_a=float(args.max_slab_thickness_a),
        max_surface_area=float(args.max_surface_area),
        min_surface_area=float(args.min_surface_area),
        layer_tol_a=float(args.layer_tol_a),
        max_terminations_per_facet=int(args.max_terminations_per_facet),
        write_all_initial_slabs=bool(args.write_all_initial_slabs),
        max_compensated_slab_thickness_a=float(args.max_compensated_slab_thickness_a),
        write_modeling_slabs=not bool(args.no_modeling_slabs),
        inplane_rectangularize=not bool(args.no_inplane_rectangularize),
        max_inplane_area_multiplier=int(args.max_inplane_area_multiplier),
        orthogonal_angle_tol_deg=float(args.orthogonal_angle_tol_deg),
        run_mlip=bool(args.run_mlip),
        ml_model=str(args.ml_model),
        ml_checkpoint=args.ml_checkpoint,
        ml_device=str(args.ml_device),
        ml_optimizer=str(args.ml_optimizer),
        ml_fmax=float(args.ml_fmax),
        ml_steps=int(args.ml_steps),
        ml_maxstep=float(args.ml_maxstep),
        relax_surface_layers=int(args.relax_surface_layers),
        max_displacement_a=float(args.max_displacement_a),
        surface_rmsd_a=float(args.surface_rmsd_a),
        interlayer_rel_change=float(args.interlayer_rel_change),
    )
    summary = run_surface_screen(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
