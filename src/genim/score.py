from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ase.io import read
from tqdm import tqdm

from .composition import atoms_counts, reduced_counts, reduced_formula
from .config import load_config
from .hull import build_or_load_hull_references, compute_e_above_hull_eV_per_atom
from .ml_relax import BulkRelaxSettings, relax_bulk_atoms, relax_settings_from_cfg
from .mlip import build_omat24_calculator, resolve_device, resolve_omat24_checkpoint


@dataclass
class ScoreRow:
    sid: str
    chemsys: str
    formula_reduced: str
    composition: str
    composition_ratio: str
    composition_percent: str
    composition_counts: str
    natoms: int
    n_elements: int

    energy_eV: Optional[float]
    energy_per_atom_eV: Optional[float]

    e_above_hull_eV_per_atom: Optional[float]
    stable_200meV: Optional[bool]

    converged: bool
    used_cell_relax: bool
    nsteps: int
    fmax_final: Optional[float]
    wall_time_s: Optional[float]

    cif_in: str
    cif_relaxed: str
    error: str


def _chemsys_from_elements(elements: Sequence[str]) -> str:
    return "-".join(sorted({str(e).strip() for e in elements if str(e).strip()}))


def _find_cifs(cif_dir: Path) -> List[Path]:
    cif_dir = Path(cif_dir).resolve()
    cands = list(cif_dir.glob("*.cif")) + list(cif_dir.glob("*.CIF"))
    uniq: Dict[str, Path] = {}
    for p in cands:
        if not p.is_file():
            continue
        try:
            k = str(p.resolve())
        except Exception:
            k = str(p)
        uniq.setdefault(k, p)
    files = list(uniq.values())
    files.sort(key=lambda p: p.name)
    return files


def _composition_display_fields(counts: Mapping[str, Any]) -> tuple[str, str, str]:
    clean: Dict[str, int] = {}
    for k, v in counts.items():
        try:
            n = int(v)
        except Exception:
            continue
        if n <= 0:
            continue
        clean[str(k)] = n

    if not clean:
        return "", "", ""

    total = int(sum(clean.values()))
    red = reduced_counts(clean)
    elems = sorted(clean)

    ratio = f"{':'.join(elems)} = {':'.join(str(int(red[e])) for e in elems)}" if red else ""
    percent = "; ".join(f"{e}={100.0 * float(clean[e]) / float(total):.4f}%" for e in elems) if total > 0 else ""
    counts_txt = "; ".join(f"{e}={int(clean[e])}" for e in elems)
    return ratio, percent, counts_txt


def _cfg_get(cfg: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(k, None)
    return default if cur is None else cur


def score_cif_dir(
    *,
    conf_path: Path,
    cif_dir: Path,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Relax CIFs with an MLIP (eSEN/OMAT24 via fairchem-core), compute ML-based Energy Above Hull,
    and write a CSV summary.

    Returns the CSV path.
    """
    cfg = load_config(Path(conf_path))
    cif_dir = Path(cif_dir).expanduser().resolve()
    if not cif_dir.is_dir():
        raise FileNotFoundError(cif_dir)

    score_sec = cfg.get("score", {}) if isinstance(cfg, dict) else {}
    if not isinstance(score_sec, dict):
        score_sec = {}

    if out_dir is None:
        out0 = score_sec.get("out_dir", None)
        out_dir = Path(out0).expanduser().resolve() if isinstance(out0, str) and out0.strip() else (cif_dir / "ml_score").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    relaxed_subdir0 = score_sec.get("relaxed_subdir", None)
    if not (isinstance(relaxed_subdir0, str) and relaxed_subdir0.strip()):
        relaxed_subdir0 = _cfg_get(cfg, ("relax", "relaxed_subdir"), "relaxed")
    relaxed_subdir = str(relaxed_subdir0 or "relaxed").strip() or "relaxed"
    relaxed_dir = (out_dir / relaxed_subdir).resolve()
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    csv_name = str(score_sec.get("csv_name", "score.csv") or "score.csv").strip() or "score.csv"
    csv_path = (out_dir / csv_name).resolve()

    # --- ML model ---
    ml_sec = cfg.get("ml", {}) if isinstance(cfg, dict) else {}
    if not isinstance(ml_sec, dict):
        ml_sec = {}

    ml_model = str(ml_sec.get("model", "eSEN-30M-OAM") or "eSEN-30M-OAM").strip()
    ml_checkpoint = ml_sec.get("checkpoint", None)
    ml_device = str(ml_sec.get("device", "auto") or "auto").strip()
    ml_quiet = bool(ml_sec.get("quiet", True))

    resolved_device = resolve_device(ml_device)
    ckpt_path = resolve_omat24_checkpoint(model=ml_model, checkpoint=str(ml_checkpoint) if ml_checkpoint else None)
    calc = build_omat24_calculator(model=ml_model, device=resolved_device, checkpoint=str(ckpt_path), quiet=ml_quiet)

    model_meta = {"model": ml_model, "checkpoint": str(ckpt_path), "device": resolved_device}

    # --- Relax settings (generated structures) ---
    relax_enabled = bool(_cfg_get(cfg, ("relax", "enabled"), True))
    relax_write_cif = bool(_cfg_get(cfg, ("relax", "write_relaxed_cif"), True))
    relax_settings: BulkRelaxSettings = relax_settings_from_cfg(cfg, prefix="relax")

    # --- Hull settings ---
    hull_enabled = bool(_cfg_get(cfg, ("hull", "enabled"), True))
    stable_thr = float(_cfg_get(cfg, ("hull", "stable_threshold_eV_per_atom"), 0.2))

    # --- Relax candidates ---
    cifs = _find_cifs(cif_dir)
    if not cifs:
        raise FileNotFoundError(f"No CIFs found under: {cif_dir}")

    cand_rows: List[ScoreRow] = []
    cand_by_chemsys: Dict[str, List[int]] = {}

    for p in tqdm(cifs, desc=f"Relax {cif_dir.name}", unit="cif"):
        sid = p.stem
        try:
            atoms = read(str(p))
        except Exception as exc:
            row = ScoreRow(
                sid=sid,
                chemsys="",
                formula_reduced="",
                composition="{}",
                composition_ratio="",
                composition_percent="",
                composition_counts="",
                natoms=0,
                n_elements=0,
                energy_eV=None,
                energy_per_atom_eV=None,
                e_above_hull_eV_per_atom=None,
                stable_200meV=None,
                converged=False,
                used_cell_relax=False,
                nsteps=0,
                fmax_final=None,
                wall_time_s=None,
                cif_in=str(p),
                cif_relaxed="",
                error=f"read_failed: {type(exc).__name__}: {exc}",
            )
            cand_rows.append(row)
            continue

        # Composition / chemsys
        try:
            counts = atoms_counts(atoms)
            red = reduced_counts(counts)
            formula = reduced_formula(red)
            composition = json.dumps(red, ensure_ascii=False, sort_keys=True)
            composition_ratio, composition_percent, composition_counts = _composition_display_fields(counts)
            elems = list(red.keys())
            chemsys = _chemsys_from_elements(elems)
        except Exception:
            counts = {}
            red = {}
            formula = ""
            composition = "{}"
            composition_ratio = ""
            composition_percent = ""
            composition_counts = ""
            elems = []
            chemsys = ""

        energy_eV = None
        cif_relaxed = ""
        err = ""
        conv = False
        used_cell = False
        nsteps = 0
        fmax_final = None
        wall_s = None

        if relax_enabled:
            r, atoms_rel = relax_bulk_atoms(
                atoms,
                calc=calc,
                settings=relax_settings,
                out_dir=relaxed_dir if relax_write_cif else None,
                relaxed_cif_name=f"{sid}.cif" if relax_write_cif else None,
                label=sid,
            )
            energy_eV = r.energy_final_eV if r.energy_final_eV is not None else r.energy_init_eV
            cif_relaxed = r.relaxed_cif
            err = str(r.error or "")
            conv = bool(r.converged)
            used_cell = bool(r.used_cell_relax)
            nsteps = int(r.nsteps or 0)
            fmax_final = r.fmax_final
            wall_s = r.wall_time_s
        else:
            try:
                atoms.calc = calc
                energy_eV = float(atoms.get_potential_energy())
            except Exception as exc:
                err = f"energy_failed: {type(exc).__name__}: {exc}"
                energy_eV = None

        natoms = int(len(atoms)) if hasattr(atoms, "__len__") else 0
        epa = (float(energy_eV) / float(natoms)) if (energy_eV is not None and natoms > 0) else None

        row = ScoreRow(
            sid=sid,
            chemsys=chemsys,
            formula_reduced=formula,
            composition=composition,
            composition_ratio=composition_ratio,
            composition_percent=composition_percent,
            composition_counts=composition_counts,
            natoms=natoms,
            n_elements=int(len(elems)),
            energy_eV=float(energy_eV) if energy_eV is not None else None,
            energy_per_atom_eV=float(epa) if epa is not None else None,
            e_above_hull_eV_per_atom=None,
            stable_200meV=None,
            converged=conv,
            used_cell_relax=used_cell,
            nsteps=nsteps,
            fmax_final=fmax_final,
            wall_time_s=wall_s,
            cif_in=str(p),
            cif_relaxed=str(cif_relaxed or ""),
            error=err,
        )
        idx = len(cand_rows)
        cand_rows.append(row)

        if chemsys and (epa is not None):
            cand_by_chemsys.setdefault(chemsys, []).append(idx)

    # --- Hull scoring ---
    if hull_enabled:
        for chemsys, idxs in tqdm(list(cand_by_chemsys.items()), desc="Hull", unit="system"):
            elements = chemsys.split("-")
            try:
                refs = build_or_load_hull_references(elements=list(elements), model_meta=model_meta, calc=calc, conf=cfg)
            except Exception as exc:
                for i in idxs:
                    cand_rows[i].error = (cand_rows[i].error + "; " if cand_rows[i].error else "") + f"hull_refs_failed: {type(exc).__name__}: {exc}"
                continue

            for i in idxs:
                row = cand_rows[i]
                if row.energy_per_atom_eV is None:
                    continue
                try:
                    comp_counts = json.loads(row.composition) if row.composition else {}
                except Exception:
                    comp_counts = {}
                cand = {"composition": comp_counts, "energy_per_atom_eV": float(row.energy_per_atom_eV)}
                eah = compute_e_above_hull_eV_per_atom(candidate=cand, references=refs)
                row.e_above_hull_eV_per_atom = float(eah) if eah is not None else None
                if eah is not None:
                    row.stable_200meV = bool(float(eah) <= float(stable_thr) + 1e-12)
                else:
                    row.stable_200meV = None

    # --- Write CSV ---
    fields = list(asdict(cand_rows[0]).keys()) if cand_rows else []
    out_csv = csv_path
    last_exc: Exception | None = None
    for k in range(0, 100):
        try:
            with out_csv.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                for r in cand_rows:
                    w.writerow(asdict(r))
            last_exc = None
            break
        except PermissionError as exc:
            last_exc = exc
            # Common on Windows if the user opened the CSV in Excel.
            if k == 0:
                out_csv = csv_path.with_name(f"{csv_path.stem}_1{csv_path.suffix}")
            else:
                out_csv = csv_path.with_name(f"{csv_path.stem}_{k+1}{csv_path.suffix}")

    if last_exc is not None:
        raise last_exc

    return out_csv
