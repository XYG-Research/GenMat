from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ase import Atoms
from tqdm import tqdm

from .composition import atoms_counts, composition_key
from .ml_relax import BulkRelaxSettings, relax_bulk_atoms
from .mp_download import _iter_mp_summary, _mp_session
from .structure_format import parse_pymatgen_mson_structure


@dataclass(frozen=True)
class HullRefConfig:
    cache_dir: Path
    force_rebuild: bool

    mp_max_atoms: int
    mp_eah_max: float
    mp_limit_per_subset: int
    mp_per_page: int
    mp_timeout: float
    mp_max_structures_per_composition: int

    relax_enabled: bool
    relax_settings: BulkRelaxSettings


def _chemsys_key(elements: Iterable[str]) -> str:
    return "-".join(sorted({str(e).strip() for e in elements if str(e).strip()}))


def _iter_element_subsets(elements: List[str]) -> Iterable[Tuple[str, ...]]:
    elems = sorted({str(e).strip() for e in elements if str(e).strip()})
    for r in range(1, len(elems) + 1):
        for comb in itertools.combinations(elems, r):
            yield tuple(comb)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).strip())
        except Exception:
            return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _meta_matches(meta: Any, want: Mapping[str, Any]) -> bool:
    if not isinstance(meta, Mapping):
        return False
    try:
        return dict(meta) == dict(want)
    except Exception:
        return False


def hull_ref_cfg_from_conf(conf: Mapping[str, Any]) -> HullRefConfig:
    hull = conf.get("hull", {}) if isinstance(conf, Mapping) else {}
    if not isinstance(hull, Mapping):
        hull = {}

    cache_dir = Path(str(hull.get("cache_dir", "output/hull_cache"))).expanduser().resolve()
    force = bool(hull.get("force_rebuild", False))

    mp = hull.get("mp", {}) if isinstance(hull.get("mp", {}), Mapping) else {}
    mp_max_atoms = int(mp.get("max_atoms", 80))
    mp_eah_max = float(mp.get("eah_max", 0.5))
    mp_limit_per_subset = int(mp.get("limit_per_subset", 2000))
    mp_per_page = int(mp.get("per_page", 200))
    mp_timeout = float(mp.get("timeout", 60.0))
    mp_max_structures_per_composition = int(mp.get("max_structures_per_composition", 2))

    relax = hull.get("relax", {}) if isinstance(hull.get("relax", {}), Mapping) else {}
    relax_enabled = bool(relax.get("enabled", True))

    # Reuse BulkRelaxSettings schema under hull.relax.
    relax_settings = BulkRelaxSettings(
        optimizer=str(relax.get("optimizer", "LBFGS")),
        fmax=float(relax.get("fmax", 0.05)),
        steps=int(relax.get("steps", 120)),
        maxstep=relax.get("maxstep", 0.05),
        relax_cell=bool(relax.get("relax_cell", True)),
        cell_filter=str(relax.get("cell_filter", "expcell")),
        hydrostatic_strain=bool(relax.get("hydrostatic_strain", False)),
        constant_volume=bool(relax.get("constant_volume", False)),
        scalar_pressure_GPa=float(relax.get("scalar_pressure_GPa", 0.0)),
        logfile_name=None,
    )

    return HullRefConfig(
        cache_dir=cache_dir,
        force_rebuild=force,
        mp_max_atoms=mp_max_atoms,
        mp_eah_max=mp_eah_max,
        mp_limit_per_subset=mp_limit_per_subset,
        mp_per_page=mp_per_page,
        mp_timeout=mp_timeout,
        mp_max_structures_per_composition=mp_max_structures_per_composition,
        relax_enabled=relax_enabled,
        relax_settings=relax_settings,
    )


def build_or_load_hull_references(
    *,
    elements: List[str],
    model_meta: Mapping[str, Any],
    calc: Any,
    conf: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build (or load cached) MLIP reference energies for a chemical system and all subsystems.

    Returns a list of dict entries:
      {"composition": {el:int}, "energy_per_atom_eV": float, "source": "...", ...}
    """
    cfg = hull_ref_cfg_from_conf(conf)
    key = _chemsys_key(elements)
    if not key:
        raise ValueError("Empty chemical system elements.")

    sys_dir = (cfg.cache_dir / key).resolve()
    meta_path = sys_dir / "meta.json"
    refs_path = sys_dir / "refs.jsonl"

    want_meta = {
        "chemsys": key,
        "model": dict(model_meta),
        "mp": {
            "max_atoms": int(cfg.mp_max_atoms),
            "eah_max": float(cfg.mp_eah_max),
            "limit_per_subset": int(cfg.mp_limit_per_subset),
            "per_page": int(cfg.mp_per_page),
            "timeout": float(cfg.mp_timeout),
            "max_structures_per_composition": int(cfg.mp_max_structures_per_composition),
        },
        "relax": {"enabled": bool(cfg.relax_enabled), **cfg.relax_settings.__dict__},
    }

    if (not cfg.force_rebuild) and refs_path.exists() and meta_path.exists():
        meta = _load_json(meta_path)
        if _meta_matches(meta, want_meta):
            out: List[Dict[str, Any]] = []
            for line in refs_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    out.append(row)
            if out:
                return out

    # Build fresh.
    sys_dir.mkdir(parents=True, exist_ok=True)

    session = _mp_session(cfg.mp_timeout)

    try:
        from pymatgen.core.composition import Composition  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Hull reference selection requires `pymatgen` (install pymatgen).") from exc

    max_per_comp = int(cfg.mp_max_structures_per_composition)
    if max_per_comp < 1:
        raise ValueError("hull.mp.max_structures_per_composition must be >= 1")

    # First pass: select a small number of candidate structures per reduced composition
    # using MP's DFT energy_above_hull as a proxy (cheap), then evaluate ML energies only
    # for those candidates (expensive).
    cand_by_comp: Dict[Tuple[Tuple[str, int], ...], List[Tuple[Tuple[float, int], Dict[str, Any]]]] = {}

    best_by_comp: Dict[Tuple[Tuple[str, int], ...], Dict[str, Any]] = {}

    subsets = list(_iter_element_subsets(list(elements)))
    for subset in subsets:
        chemsys = _chemsys_key(subset)
        if not chemsys:
            continue

        params: Dict[str, Any] = {
            "_fields": ",".join(
                [
                    "material_id",
                    "formula_pretty",
                    "nsites",
                    "elements",
                    "energy_above_hull",
                    "structure",
                ]
            ),
            "chemsys": chemsys,
            "nsites_max": int(cfg.mp_max_atoms),
            "energy_above_hull_max": float(cfg.mp_eah_max),
            "nelements_min": int(len(subset)),
            "nelements_max": int(len(subset)),
        }

        it = _iter_mp_summary(
            session=session,
            params=params,
            limit=int(cfg.mp_limit_per_subset),
            per_page=int(cfg.mp_per_page),
        )

        for doc in tqdm(it, total=int(cfg.mp_limit_per_subset), desc=f"MP refs {chemsys}", leave=False):
            # Determine reduced composition key (prefer formula_pretty; fall back to structure).
            key_comp: Optional[Tuple[Tuple[str, int], ...]] = None
            try:
                fp = doc.get("formula_pretty", None)
                if isinstance(fp, str) and fp.strip():
                    comp = Composition(fp)
                    red = comp.reduced_composition.as_dict()
                    red_i = {str(k): int(round(float(v))) for k, v in red.items() if float(v) > 0}
                    key_comp = composition_key(red_i)
            except Exception:
                key_comp = None

            if key_comp is None:
                try:
                    struct_obj = doc.get("structure", None)
                    if not isinstance(struct_obj, dict):
                        continue
                    rec = parse_pymatgen_mson_structure(struct_obj)
                    key_comp = composition_key(atoms_counts(rec.to_ase()))
                except Exception:
                    continue

            eah = _safe_float(doc.get("energy_above_hull", None))
            nsites = int(doc.get("nsites", 0) or 0)
            rank = (float(eah) if eah is not None else 1e9, int(nsites) if nsites > 0 else 10**9)

            lst = cand_by_comp.setdefault(key_comp, [])
            d0 = dict(doc)
            d0["mp_chemsys"] = chemsys
            lst.append((rank, d0))
            lst.sort(key=lambda x: x[0])
            if len(lst) > max_per_comp:
                del lst[max_per_comp:]

    # Second pass: evaluate ML energies for the selected candidates.
    for key_comp, docs in tqdm(list(cand_by_comp.items()), desc=f"ML refs {key}", unit="comp"):
        red = dict(key_comp)
        for _rank, doc in docs:
            try:
                struct_obj = doc.get("structure", None)
                if not isinstance(struct_obj, dict):
                    continue
                rec = parse_pymatgen_mson_structure(struct_obj)
                atoms0 = rec.to_ase()
            except Exception:
                continue

            energy_eV = None
            if cfg.relax_enabled:
                r, _atoms_rel = relax_bulk_atoms(
                    atoms0,
                    calc=calc,
                    settings=cfg.relax_settings,
                    out_dir=None,
                    relaxed_cif_name=None,
                    label=str(doc.get("material_id", "") or ""),
                )
                energy_eV = _safe_float(r.energy_final_eV) or _safe_float(r.energy_init_eV)
                atoms_use = _atoms_rel if _atoms_rel is not None else atoms0
            else:
                try:
                    atoms0.calc = calc
                    energy_eV = float(atoms0.get_potential_energy())
                except Exception:
                    energy_eV = None
                atoms_use = atoms0

            if energy_eV is None:
                continue

            try:
                n = int(len(atoms_use))
                if n <= 0:
                    continue
                epa = float(energy_eV) / float(n)
            except Exception:
                continue

            prev = best_by_comp.get(key_comp, None)
            if prev is None or float(epa) < float(prev["energy_per_atom_eV"]):
                best_by_comp[key_comp] = {
                    "composition": dict(red),
                    "energy_per_atom_eV": float(epa),
                    "source": "mp",
                    "material_id": doc.get("material_id", None),
                    "formula_pretty": doc.get("formula_pretty", None),
                    "mp_energy_above_hull": doc.get("energy_above_hull", None),
                    "mp_chemsys": str(doc.get("mp_chemsys", "")) or "",
                }

    refs = list(best_by_comp.values())

    # Persist
    _write_json(meta_path, want_meta)
    try:
        refs_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in refs) + "\n", encoding="utf-8")
    except Exception:
        pass

    return refs


def compute_e_above_hull_eV_per_atom(
    *,
    candidate: Mapping[str, Any],
    references: List[Mapping[str, Any]],
) -> Optional[float]:
    """
    Compute Energy Above Hull (eV/atom) for a candidate entry against a set of reference entries.

    candidate requires:
      - composition: {el:int}
      - energy_per_atom_eV: float
    references entries require the same fields.
    """
    try:
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram  # type: ignore
        from pymatgen.core.composition import Composition  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Energy Above Hull requires `pymatgen` (install pymatgen).") from exc

    def _to_entry(row: Mapping[str, Any]) -> Optional[PDEntry]:
        comp0 = row.get("composition", None)
        epa0 = _safe_float(row.get("energy_per_atom_eV", None))
        if not isinstance(comp0, Mapping) or epa0 is None:
            return None
        comp = Composition({str(k): float(v) for k, v in comp0.items() if _safe_float(v) and float(v) > 0})
        if comp.num_atoms <= 0:
            return None
        energy_total = float(epa0) * float(comp.num_atoms)
        name = str(row.get("material_id", row.get("source", "")) or "")
        return PDEntry(comp, float(energy_total), name=name)

    ref_entries: List[PDEntry] = []
    for r in references:
        e = _to_entry(r)
        if e is not None:
            ref_entries.append(e)
    if not ref_entries:
        return None

    cand_entry = _to_entry(candidate)
    if cand_entry is None:
        return None

    try:
        pd = PhaseDiagram(ref_entries)
        eah = pd.get_e_above_hull(cand_entry, allow_negative=True, check_stable=False, on_error="ignore")
    except Exception:
        return None
    if eah is None:
        return None

    # Energy above hull should be >= 0; allow tiny negatives due to numerical noise.
    try:
        x = float(eah)
    except Exception:
        return None
    if x < 0 and abs(x) < 1e-6:
        x = 0.0
    return max(0.0, float(x))
