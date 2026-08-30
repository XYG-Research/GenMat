from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ase import Atoms, units
from ase.filters import ExpCellFilter, UnitCellFilter
from ase.io import write
from ase.optimize import BFGS, FIRE, LBFGS, MDMin


@dataclass(frozen=True)
class BulkRelaxSettings:
    """
    Bulk (periodic) relaxation settings for ASE optimizers.

    Notes:
      - Supports optional cell relaxation via ExpCellFilter/UnitCellFilter (requires stress).
      - Energies are in eV, forces in eV/Å.
    """

    optimizer: str = "LBFGS"  # FIRE | BFGS | LBFGS | MDMin
    fmax: float = 0.05
    steps: int = 300
    maxstep: Optional[float] = 0.05

    relax_cell: bool = True
    cell_filter: str = "expcell"  # expcell | unitcell | none
    hydrostatic_strain: bool = False
    constant_volume: bool = False
    scalar_pressure_GPa: float = 0.0

    # Output controls (optional)
    logfile_name: Optional[str] = None


@dataclass
class BulkRelaxResult:
    label: str = ""
    converged: bool = False
    used_cell_relax: bool = False
    nsteps: int = 0
    fmax_final: Optional[float] = None
    energy_init_eV: Optional[float] = None
    energy_final_eV: Optional[float] = None
    wall_time_s: Optional[float] = None
    error: str = ""
    relaxed_cif: str = ""


def _get_optimizer(name: str):
    key = str(name or "").strip().upper()
    table = {
        "FIRE": FIRE,
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "MDMIN": MDMin,
    }
    if key not in table:
        raise ValueError("optimizer must be one of: FIRE, BFGS, LBFGS, MDMin")
    return table[key]


def _build_cell_filter(atoms: Atoms, *, settings: BulkRelaxSettings) -> Tuple[Any, bool]:
    mode = str(settings.cell_filter or "expcell").strip().lower() or "expcell"
    if not bool(settings.relax_cell) or mode == "none":
        return atoms, False

    scalar_pressure = float(settings.scalar_pressure_GPa) * float(units.GPa)
    common_kwargs: Dict[str, Any] = {
        "hydrostatic_strain": bool(settings.hydrostatic_strain),
        "constant_volume": bool(settings.constant_volume),
        "scalar_pressure": float(scalar_pressure),
    }

    if mode == "expcell":
        return ExpCellFilter(atoms, **common_kwargs), True
    if mode == "unitcell":
        return UnitCellFilter(atoms, **common_kwargs), True
    raise ValueError("cell_filter must be one of: expcell, unitcell, none")


def relax_bulk_atoms(
    atoms_in: Atoms,
    *,
    calc: Any,
    settings: BulkRelaxSettings,
    out_dir: Optional[Path] = None,
    relaxed_cif_name: Optional[str] = None,
    label: str = "",
) -> Tuple[BulkRelaxResult, Optional[Atoms]]:
    """
    Relax a periodic bulk structure using an ASE calculator (e.g., fairchem OMAT24 eSEN).

    Returns (result, relaxed_atoms). relaxed_atoms is None on failure.
    """
    res = BulkRelaxResult(label=str(label or "").strip())
    t0 = time.perf_counter()

    try:
        atoms = atoms_in.copy()
    except Exception:
        atoms = atoms_in

    try:
        atoms.set_pbc(True)
    except Exception:
        pass

    try:
        atoms.calc = calc
    except Exception as exc:
        res.error = f"attach_calc_failed: {type(exc).__name__}: {exc}"
        res.wall_time_s = float(time.perf_counter() - t0)
        return res, None

    try:
        res.energy_init_eV = float(atoms.get_potential_energy())
    except Exception:
        res.energy_init_eV = None

    try:
        target, used_cell_relax = _build_cell_filter(atoms, settings=settings)
        res.used_cell_relax = bool(used_cell_relax)
    except Exception as exc:
        res.error = f"cell_filter_failed: {type(exc).__name__}: {exc}"
        res.wall_time_s = float(time.perf_counter() - t0)
        return res, None

    opt_cls = _get_optimizer(settings.optimizer)

    init_kwargs: Dict[str, Any] = {}
    if settings.logfile_name and out_dir is not None:
        try:
            out_dir = Path(out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            init_kwargs["logfile"] = str(out_dir / str(settings.logfile_name))
        except Exception:
            init_kwargs["logfile"] = None
    else:
        init_kwargs["logfile"] = None

    # Pass maxstep only when supported by the optimizer.
    if settings.maxstep is not None:
        try:
            import inspect

            if "maxstep" in inspect.signature(opt_cls.__init__).parameters:
                init_kwargs["maxstep"] = float(settings.maxstep)
        except Exception:
            pass

    try:
        dyn = opt_cls(target, **init_kwargs)
    except Exception as exc:
        res.error = f"optimizer_init_failed: {type(exc).__name__}: {exc}"
        res.wall_time_s = float(time.perf_counter() - t0)
        return res, None

    try:
        converged = bool(dyn.run(fmax=float(settings.fmax), steps=int(settings.steps)))
        res.converged = bool(converged)
    except Exception as exc:
        res.error = f"optimizer_run_failed: {type(exc).__name__}: {exc}"
        res.wall_time_s = float(time.perf_counter() - t0)
        return res, None

    try:
        res.nsteps = int(getattr(dyn, "nsteps", 0) or 0)
    except Exception:
        res.nsteps = 0

    try:
        res.energy_final_eV = float(atoms.get_potential_energy())
    except Exception:
        res.energy_final_eV = None

    try:
        f = atoms.get_forces()
        res.fmax_final = float(np.linalg.norm(f, axis=1).max()) if len(f) else None
    except Exception:
        res.fmax_final = None

    # Always wrap scaled positions for neat CIF output.
    try:
        sp = atoms.get_scaled_positions(wrap=True)
        atoms.set_scaled_positions(sp)
    except Exception:
        try:
            atoms.wrap()
        except Exception:
            pass

    # Optional CIF write.
    if out_dir is not None and relaxed_cif_name:
        try:
            out_dir = Path(out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            cif_path = out_dir / str(relaxed_cif_name)
            write(str(cif_path), atoms, format="cif")
            res.relaxed_cif = str(cif_path)
        except Exception as exc:
            # Do not fail relaxation just because output failed.
            res.relaxed_cif = ""
            if not res.error:
                res.error = f"write_cif_failed: {type(exc).__name__}: {exc}"

    res.wall_time_s = float(time.perf_counter() - t0)
    return res, atoms


def relax_settings_from_cfg(cfg: Dict[str, Any], *, prefix: str = "relax") -> BulkRelaxSettings:
    """
    Build BulkRelaxSettings from conf.yml mapping.
    """
    sec = cfg.get(prefix, {}) if isinstance(cfg, dict) else {}
    if not isinstance(sec, dict):
        sec = {}

    def _get(k: str, default):
        v = sec.get(k, default)
        return default if v is None else v

    return BulkRelaxSettings(
        optimizer=str(_get("optimizer", BulkRelaxSettings.optimizer)),
        fmax=float(_get("fmax", BulkRelaxSettings.fmax)),
        steps=int(_get("steps", BulkRelaxSettings.steps)),
        maxstep=_get("maxstep", BulkRelaxSettings.maxstep),
        relax_cell=bool(_get("relax_cell", BulkRelaxSettings.relax_cell)),
        cell_filter=str(_get("cell_filter", BulkRelaxSettings.cell_filter)),
        hydrostatic_strain=bool(_get("hydrostatic_strain", BulkRelaxSettings.hydrostatic_strain)),
        constant_volume=bool(_get("constant_volume", BulkRelaxSettings.constant_volume)),
        scalar_pressure_GPa=float(_get("scalar_pressure_GPa", BulkRelaxSettings.scalar_pressure_GPa)),
        logfile_name=sec.get("logfile_name", BulkRelaxSettings.logfile_name),
    )


def relax_settings_to_dict(settings: BulkRelaxSettings) -> Dict[str, Any]:
    return dict(asdict(settings))
