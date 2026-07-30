from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        # MP dataset artifacts
        "mp_jsonl": "data/mp_train.jsonl",
        "tokens_pt": "data/mp_train.tokens.pt",
        # Checkpoint to use for generation
        "ckpt": "checkpoints/mp_train_fullsg_60.pt",
        # Output directory base
        "out_dir": "output",
    },
    "mp_download": {
        "chemsys": None,
        "elements": None,
        # any | metallic | intermetallic
        "chemistry_filter": "any",
        # If true: download a balanced dataset across all 230 spacegroups.
        "balance_spacegroups": False,
        "per_spacegroup": 50,
        # Optional: filter by a single spacegroup number (1..230).
        "spacegroup_number": None,
        "max_atoms": 200,
        "eah_max": 1.0,
        "nelements_min": 1,
        "nelements_max": 5,
        "limit": 100000,
        "per_page": 500,
        "timeout": 60.0,
        "include_metalloids": True,
    },
    "preprocess": {
        "max_sites": 25,
        "symprec": 1e-2,
        "coord_bins": 128,
        "len_bins": 240,
        "len_min": 1.5,
        "len_max": 30.0,
        "ang_bins": 181,
        "ang_min": 30.0,
        "ang_max": 150.0,
        "seed_all_elements": True,
        "seed_all_hall": True,
        # any | metallic | intermetallic
        "chemistry_mode": "any",
        "include_metalloids": True,
        "allowed_elements": None,
        "excluded_elements": None,
    },
    "train": {
        "steps": 20000,
        "batch": 64,
        "lr": 3e-4,
        "d_model": 256,
        "layers": 6,
        "heads": 8,
        "dropout": 0.1,
        "seed": 7,
        "val_fraction": 0.1,
        "element_emb": "features",
        # legacy4 keeps old checkpoint compatibility; periodic8 adds period,
        # group and broad chemical-class indicators for newly trained models.
        "element_feature_set": "periodic8",
    },
    "generate": {
        # Maximum number of structures to emit (acts like --n max)
        "n_max": 200,
        "max_sites": 25,
        "temperature": 1.0,
        "top_k": 0,
        # General-purpose composition policy. Use "intermetallic" to reproduce
        # the historical GenIM behaviour.
        "chemistry_mode": "any",
        "include_metalloids": True,
        "allowed_elements": None,
        "excluded_elements": None,
        # If true: isotropically rescale the cell to satisfy validate.* bounds before final validation.
        "autoscale_cell": False,
        # Prototype selection strategy for `genim synth`:
        # - target: sample directly in the requested chemistry (recommended; more realistic lengths)
        # - random: sample a random prototype chemistry and substitute to the target (more diverse but less physical)
        "prototype_mode": "target",
        # Space-group sampling: model | uniform_230 | uniform_530
        "hall_mode": "uniform_230",
        # If set, overrides hall_mode
        "fixed_hall": None,
        "fixed_spacegroup": None,
        # If nelements_total > len(required_elements), extra elements are drawn
        # from this pool: chemistry | any | metallic | intermetallic.
        "random_pool": "chemistry",
        # Prototype palette used by `genim synth`: restrict sampling to a small, common element set,
        # then substitute to the target chemistry. This dramatically reduces prototype_nelements_mismatch.
        # If set, `genim synth` can use this palette to condition prototype sampling.
        # If null/None, all vocabulary elements allowed by chemistry_mode are used.
        "prototype_elements": None,
        "force_distinct_first_sites": True,
        # Deduplication (hash on standardized cell + rounded frac coords).
        "dedup": {
            "enabled": True,
            # full | prototype (prototype ignores cell metrics for stricter de-dup of near-identical structures)
            "mode": "prototype",
            "symprec": 1e-2,
            "frac_tol": 1e-2,
            "cell_tol": 2e-1,
        },
        # Attempts control: max_attempts = n_max * attempts_factor
        "attempts_factor": 200,
        # Stop early if we fail to find any new unique structure for this many attempts.
        "stop_after_no_new": 2000,
    },
    "validate": {
        "min_dist": 0.5,
        "symprec": 1e-2,
        "min_dist_factor": 0.55,
        # Reject overly sparse structures: require min(d_ij/(r_i+r_j)) <= max_dist_factor.
        "max_dist_factor": None,
        # Upper-bound neighbor sanity check (reject isolated atoms):
        # require each atom has >= min_coordination neighbors within
        # max_nn_factor * (r_cov(i) + r_cov(j)).
        "max_nn_factor": 1.5,
        "min_coordination": 1,
        # Require the neighbor graph to be connected (avoids split clusters in large cells).
        "require_connected": False,
        "max_atoms": 500,
        "vol_per_atom_min": 1.0,
        "vol_per_atom_max": 100.0,
    },
    "synth": {
        # Element ratio control for required elements (aligned with `genim synth --elements` order).
        # When null, no ratio constraints are applied.
        "ratios": None,
        # ratio | percent (legacy auto is still accepted when loading older configs)
        "ratio_mode": "ratio",
        # Only used when ratio_mode resolves to "percent" and nelements_total > len(required).
        # Absolute tolerance on atomic fractions (e.g. 0.1 means ±10%).
        "percent_tol": 0.1,
        # Allow exact ratio realization by repeating the parent cell and splitting
        # symmetry orbits across translationally equivalent copies.
        "allow_supercell_ratio": True,
        # Cap on the total number of repeated parent-cell copies used above.
        "max_supercell_copies": 8,
        # Allow prototype sampling with fewer distinct elements than the final target.
        "flexible_prototype_nelements": True,
    },
    # --- MLIP backend (eSEN/OMAT24 via fairchem-core) ---
    "ml": {
        # OMAT24 alias or a local checkpoint path/filename, e.g. "eSEN-30M-OAM" or "esen_30m_oam.pt"
        "model": "eSEN-30M-OAM",
        # Optional explicit local checkpoint path (recommended to avoid HF download/gating issues).
        "checkpoint": None,
        # auto | cpu | cuda
        "device": "auto",
        # Suppress noisy stdout/stderr during model load.
        "quiet": True,
    },
    # --- Bulk relaxation for generated structures ---
    "relax": {
        "enabled": True,
        "write_relaxed_cif": True,
        # Output subdir inside `score.out_dir` (or default <cif_dir>/ml_score)
        "relaxed_subdir": "relaxed",
        # BulkRelaxSettings knobs
        "optimizer": "LBFGS",
        "fmax": 0.05,
        "steps": 300,
        "maxstep": 0.05,
        "relax_cell": True,
        # expcell | unitcell | none
        "cell_filter": "expcell",
        "hydrostatic_strain": False,
        "constant_volume": False,
        # External pressure (GPa). 0.0 => zero pressure relaxation.
        "scalar_pressure_GPa": 0.0,
        # Optional optimizer logfile name (relative to output dir). null disables.
        "logfile_name": None,
    },
    # --- Energy Above Hull (ML-based) ---
    "hull": {
        "enabled": True,
        # Stable if E_above_hull <= threshold (eV/atom). 0.2 eV/atom = 200 meV/atom.
        "stable_threshold_eV_per_atom": 0.2,
        "cache_dir": "output/hull_cache",
        "force_rebuild": False,
        # MP reference pool used to build the hull entries (energies computed by MLIP).
        "mp": {
            "max_atoms": 80,
            # MP energy_above_hull_max server-side filter for *reference selection only* (eV/atom).
            "eah_max": 0.5,
            "limit_per_subset": 2000,
            # Keep only the top-k MP structures (by MP DFT energy_above_hull, then nsites)
            # per reduced composition to evaluate with MLIP. This keeps hull-building fast.
            "max_structures_per_composition": 2,
            "per_page": 200,
            "timeout": 60.0,
        },
        # Reference relaxation (applied to MP structures before using their ML energies on the hull).
        "relax": {
            "enabled": True,
            "optimizer": "LBFGS",
            "fmax": 0.05,
            "steps": 120,
            "maxstep": 0.05,
            "relax_cell": True,
            "cell_filter": "expcell",
            "hydrostatic_strain": False,
            "constant_volume": False,
            "scalar_pressure_GPa": 0.0,
        },
    },
    # --- Scoring output ---
    "score": {
        # If null, defaults to <cif_dir>/ml_score
        "out_dir": None,
        "csv_name": "score.csv",
        # If null, inherits from relax.relaxed_subdir
        "relaxed_subdir": "relaxed",
    },
}


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: Path) -> dict[str, Any]:
    """
    Load and deep-merge YAML into DEFAULT_CONFIG.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("conf.yml must contain a YAML mapping at the top level")
    merged = _deep_update(copy.deepcopy(DEFAULT_CONFIG), data)
    return merged


def write_default_config(path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False, allow_unicode=True), encoding="utf-8")


@dataclass(frozen=True)
class SynthRequest:
    required_elements: list[str]
    nelements_total: int
    ratios: list[float | str | None] | None = None
    ratio_mode: str | None = None
    n_max: int | None = None
