from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import spglib
import torch
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read, write

from .chem import ChemistryPolicy
from .checkpoints import load_model_checkpoint
from .config import SynthRequest, load_config
from .autoscale import autoscale_cell_isotropic
from .dedup import atoms_hash
from .decode import DecodeConfig, decode_tokens_to_atoms
from .generate import sample_sequence
from .validate import validate_atoms


@dataclass(frozen=True)
class SynthResult:
    out_dir: Path
    written: int
    requested_max: int
    attempts: int


@dataclass(frozen=True)
class _SupercellPlan:
    repeats: tuple[int, int, int]
    orbit_assignments: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _RatioConstraint:
    mode: str
    specified_indices: tuple[int, ...]
    specified_values: tuple[float | int, ...]
    remainder: object


def _validate_element_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    for s in symbols:
        s2 = str(s).strip()
        if not s2:
            continue
        if s2 not in atomic_numbers:
            raise ValueError(f"Unknown element symbol: {s2!r}")
        out.append(s2)
    if not out:
        raise ValueError("No valid element symbols provided")
    # de-dup while preserving order
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _lcm(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def _int_weights_from_fractions(fracs: list[Fraction]) -> list[int]:
    den = 1
    for f in fracs:
        den = _lcm(den, int(f.denominator))
    weights = [int(f * den) for f in fracs]

    g = 0
    for w in weights:
        g = math.gcd(g, abs(int(w)))
    if g > 1:
        weights = [int(w // g) for w in weights]
    return [int(w) for w in weights]


def _parse_ratio_values(ratios: list[float | str | None] | None) -> list[float | None] | None:
    if ratios is None:
        return None

    out: list[float | None] = []
    for raw in ratios:
        if raw is None:
            out.append(None)
            continue

        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            val = float(raw)
        else:
            txt = str(raw).strip()
            if not txt:
                raise ValueError("ratios entries must be numbers or X")
            if txt.lower() in {"x", "*"}:
                out.append(None)
                continue
            try:
                val = float(txt)
            except Exception as exc:
                raise ValueError("ratios entries must be numbers or X") from exc

        if not math.isfinite(float(val)):
            raise ValueError("ratios entries must be finite numbers or X")
        out.append(float(val))

    if not any(v is not None for v in out):
        raise ValueError("ratios must contain at least one numeric entry; use X only for unconstrained elements")
    return out


def _ratio_constraint(
    *,
    ratios: list[float | str | None] | None,
    ratio_mode: str | None,
    percent_tol: float | None,
    required_n: int,
    nelements_total: int,
) -> _RatioConstraint | None:
    parsed = _parse_ratio_values(ratios)
    if parsed is None:
        return None
    if len(parsed) != required_n:
        raise ValueError("ratios length must match the number of required elements")

    specified = [(i, float(x)) for i, x in enumerate(parsed) if x is not None]
    if not specified:
        raise ValueError("ratios must contain at least one numeric entry")

    indices = tuple(int(i) for i, _ in specified)
    values = [float(x) for _, x in specified]

    mode = str(ratio_mode or "ratio").strip().lower()
    if mode not in ("auto", "ratio", "percent"):
        raise ValueError("ratio_mode must be one of: ratio, percent (legacy auto is also accepted)")

    if mode == "auto":
        # Preserve the legacy "full strict ratio" behavior only when every listed element has a numeric value.
        if nelements_total == required_n and len(specified) == required_n:
            mode = "ratio"
        else:
            # Heuristic: values > 10 are most commonly expressed as percentages (e.g. 25, 25, X, X).
            mode = "percent" if max(values) > 10.0 else "ratio"

    if mode == "ratio":
        if len(values) < 2:
            raise ValueError("ratio mode requires at least two numeric entries; use percent mode for single-element percentages")
        fracs = []
        for x in values:
            f = Fraction(str(float(x))).limit_denominator(1000)
            if f <= 0:
                raise ValueError("ratios must be > 0")
            fracs.append(f)
        return _RatioConstraint(
            mode="ratio",
            specified_indices=indices,
            specified_values=tuple(_int_weights_from_fractions(fracs)),
            remainder=None,
        )

    # percent mode: numeric entries are fixed total-atom percentages, and the remainder goes to X-marked
    # listed elements plus any extra elements introduced by nelements_total > len(required).
    fracs_req = []
    for x in values:
        f = Fraction(str(float(x))).limit_denominator(1000) / Fraction(100, 1)
        if f <= 0:
            raise ValueError("percent ratios must be > 0")
        fracs_req.append(f)
    rem = Fraction(1, 1) - sum(fracs_req, Fraction(0, 1))
    free_elements = (required_n - len(specified)) + max(0, nelements_total - required_n)
    if free_elements <= 0:
        if rem != 0:
            raise ValueError("percent ratios must sum to 100 when all listed elements are specified")
    elif rem <= 0:
        raise ValueError("percent ratios must sum to < 100 when some elements are left as X or when extra elements are needed")

    tol = float(percent_tol or 0.0)
    if tol < 0:
        raise ValueError("percent_tol must be >= 0")

    if free_elements <= 0:
        return _RatioConstraint(
            mode="ratio",
            specified_indices=indices,
            specified_values=tuple(_int_weights_from_fractions(fracs_req)),
            remainder=None,
        )

    if tol > 0:
        targets = tuple(float(f) for f in fracs_req)
        return _RatioConstraint(
            mode="percent_tol",
            specified_indices=indices,
            specified_values=targets,
            remainder=(float(rem), float(tol)),
        )

    weights = _int_weights_from_fractions(fracs_req + [rem])
    return _RatioConstraint(
        mode="percent",
        specified_indices=indices,
        specified_values=tuple(weights[:-1]),
        remainder=int(weights[-1]),
    )


def _counts_match_weights(counts: list[int], weights: list[int]) -> bool:
    if len(counts) != len(weights):
        raise ValueError("counts/weights length mismatch")
    if any(int(c) <= 0 for c in counts):
        return False
    if any(int(w) <= 0 for w in weights):
        return False
    # weights should be reduced by gcd => guarantees integer proportionality.
    g = 0
    for w in weights:
        g = math.gcd(g, abs(int(w)))
    if g != 1:
        weights = [int(w // g) for w in weights]

    c0 = int(counts[0])
    w0 = int(weights[0])
    for c, w in zip(counts[1:], weights[1:]):
        if int(c) * w0 != c0 * int(w):
            return False
    return True


def _atoms_to_spglib_cell(atoms: Atoms) -> tuple[object, object, object]:
    return (
        atoms.cell.array,
        atoms.get_scaled_positions(wrap=True),
        atoms.get_atomic_numbers(),
    )


def _symmetry_orbits(atoms: Atoms, *, symprec: float) -> list[list[int]]:
    try:
        dataset = spglib.get_symmetry_dataset(_atoms_to_spglib_cell(atoms), symprec=float(symprec))
    except Exception:
        dataset = None

    if dataset is None:
        return [[int(i)] for i in range(len(atoms))]

    eq = getattr(dataset, "equivalent_atoms", None)
    if eq is None:
        eq = dataset["equivalent_atoms"]

    groups: dict[int, list[int]] = {}
    for i, rep in enumerate(eq):
        groups.setdefault(int(rep), []).append(int(i))

    orbits = list(groups.values())
    orbits.sort(key=lambda idxs: (-len(idxs), idxs))
    return orbits


def _allocations_for_total(total: int, capacities: tuple[int, ...]) -> list[tuple[int, ...]]:
    if total < 0:
        return []
    if not capacities:
        return [()] if total == 0 else []

    out: list[tuple[int, ...]] = []
    prefix: list[int] = []
    n = len(capacities)

    def _rec(i: int, remain: int) -> None:
        if i == n - 1:
            if 0 <= remain <= capacities[i]:
                out.append(tuple(prefix + [int(remain)]))
            return

        suffix_cap = sum(capacities[i + 1 :])
        lo = max(0, remain - suffix_cap)
        hi = min(int(capacities[i]), remain)
        for v in range(hi, lo - 1, -1):
            prefix.append(int(v))
            _rec(i + 1, remain - int(v))
            prefix.pop()

    _rec(0, int(total))
    out.sort(key=lambda xs: (sum(1 for x in xs if int(x) > 0), tuple(-int(x) for x in xs)))
    return out


def _solve_orbit_copy_assignment(
    orbit_sizes: list[int],
    *,
    copies: int,
    target_counts: tuple[int, ...],
) -> list[tuple[int, ...]] | None:
    if not orbit_sizes:
        return [] if all(int(x) == 0 for x in target_counts) else None

    order = sorted(range(len(orbit_sizes)), key=lambda i: int(orbit_sizes[i]), reverse=True)
    sizes = tuple(int(orbit_sizes[i]) for i in order)
    ntarget = len(target_counts)
    copies = int(copies)
    target_counts = tuple(int(x) for x in target_counts)

    @lru_cache(maxsize=None)
    def _search(iorbit: int, remain: tuple[int, ...]) -> tuple[tuple[int, ...], ...] | None:
        if iorbit >= len(sizes):
            return tuple() if all(int(x) == 0 for x in remain) else None

        size = int(sizes[iorbit])
        capacities = tuple(int(x) // size for x in remain)
        if sum(capacities) < copies:
            return None

        for alloc in _allocations_for_total(copies, capacities):
            next_remain = tuple(int(r) - int(a) * size for r, a in zip(remain, alloc))
            if any(int(x) < 0 for x in next_remain):
                continue
            tail = _search(iorbit + 1, next_remain)
            if tail is not None:
                return (tuple(int(x) for x in alloc),) + tail
        return None

    found = _search(0, target_counts)
    if found is None:
        return None

    allocs: list[tuple[int, ...]] = [tuple(0 for _ in range(ntarget)) for _ in orbit_sizes]
    for ord_idx, alloc in zip(order, found):
        allocs[int(ord_idx)] = tuple(int(x) for x in alloc)
    return allocs


def _factor_triplets(total: int) -> list[tuple[int, int, int]]:
    total = int(total)
    out: list[tuple[int, int, int]] = []
    for a in range(1, total + 1):
        if total % a != 0:
            continue
        rem = total // a
        for b in range(1, rem + 1):
            if rem % b != 0:
                continue
            c = rem // b
            out.append((int(a), int(b), int(c)))
    return out


def _best_repeat_triplet(atoms: Atoms, total_copies: int) -> tuple[int, int, int]:
    total_copies = int(total_copies)
    if total_copies <= 1:
        return (1, 1, 1)

    lens = [float(x) for x in atoms.cell.lengths()]
    if any((not math.isfinite(x)) or x <= 1e-8 for x in lens):
        lens = [1.0, 1.0, 1.0]

    cands = _factor_triplets(total_copies)
    cands.sort(
        key=lambda rep: (
            sum((float(rep[i]) * lens[i] - (sum(float(rep[j]) * lens[j] for j in range(3)) / 3.0)) ** 2 for i in range(3)),
            max(rep) - min(rep),
            max(rep),
        )
    )
    return cands[0]


def _expand_orbit_assignment(
    *,
    target: list[str],
    orbit_allocs: list[tuple[int, ...]],
) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for alloc in orbit_allocs:
        seq: list[str] = []
        for sym, ncopy in zip(target, alloc):
            seq.extend([str(sym)] * int(ncopy))
        out.append(tuple(seq))
    return tuple(out)


def _apply_supercell_plan(
    atoms: Atoms,
    *,
    orbits: list[list[int]],
    plan: _SupercellPlan,
) -> Atoms:
    repeats = tuple(int(x) for x in plan.repeats)
    total_copies = int(math.prod(repeats))
    natoms0 = int(len(atoms))
    expanded = atoms.repeat(repeats)
    symbols = expanded.get_chemical_symbols()

    for copy_idx in range(total_copies):
        offset = copy_idx * natoms0
        for orbit_indices, per_copy_species in zip(orbits, plan.orbit_assignments):
            sym = str(per_copy_species[copy_idx])
            for idx in orbit_indices:
                symbols[offset + int(idx)] = sym

    expanded.set_chemical_symbols(symbols)
    return expanded


def _plan_ratio_supercell(
    atoms: Atoms,
    *,
    target: list[str],
    weights: list[int],
    symprec: float,
    max_atoms: int | None,
    max_total_copies: int,
) -> tuple[list[list[int]], _SupercellPlan] | None:
    ntarget = len(target)
    if ntarget != len(weights) or ntarget <= 0:
        return None

    natoms0 = int(len(atoms))
    if natoms0 <= 0:
        return None

    if max_atoms is not None and natoms0 > int(max_atoms):
        return None

    total_weight = int(sum(int(w) for w in weights))
    if total_weight <= 0:
        return None

    max_copies = max(1, int(max_total_copies))
    if max_atoms is not None:
        max_copies = min(max_copies, max(1, int(max_atoms) // natoms0))

    orbits = _symmetry_orbits(atoms, symprec=float(symprec))
    orbit_sizes = [int(len(orbit)) for orbit in orbits]

    for total_copies in range(1, max_copies + 1):
        total_atoms = natoms0 * int(total_copies)
        if total_atoms % total_weight != 0:
            continue

        scale = total_atoms // total_weight
        target_counts = tuple(int(w) * int(scale) for w in weights)
        orbit_allocs = _solve_orbit_copy_assignment(orbit_sizes, copies=total_copies, target_counts=target_counts)
        if orbit_allocs is None:
            continue

        repeats = _best_repeat_triplet(atoms, total_copies)
        assignments = _expand_orbit_assignment(target=target, orbit_allocs=orbit_allocs)
        return orbits, _SupercellPlan(repeats=repeats, orbit_assignments=assignments)

    return None


def synth_cifs(*, conf_path: Path, req: SynthRequest) -> SynthResult:
    cfg = load_config(conf_path)

    required = _validate_element_symbols(req.required_elements)
    nelements_total = int(req.nelements_total)
    if nelements_total < 1:
        raise ValueError("--nelements must be >= 1")
    if nelements_total < len(required):
        raise ValueError("--nelements must be >= number of required elements")

    out_base = Path(cfg["paths"]["out_dir"]).expanduser().resolve()
    req_name = "-".join(required)
    out_dir = out_base / f"{req_name}_{nelements_total}el"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = cfg["generate"]
    val = cfg["validate"]
    syn = cfg.get("synth", {}) or {}

    ratios = req.ratios if req.ratios is not None else syn.get("ratios", None)
    ratio_mode = req.ratio_mode if req.ratio_mode is not None else syn.get("ratio_mode", "ratio")
    percent_tol = syn.get("percent_tol", 0.1)
    constraint = _ratio_constraint(
        ratios=list(ratios) if isinstance(ratios, (list, tuple)) else None,
        ratio_mode=str(ratio_mode) if ratio_mode is not None else None,
        percent_tol=float(percent_tol) if percent_tol is not None else None,
        required_n=len(required),
        nelements_total=nelements_total,
    )
    strict_ratio_target = bool(
        constraint is not None
        and constraint.mode == "ratio"
        and nelements_total == len(required)
        and constraint.specified_indices == tuple(range(len(required)))
    )
    ratio_weights_all = [int(x) for x in constraint.specified_values] if strict_ratio_target else None
    allow_supercell_ratio = bool(syn.get("allow_supercell_ratio", True))
    max_supercell_copies = int(syn.get("max_supercell_copies", 8))
    flexible_prototype_nelements = bool(syn.get("flexible_prototype_nelements", True))
    if max_supercell_copies < 1:
        raise ValueError("synth.max_supercell_copies must be >= 1")

    ckpt_path = Path(cfg["paths"]["ckpt"]).expanduser().resolve()
    loaded = load_model_checkpoint(ckpt_path, device="auto")
    ckpt = loaded.blob
    vocab = loaded.vocab
    id_to_token = loaded.id_to_token
    pad_id = loaded.pad_id
    model_cfg = loaded.model_config
    vocab_elems = {t[2:] for t in vocab if t.startswith("E_")}
    device = loaded.device
    model = loaded.model

    tok_cfg = ckpt["tokenize_config"]
    dec_cfg = DecodeConfig(
        coord_bins=int(tok_cfg["coord_bins"]),
        len_bins=int(tok_cfg["len_bins"]),
        len_min=float(tok_cfg["len_min"]),
        len_max=float(tok_cfg["len_max"]),
        ang_bins=int(tok_cfg["ang_bins"]),
        ang_min=float(tok_cfg["ang_min"]),
        ang_max=float(tok_cfg["ang_max"]),
    )

    # Controls
    n_max = int(gen["n_max"])
    if req.n_max is not None:
        n_max = int(req.n_max)
    attempts_factor = int(gen["attempts_factor"])
    stop_after_no_new = int(gen["stop_after_no_new"])
    temperature = float(gen["temperature"])
    top_k = int(gen["top_k"])
    max_sites = int(gen["max_sites"])
    include_metalloids = bool(gen["include_metalloids"])
    autoscale_cell = bool(gen.get("autoscale_cell", True))
    proto_mode = str(gen.get("prototype_mode", "target") or "target").strip().lower()
    if proto_mode not in ("target", "random"):
        raise ValueError("generate.prototype_mode must be one of: target, random")

    if n_max < 0:
        raise ValueError("generate.n_max must be >= 0")
    if attempts_factor < 1:
        raise ValueError("generate.attempts_factor must be >= 1")
    if stop_after_no_new < 0:
        raise ValueError("generate.stop_after_no_new must be >= 0")

    # Space-group forcing (optional)
    hall_mode = str(gen.get("hall_mode", "model") or "model")
    fixed_hall = gen.get("fixed_hall", None)
    fixed_spacegroup = gen.get("fixed_spacegroup", None)

    hall_to_id = {int(t[5:]): int(vocab[t]) for t in vocab if t.startswith("HALL_")}
    hall_ids = sorted(hall_to_id.values())
    sym_stats = ckpt.get("symmetry_stats", None)
    halls_by_sg_trained = None
    if isinstance(sym_stats, dict):
        hbs = sym_stats.get("halls_by_sg", None)
        if isinstance(hbs, dict):
            halls_by_sg_trained = {int(k): [int(x) for x in v] for k, v in hbs.items() if isinstance(v, list)}

    def _representative_hall_by_sg() -> dict[int, int]:
        import spglib

        rep: dict[int, int] = {}
        for hall in range(1, 531):
            t = spglib.get_spacegroup_type(hall)
            if t is None:
                continue
            sg = int(getattr(t, "number") if hasattr(t, "number") else t["number"])
            rep.setdefault(sg, hall)
        return rep

    rep_hall_by_sg: dict[int, int] | None = None

    def _pick_forced_hall_id() -> int | None:
        nonlocal rep_hall_by_sg

        if fixed_hall is not None:
            hid = hall_to_id.get(int(fixed_hall))
            if hid is None:
                raise ValueError(f"generate.fixed_hall={fixed_hall} not present in vocab (seed all hall tokens)")
            return int(hid)

        if fixed_spacegroup is not None:
            if rep_hall_by_sg is None:
                rep_hall_by_sg = _representative_hall_by_sg()
            sg = int(fixed_spacegroup)
            if halls_by_sg_trained is not None and sg in halls_by_sg_trained:
                for hall in halls_by_sg_trained[sg]:
                    hid = hall_to_id.get(int(hall))
                    if hid is not None:
                        return int(hid)

            hall = rep_hall_by_sg.get(sg)
            if hall is None:
                raise ValueError(f"generate.fixed_spacegroup={fixed_spacegroup} unknown (expected 1..230)")
            hid = hall_to_id.get(int(hall))
            if hid is None:
                raise ValueError(f"Representative hall={hall} for SG={fixed_spacegroup} not in vocab")
            return int(hid)

        mode = str(hall_mode).strip().lower()
        if mode == "model":
            return None
        if mode == "uniform_530":
            if not hall_ids:
                return None
            return int(hall_ids[torch.randint(low=0, high=len(hall_ids), size=(1,)).item()])
        if mode == "uniform_230":
            if halls_by_sg_trained is not None:
                sg_list = [sg for sg in range(1, 231) if sg in halls_by_sg_trained]
                if sg_list:
                    sg = int(sg_list[torch.randint(low=0, high=len(sg_list), size=(1,)).item()])
                    halls = [h for h in halls_by_sg_trained.get(sg, []) if int(h) in hall_to_id]
                    if halls:
                        hall = int(halls[torch.randint(low=0, high=len(halls), size=(1,)).item()])
                        return int(hall_to_id[hall])

            if rep_hall_by_sg is None:
                rep_hall_by_sg = _representative_hall_by_sg()
            sg_list = [sg for sg in range(1, 231) if sg in rep_hall_by_sg and rep_hall_by_sg[sg] in hall_to_id]
            if not sg_list:
                return None
            sg = int(sg_list[torch.randint(low=0, high=len(sg_list), size=(1,)).item()])
            return int(hall_to_id[int(rep_hall_by_sg[sg])])
        raise ValueError(f"Unknown generate.hall_mode: {hall_mode!r}")

    chemistry_mode = str(gen.get("chemistry_mode", "any") or "any").strip().lower()
    chemistry = ChemistryPolicy(
        mode=chemistry_mode,
        include_metalloids=include_metalloids,
        allowed_elements=gen.get("allowed_elements", None),
        excluded_elements=gen.get("excluded_elements", None),
        min_elements=max(nelements_total, 2 if chemistry_mode == "intermetallic" else 1),
        max_elements=nelements_total,
    )
    for e in required:
        if not chemistry.is_allowed_element(e):
            raise ValueError(f"Required element {e} is rejected by generate.chemistry_mode={chemistry.mode!r}")

    pool_mode = str(gen.get("random_pool", "chemistry") or "chemistry").strip().lower()
    pool_policy = chemistry
    if pool_mode != "chemistry":
        pool_policy = ChemistryPolicy(mode=pool_mode, include_metalloids=include_metalloids)

    pool = [e for e in pool_policy.available_elements() if e not in set(required) and chemistry.is_allowed_element(e)]
    if nelements_total > len(required) and len(pool) < (nelements_total - len(required)):
        raise ValueError("Element pool is too small to satisfy requested nelements_total")

    force_distinct = bool(gen.get("force_distinct_first_sites", True))

    proto_candidates = gen.get("prototype_elements", None)
    if proto_candidates is None:
        proto_pool = [
            sym
            for sym in chemistry.available_elements()
            if f"E_{sym}" in vocab
        ]
    elif isinstance(proto_candidates, (list, tuple)):
        proto_pool = []
        for s in proto_candidates:
            sym = str(s).strip()
            if not sym:
                continue
            if not chemistry.is_allowed_element(sym):
                continue
            if f"E_{sym}" not in vocab:
                continue
            if sym not in proto_pool:
                proto_pool.append(sym)
    else:
        raise ValueError("generate.prototype_elements must be a YAML list or null")

    if len(proto_pool) < nelements_total:
        raise RuntimeError("Prototype element pool too small; re-run preprocess with --seed-all-elements or adjust generate.prototype_elements.")

    proto_nelements_choices = [int(nelements_total)]
    if strict_ratio_target and allow_supercell_ratio and flexible_prototype_nelements and nelements_total > 2:
        proto_nelements_choices = list(range(2, int(nelements_total) + 1))

    # Dedup config.
    ded_cfg = gen.get("dedup", {}) or {}
    dedup_enabled = bool(ded_cfg.get("enabled", True))
    dedup_mode = str(ded_cfg.get("mode", "prototype") or "prototype").strip().lower()
    dedup_symprec = float(ded_cfg.get("symprec", 1e-2))
    dedup_frac_tol = float(ded_cfg.get("frac_tol", 1e-3))
    dedup_cell_tol = float(ded_cfg.get("cell_tol", 1e-2))

    seen: set[str] = set()
    reject: dict[str, int] = {}

    def _reject(reason: str) -> None:
        reject[reason] = reject.get(reason, 0) + 1

    # Output naming
    def _next_path(idx: int) -> Path:
        return out_dir / f"genmat_{idx:05d}.cif"

    def _output_index_exists(idx: int) -> bool:
        return any(
            (out_dir / f"{prefix}_{idx:05d}.cif").exists()
            for prefix in ("genmat", "genim")
        )

    next_idx = 0
    while _output_index_exists(next_idx):
        next_idx += 1

    if dedup_enabled:
        for p in sorted([p for p in out_dir.glob("*.cif") if p.is_file()]):
            try:
                a = read(str(p))
                h = atoms_hash(a, symprec=dedup_symprec, frac_tol=dedup_frac_tol, cell_tol=dedup_cell_tol, mode=dedup_mode)
                seen.add(h)
            except Exception:
                continue

    rng = random.Random()
    extra_k = nelements_total - len(required)
    extra_cycle = pool.copy()
    rng.shuffle(extra_cycle)
    extra_cycle_i = 0

    def _pick_target_elements() -> list[str]:
        nonlocal extra_cycle_i, extra_cycle
        if extra_k <= 0:
            target = list(required)
        elif extra_k == 1:
            if extra_cycle_i >= len(extra_cycle):
                extra_cycle = pool.copy()
                rng.shuffle(extra_cycle)
                extra_cycle_i = 0
            target = list(required) + [extra_cycle[extra_cycle_i]]
            extra_cycle_i += 1
        else:
            target = list(required) + rng.sample(pool, k=extra_k)
        return target

    def _pick_prototype_elements(target: list[str]) -> tuple[list[str], bool]:
        proto_k = int(proto_nelements_choices[rng.randrange(len(proto_nelements_choices))])
        need_substitution = False

        if proto_mode == "target":
            if proto_k == len(target) and all(str(sym) in vocab_elems for sym in target):
                return list(target), False
            if proto_k < len(target) and all(str(sym) in vocab_elems for sym in target):
                return rng.sample(list(target), k=proto_k), True
            need_substitution = True
            return rng.sample(proto_pool, k=proto_k), need_substitution

        need_substitution = True
        return rng.sample(proto_pool, k=proto_k), need_substitution

    max_attempts = n_max * attempts_factor
    max_len = int(model_cfg.max_len)
    written = 0
    attempts = 0
    no_new = 0

    def _finalize_candidate(a2: Atoms) -> Atoms | None:
        if autoscale_cell:
            a2, _scale = autoscale_cell_isotropic(
                a2,
                min_dist=float(val["min_dist"]),
                min_dist_factor=val.get("min_dist_factor", None),
                max_dist_factor=val.get("max_dist_factor", None),
                vol_per_atom_min=val.get("vol_per_atom_min", None),
                vol_per_atom_max=val.get("vol_per_atom_max", None),
            )

        ok, _reason = validate_atoms(
            a2,
            min_dist=float(val["min_dist"]),
            symprec=float(val["symprec"]),
            min_dist_factor=val.get("min_dist_factor", None),
            max_dist_factor=val.get("max_dist_factor", None),
            max_nn_factor=val.get("max_nn_factor", None),
            min_coordination=val.get("min_coordination", None),
            require_connected=val.get("require_connected", None),
            max_atoms=val.get("max_atoms", None),
            vol_per_atom_min=val.get("vol_per_atom_min", None),
            vol_per_atom_max=val.get("vol_per_atom_max", None),
        )
        if not ok:
            return None

        elems = set(a2.get_chemical_symbols())
        if not chemistry.accepts_elements(elems):
            return None
        return a2

    while written < n_max and attempts < max_attempts:
        attempts += 1
        forced_hall_id = _pick_forced_hall_id()

        target = _pick_target_elements()
        if len(set(target)) != nelements_total:
            _reject("target_nelements_mismatch")
            no_new += 1
            if stop_after_no_new and no_new >= stop_after_no_new:
                break
            continue

        proto_elements, need_substitution = _pick_prototype_elements(target)
        forced_ids_by_pos = None
        if force_distinct:
            forced_ids_by_pos = {}
            for i, sym in enumerate(proto_elements):
                # Element token positions: 8, 13, 18, ... (each site block is 5 tokens).
                pos = 8 + 5 * i
                forced_ids_by_pos[pos] = int(vocab[f"E_{sym}"])

        try:
            ids = sample_sequence(
                model,
                vocab=vocab,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                max_sites=max_sites,
                min_sites=max(int(chemistry.min_elements), len(proto_elements)),
                restrict_elements=list(proto_elements),
                device=device,
                forced_hall_id=forced_hall_id,
                forced_ids_by_pos=forced_ids_by_pos,
            )
        except Exception:
            _reject("sample_error")
            no_new += 1
            if stop_after_no_new and no_new >= stop_after_no_new:
                break
            continue

        tokens = [id_to_token[i] for i in ids if i != pad_id]
        try:
            atoms = decode_tokens_to_atoms(tokens, cfg=dec_cfg, max_sites=max_sites)
        except Exception:
            _reject("decode_error")
            no_new += 1
            if stop_after_no_new and no_new >= stop_after_no_new:
                break
            continue

        if not math.isfinite(float(atoms.get_volume())) or float(atoms.get_volume()) <= 1e-6:
            _reject("invalid_volume")
            no_new += 1
            if stop_after_no_new and no_new >= stop_after_no_new:
                break
            continue

        src = sorted(set(atoms.get_chemical_symbols()), key=lambda s: atomic_numbers.get(s, 999))
        if strict_ratio_target and allow_supercell_ratio:
            if len(src) < 2:
                _reject("prototype_nelements_mismatch")
                no_new += 1
                if stop_after_no_new and no_new >= stop_after_no_new:
                    break
                continue
        elif len(src) != nelements_total:
            _reject("prototype_nelements_mismatch")
            no_new += 1
            if stop_after_no_new and no_new >= stop_after_no_new:
                break
            continue

        src_counts = Counter(atoms.get_chemical_symbols())
        src_species = sorted(src_counts.keys(), key=lambda s: atomic_numbers.get(s, 999))
        total_atoms = int(sum(int(v) for v in src_counts.values()))

        applied = None
        ratio_passed_any = False

        if strict_ratio_target and allow_supercell_ratio and ratio_weights_all is not None:
            planned = _plan_ratio_supercell(
                atoms,
                target=list(required),
                weights=list(ratio_weights_all),
                symprec=float(val["symprec"]),
                max_atoms=val.get("max_atoms", None),
                max_total_copies=max_supercell_copies,
            )
            if planned is not None:
                orbits, plan = planned
                ratio_passed_any = True
                a2 = _apply_supercell_plan(atoms, orbits=orbits, plan=plan)
                applied = _finalize_candidate(a2)
        else:
            perms = [tuple(src_species)] if (constraint is None and not need_substitution) else [tuple(target)]
            if constraint is not None:
                perms = list(itertools.permutations(target))

            for perm in perms:
                mapping = dict(zip(src_species, perm))
                inv = {t: s for s, t in mapping.items()}

                if constraint is not None:
                    req_counts = [int(src_counts[inv[e]]) for e in required]
                    sel_counts = [req_counts[i] for i in constraint.specified_indices]
                    if constraint.mode == "ratio":
                        if not _counts_match_weights(sel_counts, [int(x) for x in constraint.specified_values]):
                            continue
                        ratio_passed_any = True
                    elif constraint.mode == "percent":
                        rest = int(total_atoms - sum(sel_counts))
                        if constraint.remainder is None:
                            raise RuntimeError("percent constraint missing remainder weight")
                        if not _counts_match_weights(
                            sel_counts + [rest],
                            [int(x) for x in constraint.specified_values] + [int(constraint.remainder)],
                        ):
                            continue
                        ratio_passed_any = True
                    elif constraint.mode == "percent_tol":
                        if not isinstance(constraint.remainder, tuple) or len(constraint.remainder) != 2:
                            raise RuntimeError("percent_tol constraint missing (rest,target_tol)")
                        rest_target, tol = float(constraint.remainder[0]), float(constraint.remainder[1])
                        if tol < 0:
                            raise ValueError("percent_tol must be >= 0")
                        rest = int(total_atoms - sum(sel_counts))
                        fracs_req = [float(c) / float(total_atoms) for c in sel_counts]
                        frac_rest = float(rest) / float(total_atoms)

                        ok = True
                        for f, t in zip(fracs_req, constraint.specified_values):
                            if abs(float(f) - float(t)) > tol:
                                ok = False
                                break
                        if ok and abs(float(frac_rest) - float(rest_target)) > tol:
                            ok = False
                        if not ok:
                            continue
                        ratio_passed_any = True
                    else:
                        raise RuntimeError(f"Unknown constraint mode: {constraint.mode!r}")

                a2 = atoms.copy()
                if mapping and any(k != v for k, v in mapping.items()):
                    a2.set_chemical_symbols([mapping[s] for s in a2.get_chemical_symbols()])

                applied = _finalize_candidate(a2)
                if applied is not None:
                    break

        if constraint is not None and not ratio_passed_any:
            _reject("composition_ratio_mismatch")
            no_new += 1
            if stop_after_no_new and no_new >= stop_after_no_new:
                break
            continue

        if applied is None:
            _reject("no_valid_mapping")
            no_new += 1
            if stop_after_no_new and no_new >= stop_after_no_new:
                break
            continue

        atoms = applied

        if dedup_enabled:
            h = atoms_hash(atoms, symprec=dedup_symprec, frac_tol=dedup_frac_tol, cell_tol=dedup_cell_tol, mode=dedup_mode)
            if h in seen:
                _reject("duplicate")
                no_new += 1
                if stop_after_no_new and no_new >= stop_after_no_new:
                    break
                continue
            seen.add(h)

        path = _next_path(next_idx)
        write(str(path), atoms, format="cif")
        written += 1
        next_idx += 1
        no_new = 0

    print(f"[synth] Generated {written}/{n_max} unique CIFs under {out_dir} (attempts={attempts}, max_attempts={max_attempts}).")
    if reject:
        top = sorted(reject.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("[synth] Top rejections:")
        for k, v in top:
            print(f"  - {k}: {v}")

    return SynthResult(out_dir=out_dir, written=written, requested_max=n_max, attempts=attempts)
