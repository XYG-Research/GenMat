from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from ase.data import atomic_numbers
from ase.io import read, write

from .chem import IntermetallicFilter, allowed_intermetallic_elements
from .autoscale import autoscale_cell_isotropic
from .dedup import atoms_hash
from .decode import DecodeConfig, decode_tokens_to_atoms
from .model import CausalTransformerLM, ModelConfig
from .validate import validate_atoms


@dataclass(frozen=True)
class TokenGroups:
    hall: list[int]
    length: list[int]
    angle: list[int]
    elem: list[int]
    wyckoff: list[int]
    coord: list[int]


def _build_groups(vocab: dict[str, int], *, restrict_elements: list[str] | None) -> TokenGroups:
    hall = []
    length = []
    angle = []
    elem = []
    wyckoff = []
    coord = []

    restrict = None
    if restrict_elements:
        restrict = set(restrict_elements)

    for tok, tid in vocab.items():
        if tok.startswith("HALL_"):
            hall.append(tid)
        elif tok.startswith("LEN_"):
            length.append(tid)
        elif tok.startswith("ANG_"):
            angle.append(tid)
        elif tok.startswith("E_"):
            if restrict is None or tok[2:] in restrict:
                elem.append(tid)
        elif tok.startswith("W_"):
            wyckoff.append(tid)
        elif tok.startswith("COORD_"):
            coord.append(tid)

    return TokenGroups(hall=sorted(hall), length=sorted(length), angle=sorted(angle), elem=sorted(elem), wyckoff=sorted(wyckoff), coord=sorted(coord))


def _top_k_filter(logits: torch.Tensor, *, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    cutoff = v[-1]
    return torch.where(logits < cutoff, torch.full_like(logits, -1e9), logits)


def _sample_next_id(
    logits: torch.Tensor,
    *,
    allowed: list[int],
    temperature: float,
    top_k: int,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / float(temperature)

    # Mask disallowed tokens.
    mask = torch.full_like(logits, -1e9)
    mask[allowed] = 0.0
    logits = logits + mask

    logits = _top_k_filter(logits, k=top_k)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _allowed_ids_for_pos(
    pos: int,
    *,
    groups: TokenGroups,
    eos_id: int,
    min_sites: int,
    max_sites: int,
) -> list[int]:
    # pos is the index of the *next* token to be generated, starting from 1 after BOS.
    if pos == 1:
        return groups.hall
    if pos in (2, 3, 4):
        return groups.length
    if pos in (5, 6, 7):
        return groups.angle
    if pos >= 8:
        offset = pos - 8
        mod = offset % 5
        if mod == 0:
            # At site boundary: either start new site (E_*) or end sequence (EOS).
            site_count = offset // 5
            if site_count >= max_sites:
                return [eos_id]
            allowed = list(groups.elem)
            if site_count >= min_sites:
                allowed.append(eos_id)
            return allowed
        if mod == 1:
            return groups.wyckoff
        return groups.coord
    raise RuntimeError("Unhandled position")


def sample_sequence(
    model: CausalTransformerLM,
    *,
    vocab: dict[str, int],
    max_len: int,
    temperature: float,
    top_k: int,
    max_sites: int,
    min_sites: int = 1,
    restrict_elements: list[str] | None,
    device: torch.device,
    forced_hall_id: int | None = None,
    forced_ids_by_pos: dict[int, int] | None = None,
) -> list[int]:
    pad_id = int(vocab["<PAD>"])
    bos_id = int(vocab["<BOS>"])
    eos_id = int(vocab["<EOS>"])

    groups = _build_groups(vocab, restrict_elements=restrict_elements)
    if not groups.hall or not groups.length or not groups.angle or not groups.elem or not groups.wyckoff or not groups.coord:
        raise ValueError("Vocabulary missing required token groups; check preprocess output.")

    forced: dict[int, int] = dict(forced_ids_by_pos or {})
    if forced_hall_id is not None:
        forced.setdefault(1, int(forced_hall_id))

    seq: list[int] = [bos_id]
    min_sites = int(min_sites)
    if min_sites < 1:
        raise ValueError("min_sites must be >= 1")
    if min_sites > int(max_sites):
        raise ValueError("min_sites must be <= max_sites")

    for pos in range(1, max_len):
        if pos in forced:
            next_id = int(forced[pos])
            allowed = _allowed_ids_for_pos(pos, groups=groups, eos_id=eos_id, min_sites=min_sites, max_sites=max_sites)
            if next_id not in allowed:
                raise ValueError(f"Forced token id {next_id} is not allowed at pos={pos}")
            seq.append(next_id)
            if next_id == eos_id:
                break
            continue
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        key_padding_mask = input_ids.eq(pad_id)
        with torch.no_grad():
            logits = model(input_ids, key_padding_mask=key_padding_mask)[0, -1]  # (V,)

        allowed = _allowed_ids_for_pos(pos, groups=groups, eos_id=eos_id, min_sites=min_sites, max_sites=max_sites)
        next_id = _sample_next_id(logits, allowed=allowed, temperature=temperature, top_k=top_k)
        seq.append(next_id)
        if next_id == eos_id:
            break

    return seq


def generate_cifs(
    *,
    ckpt_path: Path,
    n_samples: int,
    out_dir: Path,
    max_sites: int,
    temperature: float,
    top_k: int,
    restrict_elements: list[str] | None,
    nelements_min: int,
    nelements_max: int | None,
    include_metalloids: bool,
    substitute_elements: list[str] | None = None,
    validate_min_dist: float = 1.5,
    validate_symprec: float = 1e-2,
    validate_min_dist_factor: float | None = 0.75,
    validate_max_dist_factor: float | None = None,
    validate_max_nn_factor: float | None = 1.35,
    validate_min_coordination: int | None = 1,
    validate_require_connected: bool | None = None,
    validate_max_atoms: int | None = None,
    validate_vol_per_atom_min: float | None = 5.0,
    validate_vol_per_atom_max: float | None = 40.0,
    autoscale_cell: bool = True,
    dedup_enabled: bool = True,
    dedup_mode: str = "prototype",
    dedup_symprec: float = 1e-2,
    dedup_frac_tol: float = 1e-2,
    dedup_cell_tol: float = 2e-1,
    attempts_factor: int = 50,
    stop_after_no_new: int = 0,
    hall_mode: str = "model",
    fixed_hall: int | None = None,
    fixed_spacegroup: int | None = None,
) -> None:
    if nelements_min < 1:
        raise ValueError("nelements_min must be >= 1")
    if nelements_max is not None and nelements_max < nelements_min:
        raise ValueError("nelements_max must be >= nelements_min")
    if restrict_elements is not None and nelements_min > len(set(restrict_elements)):
        raise ValueError("restrict_elements has fewer unique elements than nelements_min")
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if attempts_factor < 1:
        raise ValueError("attempts_factor must be >= 1")
    if stop_after_no_new < 0:
        raise ValueError("stop_after_no_new must be >= 0")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    id_to_token = ckpt["id_to_token"]
    pad_id = int(ckpt["pad_id"])
    model_cfg = ModelConfig(**ckpt["model_config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerLM(model_cfg, id_to_token=id_to_token).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

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

    out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute HALL token mappings for symmetry sampling.
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
                raise ValueError(f"fixed_hall={fixed_hall} not present in vocab (did you seed all hall tokens?)")
            return int(hid)

        if fixed_spacegroup is not None:
            sg = int(fixed_spacegroup)
            if halls_by_sg_trained is not None and sg in halls_by_sg_trained:
                for hall in halls_by_sg_trained[sg]:
                    hid = hall_to_id.get(int(hall))
                    if hid is not None:
                        return int(hid)

            if rep_hall_by_sg is None:
                rep_hall_by_sg = _representative_hall_by_sg()
            hall = rep_hall_by_sg.get(sg)
            if hall is None:
                raise ValueError(f"Unknown fixed_spacegroup={fixed_spacegroup} (expected 1..230)")
            hid = hall_to_id.get(int(hall))
            if hid is None:
                raise ValueError(f"Representative hall={hall} for SG={fixed_spacegroup} not present in vocab")
            return int(hid)

        mode = str(hall_mode or "model").strip().lower()
        if mode == "model":
            return None
        if mode == "uniform_530":
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

        raise ValueError(f"Unknown hall_mode: {hall_mode!r}")

    written = 0
    attempts = 0
    max_attempts = int(n_samples) * int(attempts_factor)
    max_len = int(model_cfg.max_len)
    elem_filter = IntermetallicFilter(include_metalloids=include_metalloids)
    rng = random.Random()
    seen: set[str] = set()
    reject: dict[str, int] = {}
    no_new = 0

    def _next_path(idx: int) -> Path:
        return out_dir / f"genim_{idx:05d}.cif"

    next_idx = 0
    while _next_path(next_idx).exists():
        next_idx += 1

    if dedup_enabled:
        for p in sorted([p for p in out_dir.glob("*.cif") if p.is_file()]):
            try:
                a = read(str(p))
                h = atoms_hash(
                    a,
                    symprec=float(dedup_symprec),
                    frac_tol=float(dedup_frac_tol),
                    cell_tol=float(dedup_cell_tol),
                    mode=str(dedup_mode),
                )
                seen.add(h)
            except Exception:
                continue

    k_sub = None
    proto_pool: list[str] | None = None
    if substitute_elements is not None and restrict_elements is None:
        target_sorted = sorted(set(substitute_elements), key=lambda s: atomic_numbers.get(s, 999))
        k_sub = int(len(target_sorted))
        proto_pool = [sym for sym in allowed_intermetallic_elements(include_metalloids=include_metalloids) if f"E_{sym}" in vocab]
        if len(proto_pool) < k_sub:
            raise RuntimeError("Prototype element pool too small for substitution; re-run preprocess with --seed-all-elements.")

    while written < n_samples and attempts < max_attempts:
        attempts += 1
        forced_hall_id = _pick_forced_hall_id()

        restrict_this = restrict_elements
        forced_ids_by_pos = None
        min_sites = int(nelements_min)
        if k_sub is not None and proto_pool is not None:
            min_sites = max(min_sites, int(k_sub))
            proto_elements = rng.sample(proto_pool, k=int(k_sub))
            restrict_this = list(proto_elements)
            forced_ids_by_pos = {}
            for i, sym in enumerate(proto_elements):
                pos = 8 + 5 * i
                forced_ids_by_pos[pos] = int(vocab[f"E_{sym}"])

        ids = sample_sequence(
            model,
            vocab=vocab,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            max_sites=max_sites,
            min_sites=min_sites,
            restrict_elements=restrict_this,
            device=device,
            forced_hall_id=forced_hall_id,
            forced_ids_by_pos=forced_ids_by_pos,
        )
        tokens = [id_to_token[i] for i in ids if i != pad_id]
        try:
            atoms = decode_tokens_to_atoms(tokens, cfg=dec_cfg, max_sites=max_sites)
        except Exception:
            reject["decode_error"] = reject.get("decode_error", 0) + 1
            no_new += 1
            if stop_after_no_new and no_new >= int(stop_after_no_new):
                break
            continue

        # Basic sanity: avoid totally collapsed cells.
        if not math.isfinite(float(atoms.get_volume())) or float(atoms.get_volume()) <= 1e-6:
            reject["invalid_volume"] = reject.get("invalid_volume", 0) + 1
            no_new += 1
            if stop_after_no_new and no_new >= int(stop_after_no_new):
                break
            continue

        if substitute_elements is not None:
            target = list(substitute_elements)
            if len(set(target)) < len(target):
                reject["bad_substitution_elements"] = reject.get("bad_substitution_elements", 0) + 1
                no_new += 1
                if stop_after_no_new and no_new >= int(stop_after_no_new):
                    break
                continue

            src = sorted(set(atoms.get_chemical_symbols()), key=lambda s: atomic_numbers.get(s, 999))
            target_sorted = sorted(set(target), key=lambda s: atomic_numbers.get(s, 999))
            if len(src) != len(target_sorted):
                reject["substitution_mismatch"] = reject.get("substitution_mismatch", 0) + 1
                no_new += 1
                if stop_after_no_new and no_new >= int(stop_after_no_new):
                    break
                continue
            mapping = dict(zip(src, target_sorted))
            atoms = atoms.copy()
            atoms.set_chemical_symbols([mapping[s] for s in atoms.get_chemical_symbols()])

        if autoscale_cell:
            atoms, _scale = autoscale_cell_isotropic(
                atoms,
                min_dist=float(validate_min_dist),
                min_dist_factor=validate_min_dist_factor,
                max_dist_factor=validate_max_dist_factor,
                vol_per_atom_min=validate_vol_per_atom_min,
                vol_per_atom_max=validate_vol_per_atom_max,
            )

        ok, reason = validate_atoms(
            atoms,
            min_dist=float(validate_min_dist),
            symprec=float(validate_symprec),
            min_dist_factor=validate_min_dist_factor,
            max_dist_factor=validate_max_dist_factor,
            max_nn_factor=validate_max_nn_factor,
            min_coordination=validate_min_coordination,
            require_connected=validate_require_connected,
            max_atoms=validate_max_atoms,
            vol_per_atom_min=validate_vol_per_atom_min,
            vol_per_atom_max=validate_vol_per_atom_max,
        )
        if not ok:
            reject[reason] = reject.get(reason, 0) + 1
            no_new += 1
            if stop_after_no_new and no_new >= int(stop_after_no_new):
                break
            continue

        elems = set(atoms.get_chemical_symbols())
        if not elem_filter.is_intermetallic(elems):
            reject["not_intermetallic"] = reject.get("not_intermetallic", 0) + 1
            no_new += 1
            if stop_after_no_new and no_new >= int(stop_after_no_new):
                break
            continue
        if len(elems) < nelements_min:
            reject["nelements_too_few"] = reject.get("nelements_too_few", 0) + 1
            no_new += 1
            if stop_after_no_new and no_new >= int(stop_after_no_new):
                break
            continue
        if nelements_max is not None and len(elems) > nelements_max:
            reject["nelements_too_many"] = reject.get("nelements_too_many", 0) + 1
            no_new += 1
            if stop_after_no_new and no_new >= int(stop_after_no_new):
                break
            continue

        if dedup_enabled:
            h = atoms_hash(
                atoms,
                symprec=float(dedup_symprec),
                frac_tol=float(dedup_frac_tol),
                cell_tol=float(dedup_cell_tol),
                mode=str(dedup_mode),
            )
            if h in seen:
                reject["duplicate"] = reject.get("duplicate", 0) + 1
                no_new += 1
                if stop_after_no_new and no_new >= int(stop_after_no_new):
                    break
                continue
            seen.add(h)

        path = _next_path(next_idx)
        write(str(path), atoms, format="cif")
        written += 1
        next_idx += 1
        no_new = 0

    print(f"Generated {written}/{n_samples} unique CIFs under {out_dir} (attempts={attempts}, max_attempts={max_attempts}).")
    if reject:
        top = sorted(reject.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("Top rejections:")
        for k, v in top:
            print(f"  - {k}: {v}")
