from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .structure_format import StructureRecord
from .symmetry import extract_wyckoff_structure


@dataclass(frozen=True)
class InspectConfig:
    top_k: int = 15
    wyckoff: bool = False
    symprec: float = 1e-2


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_jsonl(path: Path, *, cfg: InspectConfig) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    n_lines = 0
    nsites = Counter()
    nelements = Counter()
    elem_freq = Counter()
    eah_vals = []

    wy_sites = Counter()
    hall_freq = Counter()

    with path.open("r", encoding="utf-8") as f:
        it = f
        if cfg.wyckoff:
            it = tqdm(f, desc="inspect(wyckoff)")

        for line in it:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)

            nsites[int(row.get("nsites") or 0)] += 1
            eah = row.get("energy_above_hull", None)
            if eah is not None:
                try:
                    eah_vals.append(float(eah))
                except Exception:
                    pass

            s = row.get("structure", {})
            sp = list(s.get("species", []))
            uniq = sorted(set(sp))
            nelements[len(uniq)] += 1
            for e in uniq:
                elem_freq[str(e)] += 1

            if cfg.wyckoff:
                rec = StructureRecord(
                    lattice=np.asarray(s["lattice"], dtype=float),
                    frac_coords=np.asarray(s["frac_coords"], dtype=float),
                    species=sp,
                )
                wy = extract_wyckoff_structure(rec.to_ase(), symprec=float(cfg.symprec))
                wy_sites[len(wy.sites)] += 1
                hall_freq[int(wy.hall_number)] += 1

    print(f"[JSONL] {path}")
    print(f"  structures: {n_lines}")
    if nsites:
        print(f"  nsites: min={min(nsites)} max={max(nsites)} top={nsites.most_common(5)}")
    if nelements:
        print(f"  nelements(distinct): {dict(sorted(nelements.items()))}")

    if elem_freq:
        print(f"  unique elements: {len(elem_freq)}")
        print(f"  top elements: {elem_freq.most_common(int(cfg.top_k))}")

    if eah_vals:
        a = np.asarray(eah_vals, dtype=float)
        print(
            "  energy_above_hull(eV/atom): "
            f"min={float(a.min()):.3f} p50={float(np.median(a)):.3f} p95={float(np.quantile(a, 0.95)):.3f} max={float(a.max()):.3f}"
        )

    if cfg.wyckoff:
        print(f"  wyckoff representative sites: top={wy_sites.most_common(10)} max={max(wy_sites) if wy_sites else None}")
        print(f"  hall numbers: unique={len(hall_freq)} top={hall_freq.most_common(10)}")


def inspect_tokens(path: Path, *, cfg: InspectConfig) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    blob = _safe_torch_load(path)
    seq = blob["sequences"]
    lengths = blob["lengths"]
    vocab = blob["vocab"]
    id_to_token = blob["id_to_token"]
    tok_cfg = blob.get("config", {})
    stats = blob.get("stats", {})
    sym_stats = blob.get("symmetry_stats", None)

    hall_tokens = [t for t in vocab if t.startswith("HALL_")]
    elem_tokens = [t for t in vocab if t.startswith("E_")]

    site_counts = Counter()
    hall_freq = Counter()
    elem_freq = Counter()

    pad_id = int(vocab.get("<PAD>", 0))

    for ids, L in zip(seq.tolist(), lengths.tolist()):
        ids = ids[:L]
        toks = [id_to_token[i] for i in ids if i != pad_id]
        if len(toks) < 10:
            continue
        # HALL is token 1
        h = toks[1]
        if h.startswith("HALL_"):
            try:
                hall_freq[int(h[5:])] += 1
            except Exception:
                pass
        # count sites
        n = L - (1 + 1 + 3 + 3 + 1)
        if n >= 0 and n % 5 == 0:
            site_counts[n // 5] += 1
        # count elements (distinct)
        elems = set()
        i = 8
        while i < len(toks) and toks[i] != "<EOS>":
            if toks[i].startswith("E_"):
                elems.add(toks[i][2:])
            i += 5
        for e in elems:
            elem_freq[e] += 1

    print(f"[TOKENS] {path}")
    print(f"  sequences: {int(seq.shape[0])}  max_len: {int(seq.shape[1])}  vocab: {len(vocab)}")
    if stats:
        print(f"  stats: {stats}")
    if tok_cfg:
        print(f"  tokenize_config: {tok_cfg}")
    print(f"  hall tokens in vocab: {len(hall_tokens)}")
    print(f"  element tokens in vocab: {len(elem_tokens)}")

    if site_counts:
        print(f"  site_count(dist): top={site_counts.most_common(10)} max={max(site_counts)}")
    if elem_freq:
        print(f"  unique elements in sequences: {len(elem_freq)}")
        print(f"  top elements: {elem_freq.most_common(int(cfg.top_k))}")
    if hall_freq:
        print(f"  hall numbers in sequences: unique={len(hall_freq)} top={hall_freq.most_common(10)}")

        # Derive spacegroup coverage from hall numbers.
        try:
            import spglib

            sg_freq = Counter()
            for hall, n in hall_freq.items():
                t = spglib.get_spacegroup_type(int(hall))
                if t is None:
                    continue
                sg = int(getattr(t, "number") if hasattr(t, "number") else t["number"])
                sg_freq[sg] += int(n)
            if sg_freq:
                print(f"  spacegroups in sequences: unique={len(sg_freq)} top={sg_freq.most_common(10)}")
        except Exception:
            pass

    if isinstance(sym_stats, dict):
        sg_counts = sym_stats.get("sg_counts", None)
        if isinstance(sg_counts, dict):
            try:
                sg_freq2 = Counter({int(k): int(v) for k, v in sg_counts.items()})
                print(f"  (symmetry_stats) spacegroups: unique={len(sg_freq2)}")
            except Exception:
                pass


def inspect_any(path: Path, *, cfg: InspectConfig) -> None:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        inspect_jsonl(path, cfg=cfg)
        return
    if suf == ".pt":
        inspect_tokens(path, cfg=cfg)
        return
    raise ValueError(f"Unsupported input type: {path} (expected .jsonl or .pt)")
