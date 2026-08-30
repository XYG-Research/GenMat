from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .checkpoints import sha256_file
from .chem import available_elements
from .structure_format import StructureRecord
from .symmetry import extract_wyckoff_structure
from .tokenizer import Vocab, bin_frac, bin_range


@dataclass(frozen=True)
class TokenizeConfig:
    max_sites: int
    symprec: float
    coord_bins: int
    len_bins: int
    len_min: float
    len_max: float
    ang_bins: int
    ang_min: float
    ang_max: float
    seed_all_elements: bool = False
    seed_all_hall: bool = False
    chemistry_mode: str = "any"
    include_metalloids: bool = True
    allowed_elements: tuple[str, ...] | None = None
    excluded_elements: tuple[str, ...] | None = None

    @property
    def max_len(self) -> int:
        # BOS + HALL + 3*LEN + 3*ANG + max_sites*(E + W + 3*COORD) + EOS
        return 1 + 1 + 3 + 3 + self.max_sites * 5 + 1


def _ensure_base_tokens(vocab: Vocab, cfg: TokenizeConfig) -> None:
    for i in range(cfg.len_bins):
        vocab.add(f"LEN_{i}")
    for i in range(cfg.ang_bins):
        vocab.add(f"ANG_{i}")
    for i in range(cfg.coord_bins):
        vocab.add(f"COORD_{i}")
    vocab.add("W_?")
    for c in "abcdefghijklmnopqrstuvwxyz":
        vocab.add(f"W_{c}")

    if cfg.seed_all_hall:
        for h in range(1, 531):
            vocab.add(f"HALL_{h}")

    if cfg.seed_all_elements:
        for el in available_elements(
            mode=cfg.chemistry_mode,
            include_metalloids=cfg.include_metalloids,
            allowed_elements=cfg.allowed_elements,
            excluded_elements=cfg.excluded_elements,
        ):
            vocab.add(f"E_{el}")


def _encode_one(rec: StructureRecord, *, vocab: Vocab, cfg: TokenizeConfig) -> tuple[list[int], int, int] | None:
    wy = extract_wyckoff_structure(rec.to_ase(), symprec=cfg.symprec)
    if len(wy.sites) == 0 or len(wy.sites) > cfg.max_sites:
        return None

    tokens: list[str] = [vocab.bos_token]
    tokens.append(f"HALL_{wy.hall_number}")

    a, b, c, alpha, beta, gamma = [float(x) for x in wy.cellpar.tolist()]
    tokens.append(f"LEN_{bin_range(a, lo=cfg.len_min, hi=cfg.len_max, bins=cfg.len_bins)}")
    tokens.append(f"LEN_{bin_range(b, lo=cfg.len_min, hi=cfg.len_max, bins=cfg.len_bins)}")
    tokens.append(f"LEN_{bin_range(c, lo=cfg.len_min, hi=cfg.len_max, bins=cfg.len_bins)}")
    tokens.append(f"ANG_{bin_range(alpha, lo=cfg.ang_min, hi=cfg.ang_max, bins=cfg.ang_bins)}")
    tokens.append(f"ANG_{bin_range(beta, lo=cfg.ang_min, hi=cfg.ang_max, bins=cfg.ang_bins)}")
    tokens.append(f"ANG_{bin_range(gamma, lo=cfg.ang_min, hi=cfg.ang_max, bins=cfg.ang_bins)}")

    for s in wy.sites:
        tokens.append(f"E_{s.element}")
        tokens.append(f"W_{s.wyckoff}" if len(s.wyckoff) == 1 and s.wyckoff.isalpha() else "W_?")
        tokens.append(f"COORD_{bin_frac(float(s.frac[0]), bins=cfg.coord_bins)}")
        tokens.append(f"COORD_{bin_frac(float(s.frac[1]), bins=cfg.coord_bins)}")
        tokens.append(f"COORD_{bin_frac(float(s.frac[2]), bins=cfg.coord_bins)}")

    tokens.append(vocab.eos_token)

    # Extend vocab for any new HALL/E_ tokens.
    for t in tokens:
        if t not in vocab.token_to_id:
            vocab.add(t)

    return vocab.encode(tokens), int(wy.hall_number), int(wy.spacegroup_number)


def preprocess_jsonl_to_tokens(
    *,
    in_path: Path,
    out_path: Path,
    max_sites: int,
    symprec: float,
    coord_bins: int,
    len_bins: int,
    len_min: float,
    len_max: float,
    ang_bins: int,
    ang_min: float,
    ang_max: float,
    seed_all_elements: bool = False,
    seed_all_hall: bool = False,
    chemistry_mode: str = "any",
    include_metalloids: bool = True,
    allowed_elements: list[str] | None = None,
    excluded_elements: list[str] | None = None,
) -> None:
    cfg = TokenizeConfig(
        max_sites=max_sites,
        symprec=symprec,
        coord_bins=coord_bins,
        len_bins=len_bins,
        len_min=len_min,
        len_max=len_max,
        ang_bins=ang_bins,
        ang_min=ang_min,
        ang_max=ang_max,
        seed_all_elements=seed_all_elements,
        seed_all_hall=seed_all_hall,
        chemistry_mode=chemistry_mode,
        include_metalloids=include_metalloids,
        allowed_elements=None if allowed_elements is None else tuple(allowed_elements),
        excluded_elements=None if excluded_elements is None else tuple(excluded_elements),
    )
    vocab = Vocab.new()
    _ensure_base_tokens(vocab, cfg)

    sequences: list[list[int]] = []
    lengths: list[int] = []
    hall_counts: dict[int, int] = {}
    sg_counts: dict[int, int] = {}
    halls_by_sg: dict[int, set[int]] = {}

    kept = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="preprocess"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                s = row.get("structure", None)
                if not isinstance(s, dict):
                    skipped += 1
                    continue
                rec = StructureRecord(
                    lattice=np.asarray(s["lattice"], dtype=float),
                    frac_coords=np.asarray(s["frac_coords"], dtype=float),
                    species=list(s["species"]),
                )
                enc = _encode_one(rec, vocab=vocab, cfg=cfg)
                if enc is None:
                    skipped += 1
                    continue
                ids, hall, sg = enc
                if len(ids) > cfg.max_len:
                    skipped += 1
                    continue

                pad = vocab.pad_id
                padded = ids + [pad] * (cfg.max_len - len(ids))
                sequences.append(padded)
                lengths.append(len(ids))
                kept += 1
                hall_counts[hall] = hall_counts.get(hall, 0) + 1
                sg_counts[sg] = sg_counts.get(sg, 0) + 1
                halls_by_sg.setdefault(sg, set()).add(hall)
            except Exception:
                skipped += 1
                continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format_version": 1,
        "provenance": {
            "source_name": in_path.name,
            "source_sha256": sha256_file(in_path),
        },
        "sequences": torch.tensor(sequences, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "vocab": vocab.token_to_id,
        "id_to_token": vocab.id_to_token,
        "config": cfg.__dict__,
        "stats": {"kept": kept, "skipped": skipped},
        "symmetry_stats": {
            "hall_counts": hall_counts,
            "sg_counts": sg_counts,
            "halls_by_sg": {int(k): sorted([int(x) for x in v]) for k, v in halls_by_sg.items()},
        },
    }
    torch.save(blob, out_path)
