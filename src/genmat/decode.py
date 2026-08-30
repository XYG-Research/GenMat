from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import cellpar_to_cell

from .tokenizer import unbin_frac, unbin_range


@dataclass(frozen=True)
class DecodeConfig:
    coord_bins: int
    len_bins: int
    len_min: float
    len_max: float
    ang_bins: int
    ang_min: float
    ang_max: float


def _parse_int_suffix(tok: str, prefix: str) -> int:
    if not tok.startswith(prefix):
        raise ValueError(f"Expected {prefix}*, got: {tok}")
    return int(tok[len(prefix) :])


def _unique_frac(fracs: np.ndarray, *, tol: float = 1e-5) -> np.ndarray:
    fracs = np.asarray(fracs, dtype=float) % 1.0
    # Quantize to a grid for stable de-dup.
    q = np.round(fracs / tol).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return fracs[np.sort(idx)]


def _orbit_from_rep(rep: np.ndarray, *, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    rep = np.asarray(rep, dtype=float)
    pts = []
    for r, t in zip(rotations, translations):
        p = (r @ rep + t) % 1.0
        pts.append(p)
    pts = np.asarray(pts, dtype=float)
    return _unique_frac(pts)


def decode_tokens_to_atoms(tokens: list[str], *, cfg: DecodeConfig, max_sites: int) -> Atoms:
    if not tokens or tokens[0] != "<BOS>":
        raise ValueError("Sequence must start with <BOS>")
    if "<EOS>" not in tokens:
        raise ValueError("Missing <EOS>")

    # Trim at EOS.
    eos_i = tokens.index("<EOS>")
    tokens = tokens[: eos_i + 1]

    if len(tokens) < 1 + 1 + 6 + 1:
        raise ValueError("Sequence too short")

    hall = _parse_int_suffix(tokens[1], "HALL_")

    len_a = _parse_int_suffix(tokens[2], "LEN_")
    len_b = _parse_int_suffix(tokens[3], "LEN_")
    len_c = _parse_int_suffix(tokens[4], "LEN_")
    ang_a = _parse_int_suffix(tokens[5], "ANG_")
    ang_b = _parse_int_suffix(tokens[6], "ANG_")
    ang_g = _parse_int_suffix(tokens[7], "ANG_")

    a = unbin_range(len_a, lo=cfg.len_min, hi=cfg.len_max, bins=cfg.len_bins)
    b = unbin_range(len_b, lo=cfg.len_min, hi=cfg.len_max, bins=cfg.len_bins)
    c = unbin_range(len_c, lo=cfg.len_min, hi=cfg.len_max, bins=cfg.len_bins)
    alpha = unbin_range(ang_a, lo=cfg.ang_min, hi=cfg.ang_max, bins=cfg.ang_bins)
    beta = unbin_range(ang_b, lo=cfg.ang_min, hi=cfg.ang_max, bins=cfg.ang_bins)
    gamma = unbin_range(ang_g, lo=cfg.ang_min, hi=cfg.ang_max, bins=cfg.ang_bins)

    cell = np.asarray(cellpar_to_cell([a, b, c, alpha, beta, gamma]), dtype=float)

    ops = spglib.get_symmetry_from_database(hall)
    if ops is None:
        raise ValueError(f"spglib database missing hall_number={hall}")
    rotations = np.asarray(ops["rotations"], dtype=int)
    translations = np.asarray(ops["translations"], dtype=float)

    symbols: list[str] = []
    fracs: list[np.ndarray] = []

    # Sites start at token index 8.
    i = 8
    site_count = 0
    while i < len(tokens):
        if tokens[i] == "<EOS>":
            break
        if site_count >= max_sites:
            raise ValueError("Exceeded max_sites during decode")
        if i + 4 >= len(tokens):
            raise ValueError("Incomplete site block")

        el_tok = tokens[i]
        if not el_tok.startswith("E_"):
            raise ValueError(f"Expected element token at {i}, got {el_tok}")
        element = el_tok[2:]

        # W_* token is currently not needed for reconstruction, but we consume it for grammar.
        w_tok = tokens[i + 1]
        if not w_tok.startswith("W_"):
            raise ValueError(f"Expected Wyckoff token at {i+1}, got {w_tok}")

        x_bin = _parse_int_suffix(tokens[i + 2], "COORD_")
        y_bin = _parse_int_suffix(tokens[i + 3], "COORD_")
        z_bin = _parse_int_suffix(tokens[i + 4], "COORD_")
        rep = np.array(
            [
                unbin_frac(x_bin, bins=cfg.coord_bins),
                unbin_frac(y_bin, bins=cfg.coord_bins),
                unbin_frac(z_bin, bins=cfg.coord_bins),
            ],
            dtype=float,
        )

        orbit = _orbit_from_rep(rep, rotations=rotations, translations=translations)
        for p in orbit:
            symbols.append(element)
            fracs.append(p)

        site_count += 1
        i += 5

    if site_count == 0:
        raise ValueError("No sites decoded")

    frac_arr = np.asarray(fracs, dtype=float) % 1.0
    atoms = Atoms(symbols=symbols, cell=cell, scaled_positions=frac_arr, pbc=True)
    return atoms
