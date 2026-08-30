from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"

    @staticmethod
    def new() -> "Vocab":
        v = Vocab(token_to_id={}, id_to_token=[])
        v.add(v.pad_token)
        v.add(v.bos_token)
        v.add(v.eos_token)
        return v

    def add(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        idx = len(self.id_to_token)
        self.token_to_id[tok] = idx
        self.id_to_token.append(tok)
        return idx

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]


def bin_frac(x: float, *, bins: int) -> int:
    x = float(x) % 1.0
    i = int(np.floor(x * bins))
    return int(np.clip(i, 0, bins - 1))


def unbin_frac(i: int, *, bins: int) -> float:
    i = int(np.clip(i, 0, bins - 1))
    return float(i) / float(bins)


def bin_range(x: float, *, lo: float, hi: float, bins: int) -> int:
    if not (hi > lo):
        raise ValueError("hi must be > lo")
    x = float(x)
    t = (x - lo) / (hi - lo)
    i = int(np.floor(t * bins))
    return int(np.clip(i, 0, bins - 1))


def unbin_range(i: int, *, lo: float, hi: float, bins: int) -> float:
    if not (hi > lo):
        raise ValueError("hi must be > lo")
    i = int(np.clip(i, 0, bins - 1))
    # Use bin center for continuous values.
    t = (float(i) + 0.5) / float(bins)
    return float(lo + t * (hi - lo))
