from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import nn
from ase.data import atomic_masses, atomic_numbers, covalent_radii

from .chem import METALLOIDS, NON_METALS


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_len: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    element_emb: str = "token"  # token | features
    # Legacy checkpoints omit this field and therefore retain their 4-feature
    # projection shape. New training uses periodic8.
    element_feature_set: str = "legacy4"  # legacy4 | periodic8


def _period_and_group(z: int) -> tuple[int, int]:
    """Return conventional period/group numbers from atomic number.

    Lanthanides and actinides use group 3, which is the useful continuous
    embedding convention here rather than a claim about nomenclature.
    """

    if z == 1:
        return 1, 1
    if z == 2:
        return 1, 18
    rows = (
        (2, 3, 10, (1, 2, 13, 14, 15, 16, 17, 18)),
        (3, 11, 18, (1, 2, 13, 14, 15, 16, 17, 18)),
        (4, 19, 36, tuple(range(1, 19))),
        (5, 37, 54, tuple(range(1, 19))),
    )
    for period, lo, hi, groups in rows:
        if lo <= z <= hi:
            return period, int(groups[z - lo])
    if 55 <= z <= 86:
        if z == 55:
            return 6, 1
        if z == 56:
            return 6, 2
        if 57 <= z <= 71:
            return 6, 3
        return 6, z - 68  # Hf..Rn -> groups 4..18
    if 87 <= z <= 118:
        if z == 87:
            return 7, 1
        if z == 88:
            return 7, 2
        if 89 <= z <= 103:
            return 7, 3
        return 7, z - 100  # Rf..Og -> groups 4..18
    raise ValueError(f"Atomic number out of range: {z}")


class CausalTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig, *, id_to_token: list[str] | None = None):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self._use_element_features = cfg.element_emb == "features"
        if cfg.element_emb not in ("token", "features"):
            raise ValueError("element_emb must be one of: token, features")

        if self._use_element_features:
            if id_to_token is None:
                raise ValueError("id_to_token is required when element_emb='features'")
            if len(id_to_token) != cfg.vocab_size:
                raise ValueError("id_to_token length must match vocab_size")

            feature_set = str(cfg.element_feature_set or "legacy4").strip().lower()
            if feature_set not in {"legacy4", "periodic8"}:
                raise ValueError("element_feature_set must be one of: legacy4, periodic8")
            feature_dim = 4 if feature_set == "legacy4" else 8
            feat_table = torch.zeros(cfg.vocab_size, feature_dim, dtype=torch.float32)
            is_elem = torch.zeros(cfg.vocab_size, dtype=torch.bool)
            elem_ids: list[int] = []

            for tid, tok in enumerate(id_to_token):
                if not tok.startswith("E_"):
                    continue
                sym = tok[2:]
                z = atomic_numbers.get(sym, None)
                if not z:
                    continue
                r = float(covalent_radii[z]) if z < len(covalent_radii) else float("nan")
                if not math.isfinite(r):
                    r = 0.0
                m = float(atomic_masses[z]) if z < len(atomic_masses) else float("nan")
                if not math.isfinite(m):
                    m = 0.0

                features = [
                    float(z) / 118.0,
                    float(r) / 3.0,
                    float(m) / 300.0,
                ]
                if feature_set == "legacy4":
                    features.append(1.0 if sym in METALLOIDS else 0.0)
                else:
                    period, group = _period_and_group(int(z))
                    is_metalloid = sym in METALLOIDS
                    is_nonmetal = sym in NON_METALS
                    features.extend(
                        [
                            float(period) / 7.0,
                            float(group) / 18.0,
                            0.0 if (is_metalloid or is_nonmetal) else 1.0,
                            1.0 if is_metalloid else 0.0,
                            1.0 if is_nonmetal else 0.0,
                        ]
                    )
                feat_table[tid] = torch.tensor(features, dtype=torch.float32)
                is_elem[tid] = True
                elem_ids.append(tid)

            self.register_buffer("_elem_feat_table", feat_table, persistent=False)
            self.register_buffer("_is_elem", is_elem, persistent=False)
            self.register_buffer("_elem_ids", torch.tensor(elem_ids, dtype=torch.long), persistent=False)
            self.elem_proj = nn.Linear(feature_dim, cfg.d_model, bias=True)
        else:
            self._elem_feat_table = None
            self._is_elem = None
            self._elem_ids = None
            self.elem_proj = None

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        try:
            self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        except TypeError:
            self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Precompute max causal mask (True = masked).
        mask = torch.triu(torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_mask", mask, persistent=False)

    def forward(self, input_ids: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids: (B, T)
        key_padding_mask: (B, T) bool, True for PAD positions.
        returns logits: (B, T, V)
        """
        b, t = input_ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"Sequence too long: {t} > max_len={self.cfg.max_len}")

        pos = torch.arange(t, device=input_ids.device)
        tok = self.tok_emb(input_ids)
        if self._use_element_features:
            assert self._elem_feat_table is not None
            assert self._is_elem is not None
            assert self.elem_proj is not None
            feat = self._elem_feat_table[input_ids]
            elem_mask = self._is_elem[input_ids]
            elem_emb = self.elem_proj(feat)
            tok = torch.where(elem_mask.unsqueeze(-1), elem_emb, tok)

        x = tok + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        attn_mask = self._causal_mask[:t, :t]
        h = self.enc(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(h)

        if self._use_element_features:
            assert self._elem_feat_table is not None
            assert self._elem_ids is not None
            assert self.elem_proj is not None
            if int(self._elem_ids.numel()) > 0:
                elem_w = self.elem_proj(self._elem_feat_table[self._elem_ids])  # (E, D)
                elem_logits = torch.einsum("btd,ed->bte", h, elem_w)  # (B, T, E)
                logits = logits.index_copy(2, self._elem_ids, elem_logits)

        return logits
