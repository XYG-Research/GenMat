from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .model import CausalTransformerLM, ModelConfig


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_lm(
    *,
    data_path: Path,
    out_ckpt: Path,
    steps: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    element_emb: str,
    seed: int,
) -> None:
    _set_seed(seed)
    try:
        blob = torch.load(data_path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(data_path, map_location="cpu")
    seq_cpu = blob["sequences"]  # (N, L) on CPU
    vocab = blob["vocab"]
    id_to_token = blob["id_to_token"]
    cfg = blob["config"]

    pad_id = int(vocab["<PAD>"])
    max_len = int(seq_cpu.shape[1])
    vocab_size = int(len(id_to_token))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Speed up host→device transfers when we keep the dataset on CPU.
        try:
            seq_cpu = seq_cpu.pin_memory()
        except Exception:
            pass
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        element_emb=element_emb,
    )
    model = CausalTransformerLM(model_cfg, id_to_token=id_to_token).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    n = int(seq_cpu.shape[0])
    if n == 0:
        raise ValueError("Empty dataset")

    model.train()
    pbar = tqdm(range(steps), desc="train")
    for step in pbar:
        # Keep the full dataset on CPU for scalability; move only minibatches to GPU.
        idx = torch.randint(0, n, (batch_size,), device="cpu")
        batch = seq_cpu[idx].to(device, non_blocking=True)  # (B, L)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]

        key_padding_mask = inp.eq(pad_id)
        logits = model(inp, key_padding_mask=key_padding_mask)
        loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 50 == 0 or step == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu()))

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": asdict(model_cfg),
        "vocab": vocab,
        "id_to_token": id_to_token,
        "tokenize_config": cfg,
        "pad_id": pad_id,
        "data_stats": blob.get("stats", None),
        "symmetry_stats": blob.get("symmetry_stats", None),
    }
    torch.save(ckpt, out_ckpt)
