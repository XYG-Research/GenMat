from __future__ import annotations

import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .model import CausalTransformerLM, ModelConfig
from . import __version__
from .checkpoints import CHECKPOINT_FORMAT_VERSION, safe_torch_load, sha256_file


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
    element_feature_set: str = "periodic8",
    val_fraction: float = 0.1,
) -> None:
    _set_seed(seed)
    blob = safe_torch_load(data_path)
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
        element_feature_set=element_feature_set,
    )
    model = CausalTransformerLM(model_cfg, id_to_token=id_to_token).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    n = int(seq_cpu.shape[0])
    if n == 0:
        raise ValueError("Empty dataset")
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    if not (0.0 <= float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")

    split_generator = torch.Generator(device="cpu")
    split_generator.manual_seed(int(seed))
    order = torch.randperm(n, generator=split_generator)
    val_n = 0
    if n >= 2 and float(val_fraction) > 0:
        val_n = max(1, int(round(n * float(val_fraction))))
        val_n = min(val_n, n - 1)
    val_indices = order[:val_n]
    train_indices = order[val_n:]

    def _evaluate(indices: torch.Tensor) -> float | None:
        if int(indices.numel()) == 0:
            return None
        model.eval()
        loss_sum = 0.0
        token_count = 0
        eval_batch = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, int(indices.numel()), eval_batch):
                chosen = indices[start : start + eval_batch]
                batch = seq_cpu[chosen].to(device, non_blocking=True)
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp, key_padding_mask=inp.eq(pad_id))
                flat_tgt = tgt.reshape(-1)
                loss_sum += float(
                    nn.functional.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        flat_tgt,
                        ignore_index=pad_id,
                        reduction="sum",
                    ).detach().cpu()
                )
                token_count += int(flat_tgt.ne(pad_id).sum().item())
        model.train()
        return loss_sum / token_count if token_count else None

    model.train()
    pbar = tqdm(range(steps), desc="train")
    last_train_loss: float | None = None
    for step in pbar:
        # Keep the full dataset on CPU for scalability; move only minibatches to GPU.
        draw = torch.randint(0, int(train_indices.numel()), (batch_size,), device="cpu")
        idx = train_indices[draw]
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

        last_train_loss = float(loss.detach().cpu())
        if (step + 1) % 50 == 0 or step == 0:
            pbar.set_postfix(loss=last_train_loss)

    validation_loss = _evaluate(val_indices)

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": {
            "genmat_version": __version__,
            "genim_version": __version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_seed": int(seed),
            "training_steps": int(steps),
            "train_size": int(train_indices.numel()),
            "validation_size": int(val_indices.numel()),
            "validation_fraction": float(val_fraction),
            "last_batch_train_loss": last_train_loss,
            "validation_loss": validation_loss,
            "element_feature_set": str(element_feature_set),
            "dataset_name": Path(data_path).name,
            "dataset_sha256": sha256_file(data_path),
            "dataset_provenance": blob.get("provenance", None),
        },
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
