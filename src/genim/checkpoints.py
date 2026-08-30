from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch

from .model import CausalTransformerLM, ModelConfig

CHECKPOINT_FORMAT_VERSION = 2
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {"model_state", "model_config", "vocab", "id_to_token", "tokenize_config", "pad_id"}
)


class CheckpointError(ValueError):
    """Raised when a checkpoint is unreadable or violates the GenMat schema."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        blob = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        blob = torch.load(path, map_location=map_location)
    if not isinstance(blob, dict):
        raise CheckpointError(f"Checkpoint root must be a mapping, got {type(blob).__name__}")
    return blob


def validate_checkpoint_blob(blob: dict[str, Any]) -> int:
    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(blob))
    if missing:
        raise CheckpointError(f"Checkpoint is missing required keys: {', '.join(missing)}")
    version = int(blob.get("format_version", 1))
    if version < 1 or version > CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint format {version}; this GenMat release supports formats 1..{CHECKPOINT_FORMAT_VERSION}"
        )
    vocab = blob["vocab"]
    id_to_token = blob["id_to_token"]
    if not isinstance(vocab, dict) or not isinstance(id_to_token, (list, tuple)):
        raise CheckpointError("Checkpoint vocab/id_to_token have invalid types")
    for token in ("<PAD>", "<BOS>", "<EOS>"):
        if token not in vocab:
            raise CheckpointError(f"Checkpoint vocabulary is missing {token}")
    if len(vocab) != len(id_to_token):
        raise CheckpointError("Checkpoint vocab and id_to_token lengths differ")
    for token, index in vocab.items():
        index = int(index)
        if not (0 <= index < len(id_to_token)):
            raise CheckpointError(f"Vocabulary id is out of range for token {token!r}")
        if str(id_to_token[index]) != str(token):
            raise CheckpointError(f"vocab/id_to_token disagree for token {token!r}")
    if int(blob["pad_id"]) != int(vocab["<PAD>"]):
        raise CheckpointError("Checkpoint pad_id does not match vocab['<PAD>']")
    try:
        model_config = ModelConfig(**blob["model_config"])
    except Exception as exc:
        raise CheckpointError(f"Invalid model_config: {exc}") from exc
    if int(model_config.vocab_size) != len(vocab):
        raise CheckpointError("model_config.vocab_size does not match the vocabulary")
    tokenizer = blob["tokenize_config"]
    required_tokenizer = {"coord_bins", "len_bins", "len_min", "len_max", "ang_bins", "ang_min", "ang_max"}
    if not isinstance(tokenizer, dict) or not required_tokenizer.issubset(tokenizer):
        raise CheckpointError("Checkpoint tokenize_config is incomplete")
    return version


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    name = str(device or "auto").strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    sha256: str
    format_version: int
    blob: dict[str, Any]
    model: CausalTransformerLM
    model_config: ModelConfig
    vocab: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    device: torch.device

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.blob.get("metadata") or {})


def load_model_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "auto",
    expected_sha256: str | None = None,
) -> LoadedCheckpoint:
    path = Path(path).expanduser().resolve()
    checksum = sha256_file(path)
    if expected_sha256 is not None and checksum.lower() != str(expected_sha256).lower():
        raise CheckpointError(f"SHA256 mismatch for {path.name}: expected {expected_sha256}, got {checksum}")
    blob = safe_torch_load(path)
    version = validate_checkpoint_blob(blob)
    target = resolve_device(device)
    model_config = ModelConfig(**blob["model_config"])
    id_to_token = [str(value) for value in blob["id_to_token"]]
    model = CausalTransformerLM(model_config, id_to_token=id_to_token).to(target)
    try:
        model.load_state_dict(blob["model_state"], strict=True)
    except Exception as exc:
        raise CheckpointError(f"Model weights are incompatible with model_config: {exc}") from exc
    model.eval()
    return LoadedCheckpoint(
        path=path,
        sha256=checksum,
        format_version=version,
        blob=blob,
        model=model,
        model_config=model_config,
        vocab={str(k): int(v) for k, v in blob["vocab"].items()},
        id_to_token=id_to_token,
        pad_id=int(blob["pad_id"]),
        device=target,
    )


def download_checkpoint(
    url: str,
    *,
    cache_dir: Path,
    filename: str | None = None,
    sha256: str | None = None,
    timeout: float = 120.0,
) -> Path:
    """Download a checkpoint atomically and optionally verify its SHA256."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = filename or Path(str(url).split("?", 1)[0]).name
    if not name:
        raise ValueError("filename could not be inferred from URL")
    target = cache_dir / name
    if target.is_file() and (sha256 is None or sha256_file(target).lower() == sha256.lower()):
        return target

    fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=cache_dir)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with requests.get(url, stream=True, timeout=float(timeout)) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if sha256 is not None:
            actual = sha256_file(tmp)
            if actual.lower() != sha256.lower():
                raise CheckpointError(f"SHA256 mismatch for download: expected {sha256}, got {actual}")
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "LoadedCheckpoint",
    "download_checkpoint",
    "load_model_checkpoint",
    "resolve_device",
    "safe_torch_load",
    "sha256_file",
    "validate_checkpoint_blob",
]
