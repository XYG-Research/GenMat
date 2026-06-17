from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def resolve_device(device: str) -> str:
    dev = str(device or "").strip().lower() or "auto"
    if dev != "auto":
        return dev
    try:
        import torch  # type: ignore

        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


def _normalize_key(s: str) -> str:
    return str(s or "").strip().lower().replace("-", "_")


def _try_resolve_local_checkpoint(path_str: str) -> Optional[Path]:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    try:
        p = Path(raw).expanduser()
    except Exception:
        return None
    try:
        if p.exists() and p.is_file():
            return p.resolve()
    except Exception:
        return None
    return None


def _omat24_checkpoint_filename(model_name: str) -> str:
    """
    Map OMAT24 model aliases to checkpoint filenames.
    """
    key = _normalize_key(str(model_name or "").strip())
    if key == "esen_30m_oam":
        return "esen_30m_oam.pt"
    if key == "esen_30m_mptrj":
        return "esen_30m_mptrj.pt"
    if key == "esen_30m_omat":
        return "esen_30m_omat.pt"
    raw = str(model_name or "").strip()
    return Path(raw).name


def _find_local_omat24_checkpoint(*, filename: str) -> Optional[Path]:
    """
    Best-effort search for an OMAT24 checkpoint on disk.

    Supports using adspp-style local checkpoint layouts, e.g.:
      - data/checkpoints/eSEN/esen_30m_oam.pt
      - adspp/data/checkpoints/eSEN/esen_30m_oam.pt (sibling repo)
    """
    fn = str(filename or "").strip()
    if not fn:
        return None

    bases: List[Path] = []

    # User overrides: allow multiple paths separated by os.pathsep.
    for env_var in ("GENIM_CHECKPOINT_DIR", "ADSPP_CHECKPOINT_DIR"):
        env = str(os.environ.get(env_var, "") or "").strip()
        if not env:
            continue
        for part in env.split(os.pathsep):
            p0 = str(part or "").strip().strip('"')
            if not p0:
                continue
            try:
                bases.append(Path(p0).expanduser())
            except Exception:
                continue

    # CWD.
    try:
        bases.append(Path.cwd().resolve())
    except Exception:
        bases.append(Path.cwd())

    # GenIM repo root (when run from source or editable install).
    try:
        pkg_root = Path(__file__).resolve().parents[2]
        bases.append(pkg_root)
        bases.append(pkg_root.parent)
    except Exception:
        pass

    # Common sibling adspp checkout.
    try:
        if "pkg_root" in locals():
            bases.append((pkg_root.parent / "adspp").resolve())
    except Exception:
        pass

    # De-dup bases while preserving order.
    seen = set()
    uniq_bases: List[Path] = []
    for b in bases:
        try:
            key = str(b.resolve())
        except Exception:
            key = str(b)
        if key in seen:
            continue
        seen.add(key)
        uniq_bases.append(b)

    rels = [
        Path(fn),
        Path("data") / "checkpoints" / "eSEN" / fn,
        Path("data") / "checkpoints" / "OMAT24" / fn,
        Path("data") / "checkpoints" / fn,
        Path("checkpoints") / fn,
        # adspp-like
        Path("adspp") / "data" / "checkpoints" / "eSEN" / fn,
        Path("adspp") / "data" / "checkpoints" / "OMAT24" / fn,
    ]

    for base in uniq_bases:
        for rel in rels:
            try:
                cand = (base / rel).expanduser()
            except Exception:
                continue
            try:
                if cand.exists() and cand.is_file():
                    return cand.resolve()
            except Exception:
                continue
    return None


def _download_hf_checkpoint(*, repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Loading OMAT24 checkpoints requires `huggingface-hub`.\n"
            "Install with:\n"
            "  pip install huggingface-hub\n"
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        p = hf_hub_download(repo_id=str(repo_id), filename=str(filename))
        return Path(str(p)).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download OMAT24 checkpoint '{filename}' from '{repo_id}'.\n"
            "If the repo/model is gated, accept the conditions on HuggingFace and login locally:\n"
            "  python -c \"from huggingface_hub import login; login()\"\n"
            "or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN for the current shell.\n"
            "Alternatively, download the .pt file manually and set `ml.checkpoint` to its local path."
        ) from exc


def _best_effort_torch_safe_globals() -> None:
    """
    Some checkpoints may trigger PyTorch safe-unpickling issues in newer torch versions.
    This is a best-effort compatibility hook; failures are ignored.
    """
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore

        add_safe = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
        if callable(add_safe):
            add_safe([np.core.multiarray.scalar])
    except Exception:
        return


def _import_fairchem_ocp_calculator() -> Any:
    last_exc: Optional[BaseException] = None
    for modpath in ("fairchem.core.common.relaxation.ase_utils", "fairchem.core"):
        try:
            mod = __import__(modpath, fromlist=["OCPCalculator"])
            OCPCalculator = getattr(mod, "OCPCalculator", None)
            if OCPCalculator is not None:
                return OCPCalculator
        except Exception as exc:
            last_exc = exc
            continue
    raise ImportError(
        "OMAT24 (eSEN-30M-OAM) relaxation requires `fairchem-core`.\n"
        "Install with:\n"
        "  pip install fairchem-core\n"
        f"Import error: {type(last_exc).__name__ if last_exc else 'ImportError'}: {last_exc}"
    ) from last_exc


def build_omat24_calculator(
    *,
    model: str,
    device: str = "auto",
    checkpoint: str | None = None,
    quiet: bool = True,
) -> Any:
    """
    Build an ASE calculator backed by fairchem-core OMAT24 checkpoints (e.g. eSEN-30M-OAM).

    Args:
      model: alias like "eSEN-30M-OAM" or a local *.pt path/filename.
      device: auto|cpu|cuda
      checkpoint: optional local checkpoint path; overrides model-based resolution.
      quiet: suppress noisy stdout/stderr during model load.
    """
    resolved_device = resolve_device(device)
    if resolved_device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    raw_model = str(model or "").strip()
    if not raw_model:
        raise ValueError("ml.model is empty")

    ckpt = resolve_omat24_checkpoint(model=raw_model, checkpoint=checkpoint)

    _best_effort_torch_safe_globals()
    OCPCalculator = _import_fairchem_ocp_calculator()

    # OCPCalculator API differs across fairchem versions; use signature-based kwargs.
    import inspect

    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(OCPCalculator)
        if "checkpoint_path" in sig.parameters:
            kwargs["checkpoint_path"] = str(ckpt)
        elif "checkpoint" in sig.parameters:
            kwargs["checkpoint"] = str(ckpt)
        if "cpu" in sig.parameters:
            kwargs["cpu"] = bool(resolved_device == "cpu")
        elif "device" in sig.parameters:
            kwargs["device"] = resolved_device
        # Best-effort: avoid AMP by default for determinism.
        if "disable_amp" in sig.parameters:
            kwargs["disable_amp"] = True
    except Exception:
        kwargs = {"checkpoint_path": str(ckpt), "cpu": bool(resolved_device == "cpu")}

    if not quiet:
        return OCPCalculator(**kwargs)

    # fairchem/hydra can be very noisy; keep CLI output clean by default.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return OCPCalculator(**kwargs)


def resolve_omat24_checkpoint(*, model: str, checkpoint: str | None = None) -> Path:
    """
    Resolve an OMAT24 checkpoint path.

    This mirrors the behavior of `build_omat24_calculator` but only returns the path,
    enabling stable caching metadata for hull reference sets.
    """
    raw_model = str(model or "").strip()
    if not raw_model:
        raise ValueError("model is empty")

    if checkpoint:
        ckpt = _try_resolve_local_checkpoint(str(checkpoint))
        if ckpt is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return ckpt

    local = _try_resolve_local_checkpoint(raw_model)
    if local is not None:
        return local

    filename = _omat24_checkpoint_filename(raw_model)
    local2 = _find_local_omat24_checkpoint(filename=filename)
    return local2 if local2 is not None else _download_hf_checkpoint(repo_id="facebook/OMAT24", filename=filename)
