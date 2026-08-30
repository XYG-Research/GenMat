from __future__ import annotations

import gc
import importlib
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from ase import Atoms

from ..checkpoints import CheckpointError, resolve_device, safe_torch_load, sha256_file
from ..structure_format import StructureRecord
from ..validate import ValidationReport, validate_atoms_report
from .base import (
    BackendCapabilities,
    ConstraintApplication,
    GeneratedCandidate,
    GenerationConstraints,
    GenerationSettings,
    ObservableRole,
    ScientificObservable,
    evaluate_constraints,
)


MATRA_REQUIRED_KEYS = frozenset({"state_dict", "vocab"})

_MATRA_STRUCTURAL_CAPABILITIES = {
    "elements": ConstraintApplication.CONDITIONING,
    "stoichiometry": ConstraintApplication.CONDITIONING,
    "spacegroup_number": ConstraintApplication.CONDITIONING,
    "matra_wyckoff_indices": ConstraintApplication.CONDITIONING,
}


def _capabilities_from_props(props: list[str], *, assumed: bool = False) -> BackendCapabilities:
    handling = dict(_MATRA_STRUCTURAL_CAPABILITIES)
    normalised = {str(value).strip().lower() for value in props}
    if "ehull_disc" in normalised:
        handling["stability"] = ConstraintApplication.CONDITIONING
    if "ehull" in normalised:
        handling["target_e_hull"] = ConstraintApplication.CONDITIONING
    return BackendCapabilities(
        handling,
        notes=(
            "Injected Matra model capabilities were assumed; provide an explicit declaration "
            "for production services."
            if assumed
            else "Derived from the inspected checkpoint property configuration. Prompt "
            "application does not by itself verify the decoded scientific target."
        ),
    )


@dataclass(frozen=True)
class MatraCheckpointInfo:
    path: Path
    sha256: str
    size_bytes: int
    matra_version: str
    config: dict[str, Any]
    vocab_size: int
    tensor_count: int
    parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["path"] = str(self.path)
        return record


def _matra_config(blob: dict[str, Any]) -> dict[str, Any]:
    config = dict(blob.get("config") or {})
    if not config and isinstance(blob.get("hyper_parameters"), dict):
        supported = {
            "num_layers",
            "d_model",
            "num_heads",
            "dff_ratio",
            "dropout_rate",
            "embedding",
            "props",
            "max_len",
        }
        config = {key: value for key, value in blob["hyper_parameters"].items() if key in supported}
    return {
        "num_layers": int(config.get("num_layers", 2)),
        "d_model": int(config.get("d_model", 128)),
        "num_heads": int(config.get("num_heads", 4)),
        "dff_ratio": int(config.get("dff_ratio", 1)),
        "dropout_rate": float(config.get("dropout_rate", 0.1)),
        "embedding": str(config.get("embedding", "positional")),
        "props": list(config.get("props", [])),
        "max_len": int(config.get("max_len", 150)),
    }


def inspect_matra_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> MatraCheckpointInfo:
    """Safely inspect a Matra checkpoint without importing or executing Matra."""

    checkpoint_path = Path(path).expanduser().resolve()
    checksum = sha256_file(checkpoint_path)
    if expected_sha256 is not None and checksum.lower() != str(expected_sha256).lower():
        raise CheckpointError(
            f"SHA256 mismatch for {checkpoint_path.name}: expected {expected_sha256}, got {checksum}"
        )
    blob = safe_torch_load(checkpoint_path, map_location="cpu")
    missing = sorted(MATRA_REQUIRED_KEYS.difference(blob))
    if missing:
        raise CheckpointError(f"Matra checkpoint is missing required keys: {', '.join(missing)}")
    state = blob["state_dict"]
    vocab = blob["vocab"]
    if not isinstance(state, dict) or not isinstance(vocab, dict):
        raise CheckpointError("Matra state_dict and vocab must be mappings")
    config = _matra_config(blob)
    required_tokens = {
        "PAD",
        "C*",
        "STOP",
        "GLOBALSTOP",
        "AMT",
        "ELMS",
        "STOICH",
        "SPACEGROUP",
        "WYCKOFF",
        "LATTICE",
    }
    required_tokens.update(str(prop).upper() for prop in config["props"])
    absent_tokens = sorted(required_tokens.difference(str(key) for key in vocab))
    if absent_tokens:
        raise CheckpointError(f"Matra vocabulary is missing required tokens: {', '.join(absent_tokens)}")
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    if not tensors:
        raise CheckpointError("Matra state_dict contains no tensors")
    matra_version = blob.get("__matra_version__")
    info = MatraCheckpointInfo(
        path=checkpoint_path,
        sha256=checksum,
        size_bytes=checkpoint_path.stat().st_size,
        matra_version="unknown" if matra_version is None else str(matra_version),
        config=config,
        vocab_size=len(vocab),
        tensor_count=len(tensors),
        parameter_count=sum(int(value.numel()) for value in tensors),
    )
    del blob, state, vocab, tensors
    gc.collect()
    return info


def _import_matra() -> tuple[type[Any], type[Any]]:
    try:
        matra_module = importlib.import_module("matra")
        vocab_module = importlib.import_module("matra.vocab")
    except ImportError as exc:
        raise ImportError(
            "Matra support is optional. Install the approved Matra source first, for example "
            "`python -m pip install -e ../matra-genoa-preview`. Matra is licensed for "
            "non-commercial research, education, evaluation and personal use."
        ) from exc
    return matra_module.MatraGenoa, vocab_module.Vocab


def _pymatgen_to_ase(structure: Any) -> Atoms:
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    frac = np.asarray(structure.frac_coords, dtype=float)
    species: list[str] = []
    for value in structure.species:
        symbol = getattr(value, "symbol", None)
        species.append(str(symbol or value))
    return StructureRecord(lattice=lattice, frac_coords=frac % 1.0, species=species).to_ase()


@contextmanager
def _temporary_random_seed(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _slug(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    return "-".join(part for part in text.split("-") if part) or "model"


_MATRA_OBSERVABLE_NAMES = {
    "EHULL": ("energy_above_hull", "eV/atom"),
    "EHULL_DISC": ("stability_class", None),
}


def _matra_blocks(sequence: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for chunk in re.split(r"\s+STOP\s+", str(sequence).strip()):
        tokens = chunk.split()
        if tokens:
            blocks[tokens[0].upper()] = tokens[1:]
    return blocks


def _parse_matra_value(name: str, tokens: list[str]) -> float | int | str | bool | None:
    if not tokens:
        return None
    token = tokens[0]
    if name == "EHULL_DISC":
        label = token.upper()
        if label == "EH0":
            return "below_0.075_eV_per_atom"
        if label == "EH1":
            return "at_or_above_0.075_eV_per_atom"
        return token
    numeric = token[2:] if token.startswith("C*") else token
    try:
        value = float(numeric)
    except ValueError:
        return token
    return value if np.isfinite(value) else None


def extract_matra_observables(
    sequence: str,
    *,
    prompt: str,
    properties: list[str],
    model_name: str,
    checkpoint_sha256: str | None,
) -> list[ScientificObservable]:
    """Extract checkpoint property blocks without promoting them to calculations."""

    sequence_blocks = _matra_blocks(sequence)
    prompt_blocks = _matra_blocks(prompt)
    observables: list[ScientificObservable] = []
    for raw_name in properties:
        block_name = str(raw_name).strip().upper()
        if block_name not in sequence_blocks:
            continue
        value = _parse_matra_value(block_name, sequence_blocks[block_name])
        if value is None:
            continue
        public_name, unit = _MATRA_OBSERVABLE_NAMES.get(
            block_name,
            (block_name.lower(), None),
        )
        conditioned = block_name in prompt_blocks
        observables.append(
            ScientificObservable(
                name=public_name,
                value=value,
                unit=unit,
                role=(
                    ObservableRole.CONDITIONING_TARGET
                    if conditioned
                    else ObservableRole.MODEL_EMISSION
                ),
                method=(
                    "matra.prompt_property_block"
                    if conditioned
                    else "matra.autoregressive_property_block"
                ),
                evidence_level="L0",
                source=f"matra_sequence.{block_name}",
                model_name=model_name,
                checkpoint_sha256=checkpoint_sha256,
                independently_validated=False,
                detail=(
                    "Conditioning target copied into the generated sequence; it is not an "
                    "independent energy calculation."
                    if conditioned
                    else "Property token emitted by Matra Genoa; validate with a named "
                    "energy model and compatible reference set before making a stability claim."
                ),
            )
        )
    return observables


class MatraBackend:
    """Safe, optional adapter for Matra Genoa inference checkpoints.

    Checkpoints are first loaded with PyTorch's weights-only mode, validated,
    and then used to construct an uninitialised Matra model. This deliberately
    bypasses Matra's historical ``weights_only=False`` convenience loader.
    """

    backend_name = "matra"
    capabilities = _capabilities_from_props(["ehull", "ehull_disc"], assumed=True)

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        checkpoint_sha256: str | None,
        checkpoint_info: MatraCheckpointInfo | None = None,
        capabilities: BackendCapabilities | None = None,
    ):
        self.model = model
        self.model_name = str(model_name)
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_info = checkpoint_info
        self.property_names = (
            list(checkpoint_info.config.get("props", []))
            if checkpoint_info is not None
            else list(getattr(model, "config", {}).get("props", []) or [])
        )
        self.capabilities = capabilities or _capabilities_from_props(
            self.property_names,
            assumed=True,
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        model_name: str = "injected-matra",
        checkpoint_sha256: str | None = None,
        capabilities: BackendCapabilities | None = None,
    ) -> "MatraBackend":
        """Construct an adapter around an injected model (useful for services/tests)."""

        return cls(
            model,
            model_name=model_name,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_info=None,
            capabilities=capabilities,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "auto",
        expected_sha256: str | None = None,
        model_name: str | None = None,
        work_dir: str | Path | None = None,
    ) -> "MatraBackend":
        checkpoint_path = Path(path).expanduser().resolve()
        info = inspect_matra_checkpoint(checkpoint_path, expected_sha256=expected_sha256)
        blob = safe_torch_load(checkpoint_path, map_location="cpu")
        model_class, vocab_class = _import_matra()
        config = _matra_config(blob)
        vocab = vocab_class(token2idx={str(key): int(value) for key, value in blob["vocab"].items()})
        resolved_work_dir = (
            Path(work_dir).expanduser().resolve()
            if work_dir is not None
            else Path(tempfile.gettempdir()) / "genmat-matra"
        )
        model = model_class(
            model_name=None,
            pretrained=False,
            verbose=False,
            work_dir=resolved_work_dir,
            vocab=vocab,
            **config,
        )
        state = dict(blob["state_dict"])
        state.pop("class_weight", None)
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            raise CheckpointError(
                "Matra checkpoint weights do not exactly match the reconstructed model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        target = resolve_device(device)
        model.to(target)
        model.eval()
        del blob, state, vocab
        gc.collect()
        return cls(
            model,
            model_name=model_name or checkpoint_path.stem,
            checkpoint_sha256=info.sha256,
            checkpoint_info=info,
            capabilities=_capabilities_from_props(info.config["props"]),
        )

    def propose(
        self,
        *,
        condition: GenerationConstraints,
        config: GenerationSettings,
    ) -> list[GeneratedCandidate]:
        prompt = condition.to_matra_prompt(
            supported_constraints=self.capabilities.supported_constraints
        )
        # Matra's ``condition=None`` means "stable", not unconstrained. Starting
        # the AMT block allows an actual unconstrained generation when requested.
        effective_prompt = prompt or "AMT"
        requested = set(condition.requested_fields())
        applied = self.capabilities.applied_fields(requested)
        unsupported = self.capabilities.unsupported_fields(requested)

        with _temporary_random_seed(config.seed):
            sequences = self.model.generate(
                n=int(config.n),
                forward=int(config.forward),
                T=float(config.temperature),
                batch_size=int(config.batch_size),
                condition=effective_prompt,
                order=None,
                n_jobs=0,
                progress=False,
            )
            decoded = self.model.decode(
                sequences,
                n_jobs=int(config.decode_jobs),
                only_valid=False,
                verbose=False,
                progress=False,
            )

        if len(decoded) != len(sequences):
            raise RuntimeError(
                f"Matra returned {len(decoded)} decoded results for {len(sequences)} sequences"
            )

        prefix = f"matra-{_slug(self.model_name)}"
        proposals: list[GeneratedCandidate] = []
        for index, (sequence, result) in enumerate(zip(sequences, decoded)):
            structure = getattr(result, "structure", None)
            atoms: Atoms | None = None
            error: str | None = None
            if structure is not None:
                try:
                    atoms = _pymatgen_to_ase(structure)
                except Exception as exc:
                    error = f"matra_conversion:{type(exc).__name__}:{exc}"

            backend_metrics = {
                "parse_valid": bool(getattr(result, "valid", False)),
                "bond_consistent": bool(getattr(result, "bond_consistent", False)),
                "symmetry_consistent": bool(getattr(result, "symm_consistent", False)),
                "spacegroup_consistent": bool(getattr(result, "spg_consistent", False)),
                "prompt": effective_prompt,
            }
            observables = extract_matra_observables(
                str(sequence),
                prompt=effective_prompt,
                properties=self.property_names,
                model_name=self.model_name,
                checkpoint_sha256=self.checkpoint_sha256,
            )
            backend_metrics["property_blocks"] = [
                observable.source.rsplit(".", 1)[-1] for observable in observables
            ]
            backend_consistent = all(
                backend_metrics[key]
                for key in (
                    "parse_valid",
                    "bond_consistent",
                    "symmetry_consistent",
                    "spacegroup_consistent",
                )
            )
            if error is None and config.require_backend_consistency and not backend_consistent:
                error = "matra_consistency"

            if atoms is None:
                validation = ValidationReport(False, "matra_decode_error", {})
                if error is None:
                    error = "matra_decode_error"
            else:
                validation = validate_atoms_report(atoms, **config.validation_kwargs())

            assessments = evaluate_constraints(atoms, validation, condition)
            proposals.append(
                GeneratedCandidate(
                    candidate_id=f"{prefix}-{index:05d}",
                    backend=self.backend_name,
                    model_name=self.model_name,
                    checkpoint_sha256=self.checkpoint_sha256,
                    atoms=atoms,
                    validation=validation,
                    condition=condition.as_dict(),
                    condition_checks={
                        name: assessment.value for name, assessment in assessments.items()
                    },
                    applied_constraints=applied,
                    unsupported_constraints=unsupported,
                    sampling=config.as_dict(),
                    constraint_assessments=assessments,
                    scientific_observables=observables,
                    backend_metrics=backend_metrics,
                    raw_sequence=str(sequence),
                    error=error,
                )
            )
        return proposals


__all__ = [
    "MATRA_REQUIRED_KEYS",
    "MatraBackend",
    "MatraCheckpointInfo",
    "extract_matra_observables",
    "inspect_matra_checkpoint",
]
