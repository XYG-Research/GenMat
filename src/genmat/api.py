from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from ase.io import write

from .checkpoints import LoadedCheckpoint, load_model_checkpoint
from .chem import ChemistryPolicy
from .decode import DecodeConfig, decode_tokens_to_atoms
from .generate import sample_sequences
from .validate import ValidationReport, validate_atoms_report


@dataclass(frozen=True)
class SamplingConfig:
    n: int = 1
    batch_size: int = 16
    max_sites: int = 25
    min_sites: int = 1
    temperature: float = 1.0
    top_k: int = 0
    seed: int | None = None
    fixed_hall: int | None = None
    fixed_spacegroup: int | None = None


@dataclass
class GeneratedStructure:
    token_ids: list[int]
    tokens: list[str]
    atoms: Atoms | None
    validation: ValidationReport
    checkpoint_sha256: str
    chemistry_policy: dict[str, object]
    sampling: dict[str, object]
    error: str | None = None

    @property
    def valid(self) -> bool:
        return bool(self.atoms is not None and self.validation.valid and self.error is None)

    def to_record(self) -> dict[str, Any]:
        atoms = self.atoms
        structure: dict[str, Any] | None = None
        formula: str | None = None
        if atoms is not None:
            formula = atoms.get_chemical_formula(mode="hill")
            structure = {
                "lattice": atoms.cell.array.tolist(),
                "frac_coords": atoms.get_scaled_positions(wrap=True).tolist(),
                "species": atoms.get_chemical_symbols(),
            }
        return {
            "valid": self.valid,
            "error": self.error,
            "formula": formula,
            "token_ids": list(self.token_ids),
            "validation": {
                "valid": self.validation.valid,
                "reason": self.validation.reason,
                "metrics": dict(self.validation.metrics),
            },
            "checkpoint_sha256": self.checkpoint_sha256,
            "chemistry_policy": dict(self.chemistry_policy),
            "sampling": dict(self.sampling),
            "structure": structure,
        }


class GenMat:
    """Primary Python API for checkpoint-backed crystal generation."""

    def __init__(self, checkpoint: LoadedCheckpoint):
        self.checkpoint = checkpoint
        cfg = checkpoint.blob["tokenize_config"]
        self.decode_config = DecodeConfig(
            coord_bins=int(cfg["coord_bins"]),
            len_bins=int(cfg["len_bins"]),
            len_min=float(cfg["len_min"]),
            len_max=float(cfg["len_max"]),
            ang_bins=int(cfg["ang_bins"]),
            ang_min=float(cfg["ang_min"]),
            ang_max=float(cfg["ang_max"]),
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "auto",
        expected_sha256: str | None = None,
    ) -> "GenMat":
        return cls(load_model_checkpoint(Path(path), device=device, expected_sha256=expected_sha256))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.checkpoint.path),
            "sha256": self.checkpoint.sha256,
            "format_version": self.checkpoint.format_version,
            **self.checkpoint.metadata,
        }

    def _resolve_forced_hall(self, config: SamplingConfig) -> tuple[int | None, int | None, str | None]:
        """Resolve an exact Hall-token condition for reusable checkpoint sampling."""

        if config.fixed_hall is not None and config.fixed_spacegroup is not None:
            raise ValueError("Specify only one of fixed_hall and fixed_spacegroup")

        hall_to_id = {
            int(token[5:]): int(index)
            for token, index in self.checkpoint.vocab.items()
            if token.startswith("HALL_")
        }
        if config.fixed_hall is not None:
            hall = int(config.fixed_hall)
            if not 1 <= hall <= 530:
                raise ValueError("fixed_hall must be in 1..530")
            hall_id = hall_to_id.get(hall)
            if hall_id is None:
                raise ValueError(
                    f"fixed_hall={hall} is absent from the checkpoint vocabulary; "
                    "seed or train the required Hall token"
                )
            return hall_id, hall, "explicit_hall"

        if config.fixed_spacegroup is None:
            return None, None, None

        spacegroup = int(config.fixed_spacegroup)
        if not 1 <= spacegroup <= 230:
            raise ValueError("fixed_spacegroup must be in 1..230")

        symmetry_stats = self.checkpoint.blob.get("symmetry_stats")
        if isinstance(symmetry_stats, dict):
            halls_by_sg = symmetry_stats.get("halls_by_sg")
            if isinstance(halls_by_sg, dict):
                trained_halls = halls_by_sg.get(str(spacegroup), halls_by_sg.get(spacegroup))
                if isinstance(trained_halls, list):
                    for value in trained_halls:
                        hall = int(value)
                        if hall in hall_to_id:
                            return hall_to_id[hall], hall, "checkpoint_training_statistics"

        import spglib

        for hall in range(1, 531):
            spacegroup_type = spglib.get_spacegroup_type(hall)
            if spacegroup_type is None:
                continue
            number = int(
                getattr(spacegroup_type, "number")
                if hasattr(spacegroup_type, "number")
                else spacegroup_type["number"]
            )
            if number == spacegroup and hall in hall_to_id:
                return hall_to_id[hall], hall, "representative_hall_from_spglib"

        raise ValueError(
            f"No Hall token for space group {spacegroup} is present in the checkpoint vocabulary; "
            "vocabulary coverage is required and does not by itself guarantee learned competence"
        )

    def sample(
        self,
        *,
        config: SamplingConfig | None = None,
        chemistry: ChemistryPolicy | None = None,
        restrict_elements: list[str] | None = None,
        validate: bool = True,
        validation_options: dict[str, Any] | None = None,
    ) -> list[GeneratedStructure]:
        config = config or SamplingConfig()
        chemistry = chemistry or ChemistryPolicy(mode="any")
        if int(config.n) < 0:
            raise ValueError("SamplingConfig.n must be >= 0")
        if int(config.batch_size) < 1:
            raise ValueError("SamplingConfig.batch_size must be >= 1")
        forced_hall_id, forced_hall, hall_source = self._resolve_forced_hall(config)
        required_max_len = 1 + 1 + 3 + 3 + int(config.max_sites) * 5 + 1
        if required_max_len > int(self.checkpoint.model_config.max_len):
            raise ValueError("max_sites is incompatible with the checkpoint maximum sequence length")

        vocab_elements = {token[2:] for token in self.checkpoint.vocab if token.startswith("E_")}
        if restrict_elements is None:
            restricted = [value for value in chemistry.available_elements() if value in vocab_elements]
        else:
            restricted = []
            for value in restrict_elements:
                symbol = str(value).strip()
                if not chemistry.is_allowed_element(symbol):
                    raise ValueError(f"Element {symbol!r} is rejected by the chemistry policy")
                if symbol not in vocab_elements:
                    raise ValueError(f"Element {symbol!r} is absent from the checkpoint vocabulary")
                if symbol not in restricted:
                    restricted.append(symbol)
        if not restricted:
            raise ValueError("No checkpoint vocabulary elements satisfy the chemistry policy")

        if config.seed is not None:
            torch.manual_seed(int(config.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(config.seed))

        options: dict[str, Any] = {
            "min_dist": 0.5,
            "symprec": 1e-2,
            "min_dist_factor": 0.55,
            "max_dist_factor": None,
            "max_nn_factor": 1.5,
            "min_coordination": 1,
            "require_connected": False,
            "max_atoms": 500,
            "vol_per_atom_min": 1.0,
            "vol_per_atom_max": 100.0,
        }
        options.update(validation_options or {})
        sampling_record = {
            **asdict(config),
            "forced_hall_id": forced_hall_id,
            "forced_hall_number": forced_hall,
            "hall_resolution_source": hall_source,
        }
        results: list[GeneratedStructure] = []
        remaining = int(config.n)
        while remaining:
            batch_n = min(int(config.batch_size), remaining)
            ids_batch = sample_sequences(
                self.checkpoint.model,
                n=batch_n,
                vocab=self.checkpoint.vocab,
                max_len=int(self.checkpoint.model_config.max_len),
                temperature=float(config.temperature),
                top_k=int(config.top_k),
                max_sites=int(config.max_sites),
                min_sites=int(config.min_sites),
                restrict_elements=restricted,
                forced_hall_ids=[forced_hall_id] * batch_n if forced_hall_id is not None else None,
                device=self.checkpoint.device,
            )
            for ids in ids_batch:
                tokens = [self.checkpoint.id_to_token[index] for index in ids if index != self.checkpoint.pad_id]
                atoms: Atoms | None = None
                error: str | None = None
                try:
                    atoms = decode_tokens_to_atoms(tokens, cfg=self.decode_config, max_sites=int(config.max_sites))
                    elements = set(atoms.get_chemical_symbols())
                    if not chemistry.accepts_elements(elements):
                        error = "chemistry_policy"
                    report = (
                        validate_atoms_report(atoms, **options)
                        if validate
                        else ValidationReport(True, "not_run", {})
                    )
                except Exception as exc:
                    error = f"decode_error:{type(exc).__name__}:{exc}"
                    report = ValidationReport(False, "decode_error", {})
                results.append(
                    GeneratedStructure(
                        token_ids=list(ids),
                        tokens=tokens,
                        atoms=atoms,
                        validation=report,
                        checkpoint_sha256=self.checkpoint.sha256,
                        chemistry_policy=chemistry.as_dict(),
                        sampling=sampling_record,
                        error=error,
                    )
                )
            remaining -= batch_n
        return results

    @staticmethod
    def write_valid_cifs(results: list[GeneratedStructure], out_dir: str | Path) -> list[Path]:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        next_index = 0
        while any(
            (out_dir / f"{prefix}_{next_index:05d}.cif").exists()
            for prefix in ("genmat", "genim")
        ):
            next_index += 1
        for result in results:
            if not result.valid or result.atoms is None:
                continue
            path = out_dir / f"genmat_{next_index:05d}.cif"
            write(str(path), result.atoms, format="cif")
            paths.append(path)
            next_index += 1
        return paths


class GenIM(GenMat):
    """Backward-compatible name for the pre-0.6 GenMat checkpoint API."""


__all__ = ["GenMat", "GenIM", "GeneratedStructure", "SamplingConfig"]
