from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .backends import (
    AlgorithmicSeedBackend,
    EnsembleGenerator,
    GenerationBackend,
    GenerationConstraints,
    GenerationSettings,
    GenIMBackend,
    MatraBackend,
    normalise_backend_capabilities,
)
from .composition import parse_formula_counts, ratios_to_integer_counts


SCHEMA_VERSION = 2
DEFAULT_FORMULA = "SiO2"
DEFAULT_MATRA_MED_SHA256 = "4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa"


class BackendUnavailableError(RuntimeError):
    """Raised when a caller explicitly requests an unavailable backend."""


def _optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _discover_matra_checkpoint() -> Path | None:
    candidates = [
        Path.cwd().parent / "matra-genoa-preview" / "checkpoints" / "matra-v02-med.ckpt",
        Path(__file__).resolve().parents[3]
        / "matra-genoa-preview"
        / "checkpoints"
        / "matra-v02-med.ckpt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


@dataclass(frozen=True)
class ServiceConfig:
    """Runtime configuration for the local or hosted GenIM generation service."""

    genim_checkpoint: Path | None = None
    genim_sha256: str | None = None
    matra_checkpoint: Path | None = None
    matra_sha256: str | None = None
    device: str = "auto"
    default_backend: str = "auto"
    auto_discover_checkpoints: bool = True
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        matra = _optional_path(os.environ.get("GENIM_MATRA_CHECKPOINT"))
        auto_discover = _env_flag("GENIM_API_AUTO_DISCOVER_CHECKPOINTS", True)
        if matra is None and auto_discover:
            matra = _discover_matra_checkpoint()
        matra_sha = os.environ.get("GENIM_MATRA_SHA256")
        if matra_sha is None and matra is not None and matra.name == "matra-v02-med.ckpt":
            matra_sha = DEFAULT_MATRA_MED_SHA256
        origins = tuple(
            value.strip()
            for value in os.environ.get(
                "GENIM_CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if value.strip()
        )
        return cls(
            genim_checkpoint=_optional_path(os.environ.get("GENIM_MODEL_CHECKPOINT")),
            genim_sha256=os.environ.get("GENIM_MODEL_SHA256"),
            matra_checkpoint=matra,
            matra_sha256=matra_sha,
            device=os.environ.get("GENIM_DEVICE", "auto"),
            default_backend=os.environ.get("GENIM_DEFAULT_BACKEND", "auto"),
            auto_discover_checkpoints=auto_discover,
            cors_origins=origins,
        )


def _ratio_expression_counts(text: str) -> dict[str, int]:
    tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
    elements: list[str] = []
    ratios: list[float] = []
    for token in tokens:
        match = re.fullmatch(r"([A-Z][a-z]?)\s*[:=]\s*(\d*\.?\d+)", token)
        if match is None:
            raise ValueError(f"Invalid ratio token: {token!r}")
        elements.append(match.group(1))
        ratios.append(float(match.group(2)))
    return ratios_to_integer_counts(elements, ratios)


def _composition_from_payload(payload: Mapping[str, Any]) -> dict[str, int]:
    formula = payload.get("formula")
    if isinstance(formula, str) and formula.strip():
        if ":" in formula or "=" in formula:
            return _ratio_expression_counts(formula)
        return parse_formula_counts(formula)

    elements_value = payload.get("elements")
    stoichiometry_value = payload.get("stoichiometry")
    if isinstance(elements_value, (list, tuple)) and elements_value:
        elements = [str(value) for value in elements_value]
        ratios = (
            [float(value) for value in stoichiometry_value]
            if isinstance(stoichiometry_value, (list, tuple)) and stoichiometry_value
            else [1.0] * len(elements)
        )
        return ratios_to_integer_counts(elements, ratios)

    composition = payload.get("composition")
    if isinstance(composition, Mapping) and composition:
        return ratios_to_integer_counts(
            [str(key) for key in composition],
            [float(value) for value in composition.values()],
        )
    if isinstance(composition, (list, tuple)) and composition:
        return parse_formula_counts("".join(str(value) for value in composition))
    if isinstance(composition, str) and composition.strip():
        if ":" in composition or "=" in composition:
            return _ratio_expression_counts(composition)
        return parse_formula_counts(composition)
    return parse_formula_counts(DEFAULT_FORMULA)


@dataclass(frozen=True)
class GenerationRequest:
    """Validated, backend-neutral request used by both HTTP and Python callers."""

    formula: str
    backend: str
    constraints: GenerationConstraints
    settings: GenerationSettings

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        default_backend: str = "auto",
    ) -> "GenerationRequest":
        data = dict(payload or {})
        counts = _composition_from_payload(data)
        total_sites = sum(counts.values())
        max_sites = int(data.get("max_sites", max(64, total_sites)))
        if total_sites > max_sites:
            raise ValueError(
                f"Reduced composition requires {total_sites} sites, above max_sites={max_sites}"
            )
        n = int(data.get("n", 4))
        if not 1 <= n <= 64:
            raise ValueError("n must be an integer from 1 to 64")
        backend = str(data.get("backend", default_backend) or default_backend).strip().lower()
        if backend == "genim-seed":
            backend = "algorithmic_seed"
        spacegroup_value = data.get("spacegroup_number", data.get("spacegroup"))
        spacegroup = (
            None
            if spacegroup_value is None or str(spacegroup_value).strip() == ""
            else int(spacegroup_value)
        )
        elements = tuple(counts)
        stoichiometry = tuple(float(counts[element]) for element in elements)
        constraints = GenerationConstraints(
            elements=elements,
            stoichiometry=stoichiometry,
            spacegroup_number=spacegroup,
            matra_wyckoff_indices=tuple(int(value) for value in data.get("matra_wyckoff_indices", ())),
            stability=str(data.get("stability", "any")),
            target_e_hull=(
                None if data.get("target_e_hull") is None else float(data["target_e_hull"])
            ),
            exact_elements=bool(data.get("exact_elements", True)),
        )
        settings = GenerationSettings(
            n=n,
            batch_size=int(data.get("batch_size", min(16, n))),
            max_sites=max_sites,
            min_sites=int(data.get("min_sites", 1)),
            temperature=float(data.get("temperature", 0.8)),
            top_k=int(data.get("top_k", 0)),
            seed=None if data.get("seed") is None else int(data.get("seed", 7)),
            forward=int(data.get("forward", 150)),
            decode_jobs=int(data.get("decode_jobs", 0)),
            require_backend_consistency=bool(data.get("require_backend_consistency", True)),
            validation_options=dict(data.get("validation_options") or {}),
        )
        return cls(
            formula="".join(
                f"{element}{'' if count == 1 else count}"
                for element, count in counts.items()
            ),
            backend=backend,
            constraints=constraints,
            settings=settings,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "backend": self.backend,
            "constraints": self.constraints.as_dict(),
            "settings": self.settings.as_dict(),
        }


@dataclass
class GeneratorService:
    """Lazy multi-backend service facade over GenIM's canonical schema."""

    config: ServiceConfig = field(default_factory=ServiceConfig.from_env)
    _backends: dict[str, GenerationBackend] | None = field(default=None, init=False, repr=False)
    _backend_errors: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _load_backends(self) -> dict[str, GenerationBackend]:
        if self._backends is not None:
            return self._backends
        with self._load_lock:
            if self._backends is not None:
                return self._backends
            backends: dict[str, GenerationBackend] = {
                "algorithmic_seed": AlgorithmicSeedBackend()
            }
            errors: dict[str, str] = {}
            if self.config.matra_checkpoint is not None:
                try:
                    backends["matra"] = MatraBackend.from_checkpoint(
                        self.config.matra_checkpoint,
                        device=self.config.device,
                        expected_sha256=self.config.matra_sha256,
                    )
                except Exception as exc:
                    errors["matra"] = f"{type(exc).__name__}: {exc}"
            if self.config.genim_checkpoint is not None:
                try:
                    backends["genim"] = GenIMBackend.from_checkpoint(
                        self.config.genim_checkpoint,
                        device=self.config.device,
                        expected_sha256=self.config.genim_sha256,
                    )
                except Exception as exc:
                    errors["genim"] = f"{type(exc).__name__}: {exc}"
            self._backend_errors = errors
            self._backends = backends
            return backends

    def _select_backends(self, requested: str) -> list[GenerationBackend]:
        backends = self._load_backends()
        checkpoint_backends = [
            backends[name] for name in ("matra", "genim") if name in backends
        ]
        if requested in {"auto", "default"}:
            return checkpoint_backends[:1] or [backends["algorithmic_seed"]]
        if requested in {"hybrid", "ensemble"}:
            return checkpoint_backends or [backends["algorithmic_seed"]]
        if requested in {"all", "ensemble_with_seed"}:
            return [*checkpoint_backends, backends["algorithmic_seed"]]
        if requested in backends:
            return [backends[requested]]
        detail = self._backend_errors.get(requested)
        if detail:
            raise BackendUnavailableError(f"Backend {requested!r} could not be loaded: {detail}")
        raise BackendUnavailableError(
            f"Backend {requested!r} is not configured; available backends: {', '.join(sorted(backends))}"
        )

    def capabilities(self) -> dict[str, Any]:
        backends = self._load_backends()
        return {
            "schema_version": SCHEMA_VERSION,
            "default_backend": self.config.default_backend,
            "backends": {
                name: {
                    "available": True,
                    "model_name": backend.model_name,
                    "checkpoint_sha256": backend.checkpoint_sha256,
                    "capabilities": normalise_backend_capabilities(backend.capabilities).to_record(),
                }
                for name, backend in sorted(backends.items())
            },
            "unavailable_backends": dict(sorted(self._backend_errors.items())),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "schema_version": SCHEMA_VERSION,
            "default_backend": self.config.default_backend,
            "always_available_backend": "algorithmic_seed",
            "configured_checkpoints": {
                "matra": self.config.matra_checkpoint is not None,
                "genim": self.config.genim_checkpoint is not None,
            },
        }

    def generate(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request = GenerationRequest.from_payload(
            payload,
            default_backend=self.config.default_backend,
        )
        selected_backends = self._select_backends(request.backend)
        run = EnsembleGenerator(selected_backends).run(
            condition=request.constraints,
            config=request.settings,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "generator_mode": (
                selected_backends[0].backend_name if len(selected_backends) == 1 else "ensemble"
            ),
            "request": request.to_record(),
            "report": run.report(),
            "backend_capabilities": run.backend_capabilities,
            "backend_warnings": dict(sorted(self._backend_errors.items())),
            "candidates": [candidate.to_record() for candidate in run.candidates],
        }


__all__ = [
    "BackendUnavailableError",
    "DEFAULT_FORMULA",
    "GenerationRequest",
    "GeneratorService",
    "SCHEMA_VERSION",
    "ServiceConfig",
]
