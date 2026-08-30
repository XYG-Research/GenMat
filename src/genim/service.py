from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .backends import (
    AlgorithmicSeedBackend,
    AlexandriaMatraBackend,
    EnsembleGenerator,
    GenerationBackend,
    GenerationConstraints,
    GenerationSettings,
    GenMatBackend,
    MatraBackend,
    normalise_backend_capabilities,
)
from .composition import parse_formula_counts, ratios_to_integer_counts
from .models import ModelIntegrityError, ModelRegistry, ModelRegistryError


SCHEMA_VERSION = 3
DEFAULT_FORMULA = "SiO2"
DEFAULT_MATRA_MED_SHA256 = "4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa"


class BackendUnavailableError(RuntimeError):
    """Raised when a caller explicitly requests an unavailable backend."""


def _optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def _env_value(canonical: str, legacy: str, default: str | None = None) -> str | None:
    """Read a canonical GenMat variable before its GenIM compatibility alias."""

    value = os.environ.get(canonical)
    if value is not None:
        return value
    return os.environ.get(legacy, default)


def _env_flag_alias(canonical: str, legacy: str, default: bool) -> bool:
    value = _env_value(canonical, legacy)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_accept_all_model_licenses() -> bool:
    value = _env_value("GENMAT_ACCEPT_MODEL_LICENSES", "GENIM_ACCEPT_MODEL_LICENSES")
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "all", "*"}


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
    """Runtime configuration for the local or hosted GenMat generation service.

    The ``genim_*`` fields remain in their original positions for source
    compatibility. New callers should use ``genmat_*`` or a catalog
    ``model_ref``.
    """

    genim_checkpoint: Path | None = None
    genim_sha256: str | None = None
    matra_checkpoint: Path | None = None
    matra_sha256: str | None = None
    device: str = "auto"
    default_backend: str = "auto"
    auto_discover_checkpoints: bool = True
    enable_alexandria: bool = False
    alexandria_base_url: str = "https://alexandria.icams.rub.de/mapi"
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    # New GenMat fields intentionally follow every pre-0.6 field so callers
    # that constructed ServiceConfig positionally keep their original meaning.
    genmat_checkpoint: Path | None = None
    genmat_sha256: str | None = None
    model_ref: str | None = None
    model_cache_dir: Path | None = None
    offline: bool = False
    accept_model_license: bool = False

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        matra = _optional_path(_env_value("GENMAT_MATRA_CHECKPOINT", "GENIM_MATRA_CHECKPOINT"))
        auto_discover = _env_flag_alias(
            "GENMAT_API_AUTO_DISCOVER_CHECKPOINTS",
            "GENIM_API_AUTO_DISCOVER_CHECKPOINTS",
            True,
        )
        if matra is None and auto_discover:
            matra = _discover_matra_checkpoint()
        matra_sha = _env_value("GENMAT_MATRA_SHA256", "GENIM_MATRA_SHA256")
        if matra_sha is None and matra is not None and matra.name == "matra-v02-med.ckpt":
            matra_sha = DEFAULT_MATRA_MED_SHA256
        origins = tuple(
            value.strip()
            for value in (
                _env_value(
                    "GENMAT_CORS_ORIGINS",
                    "GENIM_CORS_ORIGINS",
                    "http://localhost:3000,http://127.0.0.1:3000",
                )
                or ""
            ).split(",")
            if value.strip()
        )
        return cls(
            genmat_checkpoint=_optional_path(
                _env_value("GENMAT_MODEL_CHECKPOINT", "GENIM_MODEL_CHECKPOINT")
            ),
            genmat_sha256=_env_value("GENMAT_MODEL_SHA256", "GENIM_MODEL_SHA256"),
            matra_checkpoint=matra,
            matra_sha256=matra_sha,
            model_ref=_env_value("GENMAT_MODEL", "GENIM_MODEL"),
            model_cache_dir=_optional_path(
                _env_value("GENMAT_MODEL_CACHE", "GENIM_MODEL_CACHE")
            ),
            offline=_env_flag_alias("GENMAT_OFFLINE", "GENIM_OFFLINE", False),
            accept_model_license=_env_accept_all_model_licenses(),
            device=_env_value("GENMAT_DEVICE", "GENIM_DEVICE", "auto") or "auto",
            default_backend=(
                _env_value("GENMAT_DEFAULT_BACKEND", "GENIM_DEFAULT_BACKEND", "auto")
                or "auto"
            ),
            auto_discover_checkpoints=auto_discover,
            enable_alexandria=_env_flag_alias(
                "GENMAT_ENABLE_ALEXANDRIA", "GENIM_ENABLE_ALEXANDRIA", False
            ),
            alexandria_base_url=(
                _env_value(
                    "GENMAT_ALEXANDRIA_BASE_URL",
                    "GENIM_ALEXANDRIA_BASE_URL",
                    "https://alexandria.icams.rub.de/mapi",
                )
                or "https://alexandria.icams.rub.de/mapi"
            ),
            cors_origins=origins,
        )

    @property
    def effective_genmat_checkpoint(self) -> Path | None:
        return self.genmat_checkpoint or self.genim_checkpoint

    @property
    def effective_genmat_sha256(self) -> str | None:
        return self.genmat_sha256 or self.genim_sha256


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
    # Added after all pre-0.6 fields to preserve positional construction.
    model: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        default_backend: str = "auto",
        default_model: str | None = None,
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
        model_value = data.get("model", default_model)
        model = (
            None
            if model_value is None or not str(model_value).strip()
            else str(model_value).strip()
        )
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
            seed=(
                None
                if data.get("seed", 7) is None
                else int(data.get("seed", 7))
            ),
            forward=int(data.get("forward", 150)),
            decode_jobs=int(data.get("decode_jobs", 0)),
            require_backend_consistency=bool(data.get("require_backend_consistency", True)),
            mutation_fraction=float(data.get("mutation_fraction", 0.0)),
            mutation_strain=float(data.get("mutation_strain", 0.08)),
            mutation_displacement=float(data.get("mutation_displacement", 0.12)),
            mutation_attempts=int(data.get("mutation_attempts", 24)),
            preserve_spacegroup=bool(data.get("preserve_spacegroup", True)),
            validation_options=dict(data.get("validation_options") or {}),
        )
        return cls(
            formula="".join(
                f"{element}{'' if count == 1 else count}"
                for element, count in counts.items()
            ),
            backend=backend,
            model=model,
            constraints=constraints,
            settings=settings,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "backend": self.backend,
            "model": self.model,
            "constraints": self.constraints.as_dict(),
            "settings": self.settings.as_dict(),
        }


@dataclass
class GeneratorService:
    """Lazy multi-backend service facade over GenMat's canonical schema."""

    config: ServiceConfig = field(default_factory=ServiceConfig.from_env)
    registry: ModelRegistry | None = None
    _backends: dict[str, GenerationBackend] | None = field(default=None, init=False, repr=False)
    _model_backends: dict[str, GenerationBackend] = field(
        default_factory=dict, init=False, repr=False
    )
    _backend_errors: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _get_registry(self) -> ModelRegistry:
        if self.registry is None:
            self.registry = ModelRegistry.default(cache_dir=self.config.model_cache_dir)
        return self.registry

    def _load_backends(self) -> dict[str, GenerationBackend]:
        if self._backends is not None:
            return self._backends
        with self._load_lock:
            if self._backends is not None:
                return self._backends
            backends: dict[str, GenerationBackend] = {
                "algorithmic_seed": AlgorithmicSeedBackend()
            }
            if self.config.enable_alexandria:
                backends["alexandria_matra"] = AlexandriaMatraBackend(
                    base_url=self.config.alexandria_base_url,
                )
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
            if self.config.effective_genmat_checkpoint is not None:
                try:
                    backends["genmat"] = GenMatBackend.from_checkpoint(
                        self.config.effective_genmat_checkpoint,
                        device=self.config.device,
                        expected_sha256=self.config.effective_genmat_sha256,
                    )
                except Exception as exc:
                    errors["genmat"] = f"{type(exc).__name__}: {exc}"
            self._backend_errors = errors
            self._backends = backends
            return backends

    def _catalog_backend(self, model_ref: str) -> tuple[Any, GenerationBackend]:
        registry = self._get_registry()
        try:
            spec = registry.info(model_ref)
        except ModelRegistryError as exc:
            raise BackendUnavailableError(str(exc)) from exc
        if str(spec.task).casefold() != "generation":
            raise BackendUnavailableError(
                f"Model {spec.id!r} has task {spec.task!r}; /v1/generate accepts "
                "structure-generation models only"
            )
        if spec.id in self._model_backends:
            return spec, self._model_backends[spec.id]
        with self._load_lock:
            if spec.id in self._model_backends:
                return spec, self._model_backends[spec.id]
            try:
                backend = registry.load_backend(
                    spec,
                    device=self.config.device,
                    download=not self.config.offline,
                    offline=self.config.offline,
                    accept_license=self.config.accept_model_license,
                )
            except ModelRegistryError as exc:
                raise BackendUnavailableError(
                    f"Model {spec.id!r} could not be loaded: {exc}"
                ) from exc
            if not isinstance(backend, GenerationBackend):
                raise BackendUnavailableError(
                    f"Model {spec.id!r} did not create a GenerationBackend"
                )
            self._model_backends[spec.id] = backend
            return spec, backend

    def _select_backends(
        self,
        requested: str,
        *,
        model_ref: str | None = None,
    ) -> list[GenerationBackend]:
        if requested in {"genim", "genim_model"}:
            requested = "genmat"
        if model_ref is not None:
            spec, backend = self._catalog_backend(model_ref)
            compatible_names = {
                "auto",
                "default",
                str(spec.backend).lower(),
                str(backend.backend_name).lower(),
            }
            if str(spec.backend).lower() == "genim":
                compatible_names.add("genmat")
            if requested not in compatible_names:
                raise BackendUnavailableError(
                    f"Model {spec.id!r} uses backend {spec.backend!r}, not {requested!r}"
                )
            return [backend]

        backends = self._load_backends()
        checkpoint_backends = [
            backends[name] for name in ("matra", "genmat") if name in backends
        ]
        remote_backends = [
            backends[name] for name in ("alexandria_matra",) if name in backends
        ]
        if requested == "alexandria":
            requested = "alexandria_matra"
        if requested in {"auto", "default"}:
            return checkpoint_backends[:1] or remote_backends[:1] or [backends["algorithmic_seed"]]
        if requested in {"hybrid", "ensemble"}:
            return [*checkpoint_backends, *remote_backends] or [backends["algorithmic_seed"]]
        if requested in {"all", "ensemble_with_seed"}:
            return [*checkpoint_backends, *remote_backends, backends["algorithmic_seed"]]
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
            "default_model": self.config.model_ref,
            "backend_aliases": {"genim": "genmat"},
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
            "models": self.models()["models"],
        }

    def _model_record(self, spec: Any) -> dict[str, Any]:
        registry = self._get_registry()
        license_ready = registry.is_license_accepted(
            spec,
            accept_license=self.config.accept_model_license,
        )
        missing_dependencies = registry.missing_optional_dependencies(spec)
        available = False
        if not license_ready:
            availability = "server_license_acceptance_required"
        elif missing_dependencies:
            availability = "missing_optional_dependencies"
        elif self.config.offline and str(spec.source.get("type", "")).casefold() == "remote":
            availability = "offline_remote_unavailable"
        elif self.config.offline:
            try:
                registry.resolve(
                    spec,
                    download=False,
                    offline=True,
                    accept_license=True,
                )
            except ModelIntegrityError:
                availability = "local_asset_integrity_error"
            except ModelRegistryError:
                availability = "offline_asset_unavailable"
            else:
                available = True
                availability = "local_verified"
        else:
            available = True
            availability = "catalogued"
        return {
            **spec.to_record(),
            "available": available,
            "availability": availability,
            "missing_optional_dependencies": list(missing_dependencies),
        }

    def models(self) -> dict[str, Any]:
        records = [self._model_record(spec) for spec in self._get_registry().list()]
        return {
            "schema_version": 1,
            "default_model": self.config.model_ref,
            "cache_policy": "content-addressed-sha256",
            "models": records,
        }

    def model_info(self, model_ref: str) -> dict[str, Any]:
        try:
            registry = self._get_registry()
            spec = registry.info(model_ref)
            return self._model_record(spec)
        except ModelRegistryError as exc:
            raise BackendUnavailableError(str(exc)) from exc

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "schema_version": SCHEMA_VERSION,
            "default_backend": self.config.default_backend,
            "default_model": self.config.model_ref,
            "always_available_backend": "algorithmic_seed",
            "configured_checkpoints": {
                "matra": self.config.matra_checkpoint is not None,
                "genmat": self.config.effective_genmat_checkpoint is not None,
                "genim": self.config.effective_genmat_checkpoint is not None,
            },
            "model_catalog": {
                "count": len(self._get_registry().list()),
                "offline": self.config.offline,
            },
            "remote_services": {
                "alexandria_matra": self.config.enable_alexandria,
            },
        }

    def generate(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request = GenerationRequest.from_payload(
            payload,
            default_backend=self.config.default_backend,
            default_model=self.config.model_ref,
        )
        selected_backends = self._select_backends(
            request.backend,
            model_ref=request.model,
        )
        model_selection = None
        if request.model is not None:
            resolved_spec = self._get_registry().info(request.model)
            model_selection = {
                "requested": request.model,
                "resolved": resolved_spec.id,
                "version": resolved_spec.version,
                "provider": resolved_spec.provider,
                "backend": resolved_spec.backend,
            }
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
            "model_selection": model_selection,
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
