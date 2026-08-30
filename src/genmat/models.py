"""Unified, auditable access to GenMat model assets and backends.

The registry deliberately separates model metadata, asset resolution, and
backend construction.  A catalog entry never implies scientific competence or
license permission; callers must explicitly accept entries that require it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping, Union


_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+"
    r"(?:@[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "all", "*"})
_DEPENDENCY_IMPORT_NAMES = {
    "fairchem-core": "fairchem",
    "huggingface-hub": "huggingface_hub",
}


class ModelRegistryError(RuntimeError):
    """Base error for catalog, asset-resolution, and backend-loading failures."""


class UnknownModelError(ModelRegistryError):
    """Raised when no model ID or alias matches a requested reference."""


class ModelResolutionError(ModelRegistryError):
    """Raised when a model asset cannot be resolved to a local file."""


class ModelIntegrityError(ModelRegistryError):
    """Raised when a model asset does not match its catalog metadata."""


class ModelLicenseError(ModelRegistryError):
    """Raised when upstream terms require explicit caller acceptance."""


class ModelOfflineError(ModelResolutionError):
    """Raised when an unavailable asset or service is requested offline."""


class UnsupportedModelBackendError(ModelRegistryError):
    """Raised when a catalog entry names an unknown backend adapter."""


# PEP 604 unions are safe inside postponed function annotations, but a type
# alias right-hand side is evaluated immediately on Python 3.9.
Downloader = Callable[["ModelSpec", Path], Union[str, Path, None]]


def _safe_filename(value: str) -> str:
    filename = str(value).strip()
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or Path(filename).is_absolute()
    ):
        raise ValueError(f"Unsafe model filename: {value!r}")
    return filename


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_ref(value: str) -> str:
    return str(value).strip().casefold()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in _TRUE_VALUES


def _cache_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    if not component or component in {".", ".."}:
        raise ModelResolutionError(f"Unsafe empty cache component derived from {value!r}")
    return component


@dataclass(frozen=True)
class ModelSpec:
    """One immutable model-catalog record.

    ``capabilities`` describes how an adapter can apply a request; it is not
    evidence that a generated candidate satisfies the scientific request.
    """

    id: str
    provider: str
    backend: str
    task: str
    version: str = "unknown"
    format: str = "unknown"
    filename: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    sha256: str | None = None
    size_bytes: int | None = None
    license: str = "upstream terms"
    license_url: str | None = None
    requires_license_acceptance: bool = False
    optional_dependencies: tuple[str, ...] = ()
    capabilities: Mapping[str, str] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        model_id = str(self.id).strip()
        if not _MODEL_ID_RE.fullmatch(model_id) or ".." in model_id.split("/"):
            raise ValueError(f"Invalid model ID: {self.id!r}")
        if not str(self.provider).strip():
            raise ValueError("Model provider must not be empty")
        if not str(self.backend).strip():
            raise ValueError("Model backend must not be empty")
        if not str(self.task).strip():
            raise ValueError("Model task must not be empty")

        filename = None if self.filename is None else _safe_filename(self.filename)
        checksum = None if self.sha256 is None else str(self.sha256).strip().lower()
        if checksum is not None and not _SHA256_RE.fullmatch(checksum):
            raise ValueError(f"Invalid SHA256 for {model_id}: {self.sha256!r}")
        size = None if self.size_bytes is None else int(self.size_bytes)
        if size is not None and size < 0:
            raise ValueError("size_bytes must be >= 0")

        source = dict(self.source)
        source_type = str(source.get("type", "")).strip().casefold()
        source_filename = source.get("filename")
        if source_filename is not None:
            source["filename"] = _safe_filename(str(source_filename))
        local_paths = source.get("local_paths", ())
        if local_paths is None:
            local_paths = ()
        if not isinstance(local_paths, (list, tuple)):
            raise ValueError("source.local_paths must be a list or tuple")
        source["local_paths"] = tuple(str(value) for value in local_paths)
        if source_type in {"http", "huggingface"}:
            if checksum is None or size is None or size <= 0:
                raise ValueError(
                    f"Downloadable model {model_id!r} must declare a positive size_bytes "
                    "and SHA256 before GenMat will resolve it"
                )
        if source_type == "huggingface":
            revision = str(source.get("revision", "")).strip()
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError(
                    f"Hugging Face model {model_id!r} must pin an immutable 40-character "
                    "commit revision"
                )

        aliases = tuple(
            dict.fromkeys(str(value).strip() for value in self.aliases if str(value).strip())
        )
        object.__setattr__(self, "id", model_id)
        object.__setattr__(self, "provider", str(self.provider).strip())
        object.__setattr__(self, "backend", str(self.backend).strip())
        object.__setattr__(self, "task", str(self.task).strip())
        object.__setattr__(self, "version", str(self.version).strip())
        object.__setattr__(self, "format", str(self.format).strip())
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "sha256", checksum)
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(
            self,
            "optional_dependencies",
            tuple(str(value) for value in self.optional_dependencies),
        )
        object.__setattr__(self, "capabilities", dict(self.capabilities))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ModelSpec":
        return cls(
            id=str(record["id"]),
            provider=str(record["provider"]),
            backend=str(record["backend"]),
            task=str(record["task"]),
            version=str(record.get("version", "unknown")),
            format=str(record.get("format", "unknown")),
            filename=(
                None if record.get("filename") is None else str(record["filename"])
            ),
            source=dict(record.get("source") or {}),
            sha256=(None if record.get("sha256") is None else str(record["sha256"])),
            size_bytes=(
                None if record.get("size_bytes") is None else int(record["size_bytes"])
            ),
            license=str(record.get("license", "upstream terms")),
            license_url=(
                None if record.get("license_url") is None else str(record["license_url"])
            ),
            requires_license_acceptance=bool(
                record.get("requires_license_acceptance", False)
            ),
            optional_dependencies=tuple(record.get("optional_dependencies") or ()),
            capabilities=dict(record.get("capabilities") or {}),
            aliases=tuple(record.get("aliases") or ()),
            description=str(record.get("description", "")),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "aliases": list(self.aliases),
            "provider": self.provider,
            "backend": self.backend,
            "task": self.task,
            "version": self.version,
            "format": self.format,
            "filename": self.filename,
            "source": {
                **dict(self.source),
                "local_paths": list(self.source.get("local_paths", ())),
            },
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "license": self.license,
            "license_url": self.license_url,
            "requires_license_acceptance": self.requires_license_acceptance,
            "optional_dependencies": list(self.optional_dependencies),
            "capabilities": dict(self.capabilities),
            "description": self.description,
        }


def _default_cache_dir() -> Path:
    for name in (
        "GENMAT_MODEL_CACHE",
        "GENMAT_CHECKPOINT_DIR",
        "GENIM_MODEL_CACHE",
        "GENIM_CHECKPOINT_DIR",
    ):
        value = os.environ.get(name)
        if value and value.strip():
            return Path(value).expanduser().resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return (Path(local_app_data) / "GenMat" / "models").resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (Path(xdg_cache).expanduser() / "genmat" / "models").resolve()
    return (Path.home() / ".cache" / "genmat" / "models").resolve()


class ModelRegistry:
    """Resolve named models from a validated catalog and construct adapters."""

    def __init__(
        self,
        specs: list[ModelSpec] | tuple[ModelSpec, ...],
        *,
        cache_dir: str | Path | None = None,
        downloader: Downloader | None = None,
        catalog_version: int = 1,
    ) -> None:
        self.cache_dir = (
            _default_cache_dir()
            if cache_dir is None
            else Path(cache_dir).expanduser().resolve()
        )
        self.catalog_version = int(catalog_version)
        self._downloader = downloader
        self._specs = tuple(specs)
        self._references: dict[str, ModelSpec] = {}
        for spec in self._specs:
            for reference in (spec.id, *spec.aliases):
                key = _normalise_ref(reference)
                prior = self._references.get(key)
                if prior is not None and prior.id != spec.id:
                    raise ValueError(
                        f"Duplicate model reference {reference!r}: {prior.id!r} and {spec.id!r}"
                    )
                self._references[key] = spec

    @classmethod
    def default(
        cls,
        *,
        cache_dir: str | Path | None = None,
        downloader: Downloader | None = None,
    ) -> "ModelRegistry":
        resource = resources.files("genmat.data").joinpath("model-catalog-v1.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(
                f"Unsupported model catalog schema: {payload.get('schema_version')!r}"
            )
        records = payload.get("models")
        if not isinstance(records, list):
            raise ValueError("Model catalog must contain a models list")
        return cls(
            [ModelSpec.from_record(record) for record in records],
            cache_dir=cache_dir,
            downloader=downloader,
            catalog_version=int(payload["schema_version"]),
        )

    def list(
        self,
        *,
        provider: str | None = None,
        task: str | None = None,
        backend: str | None = None,
    ) -> list[ModelSpec]:
        provider_key = None if provider is None else _normalise_ref(provider)
        task_key = None if task is None else _normalise_ref(task)
        backend_key = None if backend is None else _normalise_ref(backend)
        return sorted(
            [
                spec
                for spec in self._specs
                if (provider_key is None or _normalise_ref(spec.provider) == provider_key)
                and (task_key is None or _normalise_ref(spec.task) == task_key)
                and (backend_key is None or _normalise_ref(spec.backend) == backend_key)
            ],
            key=lambda spec: spec.id,
        )

    def info(self, ref: str | ModelSpec) -> ModelSpec:
        if isinstance(ref, ModelSpec):
            return ref
        key = _normalise_ref(ref)
        spec = self._references.get(key)
        if spec is None:
            known = ", ".join(spec.id for spec in self.list())
            raise UnknownModelError(f"Unknown model {ref!r}; known models: {known}")
        return spec

    def _offline(self, value: bool | None) -> bool:
        if value is not None:
            return bool(value)
        return any(
            _truthy(os.environ.get(name))
            for name in (
                "GENMAT_OFFLINE",
                "GENIM_OFFLINE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        )

    def _ensure_license(self, spec: ModelSpec, *, accepted: bool) -> None:
        if not spec.requires_license_acceptance or accepted:
            return
        for name in (
            "GENMAT_ACCEPT_MODEL_LICENSES",
            "GENMAT_ACCEPT_LICENSES",
            "GENIM_ACCEPT_MODEL_LICENSES",
            "GENIM_ACCEPT_LICENSES",
        ):
            raw = os.environ.get(name)
            if not raw:
                continue
            tokens = {
                _normalise_ref(token)
                for token in re.split(r"[,;\s]+", raw)
                if token.strip()
            }
            accepted_refs = {
                _normalise_ref(spec.id),
                _normalise_ref(spec.provider),
                *(_normalise_ref(alias) for alias in spec.aliases),
            }
            if tokens.intersection(_TRUE_VALUES) or tokens.intersection(accepted_refs):
                return
        suffix = f" See {spec.license_url}." if spec.license_url else ""
        raise ModelLicenseError(
            f"Model {spec.id!r} requires explicit acceptance of: {spec.license}.{suffix} "
            "Pass accept_license=True or set GENMAT_ACCEPT_MODEL_LICENSES to the model ID."
        )

    def is_license_accepted(
        self,
        ref: str | ModelSpec,
        *,
        accept_license: bool = False,
    ) -> bool:
        """Return whether this process has acknowledged a model's upstream terms."""

        spec = self.info(ref)
        try:
            self._ensure_license(spec, accepted=bool(accept_license))
        except ModelLicenseError:
            return False
        return True

    def missing_optional_dependencies(
        self,
        ref: str | ModelSpec,
    ) -> tuple[str, ...]:
        """Return catalog dependency names whose import packages are unavailable."""

        spec = self.info(ref)
        missing: list[str] = []
        for dependency in spec.optional_dependencies:
            distribution_name = re.split(r"[<>=!~\s]", dependency, maxsplit=1)[0]
            import_name = _DEPENDENCY_IMPORT_NAMES.get(
                distribution_name.casefold(),
                distribution_name.replace("-", "_"),
            )
            try:
                available = importlib.util.find_spec(import_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
            if not available:
                missing.append(dependency)
        return tuple(missing)

    def _asset_filename(self, spec: ModelSpec) -> str:
        value = spec.filename or spec.source.get("filename")
        if value is None:
            raise ModelResolutionError(f"Model {spec.id!r} has no local asset filename")
        return _safe_filename(str(value))

    def _cache_target(self, spec: ModelSpec) -> Path:
        filename = self._asset_filename(spec)
        provider = _cache_component(spec.provider)
        model_ref = spec.id.split("/", 1)[-1]
        model_name, separator, version = model_ref.rpartition("@")
        if not separator:
            model_name, version = model_ref, spec.version
        integrity = spec.sha256[:16] if spec.sha256 else "unverified"
        target = (
            self.cache_dir
            / provider
            / _cache_component(model_name)
            / _cache_component(version)
            / integrity
            / filename
        )
        try:
            target.resolve().relative_to(self.cache_dir.resolve())
        except ValueError as exc:
            raise ModelResolutionError("Resolved model path escapes the GenMat cache") from exc
        return target

    def _search_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        for name in ("GENMAT_MODEL_PATH", "GENIM_MODEL_PATH"):
            raw = os.environ.get(name, "")
            for value in raw.split(os.pathsep):
                if value.strip():
                    roots.append(Path(value).expanduser())
        cwd = Path.cwd()
        repo_root = Path(__file__).resolve().parents[2]
        roots.extend((cwd, cwd.parent, repo_root, repo_root.parent))
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            try:
                resolved = root.resolve()
            except OSError:
                resolved = root
            key = os.path.normcase(str(resolved))
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        return tuple(unique)

    def _local_candidates(self, spec: ModelSpec) -> tuple[Path, ...]:
        candidates: list[Path] = [self._cache_target(spec)]
        for raw_path in spec.source.get("local_paths", ()):
            path = Path(str(raw_path)).expanduser()
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.extend(root / path for root in self._search_roots())
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = os.path.normcase(str(resolved))
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        return tuple(unique)

    @staticmethod
    def _verify(spec: ModelSpec, path: Path) -> None:
        if not path.is_file():
            raise ModelResolutionError(f"Model asset is not a file: {path}")
        if spec.size_bytes is not None and path.stat().st_size != spec.size_bytes:
            raise ModelIntegrityError(
                f"Size mismatch for {spec.id}: expected {spec.size_bytes}, "
                f"got {path.stat().st_size} at {path}"
            )
        if spec.sha256 is not None:
            actual = _sha256_file(path)
            if actual.casefold() != spec.sha256.casefold():
                raise ModelIntegrityError(
                    f"SHA256 mismatch for {spec.id}: expected {spec.sha256}, "
                    f"got {actual} at {path}"
                )

    def _download_asset(self, spec: ModelSpec, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            if self._downloader is not None:
                produced = self._downloader(spec, temporary)
                if produced is not None:
                    produced_path = Path(produced).expanduser().resolve()
                    if produced_path != temporary.resolve():
                        shutil.copyfile(produced_path, temporary)
            else:
                source_type = str(spec.source.get("type", "http")).casefold()
                if source_type == "http":
                    self._download_http(spec, temporary)
                elif source_type == "huggingface":
                    self._download_huggingface(spec, temporary)
                else:
                    raise ModelResolutionError(
                        f"Model {spec.id!r} has no downloadable asset source"
                    )
            self._verify(spec, temporary)
            temporary.replace(target)
            return target
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _download_http(spec: ModelSpec, destination: Path) -> None:
        url = spec.source.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ModelResolutionError(f"Model {spec.id!r} has no HTTP URL")
        import requests

        response = requests.get(
            url,
            stream=True,
            timeout=float(spec.source.get("timeout_seconds", 120.0)),
        )
        try:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _download_huggingface(spec: ModelSpec, destination: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ModelResolutionError(
                "Hugging Face model resolution requires `huggingface-hub`; "
                "install the GenMat MLIP optional dependencies."
            ) from exc
        repo_id = spec.source.get("repo_id")
        filename = spec.source.get("filename") or spec.filename
        if not repo_id or not filename:
            raise ModelResolutionError(
                f"Model {spec.id!r} has incomplete Hugging Face source metadata"
            )
        downloaded = hf_hub_download(
            repo_id=str(repo_id),
            filename=_safe_filename(str(filename)),
            revision=str(spec.source.get("revision", "main")),
        )
        shutil.copyfile(Path(downloaded), destination)

    def resolve(
        self,
        ref: str | ModelSpec,
        *,
        download: bool = True,
        offline: bool | None = None,
        accept_license: bool = False,
    ) -> Path:
        spec = self.info(ref)
        self._ensure_license(spec, accepted=bool(accept_license))
        if str(spec.source.get("type", "")).casefold() == "remote":
            raise ModelResolutionError(
                f"Model {spec.id!r} is a remote service and has no local asset"
            )

        integrity_errors: list[ModelIntegrityError] = []
        for candidate in self._local_candidates(spec):
            if not candidate.is_file():
                continue
            try:
                self._verify(spec, candidate)
                return candidate
            except ModelIntegrityError as exc:
                integrity_errors.append(exc)

        is_offline = self._offline(offline)
        if is_offline or not download:
            if integrity_errors:
                raise integrity_errors[0]
            if is_offline:
                raise ModelOfflineError(
                    f"Model {spec.id!r} is not available locally and GenMat is offline"
                )
            raise ModelResolutionError(
                f"Model {spec.id!r} is not available locally and download=False"
            )
        return self._download_asset(spec, self._cache_target(spec))

    def pull(
        self,
        ref: str | ModelSpec,
        *,
        offline: bool | None = None,
        accept_license: bool = False,
    ) -> Path:
        return self.resolve(
            ref,
            download=True,
            offline=offline,
            accept_license=accept_license,
        )

    def load_backend(
        self,
        ref: str | ModelSpec,
        *,
        device: str = "auto",
        download: bool = True,
        offline: bool | None = None,
        accept_license: bool = False,
        **kwargs: Any,
    ) -> Any:
        spec = self.info(ref)
        self._ensure_license(spec, accepted=bool(accept_license))
        backend_name = spec.backend.casefold()

        if backend_name == "alexandria_matra":
            if self._offline(offline):
                raise ModelOfflineError(
                    f"Remote model {spec.id!r} cannot be loaded while GenMat is offline"
                )
            from .backends import AlexandriaMatraBackend

            options = dict(kwargs)
            options.setdefault("base_url", str(spec.source["base_url"]))
            return AlexandriaMatraBackend(**options)

        path = self.resolve(
            spec,
            download=download,
            offline=offline,
            accept_license=accept_license,
        )
        if backend_name in {"genmat", "genim"}:
            from .backends import GenMatBackend

            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"Unexpected GenMat checkpoint backend options: {unexpected}")
            return GenMatBackend.from_checkpoint(
                path,
                device=device,
                expected_sha256=spec.sha256,
            )
        if backend_name == "matra":
            from .backends import MatraBackend

            options = dict(kwargs)
            options.setdefault("model_name", spec.id)
            return MatraBackend.from_checkpoint(
                path,
                device=device,
                expected_sha256=spec.sha256,
                **options,
            )
        if backend_name == "omat24":
            from .mlip import build_omat24_calculator

            options = dict(kwargs)
            model_alias = str(spec.source.get("model_alias", spec.id))
            return build_omat24_calculator(
                model=model_alias,
                device=device,
                checkpoint=str(path),
                **options,
            )
        raise UnsupportedModelBackendError(
            f"Model {spec.id!r} declares unsupported backend {spec.backend!r}"
        )


__all__ = [
    "ModelIntegrityError",
    "ModelLicenseError",
    "ModelOfflineError",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelResolutionError",
    "ModelSpec",
    "UnknownModelError",
    "UnsupportedModelBackendError",
]
