from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from genmat.models import (
    ModelIntegrityError,
    ModelLicenseError,
    ModelOfflineError,
    ModelRegistry,
    ModelResolutionError,
    ModelSpec,
    UnknownModelError,
    UnsupportedModelBackendError,
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _downloadable_spec(
    content: bytes,
    *,
    model_id: str = "test/tiny@1",
    backend: str = "genim",
    license_required: bool = False,
    local_paths: tuple[str, ...] = (),
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        aliases=("tiny",),
        provider="test",
        backend=backend,
        task="generation",
        version="1",
        format="test",
        filename="tiny.pt",
        source={
            "type": "http",
            "url": "https://example.invalid/tiny.pt",
            "local_paths": local_paths,
        },
        sha256=_digest(content),
        size_bytes=len(content),
        license="test upstream terms",
        requires_license_acceptance=license_required,
    )


def test_default_catalog_is_packaged_filterable_and_alias_resolvable() -> None:
    registry = ModelRegistry.default()
    assert registry.catalog_version == 1
    assert len(registry.list()) == 6
    assert len(registry.list(task="generation")) == 5
    assert [spec.id for spec in registry.list(provider="matra")] == [
        "matra/genoa-mp@0.2",
        "matra/genoa-mpas-med@0.2",
        "matra/genoa-mpas@0.2",
    ]
    assert registry.info("matra-v02-med").id == "matra/genoa-mpas-med@0.2"
    assert registry.info("ESEN_30M_OAM").backend == "omat24"
    legacy = registry.info("genim/mp-fullsg-60")
    assert legacy.backend == "genmat"
    assert "GenMat/checkpoints/mp_train_fullsg_60.pt" in legacy.source["local_paths"]
    assert "GenIM/checkpoints/mp_train_fullsg_60.pt" in legacy.source["local_paths"]
    with pytest.raises(UnknownModelError, match="Unknown model"):
        registry.info("missing/model")


def test_genmat_native_models_module_reexports_registry() -> None:
    from genmat.models import ModelRegistry as NativeRegistry
    from genmat.models import ModelSpec as NativeSpec

    assert NativeRegistry is ModelRegistry
    assert NativeSpec is ModelSpec


def test_optional_dependency_probe_uses_distribution_to_import_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = ModelSpec(
        id="test/dependencies@1",
        provider="test",
        backend="omat24",
        task="relaxation",
        optional_dependencies=("fairchem-core", "huggingface-hub", "pymatgen"),
    )
    checked: list[str] = []

    def fake_find_spec(name: str):
        checked.append(name)
        return object() if name == "pymatgen" else None

    monkeypatch.setattr("genmat.models.importlib.util.find_spec", fake_find_spec)
    registry = ModelRegistry([spec], cache_dir=tmp_path)
    assert registry.missing_optional_dependencies(spec) == (
        "fairchem-core",
        "huggingface-hub",
    )
    assert checked == ["fairchem", "huggingface_hub", "pymatgen"]


def test_model_spec_roundtrip_and_path_safety() -> None:
    spec = _downloadable_spec(b"model")
    assert ModelSpec.from_record(spec.to_record()) == spec
    with pytest.raises(ValueError, match="Unsafe model filename"):
        ModelSpec(
            id="test/escape@1",
            provider="test",
            backend="genim",
            task="generation",
            filename="../escape.pt",
        )
    with pytest.raises(ValueError, match="Invalid SHA256"):
        ModelSpec(
            id="test/hash@1",
            provider="test",
            backend="genim",
            task="generation",
            filename="model.pt",
            sha256="not-a-hash",
        )
    with pytest.raises(ValueError, match="must declare"):
        ModelSpec(
            id="test/unverified@1",
            provider="test",
            backend="genim",
            task="generation",
            filename="model.pt",
            source={"type": "http", "url": "https://example.invalid/model.pt"},
        )
    with pytest.raises(ValueError, match="immutable 40-character"):
        ModelSpec(
            id="test/mutable-hf@1",
            provider="test",
            backend="omat24",
            task="relaxation",
            filename="model.pt",
            source={
                "type": "huggingface",
                "repo_id": "example/model",
                "filename": "model.pt",
                "revision": "main",
            },
            sha256=_digest(b"model"),
            size_bytes=5,
        )


def test_genmat_cache_environment_has_priority_over_genim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    preferred = tmp_path / "preferred"
    monkeypatch.setenv("GENIM_CHECKPOINT_DIR", str(legacy))
    monkeypatch.setenv("GENMAT_MODEL_CACHE", str(preferred))
    assert ModelRegistry.default().cache_dir == preferred.resolve()

    monkeypatch.delenv("GENMAT_MODEL_CACHE")
    monkeypatch.setenv("GENMAT_CHECKPOINT_DIR", str(tmp_path / "preferred-checkpoints"))
    assert ModelRegistry.default().cache_dir == (tmp_path / "preferred-checkpoints").resolve()


def test_pull_is_atomic_hash_verified_and_reuses_cache(tmp_path: Path) -> None:
    content = b"verified model bytes"
    calls: list[str] = []

    def downloader(spec: ModelSpec, destination: Path) -> None:
        calls.append(spec.id)
        destination.write_bytes(content)

    registry = ModelRegistry(
        [_downloadable_spec(content)],
        cache_dir=tmp_path,
        downloader=downloader,
    )
    first = registry.pull("tiny")
    second = registry.resolve("test/tiny@1", offline=True)
    assert first == second
    assert first.read_bytes() == content
    assert first.resolve().is_relative_to(tmp_path.resolve())
    assert calls == ["test/tiny@1"]
    assert not list(tmp_path.rglob("*.part"))


def test_corrupt_download_is_rejected_without_publishing_target(tmp_path: Path) -> None:
    expected = b"expected"

    def downloader(spec: ModelSpec, destination: Path) -> None:
        destination.write_bytes(b"corrupt")

    registry = ModelRegistry(
        [_downloadable_spec(expected)],
        cache_dir=tmp_path,
        downloader=downloader,
    )
    with pytest.raises(ModelIntegrityError, match="mismatch"):
        registry.pull("tiny")
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.part"))


def test_offline_missing_model_never_calls_downloader(tmp_path: Path) -> None:
    calls = 0

    def downloader(spec: ModelSpec, destination: Path) -> None:
        nonlocal calls
        calls += 1

    registry = ModelRegistry(
        [_downloadable_spec(b"model")],
        cache_dir=tmp_path,
        downloader=downloader,
    )
    with pytest.raises(ModelOfflineError, match="offline"):
        registry.resolve("tiny", offline=True)
    with pytest.raises(ModelResolutionError, match="download=False"):
        registry.resolve("tiny", download=False, offline=False)
    assert calls == 0


def test_license_must_be_explicitly_accepted_or_named_in_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"licensed"
    spec = _downloadable_spec(content, license_required=True)
    asset = tmp_path / "source" / "tiny.pt"
    asset.parent.mkdir()
    asset.write_bytes(content)
    spec = ModelSpec.from_record(
        {
            **spec.to_record(),
            "source": {**spec.to_record()["source"], "local_paths": [str(asset)]},
        }
    )
    registry = ModelRegistry([spec], cache_dir=tmp_path / "cache")
    with pytest.raises(ModelLicenseError, match="explicit acceptance"):
        registry.resolve("tiny", download=False)
    assert registry.resolve("tiny", download=False, accept_license=True) == asset.resolve()

    monkeypatch.setenv("GENMAT_ACCEPT_MODEL_LICENSES", spec.id)
    assert registry.resolve("tiny", download=False) == asset.resolve()


def test_remote_models_load_without_asset_but_cannot_resolve_offline() -> None:
    registry = ModelRegistry.default()
    with pytest.raises(ModelResolutionError, match="remote service"):
        registry.resolve("alexandria", accept_license=True)
    with pytest.raises(ModelOfflineError, match="offline"):
        registry.load_backend("alexandria", offline=True, accept_license=True)

    backend = registry.load_backend("alexandria", accept_license=True, session=object())
    assert backend.backend_name == "alexandria_matra"
    assert backend.base_url == "https://alexandria.icams.rub.de/mapi"


def test_load_backend_dispatches_verified_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"checkpoint"
    asset = tmp_path / "source.pt"
    asset.write_bytes(content)
    spec = _downloadable_spec(content, local_paths=(str(asset),))
    registry = ModelRegistry([spec], cache_dir=tmp_path / "cache")
    sentinel = object()
    captured = {}

    from genim import backends

    def fake_from_checkpoint(path, *, device, expected_sha256):
        captured.update(path=Path(path), device=device, sha256=expected_sha256)
        return sentinel

    monkeypatch.setattr(
        backends.GenMatBackend,
        "from_checkpoint",
        staticmethod(fake_from_checkpoint),
    )
    assert registry.load_backend("tiny", device="cpu", download=False) is sentinel
    assert captured == {
        "path": asset.resolve(),
        "device": "cpu",
        "sha256": spec.sha256,
    }


def test_load_backend_accepts_canonical_genmat_backend_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"canonical checkpoint"
    asset = tmp_path / "source.pt"
    asset.write_bytes(content)
    spec = _downloadable_spec(content, backend="genmat", local_paths=(str(asset),))
    registry = ModelRegistry([spec], cache_dir=tmp_path / "cache")
    sentinel = object()

    from genim import backends

    monkeypatch.setattr(
        backends.GenMatBackend,
        "from_checkpoint",
        staticmethod(lambda *args, **kwargs: sentinel),
    )
    assert registry.load_backend(spec, device="cpu", download=False) is sentinel


def test_load_backend_dispatches_omat24_calculator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"omat"
    asset = tmp_path / "esen.pt"
    asset.write_bytes(content)
    spec = ModelSpec(
        id="fairchem/test@1",
        aliases=("fair-test",),
        provider="fairchem",
        backend="omat24",
        task="relaxation",
        version="1",
        filename="esen.pt",
        source={
            "type": "huggingface",
            "repo_id": "example/model",
            "filename": "esen.pt",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "model_alias": "eSEN-test",
            "local_paths": [str(asset)],
        },
        sha256=_digest(content),
        size_bytes=len(content),
    )
    registry = ModelRegistry([spec], cache_dir=tmp_path / "cache")
    sentinel = object()
    captured = {}

    import genmat.mlip as mlip

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(mlip, "build_omat24_calculator", fake_builder)
    assert registry.load_backend("fair-test", device="cpu", download=False) is sentinel
    assert captured == {
        "model": "eSEN-test",
        "device": "cpu",
        "checkpoint": str(asset.resolve()),
    }


def test_unknown_backend_is_rejected_after_verified_resolution(tmp_path: Path) -> None:
    content = b"asset"
    asset = tmp_path / "asset.pt"
    asset.write_bytes(content)
    spec = _downloadable_spec(
        content,
        model_id="test/unsupported@1",
        backend="future_backend",
        local_paths=(str(asset),),
    )
    registry = ModelRegistry([spec], cache_dir=tmp_path / "cache")
    with pytest.raises(UnsupportedModelBackendError, match="unsupported backend"):
        registry.load_backend(spec, download=False)
