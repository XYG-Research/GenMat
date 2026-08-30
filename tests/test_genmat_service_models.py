from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from genim.backends import AlgorithmicSeedBackend, GenerationConstraints, GenerationSettings
from genim.cli import main
from genim.models import ModelRegistry, ModelSpec, UnknownModelError
from genim.server import create_app
from genim.service import (
    BackendUnavailableError,
    GenerationRequest,
    GeneratorService,
    ServiceConfig,
)


class _SeedRegistry:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.spec = ModelSpec(
            id="test/algorithmic-seed@1",
            aliases=("test-seed",),
            provider="test",
            backend="algorithmic_seed",
            task="generation",
            version="1",
            format="in-process",
            source={"type": "remote"},
            license="test",
        )

    def list(self, **filters):
        return [self.spec]

    def info(self, ref):
        if isinstance(ref, ModelSpec) or str(ref) in {self.spec.id, "test-seed"}:
            return self.spec
        raise UnknownModelError(f"Unknown model {ref!r}")

    def load_backend(self, ref, **kwargs):
        self.info(ref)
        return AlgorithmicSeedBackend()


def _config(**overrides) -> ServiceConfig:
    values = {
        "auto_discover_checkpoints": False,
        "cors_origins": (),
    }
    values.update(overrides)
    return ServiceConfig(**values)


def test_genmat_environment_names_precede_genim_compatibility_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GENMAT_API_AUTO_DISCOVER_CHECKPOINTS", "0")
    monkeypatch.setenv("GENIM_DEVICE", "legacy-device")
    monkeypatch.setenv("GENMAT_DEVICE", "canonical-device")
    monkeypatch.setenv("GENIM_DEFAULT_BACKEND", "genim")
    monkeypatch.setenv("GENMAT_DEFAULT_BACKEND", "genmat")
    monkeypatch.setenv("GENIM_MODEL", "legacy/model@1")
    monkeypatch.setenv("GENMAT_MODEL", "canonical/model@2")
    monkeypatch.setenv("GENIM_MODEL_CACHE", str(tmp_path / "legacy"))
    monkeypatch.setenv("GENMAT_MODEL_CACHE", str(tmp_path / "canonical"))

    config = ServiceConfig.from_env()
    assert config.device == "canonical-device"
    assert config.default_backend == "genmat"
    assert config.model_ref == "canonical/model@2"
    assert config.model_cache_dir == (tmp_path / "canonical").resolve()
    assert config.auto_discover_checkpoints is False


def test_service_config_preserves_pre_06_positional_field_order() -> None:
    legacy = ServiceConfig(
        Path("legacy.pt"),
        "legacy-sha",
        Path("matra.ckpt"),
        "matra-sha",
        "cpu",
        "genim",
        False,
        True,
        "https://example.invalid/mapi",
        ("https://studio.example",),
    )

    assert legacy.genim_checkpoint == Path("legacy.pt")
    assert legacy.matra_checkpoint == Path("matra.ckpt")
    assert legacy.device == "cpu"
    assert legacy.default_backend == "genim"
    assert legacy.auto_discover_checkpoints is False
    assert legacy.enable_alexandria is True
    assert legacy.cors_origins == ("https://studio.example",)
    assert legacy.genmat_checkpoint is None
    assert legacy.model_ref is None


def test_generation_request_preserves_pre_06_positional_field_order() -> None:
    constraints = GenerationConstraints(elements=("Si", "O"), stoichiometry=(1.0, 2.0))
    settings = GenerationSettings(n=2)
    request = GenerationRequest("O2Si", "auto", constraints, settings)

    assert request.constraints is constraints
    assert request.settings is settings
    assert request.model is None


def test_service_selects_a_catalog_model_and_records_its_identity(tmp_path: Path) -> None:
    registry = _SeedRegistry(tmp_path / "cache")
    service = GeneratorService(_config(), registry=registry)  # type: ignore[arg-type]

    response = service.generate(
        {"formula": "NaCl", "n": 1, "model": "test-seed", "backend": "auto"}
    )
    assert response["generator_mode"] == "algorithmic_seed"
    assert response["request"]["model"] == "test-seed"
    assert response["model_selection"]["resolved"] == registry.spec.id
    assert response["candidates"]

    with pytest.raises(BackendUnavailableError, match="not 'matra'"):
        service.generate(
            {"formula": "NaCl", "n": 1, "model": "test-seed", "backend": "matra"}
        )


def test_model_catalog_is_available_through_service_and_http() -> None:
    from genmat import ModelRegistry as PublicRegistry

    assert PublicRegistry.__name__ == "ModelRegistry"
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    service = GeneratorService(_config())
    catalog = service.models()
    ids = {record["id"] for record in catalog["models"]}
    assert "genmat/mp-fullsg-legacy@0.1.0" in ids
    assert "matra/genoa-mpas-med@0.2" in ids
    assert catalog["schema_version"] == 1
    records = {record["id"]: record for record in catalog["models"]}
    assert records["genmat/mp-fullsg-legacy@0.1.0"]["available"] is True
    assert records["matra/genoa-mpas-med@0.2"]["available"] is False
    assert (
        records["matra/genoa-mpas-med@0.2"]["availability"]
        == "server_license_acceptance_required"
    )

    client = TestClient(create_app(service=service))
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert len(response.json()["models"]) == len(ids)
    info = client.get("/v1/models/genmat/mp-fullsg-legacy@0.1.0")
    assert info.status_code == 200
    assert info.json()["backend"] == "genim"
    missing = client.get("/v1/models/not/a-model")
    assert missing.status_code == 404


def test_service_capabilities_expose_models_without_loading_model_weights() -> None:
    service = GeneratorService(_config())
    capabilities = service.capabilities()
    assert capabilities["backend_aliases"]["genim"] == "genmat"
    assert len(capabilities["models"]) == 6
    assert capabilities["backends"]["algorithmic_seed"]["available"] is True


def test_offline_catalog_reports_actual_local_and_remote_readiness(tmp_path: Path) -> None:
    content = b"cached"
    cached = tmp_path / "cached.pt"
    cached.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    specs = [
        ModelSpec(
            id="test/cached@1",
            provider="test",
            backend="genim",
            task="generation",
            filename="cached.pt",
            source={
                "type": "http",
                "url": "https://example.invalid/cached.pt",
                "local_paths": [str(cached)],
            },
            sha256=digest,
            size_bytes=len(content),
        ),
        ModelSpec(
            id="test/missing@1",
            provider="test",
            backend="genim",
            task="generation",
            filename="missing.pt",
            source={"type": "http", "url": "https://example.invalid/missing.pt"},
            sha256=digest,
            size_bytes=len(content),
        ),
        ModelSpec(
            id="test/remote@1",
            provider="test",
            backend="alexandria_matra",
            task="generation",
            source={"type": "remote", "base_url": "https://example.invalid"},
        ),
    ]
    registry = ModelRegistry(specs, cache_dir=tmp_path / "cache")
    service = GeneratorService(_config(offline=True), registry=registry)
    records = {record["id"]: record for record in service.models()["models"]}

    assert records["test/cached@1"]["availability"] == "local_verified"
    assert records["test/cached@1"]["available"] is True
    assert records["test/missing@1"]["availability"] == "offline_asset_unavailable"
    assert records["test/missing@1"]["available"] is False
    assert records["test/remote@1"]["availability"] == "offline_remote_unavailable"
    assert records["test/remote@1"]["available"] is False


def test_generate_rejects_non_generation_model_before_backend_loading() -> None:
    class RelaxationRegistry:
        def __init__(self) -> None:
            self.spec = ModelSpec(
                id="test/relaxation@1",
                provider="test",
                backend="omat24",
                task="relaxation",
                source={"type": "remote"},
            )
            self.load_calls = 0

        def info(self, ref):
            return self.spec

        def load_backend(self, ref, **kwargs):
            self.load_calls += 1
            raise AssertionError("relaxation backend must not load")

    registry = RelaxationRegistry()
    service = GeneratorService(_config(), registry=registry)  # type: ignore[arg-type]
    with pytest.raises(BackendUnavailableError, match="generation models only"):
        service.generate({"formula": "NaCl", "model": registry.spec.id})
    assert registry.load_calls == 0


def test_server_license_readiness_is_scoped_to_the_named_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_id = "matra/genoa-mpas-med@0.2"
    monkeypatch.setenv("GENMAT_ACCEPT_MODEL_LICENSES", selected_id)
    monkeypatch.setenv("GENMAT_API_AUTO_DISCOVER_CHECKPOINTS", "0")
    config = ServiceConfig.from_env()
    assert config.accept_model_license is False

    records = {
        record["id"]: record
        for record in GeneratorService(config).models()["models"]
    }
    assert records[selected_id]["availability"] in {
        "catalogued",
        "missing_optional_dependencies",
    }
    assert records["matra/genoa-mpas@0.2"]["available"] is False
    assert records["alexandria/gen-crystal@remote"]["available"] is False


def test_genmat_models_cli_lists_and_inspects_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models", "list", "--provider", "matra", "--json"]) == 0
    listed = capsys.readouterr().out
    assert "matra/genoa-mpas-med@0.2" in listed
    assert "genmat/mp-fullsg-legacy@0.1.0" not in listed

    assert main(["models", "info", "genmat-legacy"]) == 0
    info = capsys.readouterr().out
    assert '"id": "genmat/mp-fullsg-legacy@0.1.0"' in info


def test_models_cli_preserves_environment_offline_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class Registry:
        def pull(self, model, **kwargs):
            captured.update(kwargs)
            return tmp_path / "cached.pt"

    monkeypatch.setenv("GENMAT_OFFLINE", "1")
    monkeypatch.setattr(
        "genim.commands.models.ModelRegistry.default",
        lambda **kwargs: Registry(),
    )
    assert main(["models", "pull", "test/model@1"]) == 0
    assert captured["offline"] is None


def test_ensemble_cli_preserves_environment_offline_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class StopAfterResolution(RuntimeError):
        pass

    class Registry:
        def load_backend(self, model, **kwargs):
            captured.update(kwargs)
            raise StopAfterResolution

    monkeypatch.setenv("GENMAT_OFFLINE", "1")
    monkeypatch.setattr(
        "genim.commands.ensemble.ModelRegistry.default",
        lambda **kwargs: Registry(),
    )
    with pytest.raises(StopAfterResolution):
        main(
            [
                "generate-ensemble",
                "--model",
                "test/model@1",
                "--out-dir",
                str(tmp_path / "output"),
            ]
        )
    assert captured["offline"] is None


def test_canonical_auxiliary_environment_names_precede_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    canonical.mkdir()
    legacy.mkdir()
    checkpoint = canonical / "esen.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("GENMAT_CHECKPOINT_DIR", str(canonical))
    monkeypatch.setenv("GENIM_CHECKPOINT_DIR", str(legacy))
    monkeypatch.setenv("GENMAT_OVITO_ROOT", str(canonical / "ovito"))
    monkeypatch.setenv("GENIM_OVITO_ROOT", str(legacy / "ovito"))

    from genim.mlip import _find_local_omat24_checkpoint
    from genim.ovito_tachyon_panel import _default_ovito_root

    assert _find_local_omat24_checkpoint(filename="esen.pt") == checkpoint.resolve()
    assert _default_ovito_root() == canonical / "ovito"
