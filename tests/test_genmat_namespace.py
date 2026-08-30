from __future__ import annotations

import runpy
import sys
from importlib.metadata import distribution

import pytest

import genim
import genmat
from genim.backends import GenIMBackend as LegacyBackend
from genim.checkpoints import load_model_checkpoint as legacy_load_model_checkpoint
from genim.service import GeneratorService as LegacyGeneratorService
from genmat.backends import GenIMBackend, GenMatBackend, GenerationConstraints
from genmat.checkpoints import load_model_checkpoint
from genmat.service import GeneratorService


def test_genmat_is_canonical_namespace_with_legacy_compatibility() -> None:
    assert genmat.GenMat is genim.GenMat
    assert genim.GenIM is genmat.GenIM
    assert genmat.__title__ == "GenMat"
    assert genmat.__version__ == "0.6.0"


def test_genmat_lightweight_submodules_reexport_public_contracts() -> None:
    assert issubclass(GenIMBackend, GenMatBackend)
    assert LegacyBackend is GenIMBackend
    assert genim.GenIMBackend is genmat.GenIMBackend
    assert GenMatBackend.backend_name == "genmat"
    assert GenIMBackend.backend_name == "genim"
    assert GenerationConstraints is genim.GenerationConstraints
    assert load_model_checkpoint is legacy_load_model_checkpoint
    assert GeneratorService is LegacyGeneratorService


def test_legacy_shims_preserve_private_helper_identity() -> None:
    from genim.mlip import _find_local_omat24_checkpoint as legacy_helper
    from genmat.mlip import _find_local_omat24_checkpoint as canonical_helper

    assert legacy_helper is canonical_helper


def test_python_m_genmat_uses_genmat_brand(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["python -m genmat", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("genmat", run_name="__main__")
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith("usage: genmat")


def test_genmat_http_api_metadata_uses_canonical_brand() -> None:
    pytest.importorskip("fastapi")
    from genmat.server import create_app

    app = create_app(
        config=genmat.ServiceConfig(
            auto_discover_checkpoints=False,
            default_backend="auto",
            cors_origins=(),
        )
    )
    assert app.title == "GenMat Generation API"
    assert app.version == "0.6.0"


def test_distribution_metadata_points_console_scripts_at_genmat() -> None:
    installed = distribution("genmat")
    assert installed.metadata["Name"] == "genmat"
    assert installed.version == "0.6.0"
    scripts = {
        entry.name: entry.value
        for entry in installed.entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["genmat"] == "genmat.cli:main"
    assert scripts["genmat-api"] == "genmat.server:main"
    assert scripts["genim"] == "genim.cli:main"
    assert scripts["genim-api"] == "genim.server:main"
