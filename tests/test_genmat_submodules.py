from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


PUBLIC_SUBMODULES = (
    "api",
    "autoscale",
    "benchmark",
    "checkpoints",
    "chem",
    "composition",
    "config",
    "decode",
    "dedup",
    "examples",
    "generate",
    "hull",
    "inspect",
    "ml_relax",
    "mlip",
    "model",
    "models",
    "mp_download",
    "ovito_tachyon_panel",
    "preprocess",
    "score",
    "server",
    "service",
    "snapshot_panel",
    "structure_format",
    "surface_screen",
    "sym_seed",
    "symmetry",
    "synth",
    "tokenizer",
    "train",
    "validate",
    "backends.alexandria",
    "backends.base",
    "backends.ensemble",
    "backends.genmat",
    "backends.genim",
    "backends.matra",
    "backends.mutation",
    "backends.ranking",
    "backends.seed",
    "commands.ensemble",
    "commands.models",
    "commands.serve",
)


def test_every_non_private_legacy_module_has_a_genmat_implementation() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"

    for relative_package in (Path(), Path("backends"), Path("commands")):
        legacy_dir = source_root / "genim" / relative_package
        canonical_dir = source_root / "genmat" / relative_package
        legacy_modules = {
            path.name
            for path in legacy_dir.glob("*.py")
            if not path.name.startswith("_")
        }
        canonical_modules = {path.name for path in canonical_dir.glob("*.py")}

        assert legacy_modules <= canonical_modules, (
            relative_package,
            sorted(legacy_modules - canonical_modules),
        )


@pytest.mark.parametrize("submodule", PUBLIC_SUBMODULES)
def test_public_genmat_submodule_has_a_canonical_import_path(submodule: str) -> None:
    assert importlib.util.find_spec(f"genmat.{submodule}") is not None


@pytest.mark.parametrize(
    ("submodule", "compatibility_submodule", "symbol"),
    (
        ("chem", "chem", "ChemistryPolicy"),
        ("composition", "composition", "parse_formula_counts"),
        ("model", "model", "CausalTransformerLM"),
        ("validate", "validate", "ValidationReport"),
        ("mlip", "mlip", "build_omat24_calculator"),
        ("backends.base", "backends.base", "GenerationBackend"),
        ("backends.genmat", "backends.genim", "GenMatBackend"),
        ("backends.matra", "backends.matra", "MatraBackend"),
        ("backends.mutation", "backends.mutation", "mutate_candidate"),
        ("backends.ranking", "backends.ranking", "rank_candidates"),
        ("commands.models", "commands.models", "add_models_parser"),
    ),
)
def test_genmat_submodules_reexport_the_same_objects(
    submodule: str,
    compatibility_submodule: str,
    symbol: str,
) -> None:
    canonical = importlib.import_module(f"genmat.{submodule}")
    compatibility = importlib.import_module(f"genim.{compatibility_submodule}")

    assert getattr(canonical, symbol) is getattr(compatibility, symbol)
    if hasattr(canonical, "__all__"):
        assert symbol in canonical.__all__


def test_importing_genmat_does_not_eagerly_load_optional_stacks() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = f"""
import importlib.util
import sys

sys.path.insert(0, {str(source_root)!r})
import genmat

# Discovering an optional feature must not execute its implementation module.
assert importlib.util.find_spec('genmat.surface_screen') is not None
assert 'genmat.surface_screen' not in sys.modules

# The server module also keeps FastAPI behind create_app().
import genmat.server

optional_roots = (
    'fairchem',
    'fastapi',
    'huggingface_hub',
    'matra',
    'ovito',
    'pymatgen',
    'uvicorn',
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in optional_roots)
)
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
