from __future__ import annotations

from collections import Counter

import pytest

from genim.composition import parse_formula_counts, ratios_to_integer_counts
from genim.server import create_app
from genim.service import BackendUnavailableError, GeneratorService, ServiceConfig


def seed_only_service() -> GeneratorService:
    return GeneratorService(
        ServiceConfig(
            auto_discover_checkpoints=False,
            default_backend="auto",
            cors_origins=(),
        )
    )


def test_formula_and_ratio_parsing_are_reduced_and_general() -> None:
    assert parse_formula_counts("Ca3(PO4)2") == {"Ca": 3, "P": 2, "O": 8}
    assert ratios_to_integer_counts(["Fe", "Ni"], [0.5, 1.0]) == {"Fe": 1, "Ni": 2}


def test_default_request_is_reproducible_and_uses_schema_v3() -> None:
    service = seed_only_service()
    first = service.generate({"n": 2, "seed": 19})
    second = service.generate({"n": 2, "seed": 19})
    assert first["schema_version"] == 3
    assert first["generator_mode"] == "algorithmic_seed"
    assert first["request"]["formula"] == "SiO2"
    assert [row["structure"] for row in first["candidates"]] == [
        row["structure"] for row in second["candidates"]
    ]
    assert first["request"]["settings"]["seed"] == 19
    assert all(row["ranking"]["score"] is not None for row in first["candidates"])


def test_omitted_seed_uses_documented_deterministic_default() -> None:
    first = seed_only_service().generate({"n": 1})
    second = seed_only_service().generate({"n": 1})
    assert first["request"]["settings"]["seed"] == 7
    assert first["candidates"][0]["structure"] == second["candidates"][0]["structure"]


def test_arbitrary_composition_has_explicit_constraint_evidence() -> None:
    response = seed_only_service().generate(
        {
            "formula": "LiFePO4",
            "spacegroup_number": 62,
            "backend": "auto",
            "n": 1,
            "seed": 7,
        }
    )
    candidate = response["candidates"][0]
    assert Counter(candidate["structure"]["species"]) == Counter(
        {"Li": 1, "Fe": 1, "P": 1, "O": 4}
    )
    assert candidate["constraint_assessments"]["elements"]["status"] == "satisfied"
    assert candidate["constraint_assessments"]["stoichiometry"]["status"] == "satisfied"
    assert candidate["constraint_assessments"]["spacegroup_number"]["status"] in {
        "satisfied",
        "violated",
    }
    assert "spacegroup_number" in candidate["unsupported_constraints"]
    assert candidate["backend_metrics"]["learned_model"] is False
    assert candidate["validation"]["metrics"]["min_covalent_distance_ratio"] >= 0.55
    assert candidate["validation"]["metrics"]["min_coordination"] >= 1


def test_unconfigured_explicit_checkpoint_backend_is_not_silently_substituted() -> None:
    with pytest.raises(BackendUnavailableError, match="not configured"):
        seed_only_service().generate({"formula": "NaCl", "backend": "matra"})


def test_http_api_has_working_defaults_and_capabilities() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(service=seed_only_service()))
    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["always_available_backend"] == "algorithmic_seed"

    capabilities = client.get("/v1/capabilities")
    assert capabilities.status_code == 200
    assert "algorithmic_seed" in capabilities.json()["backends"]

    generated = client.post("/v1/generate", json={})
    assert generated.status_code == 200
    assert generated.json()["request"]["formula"] == "SiO2"
