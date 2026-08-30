from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from genmat.backends import (
    BackendCapabilities,
    ConstraintApplication,
    ConstraintStatus,
    EnsembleGenerator,
    GeneratedCandidate,
    GenerationBackend,
    GenerationCondition,
    GenerationConstraints,
    GenerationSettings,
    MatraBackend,
    ObservableRole,
    ProposalConfig,
    ProposedStructure,
    ScientificObservable,
    inspect_matra_checkpoint,
    write_ensemble_run,
)
from genmat.backends.matra import extract_matra_observables
from genmat.backends.mutation import mutate_candidate
from genmat.backends.base import evaluate_constraints
from genmat.checkpoints import CheckpointError, sha256_file
from genmat.validate import validate_atoms_report


class _FakeSpecies:
    def __init__(self, symbol: str):
        self.symbol = symbol


class _FakePymatgenStructure:
    def __init__(self, atoms: Atoms):
        self.lattice = SimpleNamespace(matrix=np.asarray(atoms.cell.array, dtype=float))
        self.frac_coords = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
        self.species = [_FakeSpecies(symbol) for symbol in atoms.get_chemical_symbols()]


class _FakeMatraModel:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.last_condition = None

    def generate(self, *, n: int, condition: str, **kwargs):
        self.last_condition = condition
        return [f"{condition} GLOBALSTOP" for _ in range(n)]

    def decode(self, sequences, **kwargs):
        return [
            SimpleNamespace(
                structure=_FakePymatgenStructure(self.atoms),
                valid=True,
                bond_consistent=True,
                symm_consistent=True,
                spg_consistent=True,
            )
            for _ in sequences
        ]


class _StaticBackend:
    backend_name = "static"
    capabilities = frozenset()

    def __init__(self, name: str, atoms: Atoms, *, condition_checks=None):
        self.model_name = name
        self.checkpoint_sha256 = None
        self.atoms = atoms
        self.condition_checks = dict(condition_checks or {})

    def propose(self, *, condition: GenerationCondition, config: ProposalConfig):
        report = validate_atoms_report(self.atoms, **config.validation_kwargs())
        return [
            ProposedStructure(
                candidate_id=f"static-{self.model_name}-00000",
                backend=self.backend_name,
                model_name=self.model_name,
                checkpoint_sha256=None,
                atoms=self.atoms.copy(),
                validation=report,
                condition=condition.as_dict(),
                condition_checks=dict(self.condition_checks),
                applied_constraints=[],
                unsupported_constraints=[],
                sampling=config.as_dict(),
            )
        ]


class _HullEvaluator:
    evaluator_name = "test-hull"

    def evaluate(self, candidate):
        return [
            ScientificObservable(
                name="energy_above_hull",
                value=0.03,
                unit="eV/atom",
                role=ObservableRole.CALCULATED,
                method="test.reference_hull",
                evidence_level="L3",
                source="test",
                independently_validated=True,
            )
        ]


def _fake_matra_checkpoint(path: Path) -> None:
    torch.save(
        {
            "__matra_version__": "test",
            "config": {
                "num_layers": 1,
                "d_model": 8,
                "num_heads": 2,
                "dff_ratio": 2,
                "props": ["ehull"],
            },
            "vocab": {
                "PAD": 0,
                "C*": 1,
                "STOP": 2,
                "GLOBALSTOP": 3,
                "SPACEGROUP": 4,
                "WYCKOFF": 5,
                "LATTICE": 6,
                "AMT": 7,
                "ELMS": 8,
                "STOICH": 9,
                "EHULL": 10,
            },
            "state_dict": {"layer.weight": torch.ones((2, 3))},
        },
        path,
    )


def test_generation_condition_validates_and_builds_matra_prompt() -> None:
    condition = GenerationCondition(
        elements=("Na", "Cl"),
        stoichiometry=(1, 1),
        spacegroup_number=225,
        matra_wyckoff_indices=(1, 2),
        stability="stable",
        target_e_hull=0.075,
    )
    prompt = condition.to_matra_prompt()
    assert prompt is not None
    assert "EHULL_DISC EH0 STOP" in prompt
    assert "AMT 2 STOP ELMS Na Cl STOP" in prompt
    assert "STOICH 1 1 STOP" in prompt
    assert "SPACEGROUP S225 STOP" in prompt
    assert "WYCKOFF W1 W2 STOP" in prompt
    assert "EHULL C*0.075 STOP" in prompt

    with pytest.raises(ValueError, match="one value per element"):
        GenerationCondition(elements=("Na", "Cl"), stoichiometry=(1,))
    with pytest.raises(ValueError, match="1..230"):
        GenerationCondition(spacegroup_number=231)


def test_scientific_generation_names_keep_03_compatibility() -> None:
    assert GenerationCondition is GenerationConstraints
    assert ProposalConfig is GenerationSettings
    assert ProposedStructure is GeneratedCandidate
    assert GenerationBackend.__name__ == "GenerationBackend"

    capabilities = BackendCapabilities(
        {
            "elements": ConstraintApplication.SAMPLING_FILTER,
            "stability": ConstraintApplication.UNSUPPORTED,
        }
    )
    assert capabilities.supported_constraints == frozenset({"elements"})
    assert capabilities.handling_for("elements") is ConstraintApplication.SAMPLING_FILTER
    assert capabilities.handling_for("spacegroup_number") is ConstraintApplication.UNSUPPORTED


def test_matra_backend_converts_validates_and_records_provenance() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    model = _FakeMatraModel(atoms)
    backend = MatraBackend.from_model(
        model,
        model_name="fake-matra",
        checkpoint_sha256="a" * 64,
    )
    condition = GenerationCondition(
        elements=("Na", "Cl"),
        stoichiometry=(1, 1),
        spacegroup_number=225,
        stability="stable",
    )
    candidates = backend.propose(
        condition=condition,
        config=ProposalConfig(n=2, seed=7),
    )
    assert len(candidates) == 2
    assert all(candidate.valid for candidate in candidates)
    assert all(candidate.condition_compliant is None for candidate in candidates)
    assert all(candidate.condition_checks["elements"] is True for candidate in candidates)
    assert all(candidate.condition_checks["stoichiometry"] is True for candidate in candidates)
    assert all(candidate.condition_checks["spacegroup_number"] is True for candidate in candidates)
    assert all(candidate.condition_checks["stability"] is None for candidate in candidates)
    assert all(candidate.constraint_status is ConstraintStatus.NOT_EVALUATED for candidate in candidates)
    assert all(
        candidate.constraint_assessments["stability"].method == "energy_model_not_run"
        for candidate in candidates
    )
    assert all(candidate.selected for candidate in candidates)
    assert all(
        candidate.selection_reason == "selected_with_unverified_constraints"
        for candidate in candidates
    )
    assert all(candidate.checkpoint_sha256 == "a" * 64 for candidate in candidates)
    assert model.last_condition is not None and "SPACEGROUP S225" in model.last_condition
    assert "EHULL_DISC" not in model.last_condition
    assert all(candidate.unsupported_constraints == ["stability"] for candidate in candidates)


def test_matra_property_blocks_are_reported_without_becoming_energy_evidence() -> None:
    emitted = extract_matra_observables(
        "AMT 2 STOP ELMS Na Cl STOP EHULL C*0.042 STOP GLOBALSTOP",
        prompt="AMT 2 STOP ELMS Na Cl STOP",
        properties=["ehull", "ehull_disc"],
        model_name="matra-v02-med",
        checkpoint_sha256="a" * 64,
    )
    assert len(emitted) == 1
    assert emitted[0].name == "energy_above_hull"
    assert emitted[0].value == pytest.approx(0.042)
    assert emitted[0].role is ObservableRole.MODEL_EMISSION
    assert emitted[0].independently_validated is False

    conditioned = extract_matra_observables(
        "EHULL C*0.05 STOP AMT 2 STOP GLOBALSTOP",
        prompt="EHULL C*0.05 STOP AMT 2 STOP",
        properties=["ehull"],
        model_name="matra-v02-med",
        checkpoint_sha256=None,
    )
    assert conditioned[0].role is ObservableRole.CONDITIONING_TARGET


def test_population_mutation_preserves_composition_and_requested_spacegroup() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    backend = MatraBackend.from_model(
        _FakeMatraModel(atoms),
        model_name="fake-matra-population",
        checkpoint_sha256="b" * 64,
    )
    run = EnsembleGenerator([backend]).run(
        condition=GenerationConstraints(
            elements=("Na", "Cl"),
            stoichiometry=(1, 1),
            spacegroup_number=225,
        ),
        config=GenerationSettings(
            n=4,
            mutation_fraction=0.5,
            mutation_strain=0.12,
            seed=19,
            require_backend_consistency=False,
        ),
    )
    assert len(run.candidates) == 4
    mutants = [
        candidate
        for candidate in run.candidates
        if candidate.backend_metrics.get("population_role") == "mutant"
    ]
    assert len(mutants) == 2
    assert all(
        sorted(candidate.atoms.get_chemical_symbols()) == ["Cl", "Na"]
        for candidate in mutants
        if candidate.atoms is not None
    )
    assert all(candidate.condition_checks["spacegroup_number"] is True for candidate in mutants)
    assert all(candidate.validation.metrics["spacegroup_number"] == 225 for candidate in mutants)
    assert all(candidate.scientific_observables == [] for candidate in mutants)
    assert all(
        candidate.backend_metrics["mutation_method"]
        == "genim_composition_preserving_mutation_v2"
        for candidate in mutants
    )


def test_symmetry_orbit_mutation_preserves_p_minus_one_and_moves_internal_coordinate() -> None:
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0.13, 0.21, 0.34], [0.87, 0.79, 0.66]],
        cell=np.diag([4.1, 5.2, 6.3]),
        pbc=True,
    )
    settings = GenerationSettings(
        seed=11,
        mutation_strain=0.0,
        mutation_displacement=0.08,
        preserve_spacegroup=True,
        validation_options={"min_coordination": 0},
    )
    report = validate_atoms_report(atoms, **settings.validation_kwargs())
    assert report.valid
    assert report.metrics["spacegroup_number"] == 2
    condition = GenerationConstraints(elements=("Si",), spacegroup_number=2)
    constraint_assessments = evaluate_constraints(atoms, report, condition)
    parent = GeneratedCandidate(
        candidate_id="p-minus-one-parent",
        backend="static",
        model_name="symmetry-test",
        checkpoint_sha256=None,
        atoms=atoms,
        validation=report,
        condition=condition.as_dict(),
        condition_checks={
            name: assessment.value for name, assessment in constraint_assessments.items()
        },
        applied_constraints=["elements", "spacegroup_number"],
        unsupported_constraints=[],
        sampling={},
        constraint_assessments=constraint_assessments,
    )
    child = mutate_candidate(
        parent,
        condition=condition,
        config=settings,
        mutation_index=0,
        occupied_fingerprints=set(),
    )
    assert child is not None
    assert child.validation.metrics["spacegroup_number"] == 2
    assert child.condition_checks["spacegroup_number"] is True
    assert child.atoms is not None
    assert child.atoms.get_chemical_formula() == atoms.get_chemical_formula()
    assert not np.allclose(
        child.atoms.get_scaled_positions(wrap=True),
        atoms.get_scaled_positions(wrap=True),
    )
    assert child.backend_metrics["symmetry_orbit_mutation"] is True


def test_generation_settings_validate_mutation_controls() -> None:
    assert GenerationSettings(n=8, mutation_fraction=0.5).mutation_fraction == 0.5
    with pytest.raises(ValueError, match="mutation_fraction"):
        GenerationSettings(mutation_fraction=1.1)
    with pytest.raises(ValueError, match="mutation_strain"):
        GenerationSettings(mutation_strain=0.6)


def test_ensemble_deduplicates_across_backends_and_writes_audit_files(tmp_path: Path) -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    run = EnsembleGenerator(
        [_StaticBackend("one", atoms), _StaticBackend("two", atoms)]
    ).run(config=ProposalConfig(n=1))
    assert run.report()["total"] == 2
    assert run.report()["valid"] == 2
    assert run.report()["unique_valid"] == 1
    assert run.report()["schema_version"] == 3
    assert run.report()["selected"] == 1
    assert len(run.accepted) == 1
    assert run.accepted[0].rank == 1
    assert run.accepted[0].ranking_score is not None
    duplicate = [candidate for candidate in run.candidates if candidate.duplicate_of is not None]
    assert len(duplicate) == 1

    artifacts = write_ensemble_run(run, tmp_path)
    assert artifacts["cif_count"] == 1
    assert artifacts["manifest"].is_file()
    assert artifacts["report"].is_file()
    records = [
        json.loads(line)
        for line in artifacts["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert sum(record["accepted"] for record in records) == 1
    assert sum(record["selected"] for record in records) == 1
    assert all("constraint_assessments" in record for record in records)
    assert all("ranking" in record for record in records)
    with pytest.raises(FileExistsError):
        write_ensemble_run(run, tmp_path)


def test_ensemble_prefers_condition_compliant_duplicate() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    run = EnsembleGenerator(
        [
            _StaticBackend("mismatch", atoms, condition_checks={"elements": False}),
            _StaticBackend("match", atoms, condition_checks={"elements": True}),
        ]
    ).run(config=ProposalConfig(n=1))
    mismatch, match = run.candidates
    assert mismatch.duplicate_of == match.candidate_id
    assert match.duplicate_of is None
    assert match.accepted


def test_independent_evaluator_upgrades_stability_constraint_evidence() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    run = EnsembleGenerator(
        [_StaticBackend("evaluated", atoms)],
        evaluators=[_HullEvaluator()],
    ).run(
        condition=GenerationConstraints(stability="stable", target_e_hull=0.05),
        config=GenerationSettings(n=1),
    )
    candidate = run.candidates[0]
    assert candidate.condition_checks == {"stability": True, "target_e_hull": True}
    assert candidate.constraint_assessments["stability"].evidence_level == "L3"
    assert candidate.selected


def test_matra_checkpoint_inspection_is_safe_and_checksum_aware(tmp_path: Path) -> None:
    path = tmp_path / "matra.ckpt"
    _fake_matra_checkpoint(path)
    info = inspect_matra_checkpoint(path, expected_sha256=sha256_file(path))
    assert info.matra_version == "test"
    assert info.vocab_size == 11
    assert info.tensor_count == 1
    assert info.parameter_count == 6
    assert info.config["d_model"] == 8
    with pytest.raises(CheckpointError, match="SHA256 mismatch"):
        inspect_matra_checkpoint(path, expected_sha256="0" * 64)
