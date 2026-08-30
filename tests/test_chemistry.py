from __future__ import annotations

import pytest
from ase.build import bulk

from genmat.chem import ChemistryPolicy, available_elements
from genmat.validate import validate_atoms_report


def test_any_policy_accepts_general_materials_chemistry() -> None:
    policy = ChemistryPolicy(mode="any")
    assert policy.accepts_elements({"Na", "Cl"})
    assert policy.accepts_elements({"Si", "O"})
    assert policy.accepts_elements({"C"})
    assert {"H", "C", "N", "O", "F", "Cl"}.issubset(set(policy.available_elements()))


def test_legacy_intermetallic_policy_is_explicit() -> None:
    policy = ChemistryPolicy(mode="intermetallic", include_metalloids=True)
    assert policy.accepts_elements({"Fe", "Si"})
    assert not policy.accepts_elements({"Fe"})
    assert not policy.accepts_elements({"Fe", "O"})
    assert ChemistryPolicy(mode="metallic").accepts_elements({"Fe"})
    with pytest.raises(ValueError, match="requires min_elements >= 2"):
        ChemistryPolicy(mode="intermetallic", min_elements=1)


def test_exact_allow_and_exclude_lists_are_validated() -> None:
    policy = ChemistryPolicy(mode="any", allowed_elements=["Na", "Cl"], excluded_elements=[])
    assert policy.available_elements() == ["Na", "Cl"]
    assert not policy.is_allowed_element("O")
    with pytest.raises(ValueError, match="both allowed and excluded"):
        ChemistryPolicy(mode="any", allowed_elements=["O"], excluded_elements=["O"])
    with pytest.raises(ValueError, match="Unknown chemical element"):
        available_elements(mode="any", allowed_elements=["NotAnElement"])


def test_unary_primitive_cell_has_no_spurious_pair_distance_failure() -> None:
    report = validate_atoms_report(bulk("Si", "sc", a=3.0), min_dist=0.5, symprec=1e-2)
    assert report.valid, report.reason
    assert report.metrics["min_pair_distance"] is None
