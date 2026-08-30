from __future__ import annotations

import json

import pytest

from genmat.backends import (
    AlexandriaMatraBackend,
    GenerationConstraints,
    GenerationSettings,
    ObservableRole,
)


_CIF = """data_NaCl
_cell_length_a 4.00
_cell_length_b 4.00
_cell_length_c 4.00
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na Na1 0.0 0.0 0.0
Cl Cl1 0.5 0.5 0.5
"""


class _Response:
    def __init__(self, *, lines=(), text=""):
        self._lines = list(lines)
        self.text = text

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


class _Session:
    def __init__(self):
        self.post_payloads = []

    def post(self, url, *, json, stream, timeout):
        self.post_payloads.append(json)
        done = {
            "status": "done",
            "data": {
                "cif_filename": "NaCl_331.cif",
                "formula": "NaCl",
                "energy": -3.3107,
            },
        }
        return _Response(
            lines=[
                '{"status":"progress","progress":50,"message":"Relaxing structures..."}',
                json_module.dumps(done),
            ]
        )

    def get(self, url, *, timeout):
        return _Response(text=_CIF)


json_module = json


def test_alexandria_energy_is_a_conservative_postprocessed_observable() -> None:
    session = _Session()
    backend = AlexandriaMatraBackend(session=session)
    candidates = backend.propose(
        condition=GenerationConstraints(
            elements=("Na", "Cl"),
            stoichiometry=(1, 1),
        ),
        config=GenerationSettings(n=4, temperature=0.8, seed=7),
    )
    assert session.post_payloads == [
        {
            "composition": "NaCl",
            "spacegroup": "",
            "creativity": 1,
            "pool_size": 1,
        }
        for _ in range(4)
    ]
    assert len(candidates) == 4
    candidate = candidates[0]
    assert candidate.valid
    assert candidate.backend == "alexandria_matra"
    assert len(candidate.scientific_observables) == 1
    observable = candidate.scientific_observables[0]
    assert observable.value == pytest.approx(-3.3107)
    assert observable.role is ObservableRole.POSTPROCESSED_ESTIMATE
    assert observable.name == "relaxed_energy_per_atom"
    assert observable.independently_validated is False
    assert "Do not relabel" in (observable.detail or "")
    assert len({candidate.candidate_id for candidate in candidates}) == 4
