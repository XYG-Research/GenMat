from __future__ import annotations

import io
import json
import re
from typing import Any

import requests
from ase import Atoms
from ase.io import read

from ..composition import ratios_to_integer_counts
from ..validate import ValidationReport, validate_atoms_report
from .base import (
    BackendCapabilities,
    ConstraintApplication,
    GeneratedCandidate,
    GenerationConstraints,
    GenerationSettings,
    ObservableRole,
    ScientificObservable,
    evaluate_constraints,
)


class AlexandriaMatraBackend:
    """Adapter for the public Matra/Alexandria generation-and-relaxation pipeline.

    The upstream endpoint returns one chosen structure from an internal pool.
    Its public response includes an ``energy`` scalar but does not identify the
    calculator or reference zero, so GenIM exposes it conservatively as a
    postprocessed estimate rather than formation energy, hull energy, or DFT.
    """

    backend_name = "alexandria_matra"
    model_name = "alexandria-gen-crystal"
    checkpoint_sha256 = None
    capabilities = BackendCapabilities(
        {
            "elements": ConstraintApplication.CONDITIONING,
            "stoichiometry": ConstraintApplication.CONDITIONING,
            "spacegroup_number": ConstraintApplication.CONDITIONING,
        },
        notes=(
            "Public Matra/Alexandria pipeline. The endpoint generates an internal candidate "
            "pool, relaxes structures, and returns one result. It is rate-limited and does "
            "not report the energy calculator identity in its response."
        ),
    )

    def __init__(
        self,
        *,
        base_url: str = "https://alexandria.icams.rub.de/mapi",
        timeout_seconds: float = 120.0,
        session: Any | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()

    @staticmethod
    def _formula(condition: GenerationConstraints) -> str:
        if not condition.elements:
            return ""
        ratios = condition.stoichiometry or tuple(1.0 for _ in condition.elements)
        counts = ratios_to_integer_counts(list(condition.elements), list(ratios))
        return "".join(
            f"{element}{'' if count == 1 else count}"
            for element, count in counts.items()
        )

    @staticmethod
    def _creativity(temperature: float) -> int:
        if float(temperature) <= 0.9:
            return 1
        if float(temperature) <= 1.4:
            return 2
        return 3

    def _generate(self, condition: GenerationConstraints, config: GenerationSettings) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        payload = {
            "composition": self._formula(condition),
            "spacegroup": (
                "" if condition.spacegroup_number is None else str(condition.spacegroup_number)
            ),
            "creativity": self._creativity(config.temperature),
            "pool_size": max(1, min(int(config.n), 16)),
        }
        response = self.session.post(
            f"{self.base_url}/gen-crystal",
            json=payload,
            stream=True,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        progress: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not str(raw_line).strip():
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            record = json.loads(line)
            if record.get("status") == "progress":
                progress.append(dict(record))
            elif record.get("status") == "done":
                result = dict(record.get("data") or {})
            elif record.get("error"):
                raise RuntimeError(str(record["error"]))
        if result is None:
            raise RuntimeError("Alexandria response ended without a done record")
        result["request"] = payload
        return result, progress

    def _download_atoms(self, filename: str) -> tuple[Atoms, str]:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "", str(filename))
        if not safe_name or safe_name != filename:
            raise ValueError("Alexandria returned an unsafe CIF filename")
        url = f"{self.base_url}/download/{safe_name}"
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        atoms = read(io.StringIO(response.text), format="cif")
        if not isinstance(atoms, Atoms):
            raise TypeError("Alexandria CIF did not decode to one ASE Atoms object")
        return atoms, url

    def propose(
        self,
        *,
        condition: GenerationConstraints,
        config: GenerationSettings,
    ) -> list[GeneratedCandidate]:
        requested = set(condition.requested_fields())
        applied = self.capabilities.applied_fields(requested)
        unsupported = self.capabilities.unsupported_fields(requested)
        try:
            result, progress = self._generate(condition, config)
            atoms, cif_url = self._download_atoms(str(result["cif_filename"]))
            validation = validate_atoms_report(atoms, **config.validation_kwargs())
            assessments = evaluate_constraints(atoms, validation, condition)
            energy = float(result["energy"])
            observables = [
                ScientificObservable(
                    name="relaxed_energy_per_atom",
                    value=energy,
                    unit="eV/atom",
                    role=ObservableRole.POSTPROCESSED_ESTIMATE,
                    method="alexandria.mapi.gen-crystal",
                    evidence_level="L2",
                    source="alexandria.response.energy",
                    model_name=self.model_name,
                    independently_validated=False,
                    detail=(
                        "The public endpoint reports this value after its relaxation stage but "
                        "does not identify the calculator or reference zero. Do not relabel it "
                        "as formation energy, energy above hull, or DFT energy."
                    ),
                )
            ]
            candidate_id = re.sub(
                r"[^a-z0-9]+",
                "-",
                str(result.get("cif_filename", "alexandria")).lower(),
            ).strip("-")
            return [
                GeneratedCandidate(
                    candidate_id=f"alexandria-{candidate_id}",
                    backend=self.backend_name,
                    model_name=self.model_name,
                    checkpoint_sha256=None,
                    atoms=atoms,
                    validation=validation,
                    condition=condition.as_dict(),
                    condition_checks={
                        name: assessment.value for name, assessment in assessments.items()
                    },
                    applied_constraints=applied,
                    unsupported_constraints=unsupported,
                    sampling=config.as_dict(),
                    constraint_assessments=assessments,
                    scientific_observables=observables,
                    backend_metrics={
                        "generation_mode": "remote_matra_relaxation_pipeline",
                        "reported_frontend_model": "Matra-Genoa-v02-med-0.1",
                        "upstream": self.base_url,
                        "upstream_request": result.get("request"),
                        "upstream_formula": result.get("formula"),
                        "cif_filename": result.get("cif_filename"),
                        "cif_url": cif_url,
                        "progress": progress,
                        "returned_candidates": 1,
                        "internal_pool_size": result.get("request", {}).get("pool_size"),
                    },
                )
            ]
        except Exception as exc:
            validation = ValidationReport(False, "alexandria_service_error", {})
            assessments = evaluate_constraints(None, validation, condition)
            return [
                GeneratedCandidate(
                    candidate_id="alexandria-error-00000",
                    backend=self.backend_name,
                    model_name=self.model_name,
                    checkpoint_sha256=None,
                    atoms=None,
                    validation=validation,
                    condition=condition.as_dict(),
                    condition_checks={
                        name: assessment.value for name, assessment in assessments.items()
                    },
                    applied_constraints=applied,
                    unsupported_constraints=unsupported,
                    sampling=config.as_dict(),
                    constraint_assessments=assessments,
                    backend_metrics={
                        "generation_mode": "remote_matra_relaxation_pipeline",
                        "upstream": self.base_url,
                    },
                    error=f"alexandria_service:{type(exc).__name__}:{exc}",
                )
            ]


__all__ = ["AlexandriaMatraBackend"]
