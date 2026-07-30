from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ase.io import write

from ..dedup import atoms_hash
from .base import GenerationCondition, ProposalBackend, ProposalConfig, ProposedStructure


@dataclass(frozen=True)
class BackendSummary:
    requested: int
    returned: int
    valid: int
    condition_compliant: int
    condition_unknown: int
    unique_valid: int
    accepted: int
    rejection_reasons: dict[str, int]


@dataclass
class EnsembleRun:
    condition: GenerationCondition
    candidates: list[ProposedStructure]
    backend_summaries: dict[str, BackendSummary]

    @property
    def accepted(self) -> list[ProposedStructure]:
        return [candidate for candidate in self.candidates if candidate.accepted]

    def report(self) -> dict[str, Any]:
        valid = [candidate for candidate in self.candidates if candidate.valid]
        unique_valid = [candidate for candidate in valid if candidate.duplicate_of is None]
        return {
            "schema_version": 1,
            "condition": self.condition.as_dict(),
            "total": len(self.candidates),
            "valid": len(valid),
            "unique_valid": len(unique_valid),
            "accepted": len(self.accepted),
            "condition_compliant": sum(candidate.condition_compliant is True for candidate in self.candidates),
            "condition_unknown": sum(candidate.condition_compliant is None for candidate in self.candidates),
            "backends": {
                name: asdict(summary)
                for name, summary in sorted(self.backend_summaries.items())
            },
        }


def _summary_key(backend: ProposalBackend) -> str:
    return f"{backend.backend_name}:{backend.model_name}"


def _rejection_reason(candidate: ProposedStructure) -> str | None:
    if candidate.valid:
        if candidate.condition_compliant is False:
            return "condition_mismatch"
        if candidate.duplicate_of is not None:
            return "duplicate"
        return None
    return candidate.error or candidate.validation.reason or "invalid"


class EnsembleGenerator:
    """Run independent proposal models through one validation/dedup contract."""

    def __init__(
        self,
        backends: Iterable[ProposalBackend],
        *,
        dedup_symprec: float = 1e-2,
        dedup_frac_tol: float = 1e-2,
        dedup_cell_tol: float = 2e-1,
    ):
        self.backends = list(backends)
        if not self.backends:
            raise ValueError("At least one proposal backend is required")
        identities = [_summary_key(backend) for backend in self.backends]
        if len(set(identities)) != len(identities):
            raise ValueError("Proposal backend identities must be unique")
        self.dedup_symprec = float(dedup_symprec)
        self.dedup_frac_tol = float(dedup_frac_tol)
        self.dedup_cell_tol = float(dedup_cell_tol)

    def run(
        self,
        *,
        condition: GenerationCondition | None = None,
        config: ProposalConfig | None = None,
        backend_configs: dict[str, ProposalConfig] | None = None,
    ) -> EnsembleRun:
        condition = condition or GenerationCondition()
        default_config = config or ProposalConfig()
        configs = backend_configs or {}
        candidates: list[ProposedStructure] = []
        requested_by_key: dict[str, int] = {}
        used_ids: Counter[str] = Counter()

        for backend in self.backends:
            key = _summary_key(backend)
            backend_config = configs.get(key, configs.get(backend.backend_name, default_config))
            requested_by_key[key] = int(backend_config.n)
            proposed = backend.propose(condition=condition, config=backend_config)
            for candidate in proposed:
                used_ids[candidate.candidate_id] += 1
                if used_ids[candidate.candidate_id] > 1:
                    candidate.candidate_id = f"{candidate.candidate_id}-{used_ids[candidate.candidate_id]}"
                candidates.append(candidate)

        groups: dict[str, list[tuple[int, ProposedStructure]]] = {}
        for order, candidate in enumerate(candidates):
            if not candidate.valid or candidate.atoms is None:
                continue
            key = atoms_hash(
                candidate.atoms,
                symprec=self.dedup_symprec,
                frac_tol=self.dedup_frac_tol,
                cell_tol=self.dedup_cell_tol,
                mode="full",
            )
            groups.setdefault(key, []).append((order, candidate))

        compliance_priority = {True: 0, None: 1, False: 2}
        for group in groups.values():
            _, representative = min(
                group,
                key=lambda item: (compliance_priority[item[1].condition_compliant], item[0]),
            )
            for _, candidate in group:
                if candidate is not representative:
                    candidate.duplicate_of = representative.candidate_id

        summaries: dict[str, BackendSummary] = {}
        for backend in self.backends:
            key = _summary_key(backend)
            subset = [
                candidate
                for candidate in candidates
                if candidate.backend == backend.backend_name and candidate.model_name == backend.model_name
            ]
            reasons = Counter(
                reason for candidate in subset if (reason := _rejection_reason(candidate)) is not None
            )
            summaries[key] = BackendSummary(
                requested=requested_by_key[key],
                returned=len(subset),
                valid=sum(candidate.valid for candidate in subset),
                condition_compliant=sum(candidate.condition_compliant is True for candidate in subset),
                condition_unknown=sum(candidate.condition_compliant is None for candidate in subset),
                unique_valid=sum(candidate.valid and candidate.duplicate_of is None for candidate in subset),
                accepted=sum(candidate.accepted for candidate in subset),
                rejection_reasons=dict(sorted(reasons.items())),
            )
        return EnsembleRun(condition=condition, candidates=candidates, backend_summaries=summaries)


def write_ensemble_run(
    run: EnsembleRun,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write accepted CIFs plus a complete JSONL audit trail and report."""

    target = Path(out_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "candidates.jsonl"
    report_path = target / "ensemble-report.json"
    if not overwrite:
        occupied = [path for path in (manifest_path, report_path) if path.exists()]
        occupied.extend(target / f"{candidate.candidate_id}.cif" for candidate in run.accepted)
        occupied = [path for path in occupied if path.exists()]
        if occupied:
            raise FileExistsError(f"Refusing to overwrite existing ensemble artifacts: {occupied[0]}")

    written_cifs: list[Path] = []
    for candidate in run.accepted:
        assert candidate.atoms is not None
        path = target / f"{candidate.candidate_id}.cif"
        write(str(path), candidate.atoms, format="cif")
        written_cifs.append(path)

    manifest_text = "".join(
        json.dumps(candidate.to_record(), sort_keys=True) + "\n"
        for candidate in run.candidates
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    report_path.write_text(
        json.dumps(run.report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "out_dir": target,
        "manifest": manifest_path,
        "report": report_path,
        "cif_count": len(written_cifs),
    }


__all__ = [
    "BackendSummary",
    "EnsembleGenerator",
    "EnsembleRun",
    "write_ensemble_run",
]
