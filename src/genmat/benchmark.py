from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ase.io import read

from .dedup import atoms_hash
from .validate import validate_atoms_report


@dataclass(frozen=True)
class BenchmarkReport:
    total_files: int
    readable: int
    valid: int
    unique_valid: int
    validity_rate: float
    uniqueness_rate: float
    unique_formulas: int
    element_coverage: list[str]
    spacegroup_coverage: list[int]
    rejection_reasons: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_cif_dir(
    cif_dir: Path,
    *,
    min_dist: float = 0.5,
    symprec: float = 1e-2,
    dedup_symprec: float = 1e-2,
    dedup_frac_tol: float = 1e-2,
    dedup_cell_tol: float = 2e-1,
) -> BenchmarkReport:
    """Compute deterministic validity, uniqueness and coverage metrics."""

    cif_dir = Path(cif_dir).expanduser().resolve()
    paths = sorted(path for path in cif_dir.glob("*.cif") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .cif files found under {cif_dir}")

    readable = 0
    valid = 0
    formulas: set[str] = set()
    elements: set[str] = set()
    spacegroups: set[int] = set()
    unique_hashes: set[str] = set()
    reasons: Counter[str] = Counter()

    for path in paths:
        try:
            atoms = read(str(path))
            readable += 1
        except Exception:
            reasons["read_error"] += 1
            continue
        report = validate_atoms_report(atoms, min_dist=float(min_dist), symprec=float(symprec))
        if not report.valid:
            reasons[report.reason] += 1
            continue
        valid += 1
        formulas.add(atoms.get_chemical_formula(mode="hill"))
        elements.update(atoms.get_chemical_symbols())
        sg = report.metrics.get("spacegroup_number")
        if sg is not None:
            spacegroups.add(int(sg))
        unique_hashes.add(
            atoms_hash(
                atoms,
                symprec=float(dedup_symprec),
                frac_tol=float(dedup_frac_tol),
                cell_tol=float(dedup_cell_tol),
                mode="full",
            )
        )

    total = len(paths)
    return BenchmarkReport(
        total_files=total,
        readable=readable,
        valid=valid,
        unique_valid=len(unique_hashes),
        validity_rate=float(valid / total) if total else 0.0,
        uniqueness_rate=float(len(unique_hashes) / valid) if valid else 0.0,
        unique_formulas=len(formulas),
        element_coverage=sorted(elements),
        spacegroup_coverage=sorted(spacegroups),
        rejection_reasons=dict(sorted(reasons.items())),
    )


def write_benchmark_report(report: BenchmarkReport, path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = ["BenchmarkReport", "benchmark_cif_dir", "write_benchmark_report"]
