from __future__ import annotations

from pathlib import Path

from ase.build import bulk
from ase.io import write

from genmat.benchmark import benchmark_cif_dir, write_benchmark_report


def test_benchmark_reports_validity_uniqueness_and_coverage(tmp_path: Path) -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    write(str(tmp_path / "a.cif"), atoms)
    write(str(tmp_path / "duplicate.cif"), atoms)
    report = benchmark_cif_dir(tmp_path, min_dist=0.5, symprec=1e-2)
    assert report.total_files == 2
    assert report.valid == 2
    assert report.unique_valid == 1
    assert report.uniqueness_rate == 0.5
    assert report.element_coverage == ["Cl", "Na"]
    out = write_benchmark_report(report, tmp_path / "report.json")
    assert out.is_file()
    assert '"valid": 2' in out.read_text(encoding="utf-8")
