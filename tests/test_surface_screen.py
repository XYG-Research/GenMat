from __future__ import annotations

import csv

from pymatgen.core import Lattice, Structure

from genmat.surface_screen import (
    SurfaceScreenConfig,
    final_termination_class,
    json_ready,
    pick_two_layer_fallback_candidate,
    raw_termination_class,
    run_surface_screen,
)


def test_json_ready_serializes_paths(tmp_path) -> None:
    data = {"root": tmp_path, "items": [tmp_path / "a", {"b": tmp_path / "c"}]}
    got = json_ready(data)
    assert got["root"] == str(tmp_path)
    assert got["items"][0] == str(tmp_path / "a")
    assert got["items"][1]["b"] == str(tmp_path / "c")


def test_termination_class_rules() -> None:
    cfg = SurfaceScreenConfig(input_dir="in", out_dir="out")  # type: ignore[arg-type]
    metrics_direct = {
        "dipole_risk": False,
        "top_bottom_l1": 0.02,
        "stoich_l1": 0.01,
        "en_polarity_proxy": 0.01,
        "coord_loss_mean": 0.2,
        "severe_loss_fraction": 0.0,
        "surface_roughness_a": 0.05,
    }
    assert raw_termination_class(metrics_direct, cfg)[0] == "raw_good"
    assert final_termination_class(metrics_direct, compensation_found=False, cfg=cfg)[0] == "direct_surface_candidate"

    metrics_polar = dict(metrics_direct)
    metrics_polar.update({"dipole_risk": True, "top_bottom_l1": 0.5})
    assert raw_termination_class(metrics_polar, cfg)[0] == "raw_polar_or_asymmetric"
    assert final_termination_class(metrics_polar, compensation_found=False, cfg=cfg)[0] == "polar_surface_candidate"

    metrics_broken = dict(metrics_direct)
    metrics_broken.update(
        {
            "coord_loss_mean": 3.0,
            "severe_loss_fraction": 0.8,
            "coord_loss_ratio_mean": 0.5,
            "severe_loss_ratio_fraction": 0.8,
        }
    )
    assert raw_termination_class(metrics_broken, cfg)[0] == "raw_broken_skeleton"
    assert final_termination_class(metrics_broken, compensation_found=False, cfg=cfg)[0] == "rejected_broken_skeleton"


def test_run_surface_screen_smoke(tmp_path) -> None:
    structure = Structure(
        Lattice.cubic(2.9),
        ["Al", "Ni"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    cif_path = cif_dir / "alni.cif"
    structure.to(filename=str(cif_path))

    out_dir = tmp_path / "out"
    cfg = SurfaceScreenConfig(
        input_dir=cif_dir,
        out_dir=out_dir,
        max_index=1,
        min_d_hkl=0.5,
        max_surface_area=100.0,
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        write_all_initial_slabs=False,
        run_mlip=False,
    )
    summary = run_surface_screen(cfg)
    assert summary["n_input_cifs"] == 1
    assert summary["n_failed_structures"] == 0
    assert summary["n_termination_candidates"] > 0

    term_csv = out_dir / "termination_screening.csv"
    assert term_csv.exists()
    with term_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert any(row["raw_class"] for row in rows)
    assert any(row["final_class"] for row in rows)


def test_pick_two_layer_fallback_candidate_prefers_closest_matching_termination() -> None:
    cfg = SurfaceScreenConfig(input_dir="in", out_dir="out")  # type: ignore[arg-type]
    reference_metrics = {
        "top_layer_formula": "Ni",
        "bottom_layer_formula": "Al",
        "top_two_layers_formula": "AlNi",
        "bottom_two_layers_formula": "AlNi",
        "top_bottom_l1": 0.25,
        "stoich_l1": 0.02,
        "top_surface_atom_density": 0.08,
        "bottom_surface_atom_density": 0.08,
        "slab_thickness_a": 16.5,
    }
    candidates = [
        {
            "shift": 0.12,
            "slab": "best",
            "metrics": {
                "top_layer_formula": "Ni",
                "bottom_layer_formula": "Al",
                "top_two_layers_formula": "AlNi",
                "bottom_two_layers_formula": "AlNi",
                "top_bottom_l1": 0.22,
                "stoich_l1": 0.02,
                "top_surface_atom_density": 0.08,
                "bottom_surface_atom_density": 0.08,
                "slab_thickness_a": 14.2,
            },
        },
        {
            "shift": 0.62,
            "slab": "worse",
            "metrics": {
                "top_layer_formula": "Al",
                "bottom_layer_formula": "Ni",
                "top_two_layers_formula": "Al2Ni",
                "bottom_two_layers_formula": "AlNi2",
                "top_bottom_l1": 0.45,
                "stoich_l1": 0.11,
                "top_surface_atom_density": 0.05,
                "bottom_surface_atom_density": 0.05,
                "slab_thickness_a": 13.8,
            },
        },
        {
            "shift": 0.10,
            "slab": "too_thick",
            "metrics": {
                "top_layer_formula": "Ni",
                "bottom_layer_formula": "Al",
                "top_two_layers_formula": "AlNi",
                "bottom_two_layers_formula": "AlNi",
                "top_bottom_l1": 0.21,
                "stoich_l1": 0.02,
                "top_surface_atom_density": 0.08,
                "bottom_surface_atom_density": 0.08,
                "slab_thickness_a": 15.8,
            },
        },
    ]
    best = pick_two_layer_fallback_candidate(
        reference_shift=0.10,
        reference_metrics=reference_metrics,
        candidates=candidates,
        cfg=cfg,
    )
    assert best is not None
    assert best["slab"] == "best"
