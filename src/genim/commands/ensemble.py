from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from ..backends import (
    EnsembleGenerator,
    GenerationBackend,
    GenerationConstraints,
    GenerationSettings,
    GenMatBackend,
    MatraBackend,
    write_ensemble_run,
)
from ..models import ModelRegistry


def add_ensemble_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    path_type: Callable[[str], Path],
) -> None:
    parser = subparsers.add_parser(
        "generate-ensemble",
        aliases=["hybrid-generate"],
        help="Generate with one or more backends through shared validation and deduplication.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--genmat-ckpt",
        "--genim-ckpt",
        dest="genmat_ckpt",
        type=path_type,
        default=None,
        help="GenMat checkpoint (`--genim-ckpt` is the compatibility spelling).",
    )
    parser.add_argument("--matra-ckpt", type=path_type, default=None)
    parser.add_argument(
        "--genmat-sha256",
        "--genim-sha256",
        dest="genmat_sha256",
        default=None,
    )
    parser.add_argument("--matra-sha256", default=None)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Versioned GenMat catalog model ID or alias; repeat for an ensemble.",
    )
    parser.add_argument("--model-cache", type=path_type, default=None)
    parser.add_argument("--offline", action="store_true", default=None)
    parser.add_argument(
        "--accept-model-license",
        action="store_true",
        help="Acknowledge the separate upstream terms of named catalog models.",
    )
    parser.add_argument("--out-dir", required=True, type=path_type)
    parser.add_argument("--n-per-backend", type=int, default=32)
    parser.add_argument("--elements", nargs="+", default=None)
    parser.add_argument("--stoichiometry", nargs="+", type=float, default=None)
    parser.add_argument("--spacegroup", type=int, default=None)
    parser.add_argument("--matra-wyckoff", nargs="+", type=int, default=None)
    parser.add_argument("--stability", choices=["any", "stable", "unstable"], default="any")
    parser.add_argument("--target-e-hull", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sites", type=int, default=25)
    parser.add_argument("--min-sites", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--forward", type=int, default=150)
    parser.add_argument("--decode-jobs", type=int, default=0)
    parser.add_argument(
        "--mutation-fraction",
        type=float,
        default=0.0,
        help="Fraction of the final population produced as composition-preserving mutants.",
    )
    parser.add_argument("--mutation-strain", type=float, default=0.08)
    parser.add_argument("--mutation-displacement", type=float, default=0.12)
    parser.add_argument("--mutation-attempts", type=int, default=24)
    parser.add_argument(
        "--allow-symmetry-breaking-mutations",
        action="store_true",
        help="Allow coordinate/site mutations even when an exact space group was requested.",
    )
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--symprec", type=float, default=1e-2)
    parser.add_argument(
        "--allow-inconsistent-matra",
        action="store_true",
        help="Keep candidates that pass GenMat validation when Matra consistency flags fail.",
    )
    parser.add_argument("--overwrite", action="store_true")


def run_ensemble_command(args: argparse.Namespace) -> int:
    if args.genmat_ckpt is None and args.matra_ckpt is None and not args.model:
        raise ValueError(
            f"{args.cmd} requires --model, --genmat-ckpt, and/or --matra-ckpt"
        )
    condition = GenerationConstraints(
        elements=tuple(args.elements or ()),
        stoichiometry=tuple(args.stoichiometry or ()),
        spacegroup_number=args.spacegroup,
        matra_wyckoff_indices=tuple(args.matra_wyckoff or ()),
        stability=args.stability,
        target_e_hull=args.target_e_hull,
        exact_elements=True,
    )
    settings = GenerationSettings(
        n=int(args.n_per_backend),
        batch_size=int(args.batch_size),
        max_sites=int(args.max_sites),
        min_sites=int(args.min_sites),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        seed=int(args.seed) if args.seed is not None else None,
        forward=int(args.forward),
        decode_jobs=int(args.decode_jobs),
        require_backend_consistency=not bool(args.allow_inconsistent_matra),
        mutation_fraction=float(args.mutation_fraction),
        mutation_strain=float(args.mutation_strain),
        mutation_displacement=float(args.mutation_displacement),
        mutation_attempts=int(args.mutation_attempts),
        preserve_spacegroup=not bool(args.allow_symmetry_breaking_mutations),
        validation_options={"min_dist": float(args.min_dist), "symprec": float(args.symprec)},
    )
    backends = []
    if args.model:
        registry = ModelRegistry.default(cache_dir=args.model_cache)
        for model_ref in args.model:
            backend = registry.load_backend(
                model_ref,
                device=args.device,
                offline=args.offline,
                accept_license=bool(args.accept_model_license),
            )
            if not isinstance(backend, GenerationBackend):
                raise ValueError(
                    f"Catalog model {model_ref!r} is not a structure-generation backend"
                )
            backends.append(backend)
    if args.genmat_ckpt is not None:
        backends.append(
            GenMatBackend.from_checkpoint(
                args.genmat_ckpt,
                device=args.device,
                expected_sha256=args.genmat_sha256,
            )
        )
    if args.matra_ckpt is not None:
        backends.append(
            MatraBackend.from_checkpoint(
                args.matra_ckpt,
                device=args.device,
                expected_sha256=args.matra_sha256,
            )
        )
    run = EnsembleGenerator(backends).run(condition=condition, config=settings)
    artifacts = write_ensemble_run(run, args.out_dir, overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                **run.report(),
                "manifest": str(artifacts["manifest"]),
                "report": str(artifacts["report"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["add_ensemble_parser", "run_ensemble_command"]
