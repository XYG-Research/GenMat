from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from ..backends import (
    EnsembleGenerator,
    GenerationConstraints,
    GenerationSettings,
    GenIMBackend,
    MatraBackend,
    write_ensemble_run,
)


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
    parser.add_argument("--genim-ckpt", type=path_type, default=None)
    parser.add_argument("--matra-ckpt", type=path_type, default=None)
    parser.add_argument("--genim-sha256", default=None)
    parser.add_argument("--matra-sha256", default=None)
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
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--symprec", type=float, default=1e-2)
    parser.add_argument(
        "--allow-inconsistent-matra",
        action="store_true",
        help="Keep candidates that pass GenIM validation when Matra consistency flags fail.",
    )
    parser.add_argument("--overwrite", action="store_true")


def run_ensemble_command(args: argparse.Namespace) -> int:
    if args.genim_ckpt is None and args.matra_ckpt is None:
        raise ValueError(f"{args.cmd} requires --genim-ckpt and/or --matra-ckpt")
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
        validation_options={"min_dist": float(args.min_dist), "symprec": float(args.symprec)},
    )
    backends = []
    if args.genim_ckpt is not None:
        backends.append(
            GenIMBackend.from_checkpoint(
                args.genim_ckpt,
                device=args.device,
                expected_sha256=args.genim_sha256,
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

