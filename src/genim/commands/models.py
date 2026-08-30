from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..models import ModelRegistry


def _add_registry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the GenMat model cache directory.",
    )


def add_models_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "models",
        help="Discover, inspect, and obtain versioned GenMat model assets.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="models_cmd", required=True)

    list_parser = commands.add_parser(
        "list",
        help="List models known to the packaged GenMat catalog.",
        allow_abbrev=False,
    )
    _add_registry_options(list_parser)
    list_parser.add_argument("--provider", default=None)
    list_parser.add_argument("--task", default=None)
    list_parser.add_argument("--backend", default=None)
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    info_parser = commands.add_parser(
        "info",
        help="Show one model's immutable identity, provenance, and requirements.",
        allow_abbrev=False,
    )
    _add_registry_options(info_parser)
    info_parser.add_argument("model")

    pull_parser = commands.add_parser(
        "pull",
        help="Download and integrity-check one model asset into the GenMat cache.",
        allow_abbrev=False,
    )
    _add_registry_options(pull_parser)
    pull_parser.add_argument("model")
    pull_parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Acknowledge a model's separate license when the catalog requires it.",
    )
    pull_parser.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="Require an already cached, verified asset and never use the network.",
    )


def _registry(args: argparse.Namespace) -> ModelRegistry:
    cache_dir = None if args.cache_dir is None else args.cache_dir.expanduser().resolve()
    return ModelRegistry.default(cache_dir=cache_dir)


def _print_model_table(records: list[dict[str, object]]) -> None:
    headers = ("MODEL", "BACKEND", "TASK", "ACCESS")
    rows = [
        (
            str(record["id"]),
            str(record["backend"]),
            str(record["task"]),
            (
                "license acknowledgement required"
                if bool(record["requires_license_acceptance"])
                else "ready"
            ),
        )
        for record in records
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ] if rows else [len(value) for value in headers]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def run_models_command(args: argparse.Namespace) -> int:
    registry = _registry(args)
    if args.models_cmd == "list":
        records = [
            model.to_record()
            for model in registry.list(
                provider=args.provider,
                task=args.task,
                backend=args.backend,
            )
        ]
        if args.as_json:
            print(json.dumps(records, indent=2, sort_keys=True))
        else:
            _print_model_table(records)
        return 0

    if args.models_cmd == "info":
        print(json.dumps(registry.info(args.model).to_record(), indent=2, sort_keys=True))
        return 0

    if args.models_cmd == "pull":
        path = registry.pull(
            args.model,
            offline=args.offline,
            accept_license=bool(args.accept_license),
        )
        print(path)
        return 0

    raise RuntimeError(f"Unknown models command: {args.models_cmd}")


__all__ = ["add_models_parser", "run_models_command"]
