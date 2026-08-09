from __future__ import annotations

import argparse


def add_serve_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Run the GenIM HTTP generation service with safe local defaults.",
        allow_abbrev=False,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", choices=["critical", "error", "warning", "info", "debug"], default="info")


def run_serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The serve command requires the 'api' extra: pip install 'genim[api]'"
        ) from exc
    from ..server import create_app

    uvicorn.run(create_app(), host=str(args.host), port=int(args.port), log_level=str(args.log_level))
    return 0


__all__ = ["add_serve_parser", "run_serve_command"]
