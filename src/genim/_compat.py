"""Internal helpers for the legacy :mod:`genim` compatibility surface.

The implementation lives in the canonical ``genmat`` package. Public shim
modules call :func:`reexport` only when that
specific shim is imported, preserving object identity without eagerly loading
optional scientific dependencies from unrelated modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, MutableMapping


def reexport(canonical_module: str, namespace: MutableMapping[str, Any]) -> None:
    """Populate a GenIM shim with the public surface of *canonical_module*.

    An explicit ``__all__`` is respected when the implementation provides one;
    otherwise this mirrors Python's normal ``from module import *`` behaviour.
    Objects are not wrapped or copied, so classes, functions, and sentinels keep
    the same identity across the ``genmat`` and compatibility ``genim`` paths.
    """

    if canonical_module == "genim" or canonical_module.startswith("genim."):
        canonical_module = "genmat" + canonical_module[len("genim") :]
    target = import_module(canonical_module)
    public_names = tuple(
        getattr(
            target,
            "__all__",
            tuple(name for name in vars(target) if not name.startswith("_")),
        )
    )
    names = tuple(name for name in vars(target) if not name.startswith("__"))
    namespace.update({name: getattr(target, name) for name in names})
    namespace["__all__"] = list(public_names)


__all__ = ["reexport"]
