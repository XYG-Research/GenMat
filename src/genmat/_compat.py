"""Internal helpers for the canonical :mod:`genmat` compatibility surface.

GenMat currently shares implementation modules with the historical
``genim`` package.  Public shim modules call :func:`reexport` only when that
specific shim is imported, preserving object identity without eagerly loading
optional scientific dependencies from unrelated modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, MutableMapping


def reexport(legacy_module: str, namespace: MutableMapping[str, Any]) -> None:
    """Populate a GenMat shim with the public surface of *legacy_module*.

    An explicit ``__all__`` is respected when the implementation provides one;
    otherwise this mirrors Python's normal ``from module import *`` behaviour.
    Objects are not wrapped or copied, so classes, functions, and sentinels keep
    the same identity across the ``genmat`` and compatibility ``genim`` paths.
    """

    target = import_module(legacy_module)
    names = tuple(
        getattr(
            target,
            "__all__",
            tuple(name for name in vars(target) if not name.startswith("_")),
        )
    )
    namespace.update({name: getattr(target, name) for name in names})
    namespace["__all__"] = list(names)


__all__ = ["reexport"]
