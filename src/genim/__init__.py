"""Backward-compatible namespace for the former :mod:`genim` package."""

from genmat import *  # noqa: F401,F403
from genmat import __all__ as _COMPAT_ALL

__all__ = list(_COMPAT_ALL)

del _COMPAT_ALL
