"""Canonical GenMat generation backend namespace."""

from genim.backends import *  # noqa: F401,F403
from genim.backends import __all__ as _BACKEND_ALL

__all__ = list(_BACKEND_ALL)

del _BACKEND_ALL
