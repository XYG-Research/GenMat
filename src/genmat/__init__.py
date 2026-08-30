"""Canonical GenMat public namespace.

The implementation remains shared with :mod:`genim` during the compatibility
window so existing checkpoint workflows and imports continue to work.
"""

from genim import *  # noqa: F401,F403
from genim import __all__ as _COMPAT_ALL

__all__ = list(_COMPAT_ALL)

del _COMPAT_ALL
