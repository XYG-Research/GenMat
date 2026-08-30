"""Canonical re-export of the local and hosted generation service."""

from genmat.service import *  # noqa: F401,F403
from genmat.service import __all__ as _SERVICE_ALL

__all__ = list(_SERVICE_ALL)

del _SERVICE_ALL
