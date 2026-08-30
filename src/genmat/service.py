"""Canonical re-export of the local and hosted generation service."""

from genim.service import *  # noqa: F401,F403
from genim.service import __all__ as _SERVICE_ALL

__all__ = list(_SERVICE_ALL)

del _SERVICE_ALL
