"""Canonical module name for the GenMat checkpoint backend.

The implementation still lives in ``genim.backends.genim`` during the
compatibility window; this module provides the brand-correct extension path.
"""

from .._compat import reexport

reexport("genim.backends.genim", globals())
del reexport
