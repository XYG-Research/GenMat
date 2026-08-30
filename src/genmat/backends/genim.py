"""Canonical import path for the GenMat checkpoint backend."""

from .._compat import reexport

reexport("genim.backends.genim", globals())
del reexport
