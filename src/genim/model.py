"""Canonical import path for GenMat neural model definitions."""

from ._compat import reexport

reexport("genim.model", globals())
del reexport
