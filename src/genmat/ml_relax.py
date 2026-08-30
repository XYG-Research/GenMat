"""Canonical import path for atomistic relaxation utilities."""

from ._compat import reexport

reexport("genim.ml_relax", globals())
del reexport
