"""Canonical import path for the local Matra backend."""

from .._compat import reexport

reexport("genim.backends.matra", globals())
del reexport
