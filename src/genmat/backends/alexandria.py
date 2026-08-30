"""Canonical import path for the Alexandria/Matra backend."""

from .._compat import reexport

reexport("genim.backends.alexandria", globals())
del reexport
