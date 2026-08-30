"""Canonical import path for population mutation."""

from .._compat import reexport

reexport("genim.backends.mutation", globals())
del reexport
