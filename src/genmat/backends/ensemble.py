"""Canonical import path for multi-backend ensemble generation."""

from .._compat import reexport

reexport("genim.backends.ensemble", globals())
del reexport
