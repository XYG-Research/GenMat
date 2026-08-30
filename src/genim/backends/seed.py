"""Canonical import path for algorithmic seed construction."""

from .._compat import reexport

reexport("genim.backends.seed", globals())
del reexport
