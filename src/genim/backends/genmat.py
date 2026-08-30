"""Backward-compatible alias for :mod:`genmat.backends.genim`."""

from .._compat import reexport

reexport("genim.backends.genim", globals())
del reexport
