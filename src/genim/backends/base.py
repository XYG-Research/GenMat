"""Canonical import path for backend extension contracts."""

from .._compat import reexport

reexport("genim.backends.base", globals())
del reexport
