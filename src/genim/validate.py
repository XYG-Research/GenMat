"""Canonical import path for structure validation."""

from ._compat import reexport

reexport("genim.validate", globals())
del reexport
