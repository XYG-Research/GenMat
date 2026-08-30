"""Canonical import path for scientific scoring workflows."""

from ._compat import reexport

reexport("genim.score", globals())
del reexport
