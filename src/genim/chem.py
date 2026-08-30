"""Canonical import path for chemistry-domain policy."""

from ._compat import reexport

reexport("genim.chem", globals())
del reexport
