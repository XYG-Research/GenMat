"""Canonical import path for optional interatomic-potential adapters."""

from ._compat import reexport

reexport("genim.mlip", globals())
del reexport
