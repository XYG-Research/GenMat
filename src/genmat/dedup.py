"""Canonical import path for structure deduplication."""

from ._compat import reexport

reexport("genim.dedup", globals())
del reexport
