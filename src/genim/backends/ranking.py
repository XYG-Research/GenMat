"""Canonical import path for scientific triage ranking."""

from .._compat import reexport

reexport("genim.backends.ranking", globals())
del reexport
