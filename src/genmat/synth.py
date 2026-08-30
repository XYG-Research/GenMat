"""Canonical import path for the legacy synthesis workflow."""

from ._compat import reexport

reexport("genim.synth", globals())
del reexport
