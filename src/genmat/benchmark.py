"""Canonical import path for deterministic structure benchmarks."""

from ._compat import reexport

reexport("genim.benchmark", globals())
del reexport
