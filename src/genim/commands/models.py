"""Canonical import path for model-registry CLI registration."""

from .._compat import reexport

reexport("genim.commands.models", globals())
del reexport
