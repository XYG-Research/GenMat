"""Canonical import path for ensemble CLI registration."""

from .._compat import reexport

reexport("genim.commands.ensemble", globals())
del reexport
