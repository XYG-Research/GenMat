"""Canonical import path for API-server CLI registration."""

from .._compat import reexport

reexport("genim.commands.serve", globals())
del reexport
