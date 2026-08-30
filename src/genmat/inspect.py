"""Canonical import path for dataset and checkpoint inspection."""

from ._compat import reexport

reexport("genim.inspect", globals())
del reexport
