"""Canonical import path for model training."""

from ._compat import reexport

reexport("genim.train", globals())
del reexport
