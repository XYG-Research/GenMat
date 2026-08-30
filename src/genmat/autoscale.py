"""Canonical import path for cell autoscaling utilities."""

from ._compat import reexport

reexport("genim.autoscale", globals())
del reexport
