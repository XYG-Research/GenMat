"""Canonical import path for Materials Project data acquisition."""

from ._compat import reexport

reexport("genim.mp_download", globals())
del reexport
