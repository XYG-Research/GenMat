"""Brand-correct alias for the checkpoint-backed GenMat backend."""

from .genim import GenIMBackend, GenMatBackend

__all__ = ["GenMatBackend", "GenIMBackend"]
