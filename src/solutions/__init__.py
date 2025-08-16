"""Video-level solution strategies package."""

from .base import BaseSolution
from .counting import CountingSolution
from .temporal import TemporalHysteresisSolution

__all__ = [
    "BaseSolution",
    "CountingSolution",
    "TemporalHysteresisSolution",
]
