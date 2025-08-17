"""Video-level solution strategies package."""

from .base import BaseSolution
from .counting import CountingSolution
from .temporal import TemporalHysteresisSolution
from .temporal_cardaware import TemporalCardAwareSolution

__all__ = [
    "BaseSolution",
    "CountingSolution",
    "TemporalHysteresisSolution",
    "TemporalCardAwareSolution",
]
