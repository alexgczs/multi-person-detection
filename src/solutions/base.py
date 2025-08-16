from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from src.utils.config import Config


class BaseSolution(ABC):
    """Interface that aggregates frame-level detections into a video-level decision."""

    @abstractmethod
    def aggregate(self, frame_results: List[Dict], config: Config) -> Dict:
        """Aggregate per-frame dicts into a video-level result dict."""
        raise NotImplementedError
