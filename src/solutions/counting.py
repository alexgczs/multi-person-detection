from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.solutions.base import BaseSolution
from src.utils.config import Config


class CountingSolution(BaseSolution):
    """Baseline solution: ratio of frames with >1 person over total frames."""

    def aggregate(self, frame_results: List[Dict], config: Config) -> Dict:
        # Edge case: no frames
        total_frames = len(frame_results)
        if total_frames == 0:
            return {
                "has_multiple_people": False,
                "num_people": 0,
                "max_people": 0,
                "multiple_people_ratio": 0.0,
                "frames_with_multiple": 0,
                "total_frames": 0,
                "frame_results": [],
            }

        frames_with_multiple = sum(
            1 for r in frame_results if r.get("has_multiple_people")
        )
        multiple_people_ratio = frames_with_multiple / total_frames

        has_multiple_people = (
            multiple_people_ratio > float(config.MULTIPLE_PEOPLE_THRESHOLD)
        )

        avg_people = float(np.mean([r.get("num_people", 0) for r in frame_results]))
        max_people = int(max([r.get("num_people", 0) for r in frame_results]))

        return {
            "has_multiple_people": bool(has_multiple_people),
            "num_people": int(round(avg_people)),
            "max_people": max_people,
            "multiple_people_ratio": float(multiple_people_ratio),
            "frames_with_multiple": int(frames_with_multiple),
            "total_frames": int(total_frames),
            "frame_results": frame_results,
        }
