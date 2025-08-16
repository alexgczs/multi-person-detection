from __future__ import annotations

from typing import Dict, List

from src.solutions.base import BaseSolution
from src.utils.config import Config


class TemporalHysteresisSolution(BaseSolution):
    """Temporal hysteresis over the binary multi-person signal per frame.

    Rules:
    - Build a boolean sequence b_t = has_multiple_people per frame
    - Activate multi-person if there are at least TEMPORAL_MIN_CONSECUTIVE True's
      within a sliding window of size TEMPORAL_WINDOW.
    - Sticky behavior: once active, remain active for the rest of the video.
    """

    def aggregate(self, frame_results: List[Dict], config: Config) -> Dict:
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

        # min_consec used for activation
        min_consec = max(1, int(getattr(config, "TEMPORAL_MIN_CONSECUTIVE", 3)))

        flags = [bool(fr.get("has_multiple_people", False)) for fr in frame_results]
        frames_with_multiple = sum(1 for f in flags if f)
        multiple_ratio = frames_with_multiple / total_frames

        active = False
        consecutive = 0
        for f in flags:
            consecutive = consecutive + 1 if f else 0
            if consecutive >= min_consec:
                active = True
                break

        avg_people = (
            sum(int(fr.get("num_people", 0)) for fr in frame_results) / total_frames
        )
        max_people = max(int(fr.get("num_people", 0)) for fr in frame_results)

        return {
            "has_multiple_people": bool(active),
            "num_people": int(round(avg_people)),
            "max_people": int(max_people),
            "multiple_people_ratio": float(multiple_ratio),
            "frames_with_multiple": int(frames_with_multiple),
            "total_frames": int(total_frames),
            "frame_results": frame_results,
        }
