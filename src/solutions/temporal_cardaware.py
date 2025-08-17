from __future__ import annotations

from typing import Dict, List, Sequence, Optional

from src.solutions.base import BaseSolution
from src.utils.config import Config
from loguru import logger


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _aspect_ratio(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return w / h


class TemporalCardAwareSolution(BaseSolution):
    """Temporal solution with ID-card photo suppression.

    Enhanced logic:
    - Uses the LARGEST detected face from the first frame as the reference
    - Can suppress multiple ID-card-like detections per frame
    - Applies temporal hysteresis with "sticky" behavior
    """

    def aggregate(self, frame_results: List[Dict], config: Config) -> Dict:
        """Aggregate frame-level detections into video-level decision."""
        if not frame_results:
            logger.debug("card-aware: No frames to process")
            return {"has_multiple_people": False, "ratio": 0.0}

        # Get configuration parameters
        min_consecutive = config.TEMPORAL_MIN_CONSECUTIVE
        area_threshold = config.CARD_MIN_AREA_RATIO_TO_LARGEST
        square_tolerance = config.CARD_SQUARE_TOLERANCE

        logger.debug(
            f"card-aware: Config min_consec={min_consecutive}, "
            f"area_thresh={area_threshold}, square_tol={square_tolerance}"
        )

        # Initialize reference face area
        # will be set from first frame with largest detection
        first_face_area: Optional[float] = None

        # Process each frame
        processed_frames = []

        for frame_idx, frame_result in enumerate(frame_results):
            detections = frame_result.get("detections", [])

            if not detections:
                processed_frames.append(
                    {"has_multiple_people": False, "adjusted_count": 0}
                )
                logger.debug(f"card-aware: frame={frame_idx} - No detections")
                continue

            # Set reference face area from largest detection in first frame if not set
            if first_face_area is None and detections:
                # Find the largest detection in this frame
                largest_area = 0.0
                for detection in detections:
                    area = _bbox_area(detection["bbox"])
                    if area > largest_area:
                        largest_area = area

                first_face_area = largest_area
                logger.debug(
                    "card-aware: Set largest face "
                    f"reference area to {first_face_area:.1f}"
                )

            # Filter out ID-card-like detections
            valid_detections = []
            suppressed_count = 0

            for detection in detections:
                bbox = detection["bbox"]
                area = _bbox_area(bbox)
                aspect_ratio = _aspect_ratio(bbox)

                # Calculate ratio relative to largest face
                ratio = area / first_face_area if first_face_area else 1.0

                # Check if this looks like an ID card
                is_small = ratio < area_threshold
                is_near_square = abs(aspect_ratio - 1.0) <= square_tolerance

                if is_small and is_near_square:
                    suppressed_count += 1
                else:
                    valid_detections.append(detection)

            # Count remaining detections after suppression
            adjusted_count = len(valid_detections)
            has_multiple = adjusted_count > 1

            processed_frames.append({
                "has_multiple_people": has_multiple,
                "adjusted_count": adjusted_count
            })

        # Apply temporal hysteresis with sticky behavior
        active = False
        consecutive_count = 0
        max_consecutive = 0

        logger.debug("card-aware: Starting temporal aggregation...")

        for frame_idx, frame_data in enumerate(processed_frames):
            if frame_data["has_multiple_people"]:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)

                if not active and consecutive_count >= min_consecutive:
                    active = True
            else:
                consecutive_count = 0

        # Calculate final ratio for debugging
        total_frames = len(processed_frames)
        frames_with_multiple = sum(
            1 for f in processed_frames if f["has_multiple_people"]
        )
        final_ratio = frames_with_multiple / total_frames if total_frames > 0 else 0.0

        # Calculate average people count for compatibility
        total_people = sum(f["adjusted_count"] for f in processed_frames)
        avg_people = total_people / total_frames if total_frames > 0 else 0.0
        max_people = (
            max(f["adjusted_count"] for f in processed_frames)
            if processed_frames else 0
        )

        return {
            "has_multiple_people": active,
            "ratio": final_ratio,
            "max_consecutive": max_consecutive,
            "total_frames": total_frames,
            "frames_with_multiple": frames_with_multiple,
            "num_people": int(round(avg_people)),
            "max_people": max_people,
            "multiple_people_ratio": final_ratio,
            "frame_results": frame_results,
        }
