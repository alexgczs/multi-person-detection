from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Tuple

from src.solutions.base import BaseSolution
from src.utils.config import Config


class TemporalTextAwareSolution(BaseSolution):
    """Temporal hysteresis with text detection for ID card filtering.

    This solution combines temporal analysis with text detection to better identify
    ID cards and documents. It filters out faces that are near detected text,
    assuming they are likely to be ID card photos rather than real people.

    Rules:
    - Build a boolean sequence b_t = has_multiple_people per frame
    - Filter out faces that are near detected text (likely ID cards)
    - Activate multi-person if there are at least TEMPORAL_MIN_CONSECUTIVE
      consecutive frames with multi-person (after filtering)
    - Sticky behavior: once active, remain active for the rest of the video.
    """

    def __init__(self):
        super().__init__()
        self._text_detector = None
        self._text_detection_cache = {}

    def _get_text_detector(self):
        """Lazy initialization of text detector."""
        if self._text_detector is None:
            try:
                import easyocr
                # Try to detect if GPU is available, fallback to CPU if not
                use_gpu = False
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                except (ImportError, AttributeError):
                    # If torch is not available or doesn't have CUDA support
                    use_gpu = False

                self._text_detector = easyocr.Reader(['en'], gpu=use_gpu)
            except ImportError:
                raise ImportError(
                    "easyocr is required for text-aware solution. "
                    "Install with: pip install easyocr"
                )
        return self._text_detector

    def _detect_text_regions(
        self, frame: np.ndarray, config: Config
    ) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in the frame.

        Returns:
            List of (x, y, w, h) bounding boxes for text regions
        """
        try:
            reader = self._get_text_detector()
            results = reader.readtext(frame)

            text_regions = []
            confidence_threshold = getattr(config, 'TEXT_CONFIDENCE_THRESHOLD', 0.5)
            for (bbox, text, confidence) in results:
                if confidence > confidence_threshold:  # Filter low confidence
                    # Convert bbox to (x, y, w, h) format
                    points = np.array(bbox, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(points)
                    text_regions.append((x, y, w, h))

            return text_regions
        except Exception:
            # Fallback to empty list if text detection fails
            return []

    def _is_face_near_text(
        self,
        face_bbox: Tuple[int, int, int, int],
        text_regions: List[Tuple[int, int, int, int]],
        config: Config,
    ) -> bool:
        """Check if a face bounding box is near any text region.

        Args:
            face_bbox: (x, y, w, h) of face detection
            text_regions: List of (x, y, w, h) text regions
            config: Configuration object

        Returns:
            True if face is near text (likely an ID card)
        """
        fx, fy, fw, fh = face_bbox
        face_center = (fx + fw // 2, fy + fh // 2)

        for tx, ty, tw, th in text_regions:
            text_center = (tx + tw // 2, ty + th // 2)

            # Calculate distance between centers
            distance = np.sqrt(
                (face_center[0] - text_center[0])**2 +
                (face_center[1] - text_center[1])**2
            )

            # Check if face is within proximity threshold
            proximity_threshold = getattr(config, 'TEXT_PROXIMITY_THRESHOLD', 100)
            if distance < proximity_threshold:
                return True

            # Check if face overlaps with text region
            if (fx < tx + tw and fx + fw > tx and
                    fy < ty + th and fy + fh > ty):
                return True

        return False

    def _filter_id_card_faces(
        self,
        frame_results: List[Dict],
        frame: np.ndarray,
        config: Config,
    ) -> List[Dict]:
        """Filter out faces that are likely ID card photos based on text proximity.

        Args:
            frame_results: List of frame detection results
            frame: Original frame image
            config: Configuration object

        Returns:
            Filtered frame results with ID card faces removed
        """
        if not frame_results:
            return frame_results

        # Detect text regions in the frame
        text_regions = self._detect_text_regions(frame, config)

        # Filter each detection
        filtered_results = []
        for result in frame_results:
            if 'detections' not in result:
                filtered_results.append(result)
                continue

            filtered_detections = []
            for detection in result['detections']:
                # Only process person detections (class 0 in YOLO, 1 in torchvision)
                if detection.get('class') in [0, 1]:
                    # Check if this person detection is near text
                    bbox = detection.get('bbox', [])
                    if len(bbox) == 4:
                        face_bbox = (
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2] - bbox[0]),
                            int(bbox[3] - bbox[1]),
                        )

                        if not self._is_face_near_text(face_bbox, text_regions, config):
                            filtered_detections.append(detection)
                        # If face is near text, exclude it (likely ID card)
                    else:
                        # Invalid bbox, keep the detection
                        filtered_detections.append(detection)
                else:
                    # Keep non-person detections as is
                    filtered_detections.append(detection)

            # Update result with filtered detections
            filtered_result = result.copy()
            filtered_result['detections'] = filtered_detections
            filtered_result['num_people'] = len([
                d for d in filtered_detections if d.get('class') in [0, 1]
            ])
            filtered_result['has_multiple_people'] = filtered_result['num_people'] > 1

            filtered_results.append(filtered_result)

        return filtered_results

    def aggregate(self, frame_results: List[Dict], config: Config) -> Dict:
        """Aggregate frame results using temporal hysteresis.

        Text filtering is already applied at frame level.
        """
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

        # Text filtering has already been applied at frame level by PersonDetector
        # Now we just apply temporal hysteresis on the filtered results
        min_consec = max(1, config.TEMPORAL_MIN_CONSECUTIVE)

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
            "solution_type": "temporal_textaware",
        }
