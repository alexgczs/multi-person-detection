"""
Person detection model using YOLOv8.

Detects multiple people in video frames.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from src.utils.config import Config
from src.utils.video_processor import VideoProcessor


class PersonDetector:
    """
    A class for detecting multiple people in videos.

    This detector uses YOLOv8 for person detection and implements
    logic to determine if multiple people are present in a video.
    """

    def __init__(self, model_size: str = "n", device: Optional[str] = None):
        """
        Initialize the person detector.

        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and setup
        self.model = self._load_model()
        self.video_processor = VideoProcessor()
        self.config = Config()

        logger.info(f"PersonDetector initialized with model size: {model_size}")
        logger.info(f"Using device: {self.device}")

    def _load_model(self) -> YOLO:
        """Load YOLO model."""
        try:
            model_name = f"yolov8{self.model_size}.pt"
            model = YOLO(model_name)
            logger.info(f"Loaded YOLO model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def predict(self, video_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Predict whether a video contains multiple people.

        Args:
            video_path: Path to the input video file
            confidence_threshold: Minimum confidence for person detection

        Returns:
            Dictionary containing prediction results
        """
        try:
            logger.info(f"Processing video: {video_path}")

            # Get frames and process them
            frames = self.video_processor.extract_frames(video_path)
            logger.info(f"Extracted {len(frames)} frames from video")

            frame_results = []
            for i, frame in enumerate(frames):
                result = self._process_frame(frame, confidence_threshold)
                result["frame_index"] = i
                frame_results.append(result)

            final_result = self._aggregate_results(frame_results)

            logger.info(
                f"Detection complete: {final_result['num_people']} people detected"
            )
            logger.info(f"Multiple people: {final_result['has_multiple_people']}")

            return final_result

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def _process_frame(self, frame: np.ndarray, confidence_threshold: float) -> Dict:
        """
        Process a single frame to detect people.

        Args:
            frame: Input frame as numpy array
            confidence_threshold: Minimum confidence for detection

        Returns:
            Dictionary containing detection results for the frame
        """
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)

            # Filter for person detections (class 0 in COCO dataset)
            person_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0)
                        if (
                            int(box.cls) == 0
                            and float(box.conf) >= confidence_threshold
                        ):
                            person_detections.append(
                                {
                                    "bbox": box.xyxy[0].cpu().numpy(),
                                    "confidence": float(box.conf),
                                    "class": int(box.cls),
                                }
                            )

            return {
                "num_people": len(person_detections),
                "detections": person_detections,
                "has_multiple_people": len(person_detections) > 1,
            }

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # To avoid error in complete process, we return 0 people detected
            return {"num_people": 0, "detections": [], "has_multiple_people": False}

    def _aggregate_results(self, frame_results: List[Dict]) -> Dict:
        """
        Aggregate results across multiple frames.

        Args:
            frame_results: List of frame-level detection results

        Returns:
            Aggregated result for the entire video
        """
        # Count frames with multiple people
        frames_with_multiple = sum(1 for r in frame_results if r["has_multiple_people"])
        total_frames = len(frame_results)

        # Calculate percentage of frames with multiple people
        multiple_people_ratio = (
            frames_with_multiple / total_frames if total_frames > 0 else 0
        )

        # Determine if video has multiple people based on threshold
        has_multiple_people = (
            multiple_people_ratio > self.config.MULTIPLE_PEOPLE_THRESHOLD
        )

        # Calculate average number of people across frames
        avg_people = np.mean([r["num_people"] for r in frame_results])
        max_people = max([r["num_people"] for r in frame_results])

        return {
            "has_multiple_people": has_multiple_people,
            "num_people": int(round(avg_people)),
            "max_people": max_people,
            "multiple_people_ratio": multiple_people_ratio,
            "frames_with_multiple": frames_with_multiple,
            "total_frames": total_frames,
            "frame_results": frame_results,
        }
