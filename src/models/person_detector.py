"""
Person detector that aggregates frame-level detections from pluggable backends.

This class orchestrates frame extraction and delegates per-frame person
detections to a selected backend. The aggregation logic computes simple
statistics (counts and ratios) to decide whether a video likely contains
multiple persons.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from src.utils.config import Config
from src.utils.video_processor import VideoProcessor
from src.solutions.base import BaseSolution
from src.solutions.counting import CountingSolution
from src.solutions.temporal import TemporalHysteresisSolution
from src.solutions.temporal_cardaware import TemporalCardAwareSolution

from src.models.backends import (
    OpenCVHOGBackend,
    PersonDetectionBackend,
    TorchvisionFRCNNBackend,
    TorchvisionRetinaNetBackend,
    TorchvisionSSDBackend,
    YoloV8Backend,
)


class PersonDetector:
    """
    A class for detecting multiple people in videos.

    This detector uses YOLOv8 for person detection and implements
    logic to determine if multiple people are present in a video.
    """

    def __init__(
        self,
        model_size: str = "n",
        device: Optional[str] = None,
        backend: str = "yolov8",
        solution: str = "counting",
    ):
        """
        Initialize the person detector.

        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend_name = backend
        self.solution_name = solution

        # Initialize backend and utilities
        self.backend: PersonDetectionBackend = self._init_backend()
        self.video_processor = VideoProcessor()
        self.config = Config()
        self.solution: BaseSolution = self._init_solution()

        logger.info(
            f"PersonDetector initialized with backend={self.backend_name}, "
            f"model size={model_size}, device={self.device}"
        )

    def _init_backend(self) -> PersonDetectionBackend:
        if self.backend_name == "yolov8":
            return YoloV8Backend(self.model_size, self.device)
        if self.backend_name == "torchvision_frcnn":
            return TorchvisionFRCNNBackend(self.device)
        if self.backend_name == "torchvision_ssd":
            return TorchvisionSSDBackend(self.device)
        if self.backend_name == "torchvision_retinanet":
            return TorchvisionRetinaNetBackend(self.device)
        if self.backend_name == "opencv_hog":
            return OpenCVHOGBackend(self.device)
        raise ValueError(
            "Unsupported backend '" + self.backend_name + "'. Supported: "
            "['yolov8', 'torchvision_frcnn', 'torchvision_ssd', "
            "'torchvision_retinanet', 'opencv_hog']",
        )

    def _init_solution(self) -> BaseSolution:
        if self.solution_name == "counting":
            return CountingSolution()
        if self.solution_name == "temporal":
            return TemporalHysteresisSolution()
        if self.solution_name == "temporal_cardaware":
            return TemporalCardAwareSolution()
        raise ValueError(
            "Unsupported solution '" + self.solution_name + "'. Supported: "
            "['counting', 'temporal', 'temporal_cardaware']",
        )

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

            final_result = self.solution.aggregate(frame_results, self.config)

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
            # Run backend detection for person class
            person_detections = self.backend.predict_persons(
                frame, confidence_threshold
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
        return CountingSolution().aggregate(frame_results, self.config)
