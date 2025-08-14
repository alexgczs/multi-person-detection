"""
Unit tests for the PersonDetector class.

This module contains the tests for all the features of the PersonDetector class.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.models.person_detector import PersonDetector
from src.utils.config import Config


class TestPersonDetector(unittest.TestCase):
    """Test cases for the PersonDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.MULTIPLE_PEOPLE_THRESHOLD = 0.3
        self.config.FRAME_WIDTH = 640
        self.config.FRAME_HEIGHT = 480

        # Create a mock YOLO model
        self.mock_model = Mock()
        self.mock_result = Mock()
        self.mock_boxes = Mock()
        self.mock_box = Mock()

        # Configure mock box
        self.mock_box.cls = torch.tensor([0])  # Person class
        self.mock_box.conf = torch.tensor([0.8])  # confidence
        self.mock_box.xyxy = torch.tensor([[100, 100, 200, 300]])

        # Configure mock boxes
        self.mock_boxes.__iter__ = lambda x: iter([self.mock_box])

        # Configure mock result
        self.mock_result.boxes = self.mock_boxes

        # Configure mock model
        self.mock_model.return_value = [self.mock_result]

    @patch("src.models.person_detector.YOLO")
    def test_initialization(self, mock_yolo):
        """Test PersonDetector initialization."""
        mock_yolo.return_value = self.mock_model

        detector = PersonDetector(model_size="n")

        self.assertEqual(detector.model_size, "n")
        self.assertIsNotNone(detector.model)
        mock_yolo.assert_called_once_with("yolov8n.pt")

    @patch("src.models.person_detector.YOLO")
    def test_model_loading_error(self, mock_yolo):
        """Test handling of model loading errors."""
        mock_yolo.side_effect = Exception("Model loading failed")

        with self.assertRaises(Exception):
            PersonDetector(model_size="n")

    @patch("src.models.person_detector.YOLO")
    def test_process_frame_single_person(self, mock_yolo):
        """Test processing a frame with a single person."""
        mock_yolo.return_value = self.mock_model

        detector = PersonDetector(model_size="n")

        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector._process_frame(frame, confidence_threshold=0.5)

        self.assertEqual(result["num_people"], 1)
        self.assertFalse(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 1)

    @patch("src.models.person_detector.YOLO")
    def test_process_frame_multiple_people(self, mock_yolo):
        """Test processing a frame with multiple people."""
        mock_yolo.return_value = self.mock_model

        # Create a second person detection
        mock_box2 = Mock()
        mock_box2.cls = torch.tensor([0])
        mock_box2.conf = torch.tensor([0.7])
        mock_box2.xyxy = torch.tensor([[300, 100, 400, 300]])

        # Update mock boxes to include two people
        self.mock_boxes.__iter__ = lambda x: iter([self.mock_box, mock_box2])

        detector = PersonDetector(model_size="n")

        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector._process_frame(frame, confidence_threshold=0.5)

        self.assertEqual(result["num_people"], 2)
        self.assertTrue(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 2)

    @patch("src.models.person_detector.YOLO")
    def test_process_frame_no_people(self, mock_yolo):
        """Test processing a frame with no people."""
        mock_yolo.return_value = self.mock_model

        # Configure mock to return no detections
        self.mock_boxes.__iter__ = lambda x: iter([])

        detector = PersonDetector(model_size="n")

        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector._process_frame(frame, confidence_threshold=0.5)

        self.assertEqual(result["num_people"], 0)
        self.assertFalse(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 0)

    @patch("src.models.person_detector.YOLO")
    def test_process_frame_low_confidence(self, mock_yolo):
        """Test processing a frame with low confidence detections."""
        mock_yolo.return_value = self.mock_model

        # Set low confidence
        self.mock_box.conf = torch.tensor([0.3])

        detector = PersonDetector(model_size="n")

        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector._process_frame(frame, confidence_threshold=0.5)

        self.assertEqual(result["num_people"], 0)
        self.assertFalse(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 0)

    @patch("src.models.person_detector.YOLO")
    def test_aggregate_results_single_person(self, mock_yolo):
        """Test aggregating results for single person video."""
        mock_yolo.return_value = self.mock_model

        detector = PersonDetector(model_size="n")

        # Create frame results for single person
        frame_results = [
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 0, "has_multiple_people": False},
            {"num_people": 1, "has_multiple_people": False},
        ]

        result = detector._aggregate_results(frame_results)

        self.assertFalse(result["has_multiple_people"])
        self.assertEqual(result["num_people"], 1)
        self.assertEqual(result["multiple_people_ratio"], 0.0)
        self.assertEqual(result["frames_with_multiple"], 0)
        self.assertEqual(result["total_frames"], 5)

    @patch("src.models.person_detector.YOLO")
    def test_aggregate_results_multiple_people(self, mock_yolo):
        """Test aggregating results for multiple people video."""
        mock_yolo.return_value = self.mock_model

        detector = PersonDetector(model_size="n")

        # Create frame results for multiple people
        frame_results = [
            {"num_people": 2, "has_multiple_people": True},
            {"num_people": 2, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 2, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
        ]

        result = detector._aggregate_results(frame_results)

        self.assertTrue(result["has_multiple_people"])
        self.assertEqual(result["num_people"], 2)
        self.assertEqual(result["multiple_people_ratio"], 0.6)
        self.assertEqual(result["frames_with_multiple"], 3)
        self.assertEqual(result["total_frames"], 5)

    @patch("src.models.person_detector.YOLO")
    @patch("src.models.person_detector.VideoProcessor")
    def test_predict_integration(self, mock_video_processor, mock_yolo):
        """Test the complete predict method integration."""
        mock_yolo.return_value = self.mock_model

        # Mock video processor
        mock_processor = Mock()
        mock_processor.extract_frames.return_value = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        ]
        mock_video_processor.return_value = mock_processor

        detector = PersonDetector(model_size="n")

        # Create a temporary video file path
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            result = detector.predict(video_path, confidence_threshold=0.5)

            self.assertIn("has_multiple_people", result)
            self.assertIn("num_people", result)
            self.assertIn("multiple_people_ratio", result)

        finally:
            # Clean up
            if os.path.exists(video_path):
                os.unlink(video_path)

    @patch("src.models.person_detector.YOLO")
    @patch("src.models.person_detector.VideoProcessor")
    def test_predict_raises_on_extract_error(self, mock_video_processor, mock_yolo):
        """Predict should propagate exceptions from frame extraction."""
        mock_yolo.return_value = self.mock_model

        mock_processor = Mock()
        mock_processor.extract_frames.side_effect = ValueError("boom")
        mock_video_processor.return_value = mock_processor

        detector = PersonDetector(model_size="n")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
        try:
            with self.assertRaises(ValueError):
                detector.predict(video_path, confidence_threshold=0.5)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    @patch("src.models.person_detector.YOLO")
    def test_process_frame_exception_path(self, mock_yolo):
        """_process_frame returns default result if model call fails."""
        # Configure model to raise when called
        failing_model = Mock()
        failing_model.side_effect = Exception("inference failed")
        mock_yolo.return_value = failing_model

        detector = PersonDetector(model_size="n")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector._process_frame(frame, confidence_threshold=0.5)
        self.assertEqual(result["num_people"], 0)
        self.assertFalse(result["has_multiple_people"])


if __name__ == "__main__":
    unittest.main()
