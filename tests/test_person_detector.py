"""
Unit tests for the PersonDetector class.

These tests focus on orchestration (frame processing and aggregation). Model-specific
logic is covered in backend tests.
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

    @patch("src.models.person_detector.YoloV8Backend")
    def test_initialization(self, mock_backend):
        """PersonDetector initializes the chosen backend via interface."""
        fake_backend = Mock()
        mock_backend.return_value = fake_backend

        detector = PersonDetector(model_size="n", backend="yolov8")

        self.assertEqual(detector.model_size, "n")
        # Only backend interface is guaranteed
        self.assertIs(detector.backend, fake_backend)
        mock_backend.assert_called_once()

    @patch("src.models.person_detector.YoloV8Backend")
    def test_process_frame_delegates_to_backend(self, mock_backend):
        """_process_frame delegates to backend.predict_persons and maps outputs."""
        fake_backend = Mock()
        fake_backend.predict_persons.return_value = [
            {"bbox": [0, 0, 1, 1], "confidence": 0.9, "class": 0}
        ]
        mock_backend.return_value = fake_backend

        detector = PersonDetector(backend="yolov8")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector._process_frame(frame, confidence_threshold=0.5)
        self.assertEqual(result["num_people"], 1)
        self.assertFalse(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 1)

    @patch("src.models.person_detector.YoloV8Backend")
    def test_process_frame_multiple_people_via_backend(self, mock_backend):
        """Multiple detections from backend should yield has_multiple_people=True."""
        fake_backend = Mock()
        fake_backend.predict_persons.return_value = [
            {"bbox": [0, 0, 1, 1], "confidence": 0.9, "class": 0},
            {"bbox": [2, 2, 3, 3], "confidence": 0.8, "class": 0},
        ]
        mock_backend.return_value = fake_backend

        detector = PersonDetector(backend="yolov8")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector._process_frame(frame, confidence_threshold=0.5)
        self.assertEqual(result["num_people"], 2)
        self.assertTrue(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 2)

    @patch("src.models.person_detector.YoloV8Backend")
    def test_backend_initialization_error(self, mock_backend):
        """Initialization should propagate backend construction errors."""
        mock_backend.side_effect = Exception("Backend init failed")
        with self.assertRaises(Exception):
            PersonDetector(model_size="n", backend="yolov8")

    @patch("src.models.person_detector.YoloV8Backend")
    def test_process_frame_no_people(self, mock_backend):
        """No detections from backend should map to zero people."""
        fake_backend = Mock()
        fake_backend.predict_persons.return_value = []
        mock_backend.return_value = fake_backend

        detector = PersonDetector(backend="yolov8")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector._process_frame(frame, confidence_threshold=0.5)
        self.assertEqual(result["num_people"], 0)
        self.assertFalse(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 0)

    # The remaining aggregation tests stay the same

    @patch("src.models.person_detector.YoloV8Backend")
    @patch("src.models.person_detector.VideoProcessor")
    def test_predict_integration(self, mock_video_processor, mock_backend):
        """Test the complete predict method integration."""
        fake_backend = Mock()
        fake_backend.predict_persons.return_value = []
        mock_backend.return_value = fake_backend

        # Mock video processor
        mock_processor = Mock()
        mock_processor.extract_frames.return_value = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        ]
        mock_video_processor.return_value = mock_processor

        detector = PersonDetector(model_size="n", backend="yolov8")

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

    @patch("src.models.person_detector.YoloV8Backend")
    @patch("src.models.person_detector.VideoProcessor")
    def test_predict_raises_on_extract_error(self, mock_video_processor, mock_backend):
        """Predict should propagate exceptions from frame extraction."""
        fake_backend = Mock()
        fake_backend.predict_persons.return_value = []
        mock_backend.return_value = fake_backend

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

    def test_aggregate_results_single_person(self):
        """Test aggregating results for single person video."""
        detector = PersonDetector(model_size="n", backend="yolov8")

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

    def test_aggregate_results_multiple_people(self):
        """Test aggregating results for multiple people video."""
        detector = PersonDetector(model_size="n", backend="yolov8")

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

    @patch("src.models.person_detector.YoloV8Backend")
    def test_apply_text_filtering_temporal_textaware(self, mock_backend):
        """Test that text filtering is applied only for temporal_textaware solution."""
        fake_backend = Mock()
        fake_backend.predict_persons.return_value = [
            {"bbox": [0, 0, 100, 100], "confidence": 0.9, "class": 0},
            {"bbox": [200, 200, 300, 300], "confidence": 0.8, "class": 0},
        ]
        mock_backend.return_value = fake_backend

        # Test with temporal_textaware solution
        detector = PersonDetector(backend="yolov8", solution="temporal_textaware")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_result = {
            "num_people": 2,
            "has_multiple_people": True,
            "detections": [
                {"bbox": [0, 0, 100, 100], "confidence": 0.9, "class": 0},
                {"bbox": [200, 200, 300, 300], "confidence": 0.8, "class": 0},
            ]
        }

        # Mock the text-aware solution's filtering method
        with patch.object(detector.solution, '_filter_id_card_faces') as mock_filter:
            mock_filter.return_value = [{
                "num_people": 1,
                "has_multiple_people": False,
                "detections": [
                    {"bbox": [200, 200, 300, 300], "confidence": 0.8, "class": 0}
                ]
            }]

            result = detector._apply_text_filtering(frame_result, frame)

            # Should have applied filtering
            self.assertEqual(result["num_people"], 1)
            self.assertFalse(result["has_multiple_people"])
            mock_filter.assert_called_once()

    @patch("src.models.person_detector.YoloV8Backend")
    def test_apply_text_filtering_other_solutions(self, mock_backend):
        """Test that text filtering is NOT applied for other solutions."""
        fake_backend = Mock()
        mock_backend.return_value = fake_backend

        # Test with temporal solution (not text-aware)
        detector = PersonDetector(backend="yolov8", solution="temporal")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_result = {
            "num_people": 2,
            "has_multiple_people": True,
            "detections": [
                {"bbox": [0, 0, 100, 100], "confidence": 0.9, "class": 0},
                {"bbox": [200, 200, 300, 300], "confidence": 0.8, "class": 0},
            ]
        }

        result = detector._apply_text_filtering(frame_result, frame)

        # Should return unchanged result
        self.assertEqual(result["num_people"], 2)
        self.assertTrue(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 2)

    @patch("src.models.person_detector.YoloV8Backend")
    def test_apply_text_filtering_single_detection(self, mock_backend):
        """Test that text filtering is NOT applied when there's only one detection."""
        fake_backend = Mock()
        mock_backend.return_value = fake_backend

        detector = PersonDetector(backend="yolov8", solution="temporal_textaware")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_result = {
            "num_people": 1,
            "has_multiple_people": False,
            "detections": [
                {"bbox": [0, 0, 100, 100], "confidence": 0.9, "class": 0},
            ]
        }

        result = detector._apply_text_filtering(frame_result, frame)

        # Should return unchanged result (no filtering for single detection)
        self.assertEqual(result["num_people"], 1)
        self.assertFalse(result["has_multiple_people"])
        self.assertEqual(len(result["detections"]), 1)

    @patch("src.models.person_detector.YoloV8Backend")
    def test_detector_config_integration(self, mock_backend):
        """Test that detector properly uses provided config."""
        fake_backend = Mock()
        mock_backend.return_value = fake_backend

        # Create custom config
        config = Config()
        config.TEXT_CONFIDENCE_THRESHOLD = 0.8
        config.TEXT_PROXIMITY_THRESHOLD = 50
        config.TEMPORAL_MIN_CONSECUTIVE = 5

        detector = PersonDetector(
            backend="yolov8",
            solution="temporal_textaware",
            config=config
        )

        # Check that config was applied
        self.assertEqual(detector.config.TEXT_CONFIDENCE_THRESHOLD, 0.8)
        self.assertEqual(detector.config.TEXT_PROXIMITY_THRESHOLD, 50)
        self.assertEqual(detector.config.TEMPORAL_MIN_CONSECUTIVE, 5)

    @patch("src.models.person_detector.YoloV8Backend")
    def test_detector_default_config(self, mock_backend):
        """Test that detector uses default config when none provided."""
        fake_backend = Mock()
        mock_backend.return_value = fake_backend

        detector = PersonDetector(backend="yolov8", solution="temporal_textaware")

        # Check that default config was created
        self.assertIsNotNone(detector.config)
        self.assertEqual(detector.config.TEXT_CONFIDENCE_THRESHOLD, 0.5)
        self.assertEqual(detector.config.TEXT_PROXIMITY_THRESHOLD, 100)
        self.assertEqual(detector.config.TEMPORAL_MIN_CONSECUTIVE, 20)


if __name__ == "__main__":
    unittest.main()
