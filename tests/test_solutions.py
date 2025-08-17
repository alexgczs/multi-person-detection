"""
Tests for video-level solution strategies.
"""

import unittest

from src.solutions.counting import CountingSolution
from src.solutions.temporal import TemporalHysteresisSolution
from src.solutions.temporal_cardaware import TemporalCardAwareSolution
from src.solutions.temporal_textaware import TemporalTextAwareSolution
from src.utils.config import Config
import numpy as np
from unittest.mock import patch


class TestSolutions(unittest.TestCase):
    def test_counting_zero_frames(self):
        cfg = Config()
        sol = CountingSolution()
        res = sol.aggregate([], cfg)
        self.assertFalse(res["has_multiple_people"])  # no frames -> False
        self.assertEqual(res["total_frames"], 0)

    def test_counting_threshold(self):
        cfg = Config()
        cfg.MULTIPLE_PEOPLE_THRESHOLD = 0.5
        sol = CountingSolution()
        # 2 of 5 frames True -> ratio 0.4 -> below 0.5 -> False
        frames = [
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 2, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 2, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
        ]
        res = sol.aggregate(frames, cfg)
        self.assertFalse(res["has_multiple_people"])  # 0.4 <= 0.5

    def test_temporal_activation_min_consecutive(self):
        cfg = Config()
        cfg.TEMPORAL_MIN_CONSECUTIVE = 3
        sol = TemporalHysteresisSolution()
        # Only two consecutive True -> should not activate
        frames = [
            {"has_multiple_people": False},
            {"has_multiple_people": True},
            {"has_multiple_people": True},
            {"has_multiple_people": False},
        ]
        res = sol.aggregate(frames, cfg)
        self.assertFalse(res["has_multiple_people"])  # not enough consecutive

    def test_temporal_sticky_once_true(self):
        cfg = Config()
        cfg.TEMPORAL_MIN_CONSECUTIVE = 2
        sol = TemporalHysteresisSolution()
        # Two consecutive True early -> activate and stay True
        frames = [
            {"has_multiple_people": False, "num_people": 1},
            {"has_multiple_people": True, "num_people": 2},
            {"has_multiple_people": True, "num_people": 2},
            {"has_multiple_people": False, "num_people": 1},
            {"has_multiple_people": False, "num_people": 1},
        ]
        res = sol.aggregate(frames, cfg)
        self.assertTrue(res["has_multiple_people"])  # sticky behavior

    def test_temporal_cardaware_suppresses_square_small(self):
        cfg = Config()
        cfg.TEMPORAL_MIN_CONSECUTIVE = 1
        cfg.CARD_MIN_AREA_RATIO_TO_LARGEST = 0.2
        cfg.CARD_SQUARE_TOLERANCE = 0.2

        sol = TemporalCardAwareSolution()
        # Two detections: one large, one small almost square -> should suppress 1
        frames = [
            {
                "num_people": 2,
                "has_multiple_people": True,
                "detections": [
                    {"bbox": [0, 0, 100, 200],
                     "confidence": 0.9,
                     "class": 0},  # large (AR=0.5)
                    {"bbox": [10, 10, 30, 30],
                     "confidence": 0.9,
                     "class": 0},  # small, square
                ],
            }
        ]
        res = sol.aggregate(frames, cfg)
        # After suppression, adjusted count = 1 -> not multi-person
        self.assertFalse(res["has_multiple_people"])  # suppressed small square

    def test_temporal_cardaware_keeps_two_if_not_square(self):
        cfg = Config()
        cfg.TEMPORAL_MIN_CONSECUTIVE = 1
        cfg.CARD_MIN_AREA_RATIO_TO_LARGEST = 0.2
        cfg.CARD_SQUARE_TOLERANCE = 0.1

        sol = TemporalCardAwareSolution()
        # Two detections: small but not square (AR far from 1) -> do not suppress
        frames = [
            {
                "num_people": 2,
                "has_multiple_people": True,
                "detections": [
                    {"bbox": [0, 0, 100, 200],
                     "confidence": 0.9,
                     "class": 0},  # large (AR=0.5)
                    {"bbox": [10, 10, 50, 30],
                     "confidence": 0.9,
                     "class": 0},  # small, AR~2
                ],
            }
        ]
        res = sol.aggregate(frames, cfg)
        self.assertTrue(res["has_multiple_people"])  # not square -> remains 2

    def test_counting_equal_threshold_edge(self):
        cfg = Config()
        cfg.MULTIPLE_PEOPLE_THRESHOLD = 0.4
        sol = CountingSolution()
        frames = [
            {"num_people": 1, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 2, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 1, "has_multiple_people": False},
        ]  # ratio = 2/5 = 0.4 -> not strictly greater
        res = sol.aggregate(frames, cfg)
        self.assertFalse(res["has_multiple_people"])  # equality should be False

    def test_counting_above_threshold_and_metrics(self):
        cfg = Config()
        cfg.MULTIPLE_PEOPLE_THRESHOLD = 0.3
        sol = CountingSolution()
        frames = [
            {"num_people": 2, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
            {"num_people": 3, "has_multiple_people": True},
            {"num_people": 1, "has_multiple_people": False},
        ]  # ratio = 0.5 > 0.3
        res = sol.aggregate(frames, cfg)
        self.assertTrue(res["has_multiple_people"])  # above threshold
        # avg people = (2+1+3+1)/4 = 1.75 -> round to 2
        self.assertEqual(res["num_people"], 2)
        self.assertEqual(res["max_people"], 3)
        self.assertAlmostEqual(res["multiple_people_ratio"], 0.5)
        self.assertEqual(res["frames_with_multiple"], 2)
        self.assertEqual(res["total_frames"], 4)

    def test_temporal_min_consecutive_one(self):
        cfg = Config()
        cfg.TEMPORAL_MIN_CONSECUTIVE = 1
        sol = TemporalHysteresisSolution()
        frames = [
            {"has_multiple_people": False},
            {"has_multiple_people": True},
            {"has_multiple_people": False},
        ]
        res = sol.aggregate(frames, cfg)
        self.assertTrue(res["has_multiple_people"])  # activates on single True

    def test_temporal_all_false(self):
        cfg = Config()
        cfg.TEMPORAL_MIN_CONSECUTIVE = 2
        sol = TemporalHysteresisSolution()
        frames = [
            {"has_multiple_people": False, "num_people": 1},
            {"has_multiple_people": False, "num_people": 0},
            {"has_multiple_people": False, "num_people": 1},
        ]
        res = sol.aggregate(frames, cfg)
        self.assertFalse(res["has_multiple_people"])  # never activates
        self.assertEqual(res["max_people"], 1)
        self.assertEqual(res["num_people"], 1)
        self.assertAlmostEqual(res["multiple_people_ratio"], 0.0)

    def test_temporal_textaware_basic_aggregation(self):
        """Test that temporal_textaware performs basic temporal hysteresis."""
        cfg = Config()
        cfg.TEMPORAL_MIN_CONSECUTIVE = 2
        sol = TemporalTextAwareSolution()
        frames = [
            {"has_multiple_people": False, "num_people": 1},
            {"has_multiple_people": True, "num_people": 2},
            {"has_multiple_people": True, "num_people": 2},
            {"has_multiple_people": False, "num_people": 1},
        ]
        res = sol.aggregate(frames, cfg)
        self.assertTrue(res["has_multiple_people"])  # sticky behavior
        self.assertEqual(res["solution_type"], "temporal_textaware")

    def test_temporal_textaware_text_detection_mock(self):
        """Test text detection functionality with mocked EasyOCR."""
        cfg = Config()
        cfg.TEXT_CONFIDENCE_THRESHOLD = 0.5
        cfg.TEXT_PROXIMITY_THRESHOLD = 100

        sol = TemporalTextAwareSolution()

        # Mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock EasyOCR results
        mock_results = [
            ([[10, 10], [110, 10], [110, 50], [10, 50]], "TEXT", 0.8),  # High conf
            ([[200, 200], [300, 200], [300, 250], [200, 250]], "ID", 0.3),  # Low conf
        ]

        with patch.object(sol, '_get_text_detector') as mock_detector:
            mock_detector.return_value.readtext.return_value = mock_results

            text_regions = sol._detect_text_regions(frame, cfg)

            # Should only include high confidence detection
            self.assertEqual(len(text_regions), 1)
            self.assertEqual(text_regions[0], (10, 10, 101, 41))  # x, y, w, h

    def test_temporal_textaware_confidence_threshold(self):
        """Test that text confidence threshold is properly applied."""
        cfg = Config()
        cfg.TEXT_CONFIDENCE_THRESHOLD = 0.7  # High threshold

        sol = TemporalTextAwareSolution()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_results = [
            ([[10, 10], [110, 10], [110, 50], [10, 50]], "TEXT", 0.6),  # Below thresh
            ([[200, 200], [300, 200], [300, 250], [200, 250]], "ID", 0.8),  # Above
        ]

        with patch.object(sol, '_get_text_detector') as mock_detector:
            mock_detector.return_value.readtext.return_value = mock_results

            text_regions = sol._detect_text_regions(frame, cfg)

            # Should only include detection above threshold
            self.assertEqual(len(text_regions), 1)
            self.assertEqual(text_regions[0], (200, 200, 101, 51))

    def test_temporal_textaware_face_near_text(self):
        """Test face proximity detection to text regions."""
        cfg = Config()
        cfg.TEXT_PROXIMITY_THRESHOLD = 50

        sol = TemporalTextAwareSolution()

        # Text region at (100, 100, 100, 50)
        text_regions = [(100, 100, 100, 50)]

        # Face bbox close to text (center distance < threshold)
        close_face = (80, 80, 40, 40)  # Center at (100, 100), distance = 25
        far_face = (300, 300, 40, 40)  # Center at (320, 320), distance = ~311

        self.assertTrue(sol._is_face_near_text(close_face, text_regions, cfg))
        self.assertFalse(sol._is_face_near_text(far_face, text_regions, cfg))

    def test_temporal_textaware_overlapping_face_text(self):
        """Test face overlapping with text region detection."""
        cfg = Config()
        cfg.TEXT_PROXIMITY_THRESHOLD = 10  # Very small threshold

        sol = TemporalTextAwareSolution()
        text_regions = [(100, 100, 100, 50)]  # x=100-200, y=100-150

        # Face that overlaps with text region
        overlapping_face = (150, 120, 40, 40)  # x=150-190, y=120-160 (overlaps)
        non_overlapping_face = (300, 300, 40, 40)  # No overlap, far away

        self.assertTrue(sol._is_face_near_text(overlapping_face, text_regions, cfg))
        self.assertFalse(
            sol._is_face_near_text(non_overlapping_face, text_regions, cfg)
        )

    @patch(
        'src.solutions.temporal_textaware.'
        'TemporalTextAwareSolution._detect_text_regions'
    )
    def test_temporal_textaware_filter_id_card_faces(self, mock_detect_text):
        """Test filtering of ID card faces based on text proximity."""
        cfg = Config()
        cfg.TEXT_PROXIMITY_THRESHOLD = 50

        sol = TemporalTextAwareSolution()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock text detection to return one text region
        mock_detect_text.return_value = [(100, 100, 100, 50)]

        # Frame with two detections: one near text, one far
        frame_results = [{
            'detections': [
                {'bbox': [80, 80, 120, 120], 'confidence': 0.9,
                    'class': 0},  # Near text - should be filtered
                {'bbox': [300, 300, 340, 340], 'confidence': 0.9,
                    'class': 0},  # Far from text - should be kept
            ],
            'num_people': 2,
            'has_multiple_people': True
        }]

        filtered_results = sol._filter_id_card_faces(frame_results, frame, cfg)

        # Should keep only the face far from text
        self.assertEqual(len(filtered_results), 1)
        self.assertEqual(len(filtered_results[0]['detections']), 1)
        self.assertEqual(filtered_results[0]['num_people'], 1)
        self.assertFalse(filtered_results[0]['has_multiple_people'])

        # Check that the remaining detection is the far one
        remaining_bbox = filtered_results[0]['detections'][0]['bbox']
        self.assertEqual(remaining_bbox, [300, 300, 340, 340])

    @patch(
        'src.solutions.temporal_textaware.'
        'TemporalTextAwareSolution._detect_text_regions'
    )
    def test_temporal_textaware_no_text_detected(self, mock_detect_text):
        """Test behavior when no text is detected."""
        cfg = Config()

        sol = TemporalTextAwareSolution()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock no text detection
        mock_detect_text.return_value = []

        frame_results = [{
            'detections': [
                {'bbox': [80, 80, 120, 120], 'confidence': 0.9, 'class': 0},
                {'bbox': [300, 300, 340, 340], 'confidence': 0.9, 'class': 0},
            ],
            'num_people': 2,
            'has_multiple_people': True
        }]

        filtered_results = sol._filter_id_card_faces(frame_results, frame, cfg)

        # Should keep all detections when no text is found
        self.assertEqual(len(filtered_results), 1)
        self.assertEqual(len(filtered_results[0]['detections']), 2)
        self.assertEqual(filtered_results[0]['num_people'], 2)
        self.assertTrue(filtered_results[0]['has_multiple_people'])

    def test_temporal_textaware_non_person_detections(self):
        """Test that non-person detections are preserved."""
        cfg = Config()

        sol = TemporalTextAwareSolution()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_results = [{
            'detections': [
                {'bbox': [80, 80, 120, 120], 'confidence': 0.9,
                    'class': 0},  # Person (YOLO)
                {'bbox': [200, 200, 240, 240], 'confidence': 0.9,
                    'class': 1},  # Person (torchvision)
                {'bbox': [300, 300, 340, 340],
                    'confidence': 0.9, 'class': 2},  # Non-person
            ],
            'num_people': 2,
            'has_multiple_people': True
        }]

        with patch.object(sol, '_detect_text_regions', return_value=[]):
            filtered_results = sol._filter_id_card_faces(frame_results, frame, cfg)

            # Should preserve all detections including non-person
            self.assertEqual(len(filtered_results[0]['detections']), 3)

            # Person count should only consider class 0 and 1
            self.assertEqual(filtered_results[0]['num_people'], 2)


if __name__ == "__main__":
    unittest.main()
