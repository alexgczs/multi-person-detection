"""
Tests for video-level solution strategies.
"""

import unittest

from src.solutions.counting import CountingSolution
from src.solutions.temporal import TemporalHysteresisSolution
from src.utils.config import Config


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
        cfg.TEMPORAL_WINDOW = 10
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
        cfg.TEMPORAL_WINDOW = 10
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


if __name__ == "__main__":
    unittest.main()
