"""
Tests for realtime demo utilities.

We mock OpenCV windowing and camera APIs to avoid GUI/hardware dependencies.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.demo.realtime import run_webcam_demo


class FakeCap:
    def __init__(self, frames=1):
        self.frames = frames
        self.count = 0

    def isOpened(self):
        return True

    def read(self):
        if self.count < self.frames:
            self.count += 1
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return True, frame
        return False, None

    def release(self):
        return None


class TestRealtimeDemo(unittest.TestCase):
    @patch("src.demo.realtime.cv2.waitKey", return_value=ord("q"))
    @patch("src.demo.realtime.cv2.imshow")
    @patch("src.demo.realtime.cv2.namedWindow")
    @patch("src.demo.realtime.cv2.VideoCapture", return_value=FakeCap(frames=2))
    @patch("src.demo.realtime.PersonDetector")
    def test_run_webcam_demo_quick_exit(
        self, mock_detector_cls, _mock_cap, _mock_win, _mock_show, _mock_wait
    ):
        # Backend returns one person detection to exercise drawing path
        mock_detector = Mock()
        mock_detector.backend.predict_persons.return_value = [
            {"bbox": [10, 10, 100, 100], "confidence": 0.9, "class": 0}
        ]
        mock_detector_cls.return_value = mock_detector

        exit_code = run_webcam_demo(
            camera_index=0,
            backend="yolov8",
            model_size="n",
            device="cpu",
            threshold=0.5,
            sample_rate=1,
            show_confidence=True,
            solution="counting",
        )

        self.assertEqual(exit_code, 0)
        mock_detector_cls.assert_called_once()
        mock_detector.backend.predict_persons.assert_called()


if __name__ == "__main__":
    unittest.main()
