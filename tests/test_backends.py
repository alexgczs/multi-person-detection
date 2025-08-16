"""
Unit tests for detection backends.

These tests verify that each backend:
  - Satisfies the common backend interface
  - Parses raw model outputs into the `Detection` schema
  - Applies thresholding on confidences

Weights downloads are avoided by mocking constructors where applicable.
The OpenCV HOG test runs real on Linux/macOS and is mocked on Windows for
stability.
"""

import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.models.backends import (
    OpenCVHOGBackend,
    TorchvisionFRCNNBackend,
    TorchvisionRetinaNetBackend,
    TorchvisionSSDBackend,
    YoloV8Backend,
)


class TestBackends(unittest.TestCase):
    """Backend-specific tests to ensure API and parsing behavior."""

    @patch("src.models.backends.YOLO")
    def test_yolov8_backend_parsing(self, mock_yolo):
        # Fake YOLO output with two person detections and one non-person
        mock_model = Mock()
        mock_result = Mock()
        box1 = Mock()
        box1.cls = torch.tensor([0])
        box1.conf = torch.tensor([0.9])
        box1.xyxy = torch.tensor([[0, 0, 10, 10]])
        box2 = Mock()
        box2.cls = torch.tensor([0])
        box2.conf = torch.tensor([0.7])
        box2.xyxy = torch.tensor([[5, 5, 15, 15]])
        box3 = Mock()
        box3.cls = torch.tensor([2])
        box3.conf = torch.tensor([0.95])
        box3.xyxy = torch.tensor([[1, 1, 2, 2]])
        mock_result.boxes = [box1, box2, box3]
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        backend = YoloV8Backend(model_size="n", device="cpu")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = backend.predict_persons(frame, confidence_threshold=0.8)
        self.assertEqual(len(detections), 1)

    @patch("src.models.backends.YOLO")
    def test_yolov8_backend_boxes_none_path(self, mock_yolo):
        # Simulate results with boxes=None to hit the guard branch
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        backend = YoloV8Backend(model_size="n", device="cpu")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = backend.predict_persons(frame, confidence_threshold=0.5)
        self.assertEqual(detections, [])

    @patch("src.models.backends.YOLO")
    def test_yolov8_backend_device_move_warning(self, mock_yolo):
        # Make model.to raise to trigger warning path
        class Fake:
            def to(self, *_a, **_k):
                raise RuntimeError("no device")

            def __call__(self, *args, **kwargs):
                return []

        mock_yolo.return_value = Fake()
        backend = YoloV8Backend(model_size="n", device="cpu")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = backend.predict_persons(frame, 0.5)
        self.assertIsInstance(detections, list)

    @patch("src.models.backends.torchvision.models.detection.fasterrcnn_resnet50_fpn")
    def test_torchvision_frcnn_backend_parsing(self, mock_ctor):
        class Fake:
            def eval(self):
                return self

            def to(self, *_a, **_k):
                return self

            def __call__(self, xs):
                boxes = torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0]]
                )
                labels = torch.tensor([1, 3])
                scores = torch.tensor([0.9, 0.95])
                return [{"boxes": boxes, "labels": labels, "scores": scores}]

        mock_ctor.return_value = Fake()

        backend = TorchvisionFRCNNBackend(device="cpu")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = backend.predict_persons(frame, confidence_threshold=0.5)
        self.assertEqual(len(detections), 1)

    @patch("src.models.backends.torchvision.models.detection.fasterrcnn_resnet50_fpn")
    def test_torchvision_frcnn_boxes_none(self, mock_ctor):
        class Fake:
            def eval(self):
                return self

            def to(self, *_a, **_k):
                return self

            def __call__(self, xs):
                return [{"boxes": None, "labels": None, "scores": None}]

        mock_ctor.return_value = Fake()

        backend = TorchvisionFRCNNBackend(device="cpu")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = backend.predict_persons(frame, 0.5)
        self.assertEqual(detections, [])

    @patch(
            "src.models.backends.torchvision.models.detection."
            "ssdlite320_mobilenet_v3_large"
            )
    def test_torchvision_ssd_backend_parsing(self, mock_ctor):
        class Fake:
            def eval(self):
                return self

            def to(self, *_a, **_k):
                return self

            def __call__(self, xs):
                boxes = torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0]]
                )
                labels = torch.tensor([1, 1])
                scores = torch.tensor([0.9, 0.4])
                return [{"boxes": boxes, "labels": labels, "scores": scores}]

        mock_ctor.return_value = Fake()

        backend = TorchvisionSSDBackend(device="cpu")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = backend.predict_persons(frame, confidence_threshold=0.5)
        self.assertEqual(len(detections), 1)

    @patch("src.models.backends.torchvision.models.detection.retinanet_resnet50_fpn")
    def test_torchvision_retinanet_backend_parsing(self, mock_ctor):
        class Fake:
            def eval(self):
                return self

            def to(self, *_a, **_k):
                return self

            def __call__(self, xs):
                boxes = torch.tensor(
                    [
                        [0.0, 0.0, 10.0, 10.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [2.0, 2.0, 3.0, 3.0],
                    ]
                )
                labels = torch.tensor([1, 1, 1])
                scores = torch.tensor([0.9, 0.6, 0.4])
                return [{"boxes": boxes, "labels": labels, "scores": scores}]

        mock_ctor.return_value = Fake()

        backend = TorchvisionRetinaNetBackend(device="cpu")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = backend.predict_persons(frame, confidence_threshold=0.5)
        self.assertEqual(len(detections), 2)

    def test_opencv_hog_backend_smoke(self):
        backend = OpenCVHOGBackend(device="cpu")
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        if sys.platform.startswith("win"):
            with patch(
                "src.models.backends.cv2.HOGDescriptor.detectMultiScale",
                return_value=([], []),
            ):
                dets = backend.predict_persons(frame, confidence_threshold=10.0)
        else:
            dets = backend.predict_persons(frame, confidence_threshold=10.0)
        self.assertIsInstance(dets, list)


if __name__ == "__main__":
    unittest.main()
