"""
Person detection backends and base class.

This module defines a backend interface for per-frame person detection and
multiple concrete implementations (Ultralytics YOLOv8, torchvision detectors,
and OpenCV HOG). Each backend returns a uniform list of detection dicts,
so the rest of the system can remain backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TypedDict, Sequence

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger
from ultralytics import YOLO


Detection = TypedDict(
    "Detection",
    {
        # [x1, y1, x2, y2] coordinates in pixel space
        "bbox": Sequence[float],
        # Detector-specific confidence score in [0, 1]
        "confidence": float,
        # Integer class id (1 for person in torchvision, 0 in YOLO)
        "class": int,
    },
)


class PersonDetectionBackend(ABC):
    """Abstract base class for person detection backends.

    Implementations must provide a `predict_persons` method that returns a list
    of person detections following the `Detection` schema.
    """

    @abstractmethod
    def predict_persons(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        """Detect persons in a single BGR image (OpenCV format).

        Args:
            frame: Input image as a NumPy array in BGR color order.
            confidence_threshold: Minimum score required to keep a detection.

        Returns:
            List of detections with bounding boxes and confidence scores.
        """
        raise NotImplementedError


class YoloV8Backend(PersonDetectionBackend):
    """YOLOv8-based backend for person detection.

    Uses Ultralytics YOLOv8 pre-trained weights (COCO) and filters for the
    person class (class id 0 in YOLO/COCO).
    """

    def __init__(self, model_size: str, device: str):
        self.model_size = model_size
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        try:
            model_name = f"yolov8{self.model_size}.pt"
            model = YOLO(model_name)
            try:
                model.to(self.device)
                logger.info(
                    f"Moved YOLO model to device: {self.device}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not move model to device '{self.device}': {e}. "
                    "Using default device."
                )
            logger.info(f"Loaded YOLO model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def predict_persons(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        # Be explicit about device to avoid implicit defaults
        results = self.model(frame, verbose=False, device=self.device)
        person_detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
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
        return person_detections


class TorchvisionFRCNNBackend(PersonDetectionBackend):
    """Faster R-CNN (torchvision) backend for person detection.

    Uses torchvision's pre-trained Faster R-CNN (COCO) and filters for
    class id 1 (person in torchvision's COCO label mapping).
    """

    def __init__(self, device: str):
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=(
                torchvision.models.detection.
                FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
        )
        self.model.eval()
        self.model.to(self.device)
        self.person_class_id = 1

    def predict_persons(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        rgb = frame[:, :, ::-1].copy()
        tensor = torchvision.transforms.functional.to_tensor(rgb).to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model([tensor])[0]

        person_detections: List[Detection] = []
        boxes = outputs.get("boxes")
        labels = outputs.get("labels")
        scores = outputs.get("scores")
        if boxes is None or labels is None or scores is None:
            return person_detections

        for box, label, score in zip(boxes, labels, scores):
            if (
                int(label.item()) == self.person_class_id
                and float(score.item()) >= confidence_threshold
            ):
                person_detections.append(
                    {
                        "bbox": box.detach().cpu().numpy(),
                        "confidence": float(score.item()),
                        "class": int(label.item()),
                    }
                )
        return person_detections


class TorchvisionSSDBackend(PersonDetectionBackend):
    """SSDlite (MobileNetV3) backend for person detection (torchvision).

    Good balance between speed and accuracy, suitable for CPU real-time demos.
    """

    def __init__(self, device: str):
        self.device = device
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=(
                torchvision.models.detection.
                SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            )
        )
        self.model.eval()
        self.model.to(self.device)
        self.person_class_id = 1

    def predict_persons(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        rgb = frame[:, :, ::-1].copy()
        tensor = torchvision.transforms.functional.to_tensor(rgb).to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model([tensor])[0]

        person_detections: List[Detection] = []
        boxes, labels, scores = (
            outputs["boxes"],
            outputs["labels"],
            outputs["scores"],
        )
        for box, label, score in zip(boxes, labels, scores):
            if (
                int(label.item()) == self.person_class_id
                and float(score.item()) >= confidence_threshold
            ):
                person_detections.append(
                    {
                        "bbox": box.detach().cpu().numpy(),
                        "confidence": float(score.item()),
                        "class": int(label.item()),
                    }
                )
        return person_detections


class TorchvisionRetinaNetBackend(PersonDetectionBackend):
    """RetinaNet (ResNet50 FPN) backend for person detection (torchvision).

    Strong single-stage detector; heavier than SSD on CPU, works well on GPU.
    """

    def __init__(self, device: str):
        self.device = device
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=(
                torchvision.models.detection.
                RetinaNet_ResNet50_FPN_Weights.DEFAULT
            )
        )
        self.model.eval()
        self.model.to(self.device)
        self.person_class_id = 1

    def predict_persons(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        rgb = frame[:, :, ::-1].copy()
        tensor = torchvision.transforms.functional.to_tensor(rgb).to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model([tensor])[0]

        person_detections: List[Detection] = []
        boxes, labels, scores = (
            outputs["boxes"],
            outputs["labels"],
            outputs["scores"],
        )
        for box, label, score in zip(boxes, labels, scores):
            if (
                int(label.item()) == self.person_class_id
                and float(score.item()) >= confidence_threshold
            ):
                person_detections.append(
                    {
                        "bbox": box.detach().cpu().numpy(),
                        "confidence": float(score.item()),
                        "class": int(label.item()),
                    }
                )
        return person_detections


class OpenCVHOGBackend(PersonDetectionBackend):
    """OpenCV HOG+SVM person detector (pedestrian detector).

    Note:
        This detector is optimized for full-body pedestrian detection.
    """

    def __init__(self, device: str):  # device unused; kept for uniform API
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

    def predict_persons(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        # Map our threshold to HOG hitThreshold and do minimal post-filtering
        rects, weights = self.hog.detectMultiScale(
            frame,
            hitThreshold=float(confidence_threshold),
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        person_detections: List[Detection] = []
        for (x, y, w, h), score in zip(rects, weights):
            conf = float(score)
            bbox = np.array([x, y, x + w, y + h], dtype=float)
            person_detections.append({
                "bbox": bbox,
                "confidence": conf,
                "class": 1,
            })
        return person_detections
