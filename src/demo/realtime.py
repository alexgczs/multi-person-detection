"""
Real-time webcam demo utilities.

Provides a function that opens a webcam, runs the selected backend, draws
bounding boxes, and overlays person counts and FPS.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Optional

import cv2
from loguru import logger

from src.models.person_detector import PersonDetector


def run_webcam_demo(
    camera_index: int,
    backend: str,
    model_size: str,
    device: Optional[str],
    threshold: float,
    sample_rate: int,
    show_confidence: bool,
) -> int:
    """Run a real-time webcam demo.

    Args:
        camera_index: Webcam index to open
        backend: Detection backend name
        model_size: Model size for chosen backend
        device: Device string (cpu/cuda)
        threshold: Detection confidence threshold
        sample_rate: Process every Nth frame (1 = all)
        show_confidence: Whether to render confidence values

    Returns:
        Process exit code (0 = success)
    """
    logger.info("Initializing person detector for demo...")
    detector = PersonDetector(
        model_size=model_size,
        device=device,
        backend=backend,
    )

    logger.info(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %s", camera_index)
        return 1

    window_name = f"Multi-person demo [{backend}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    times = deque(maxlen=30)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.time()

            detections = []
            if frame_idx % max(1, int(sample_rate)) == 0:
                try:
                    detections = detector.backend.predict_persons(frame, threshold)
                except Exception as det_err:  # pragma: no cover (timing dependent)
                    logger.error(f"Error during detection: {det_err}")
                    detections = []

            # Draw detections
            num_people = len(detections)
            for det in detections:
                bbox = det.get("bbox")
                conf = det.get("confidence", 0.0)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if show_confidence:
                    label = f"person {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            # Overlay count and fps
            times.append(time.time() - start)
            fps = (
                1.0 / (sum(times) / len(times))
                if len(times) > 0 and sum(times) > 0
                else 0.0
            )
            header = f"People: {num_people} | FPS: {fps:.1f}"
            cv2.putText(
                frame,
                header,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 200, 255),
                2,
            )

            cv2.imshow(window_name, frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0
