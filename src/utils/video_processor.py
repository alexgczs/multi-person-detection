"""
Video processing utilities.

Frame extraction and resizing for the detector.
"""

from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from src.utils.config import Config


class VideoProcessor:
    """
    Processing utilities to extract and resize frames from videos.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def extract_frames(
        self,
        video_path: str,
        sample_rate: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file
            sample_rate: Extract every Nth frame (default from config)
            max_frames: Maximum number of frames to extract (default from config)

        Returns:
            List of frames as numpy arrays
        """
        try:
            # Use config defaults if not specified
            sample_rate = int(sample_rate or self.config.FRAME_SAMPLE_RATE)
            max_frames = int(max_frames or self.config.MAX_FRAMES)

            logger.info(f"Extracting frames from: {video_path}")

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            logger.info(
                f"Video properties: {total_frames} frames, {fps:.2f} fps, "
                f"{duration:.2f}s"
            )

            # Extract frames
            frames = []
            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames based on sample rate
                if frame_count % sample_rate == 0:
                    # Resize frame if needed
                    if (
                        frame.shape[1] != self.config.FRAME_WIDTH
                        or frame.shape[0] != self.config.FRAME_HEIGHT
                    ):
                        frame = self._resize_frame(frame)

                    frames.append(frame)
                    extracted_count += 1

                    # Stop if max frames reached
                    if extracted_count >= max_frames:
                        break

                frame_count += 1

            cap.release()

            logger.info(f"Extracted {len(frames)} frames from video")
            return frames

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            raise

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target dimensions while maintaining aspect ratio.

        Args:
            frame: Input frame

        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        target_width = self.config.FRAME_WIDTH
        target_height = self.config.FRAME_HEIGHT

        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize frame
        resized = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Pad to target size if necessary
        if new_width < target_width or new_height < target_height:
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            padded[
                y_offset : y_offset + new_height, x_offset : x_offset + new_width
            ] = resized
            return padded

        return resized
