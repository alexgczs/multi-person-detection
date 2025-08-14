"""
Configuration management for system.

Provides centralized configuration management for all parameters and settings
used throughout the project.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Video Processing
    FRAME_SAMPLE_RATE: int = 1  # process every Nth frame
    MAX_FRAMES: int = 100  # max frames per video
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480

    # Decision threshold, ratio of frames with >1 person
    # 0 threshold means that with only 1 frame with >1 person,
    # the video is considered as multiple people
    MULTIPLE_PEOPLE_THRESHOLD: float = 0.0
