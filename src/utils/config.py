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
    MAX_FRAMES: int = None  # max frames per video (None = process all frames)
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480

    # Decision threshold, ratio of frames with >1 person
    # 0 threshold means that with only 1 frame with >1 person,
    # the video is considered as multiple people
    MULTIPLE_PEOPLE_THRESHOLD: float = 0.0

    # Temporal solution parameters
    TEMPORAL_WINDOW: int = 15
    TEMPORAL_MIN_CONSECUTIVE: int = 20

    # Temporal card-aware solution parameters
    CARD_MIN_AREA_RATIO_TO_LARGEST: float = 0.90
    CARD_SQUARE_TOLERANCE: float = 0.35
