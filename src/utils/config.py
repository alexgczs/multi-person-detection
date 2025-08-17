"""
Configuration management for system.

Provides centralized configuration management for all parameters and settings
used throughout the project. Supports environment variables and CLI overrides.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Centralized configuration for the multi-person detection system.
    Configuration can be set via:
    1. Environment variables (e.g., MULTIPLE_PEOPLE_THRESHOLD=0.2)
    2. CLI parameters (e.g., --people-threshold 0.2)
    3. Default values (defined below)

    Environment variables take precedence over defaults.
    CLI parameters take precedence over environment variables.
    """

    # Video Processing
    FRAME_SAMPLE_RATE: int = 1  # process every Nth frame
    MAX_FRAMES: Optional[int] = None  # max frames per video (None = process all frames)
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480

    # Person detection confidence threshold
    PERSON_CONFIDENCE_THRESHOLD: float = 0.5

    # Decision threshold, ratio of frames with >1 person
    # 0 threshold means that with only 1 frame with >1 person,
    # the video is considered as multiple people
    MULTIPLE_PEOPLE_THRESHOLD: float = 0.0

    # Temporal solution parameters
    TEMPORAL_MIN_CONSECUTIVE: int = 20

    # Temporal card-aware solution parameters
    CARD_MIN_AREA_RATIO_TO_LARGEST: float = 0.90
    CARD_SQUARE_TOLERANCE: float = 0.35

    # Text detection parameters
    TEXT_PROXIMITY_THRESHOLD: int = 100  # pixels
    TEXT_CONFIDENCE_THRESHOLD: float = 0.5  # OCR confidence threshold

    def __post_init__(self):
        """Load configuration from environment variables."""
        # Video Processing
        if env_val := os.getenv('MULTI_DETECT_FRAME_SAMPLE_RATE'):
            self.FRAME_SAMPLE_RATE = int(env_val)

        if env_val := os.getenv('MULTI_DETECT_MAX_FRAMES'):
            self.MAX_FRAMES = int(env_val) if env_val.lower() != 'none' else None

        if env_val := os.getenv('MULTI_DETECT_FRAME_WIDTH'):
            self.FRAME_WIDTH = int(env_val)

        if env_val := os.getenv('MULTI_DETECT_FRAME_HEIGHT'):
            self.FRAME_HEIGHT = int(env_val)

        # Person detection confidence
        if env_val := os.getenv('MULTI_DETECT_PERSON_CONFIDENCE_THRESHOLD'):
            self.PERSON_CONFIDENCE_THRESHOLD = float(env_val)

        # Decision threshold
        if env_val := os.getenv('MULTI_DETECT_MULTIPLE_PEOPLE_THRESHOLD'):
            self.MULTIPLE_PEOPLE_THRESHOLD = float(env_val)

        # Temporal solution parameters
        if env_val := os.getenv('MULTI_DETECT_TEMPORAL_MIN_CONSECUTIVE'):
            self.TEMPORAL_MIN_CONSECUTIVE = int(env_val)

        # Card-aware solution parameters
        if env_val := os.getenv('MULTI_DETECT_CARD_MIN_AREA_RATIO'):
            self.CARD_MIN_AREA_RATIO_TO_LARGEST = float(env_val)

        if env_val := os.getenv('MULTI_DETECT_CARD_SQUARE_TOLERANCE'):
            self.CARD_SQUARE_TOLERANCE = float(env_val)

        # Text detection parameters
        if env_val := os.getenv('MULTI_DETECT_TEXT_PROXIMITY_THRESHOLD'):
            self.TEXT_PROXIMITY_THRESHOLD = int(env_val)

        if env_val := os.getenv('MULTI_DETECT_TEXT_CONFIDENCE_THRESHOLD'):
            self.TEXT_CONFIDENCE_THRESHOLD = float(env_val)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary (for logging/debugging)."""
        return {
            'FRAME_SAMPLE_RATE': self.FRAME_SAMPLE_RATE,
            'MAX_FRAMES': self.MAX_FRAMES,
            'FRAME_WIDTH': self.FRAME_WIDTH,
            'FRAME_HEIGHT': self.FRAME_HEIGHT,
            'PERSON_CONFIDENCE_THRESHOLD': self.PERSON_CONFIDENCE_THRESHOLD,
            'MULTIPLE_PEOPLE_THRESHOLD': self.MULTIPLE_PEOPLE_THRESHOLD,
            'TEMPORAL_MIN_CONSECUTIVE': self.TEMPORAL_MIN_CONSECUTIVE,
            'CARD_MIN_AREA_RATIO_TO_LARGEST': self.CARD_MIN_AREA_RATIO_TO_LARGEST,
            'CARD_SQUARE_TOLERANCE': self.CARD_SQUARE_TOLERANCE,
            'TEXT_PROXIMITY_THRESHOLD': self.TEXT_PROXIMITY_THRESHOLD,
            'TEXT_CONFIDENCE_THRESHOLD': self.TEXT_CONFIDENCE_THRESHOLD,
        }
