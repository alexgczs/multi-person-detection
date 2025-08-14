"""
CLI interface for multi-person detection.

Provides a single command to predict and print a label for a video.
"""

import os
import sys

import click
from loguru import logger

from src.models.person_detector import PersonDetector


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", default="INFO", help="Logging level")
def cli(verbose: bool, log_level: str):
    """Multi-person detection for identity verification videos."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level if not verbose else "DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )

    if verbose:
        logger.info("Verbose mode enabled")


@cli.command()
@click.option("--video", "-i", required=True, help="Path to input video file")
@click.option(
    "--threshold", "-t", default=0.5, help="Confidence threshold for detection"
)
@click.option("--model-size", default="n", help="YOLO model size (n, s, m, l, x)")
def predict(video: str, threshold: float, model_size: str):
    """Predict whether a video contains multiple people."""
    try:
        # Validate input
        if not os.path.exists(video):
            logger.error(f"Video file not found: {video}")
            sys.exit(1)

        # Initialize detector
        logger.info("Initializing person detector...")
        detector = PersonDetector(model_size=model_size)

        # Process video
        logger.info(f"Processing video: {video}")
        result = detector.predict(video_path=video, confidence_threshold=threshold)

        # stdout output
        click.echo(f"label predicted: {int(result['has_multiple_people'])}")

        # Always exit 0 on successful execution; label is conveyed via logs/output
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
