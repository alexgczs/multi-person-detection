"""
CLI interface for multi-person detection.

Commands to predict, evaluate videos, and generate reports.
Also provides a real-time webcam demo.
"""

import os
import sys
from datetime import datetime

import click
from loguru import logger

from src.models.person_detector import PersonDetector
from src.demo.realtime import run_webcam_demo
from src.utils.dataset_evaluator import DatasetEvaluator
from src.utils.report_generator import ReportGenerator


def setup_logging(verbose: bool = False, log_level: str = "INFO"):
    """Configure logging for the application."""
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


def apply_detector_config(
    detector: PersonDetector,
    sample_rate: int | None = None,
    max_frames: int | None = None,
    people_threshold: float | None = None,
    temporal_min_consecutive: int | None = None,
    card_min_area_ratio: float | None = None,
    card_square_tolerance: float | None = None,
    text_proximity_threshold: int | None = None,
    text_confidence_threshold: float | None = None,
) -> None:
    """Apply configuration overrides to detector consistently."""
    if sample_rate is not None:
        detector.video_processor.config.FRAME_SAMPLE_RATE = int(sample_rate)
    if max_frames is not None:
        detector.video_processor.config.MAX_FRAMES = int(max_frames)
    if people_threshold is not None:
        detector.config.MULTIPLE_PEOPLE_THRESHOLD = float(people_threshold)
    if temporal_min_consecutive is not None:
        detector.config.TEMPORAL_MIN_CONSECUTIVE = int(temporal_min_consecutive)
    if card_min_area_ratio is not None:
        detector.config.CARD_MIN_AREA_RATIO_TO_LARGEST = float(card_min_area_ratio)
    if card_square_tolerance is not None:
        detector.config.CARD_SQUARE_TOLERANCE = float(card_square_tolerance)
    if text_proximity_threshold is not None:
        detector.config.TEXT_PROXIMITY_THRESHOLD = int(text_proximity_threshold)
    if text_confidence_threshold is not None:
        detector.config.TEXT_CONFIDENCE_THRESHOLD = float(text_confidence_threshold)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", default="INFO", help="Logging level")
def cli(verbose: bool, log_level: str):
    """Multi-person detection for identity verification videos."""
    setup_logging(verbose, log_level)


@click.command()
@click.option("--video", "-i", required=True, help="Path to input video file")
@click.option(
    "--threshold", "-t", default=0.5, help="Confidence threshold for detection"
)
@click.option("--model-size", default="n", help="Model size for the selected backend")
@click.option(
    "--backend",
    default="yolov8",
    type=click.Choice([
        "yolov8",
        "torchvision_frcnn",
        "torchvision_ssd",
        "torchvision_retinanet",
        "opencv_hog",
    ], case_sensitive=False),
    help="Detection backend to use",
)
@click.option(
    "--solution",
    default="counting",
    type=click.Choice(
        ["counting", "temporal", "temporal_cardaware", "temporal_textaware"],
        case_sensitive=False
    ),
    help="Video-level solution strategy",
)
@click.option("--device", default=None, help="Computation device: cpu or cuda")
@click.option("--sample-rate", default=None, type=int, help="Process every Nth frame")
@click.option("--max-frames", default=None, type=int, help="Maximum frames per video")
@click.option(
    "--people-threshold",
    default=None,
    type=float,
    help="Ratio of frames with >1 person to classify as multi-person",
)
@click.option(
    "--temporal-min-consecutive",
    default=None,
    type=int,
    help="Minimum consecutive frames for temporal activation",
)
@click.option(
    "--card-min-area-ratio",
    default=None,
    type=float,
    help="Minimum area ratio for card detection",
)
@click.option(
    "--card-square-tolerance",
    default=None,
    type=float,
    help="Square tolerance for card detection",
)
@click.option(
    "--text-proximity-threshold",
    default=None,
    type=int,
    help="Text proximity threshold for text-aware detection",
)
@click.option(
    "--text-confidence-threshold",
    default=None,
    type=float,
    help="OCR confidence threshold for text detection",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", default="INFO", help="Logging level")
def predict(
    video: str,
    threshold: float,
    model_size: str,
    backend: str,
    solution: str,
    device: str | None,
    sample_rate: int | None,
    max_frames: int | None,
    people_threshold: float | None,
    temporal_min_consecutive: int | None,
    card_min_area_ratio: float | None,
    card_square_tolerance: float | None,
    text_proximity_threshold: int | None,
    text_confidence_threshold: float | None,
    verbose: bool,
    log_level: str,
):
    """Predict whether a video contains multiple people."""
    setup_logging(verbose, log_level)
    try:
        # Validate input
        if not os.path.exists(video):
            logger.error(f"Video file not found: {video}")
            sys.exit(1)

        # Setup detector and process video
        logger.info("Initializing person detector...")
        detector = PersonDetector(
            model_size=model_size,
            device=device,
            backend=backend,
            solution=solution,
        )
        # Apply config overrides if provided
        apply_detector_config(
            detector=detector,
            sample_rate=sample_rate,
            max_frames=max_frames,
            people_threshold=people_threshold,
            temporal_min_consecutive=temporal_min_consecutive,
            card_min_area_ratio=card_min_area_ratio,
            card_square_tolerance=card_square_tolerance,
            text_proximity_threshold=text_proximity_threshold,
            text_confidence_threshold=text_confidence_threshold,
        )

        logger.info(f"Processing video: {video}")
        result = detector.predict(video_path=video, confidence_threshold=threshold)

        # stdout output
        click.echo(f"label predicted: {int(result['has_multiple_people'])}")

        # Always exit 0 on successful execution; label is conveyed via logs/output
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)


@click.command()
@click.option("--dataset-path", "-d", required=True, help="Path to dataset directory")
@click.option("--labels-file", "-l", required=True, help="Path to labels file")
@click.option(
    "--threshold", "-t", default=0.5, help="Confidence threshold for detection"
)
@click.option("--model-size", default="n", help="Model size for the selected backend")
@click.option(
    "--backend",
    default="yolov8",
    type=click.Choice([
        "yolov8",
        "torchvision_frcnn",
        "torchvision_ssd",
        "torchvision_retinanet",
        "opencv_hog",
    ], case_sensitive=False),
    help="Detection backend to use",
)
@click.option(
    "--solution",
    default="counting",
    type=click.Choice(
        ["counting", "temporal", "temporal_cardaware", "temporal_textaware"],
        case_sensitive=False
    ),
    help="Video-level solution strategy",
)
@click.option("--device", default=None, help="Computation device: cpu or cuda")
@click.option("--sample-rate", default=None, type=int, help="Process every Nth frame")
@click.option("--max-frames", default=None, type=int, help="Maximum frames per video")
@click.option(
    "--people-threshold",
    default=None,
    type=float,
    help="Ratio of frames with >1 person to classify as multi-person",
)
@click.option(
    "--temporal-min-consecutive",
    default=None,
    type=int,
    help="Minimum consecutive frames for temporal activation",
)
@click.option(
    "--card-min-area-ratio",
    default=None,
    type=float,
    help="Minimum area ratio for card detection",
)
@click.option(
    "--card-square-tolerance",
    default=None,
    type=float,
    help="Square tolerance for card detection",
)
@click.option(
    "--text-proximity-threshold",
    default=None,
    type=int,
    help="Text proximity threshold for text-aware detection",
)
@click.option(
    "--text-confidence-threshold",
    default=None,
    type=float,
    help="OCR confidence threshold for text detection",
)
@click.option(
    "--num-workers",
    default=1,
    type=int,
    help="Parallel workers for evaluation",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    help="Show progress bar",
)
@click.option(
    "--output-dir",
    "-o",
    help="Output directory name (default: timestamp)",
)
@click.option(
    "--no-report",
    is_flag=True,
    help="Skip automatic report generation",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", default="INFO", help="Logging level")
def evaluate(
    dataset_path: str,
    labels_file: str,
    threshold: float,
    model_size: str,
    backend: str,
    solution: str,
    device: str | None,
    sample_rate: int | None,
    max_frames: int | None,
    people_threshold: float | None,
    temporal_min_consecutive: int | None,
    card_min_area_ratio: float | None,
    card_square_tolerance: float | None,
    text_proximity_threshold: int | None,
    text_confidence_threshold: float | None,
    num_workers: int,
    progress: bool,
    output_dir: str,
    no_report: bool,
    verbose: bool,
    log_level: str,
):
    """Evaluate model performance on the complete dataset and generate report."""
    setup_logging(verbose, log_level)
    try:
        # Validate inputs
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset directory not found: {dataset_path}")
            sys.exit(1)
        if not os.path.exists(labels_file):
            logger.error(f"Labels file not found: {labels_file}")
            sys.exit(1)

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"data/results/{timestamp}"
        else:
            output_dir = f"data/results/{output_dir}"

        logger.info(f"Results will be saved to: {output_dir}")

        # Initialize evaluator
        evaluator = DatasetEvaluator(
            dataset_path=dataset_path,
            labels_file=labels_file,
            model_size=model_size,
            backend=backend,
            solution=solution,
            device=device,
            num_workers=num_workers,
            confidence_threshold=threshold,
            frame_sample_rate=sample_rate,
            max_frames=max_frames,
            multiple_people_threshold=people_threshold,
            temporal_min_consecutive=temporal_min_consecutive,
            card_min_area_ratio=card_min_area_ratio,
            card_square_tolerance=card_square_tolerance,
            text_proximity_threshold=text_proximity_threshold,
            text_confidence_threshold=text_confidence_threshold,
            show_progress=progress,
        )

        # Run evaluation
        results = evaluator.evaluate()

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "evaluation_results.json")
        evaluator.save_results(results, results_file)

        logger.info(f"Evaluation complete. Results saved to: {results_file}")

        # Generate report unless disabled
        if not no_report:
            report_file = os.path.join(output_dir, "technical_report.md")
            report_generator = ReportGenerator(results_file)
            report_generator.generate_report(report_file)
            logger.info(f"Technical report generated: {report_file}")

        # Print summary
        click.echo(f"Accuracy: {results['accuracy']:.3f}")
        click.echo(f"Precision: {results['precision']:.3f}")
        click.echo(f"Recall: {results['recall']:.3f}")
        click.echo(f"F1-Score: {results['f1_score']:.3f}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)


@click.command()
@click.option("--results-file", "-r", required=True,
              help="Path to evaluation results JSON")
@click.option("--output-file", "-o", help="Output file for the report")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", default="INFO", help="Logging level")
def report(
    results_file: str,
    output_file: str | None,
    verbose: bool,
    log_level: str,
):
    """Generate a technical report from evaluation results."""
    setup_logging(verbose, log_level)

    try:
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            sys.exit(1)

        report_generator = ReportGenerator(results_file)
        report_content = report_generator.generate_report(output_file)

        if output_file is None:
            click.echo(report_content)
        else:
            logger.info(f"Report generated: {output_file}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)


@click.command()
@click.option("--camera-index", default=0, type=int, help="Camera index to use")
@click.option(
    "--backend",
    default="yolov8",
    type=click.Choice([
        "yolov8",
        "torchvision_frcnn",
        "torchvision_ssd",
        "torchvision_retinanet",
        "opencv_hog",
    ], case_sensitive=False),
    help="Detection backend to use",
)
@click.option("--model-size", default="n", help="Model size for the selected backend")
@click.option("--device", default=None, help="Computation device: cpu or cuda")
@click.option("--threshold", default=0.5, help="Detection confidence threshold")
@click.option("--sample-rate", default=1, type=int, help="Process every Nth frame")
@click.option(
    "--show-confidence/--no-show-confidence",
    default=True,
    help="Show confidence scores on bounding boxes",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", default="INFO", help="Logging level")
def demo(
    camera_index: int,
    backend: str,
    model_size: str,
    device: str | None,
    threshold: float,
    sample_rate: int,
    show_confidence: bool,
    verbose: bool,
    log_level: str,
):
    """Run a real-time webcam demo with person detection."""
    setup_logging(verbose, log_level)

    try:
        run_webcam_demo(
            camera_index=camera_index,
            backend=backend,
            model_size=model_size,
            device=device,
            threshold=threshold,
            sample_rate=sample_rate,
            show_confidence=show_confidence,
        )
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error in demo: {e}")
        sys.exit(1)


# Add commands to the CLI group
cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(report)
cli.add_command(demo)

if __name__ == "__main__":
    cli()
