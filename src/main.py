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
@click.option("--device", default=None, help="Computation device: cpu or cuda")
@click.option("--sample-rate", default=None, type=int, help="Process every Nth frame")
@click.option("--max-frames", default=None, type=int, help="Maximum frames per video")
@click.option(
    "--people-threshold",
    default=None,
    type=float,
    help="Ratio of frames with >1 person to classify as multi-person",
)
def predict(
    video: str,
    threshold: float,
    model_size: str,
    backend: str,
    device: str | None,
    sample_rate: int | None,
    max_frames: int | None,
    people_threshold: float | None,
):
    """Predict whether a video contains multiple people."""
    try:
        # Validate input
        if not os.path.exists(video):
            logger.error(f"Video file not found: {video}")
            sys.exit(1)

        # Setup detector and process video
        logger.info("Initializing person detector...")
        detector = PersonDetector(model_size=model_size, device=device, backend=backend)
        # Apply config overrides if provided
        if sample_rate is not None:
            detector.video_processor.config.FRAME_SAMPLE_RATE = int(sample_rate)
        if max_frames is not None:
            detector.video_processor.config.MAX_FRAMES = int(max_frames)
        if people_threshold is not None:
            detector.config.MULTIPLE_PEOPLE_THRESHOLD = float(people_threshold)

        logger.info(f"Processing video: {video}")
        result = detector.predict(video_path=video, confidence_threshold=threshold)

        # stdout output
        click.echo(f"label predicted: {int(result['has_multiple_people'])}")

        # Always exit 0 on successful execution; label is conveyed via logs/output
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)


@cli.command()
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
def evaluate(
    dataset_path: str,
    labels_file: str,
    threshold: float,
    model_size: str,
    backend: str,
    device: str | None,
    sample_rate: int | None,
    max_frames: int | None,
    people_threshold: float | None,
    num_workers: int,
    progress: bool,
    output_dir: str,
    no_report: bool,
):
    """Evaluate model performance on the complete dataset and generate report."""
    try:
        # Validate inputs
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset directory not found: {dataset_path}")
            sys.exit(1)

        if not os.path.exists(labels_file):
            logger.error(f"Labels file not found: {labels_file}")
            sys.exit(1)

        # Create output directory
        if output_dir:
            # Use custom directory name
            results_dir = f"data/results/{output_dir}"
        else:
            # Use timestamp as directory name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_dir = f"data/results/{timestamp}"

        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")

        # Setup evaluator and run evaluation
        logger.info("Initializing dataset evaluator...")
        evaluator = DatasetEvaluator(
            dataset_path=dataset_path,
            labels_file=labels_file,
            model_size=model_size,
            confidence_threshold=threshold,
            device=device,
            backend=backend,
            frame_sample_rate=sample_rate,
            max_frames=max_frames,
            multiple_people_threshold=people_threshold,
            num_workers=num_workers,
            show_progress=progress,
        )

        logger.info("Starting dataset evaluation...")
        results = evaluator.evaluate()

        # Print summary metrics
        click.echo("\n" + "=" * 50)
        click.echo("DATASET EVALUATION RESULTS")
        click.echo("=" * 50)
        click.echo(f"Total videos: {results['total_videos']}")
        click.echo(f"Accuracy: {results['accuracy']:.3f}")
        click.echo(f"Precision: {results['precision']:.3f}")
        click.echo(f"Recall: {results['recall']:.3f}")
        click.echo(f"F1-Score: {results['f1_score']:.3f}")
        click.echo("Confusion Matrix:")
        click.echo(
            f"  TN: {results['confusion_matrix'][0][0]}, "
            f"FP: {results['confusion_matrix'][0][1]}"
        )
        click.echo(
            f"  FN: {results['confusion_matrix'][1][0]}, "
            f"TP: {results['confusion_matrix'][1][1]}"
        )
        click.echo("=" * 50)

        # Save results and generate report
        results_file = os.path.join(results_dir, "evaluation_results.json")
        evaluator.save_results(results, results_file)
        click.echo(f"Detailed results saved to: {results_file}")

        if not no_report:
            logger.info("Generating technical report...")
            report_file = os.path.join(results_dir, "technical_report.md")
            generator = ReportGenerator(results_file)
            generator.generate_report(report_file)
            click.echo(f"Technical report saved to: {report_file}")
        else:
            click.echo("Report generation skipped (--no-report flag used)")

        # Save metadata
        metadata_file = os.path.join(results_dir, "evaluation_metadata.txt")
        with open(metadata_file, "w") as f:
            f.write(
                f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Dataset path: {dataset_path}\n")
            f.write(f"Labels file: {labels_file}\n")
            f.write(f"Model size: {model_size}\n")
            f.write(f"Confidence threshold: {threshold}\n")
            f.write(f"Total videos: {results['total_videos']}\n")
            f.write(f"Accuracy: {results['accuracy']:.3f}\n")
            f.write(f"Precision: {results['precision']:.3f}\n")
            f.write(f"Recall: {results['recall']:.3f}\n")
            f.write(f"F1-score: {results['f1_score']:.3f}\n")
        click.echo(f"Evaluation metadata saved to: {metadata_file}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error evaluating dataset: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--results-file", "-r", required=True, help="Path to evaluation results JSON file"
)
@click.option("--output-file", "-o", help="Path to save the report")
def report(results_file: str, output_file: str):
    """Generate a technical report from evaluation results.

    Use this command to regenerate reports from existing evaluation results
    without re-running the full evaluation. The 'evaluate' command automatically
    generates reports, so this is mainly useful for:
    - Regenerating reports with different parameters
    - Debugging report generation
    - Creating reports from partial evaluation results
    """
    try:
        # Validate input
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            sys.exit(1)

        # Setup generator and create report
        logger.info("Initializing report generator...")
        generator = ReportGenerator(results_file)

        logger.info("Generating technical report...")
        report_text = generator.generate_report(output_file)

        # Print report to console
        click.echo(report_text)

        if output_file:
            click.echo(f"\nReport saved to: {output_file}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)


@cli.command()
@click.option("--camera-index", default=0, type=int, help="Webcam index to open")
@click.option(
    "--backend",
    default="yolov8",
    type=click.Choice(
        [
            "yolov8",
            "torchvision_frcnn",
            "torchvision_ssd",
            "torchvision_retinanet",
            "opencv_hog",
        ],
        case_sensitive=False,
    ),
    help="Detection backend",
)
@click.option("--model-size", default="n", help="Model size for the selected backend")
@click.option("--device", default=None, help="Computation device: cpu or cuda")
@click.option("--threshold", "-t", default=0.5, type=float,
              help="Detection confidence threshold")
@click.option("--sample-rate", default=1, type=int,
              help="Process every Nth frame (1 = all)")
@click.option(
    "--show-confidence/--no-show-confidence",
    default=True,
    help="Draw confidence on boxes",
)
def demo(
    camera_index: int,
    backend: str,
    model_size: str,
    device: str | None,
    threshold: float,
    sample_rate: int,
    show_confidence: bool,
):
    """Run a real-time webcam demo drawing detected persons and counts."""
    exit_code = run_webcam_demo(
        camera_index,
        backend,
        model_size,
        device,
        threshold,
        sample_rate,
        show_confidence,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
