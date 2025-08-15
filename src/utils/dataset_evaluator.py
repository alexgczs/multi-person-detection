"""
Dataset evaluation utilities.

Evaluates model performance on complete datasets.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from src.models.person_detector import PersonDetector
from tqdm import tqdm


class DatasetEvaluator:
    """
    Evaluates model performance on the complete dataset.

    Loads ground truth labels and compares them with model predictions
    to calculate various performance metrics.
    """

    def __init__(
        self,
        dataset_path: str,
        labels_file: str,
        model_size: str = "n",
        confidence_threshold: float = 0.5,
        device: str | None = None,
        frame_sample_rate: int | None = None,
        max_frames: int | None = None,
        multiple_people_threshold: float | None = None,
        num_workers: int = 1,
        show_progress: bool = False,
    ):
        """
        Initialize the dataset evaluator.

        Args:
            dataset_path: Path to directory containing video files
            labels_file: Path to file containing ground truth labels
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Confidence threshold for detection
        """
        self.dataset_path = dataset_path
        self.labels_file = labels_file
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.frame_sample_rate = frame_sample_rate
        self.max_frames = max_frames
        self.multiple_people_threshold = multiple_people_threshold
        self.num_workers = max(1, int(num_workers))
        self.show_progress = bool(show_progress)

        # Setup detector and load labels
        self.detector = PersonDetector(model_size=model_size, device=device)

        # Apply overrides directly to detector config
        if frame_sample_rate is not None:
            self.detector.video_processor.config.FRAME_SAMPLE_RATE = int(frame_sample_rate)
        if max_frames is not None:
            self.detector.video_processor.config.MAX_FRAMES = int(max_frames)
        if multiple_people_threshold is not None:
            self.detector.config.MULTIPLE_PEOPLE_THRESHOLD = float(multiple_people_threshold)
        self.ground_truth = self._load_labels()

        logger.info(
            f"DatasetEvaluator initialized with {len(self.ground_truth)} videos"
        )

    def _load_labels(self) -> Dict[str, int]:
        """
        Load ground truth labels from the labels file.

        Returns:
            Dictionary mapping video names to labels (0 or 1)
        """
        try:
            # Read labels file
            df = pd.read_csv(self.labels_file, sep="\t")

            # Validate columns
            required_cols = {"video", "label"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(
                    f"labels file is missing required columns: {sorted(missing)}"
                )

            # Drop rows with NaNs in required columns
            df = df.dropna(subset=["video", "label"]).copy()

            # Normalize video names (strip whitespace)
            df["video"] = df["video"].astype(str).str.strip()

            # Enforce label integer and in {0,1}
            try:
                df["label"] = df["label"].astype(int)
            except Exception as e:
                raise ValueError(f"labels must be integers 0/1: {e}")
            invalid = df[~df["label"].isin([0, 1])]
            if not invalid.empty:
                raise ValueError(
                    f"labels must be 0 or 1; found invalid values: {sorted(invalid['label'].unique())}"
                )

            # Detect duplicates
            dups = df[df.duplicated(subset=["video"], keep=False)]["video"].unique()
            if len(dups) > 0:
                raise ValueError(
                    f"duplicate video entries in labels file: {sorted(map(str, dups))}"
                )

            # Create mapping from video name to label
            labels = {}
            for _, row in df.iterrows():
                video_name = row["video"]
                label = int(row["label"])
                labels[video_name] = label

            logger.info(f"Loaded {len(labels)} ground truth labels")
            logger.info(
                f"Label distribution: "
                f"{pd.Series(labels.values()).value_counts().to_dict()}"
            )

            return labels

        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            raise

    def evaluate(self) -> Dict:
        """
        Evaluate model performance on the complete dataset.

        Returns:
            Dictionary containing evaluation results and metrics
        """
        logger.info("Starting dataset evaluation...")

        predictions = []
        true_labels = []
        video_results = []

        items: List[Tuple[str, int, str]] = []
        for video_name, true_label in self.ground_truth.items():
            video_path = os.path.join(self.dataset_path, f"{video_name}.mp4")
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                continue
            items.append((video_name, true_label, video_path))

        # Progress bar disabled in non-interactive contexts
        progress_disable = not sys.stderr.isatty() or not self.show_progress

        if self.num_workers == 1:
            iterable = tqdm(items, desc="Evaluating", leave=False, disable=progress_disable)
            for video_name, true_label, video_path in iterable:
                try:
                    logger.info(f"Processing video: {video_name}")
                    result = self.detector.predict(
                        video_path=video_path,
                        confidence_threshold=self.confidence_threshold,
                    )
                    predicted_label = int(result["has_multiple_people"])
                    predictions.append(predicted_label)
                    true_labels.append(true_label)
                    video_result = {
                        "video_name": video_name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "correct": true_label == predicted_label,
                        "num_people": result["num_people"],
                        "max_people": result["max_people"],
                        "multiple_people_ratio": result["multiple_people_ratio"],
                        "frames_with_multiple": result["frames_with_multiple"],
                        "total_frames": result["total_frames"],
                    }
                    video_results.append(video_result)
                except Exception as e:
                    logger.error(f"Error processing video {video_name}: {e}")
                    continue
        else:
            # Parallel processing per video
            def _worker(args: Tuple[str, int, str]) -> Tuple[str, int, Dict]:
                vname, tlabel, vpath = args
                detector = PersonDetector(model_size=self.model_size, device=self.device)
                # Apply overrides
                if self.frame_sample_rate is not None:
                    detector.video_processor.config.FRAME_SAMPLE_RATE = int(self.frame_sample_rate)
                if self.max_frames is not None:
                    detector.video_processor.config.MAX_FRAMES = int(self.max_frames)
                if self.multiple_people_threshold is not None:
                    detector.config.MULTIPLE_PEOPLE_THRESHOLD = float(self.multiple_people_threshold)
                res = detector.predict(
                    video_path=vpath, confidence_threshold=self.confidence_threshold
                )
                return vname, tlabel, res

            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                futures = [ex.submit(_worker, it) for it in items]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Evaluating",
                    leave=False,
                    disable=progress_disable,
                ):
                    try:
                        video_name, true_label, result = fut.result()
                    except Exception as e:  # pragma: no cover (timing dependent)
                        logger.error(f"Error in worker: {e}")
                        continue
                    predicted_label = int(result["has_multiple_people"])
                    predictions.append(predicted_label)
                    true_labels.append(true_label)
                    video_result = {
                        "video_name": video_name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "correct": true_label == predicted_label,
                        "num_people": result["num_people"],
                        "max_people": result["max_people"],
                        "multiple_people_ratio": result["multiple_people_ratio"],
                        "frames_with_multiple": result["frames_with_multiple"],
                        "total_frames": result["total_frames"],
                    }
                    video_results.append(video_result)

        # Get metrics and compile results
        metrics = self._calculate_metrics(true_labels, predictions)

        results = {
            "total_videos": len(video_results),
            "video_results": video_results,
            "true_labels": true_labels,
            "predictions": predictions,
            **metrics,
        }

        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.3f}")
        return results

    def _calculate_metrics(
        self, true_labels: List[int], predictions: List[int]
    ) -> Dict:
        """
        Calculate performance metrics.

        Args:
            true_labels: List of true labels
            predictions: List of predicted labels

        Returns:
            Dictionary containing various performance metrics
        """
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        # Classification report
        # Zero division is used to avoid division by zero
        report = classification_report(
            true_labels, predictions, output_dict=True, zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def save_results(self, results: Dict, output_file: str):
        """
        Save detailed evaluation results to a JSON file.

        Args:
            results: Evaluation results dictionary
            output_file: Path to save the results
        """
        try:
            # Create a copy of results for saving (remove non-serializable objects)
            save_results = results.copy()

            # Convert numpy arrays to lists if present
            if "confusion_matrix" in save_results:
                save_results["confusion_matrix"] = results["confusion_matrix"]

            with open(output_file, "w") as f:
                json.dump(save_results, f, indent=2)

            logger.info(f"Results saved to: {output_file}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def get_error_analysis(self, results: Dict) -> Dict:
        """
        Analyze prediction errors to understand model weaknesses.

        Args:
            results: Evaluation results from evaluate() method

        Returns:
            Dictionary containing error analysis
        """
        video_results = results["video_results"]

        # Find incorrect predictions
        errors = [vr for vr in video_results if not vr["correct"]]

        # Analyze false positives (predicted 1, true 0)
        false_positives = [
            vr for vr in errors if vr["predicted_label"] == 1 and vr["true_label"] == 0
        ]

        # Analyze false negatives (predicted 0, true 1)
        false_negatives = [
            vr for vr in errors if vr["predicted_label"] == 0 and vr["true_label"] == 1
        ]

        error_analysis = {
            "total_errors": len(errors),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "false_positive_videos": [vr["video_name"] for vr in false_positives],
            "false_negative_videos": [vr["video_name"] for vr in false_negatives],
            "error_details": errors,
        }

        return error_analysis
