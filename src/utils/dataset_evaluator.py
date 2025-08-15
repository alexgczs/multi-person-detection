"""
Dataset evaluation utilities.

Evaluates model performance on complete datasets.
"""

import json
import os
from typing import Dict, List

import pandas as pd
from loguru import logger
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from src.models.person_detector import PersonDetector


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

        # Setup detector and load labels
        self.detector = PersonDetector(model_size=model_size)
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

        # Go through each video
        for video_name, true_label in self.ground_truth.items():
            video_path = os.path.join(self.dataset_path, f"{video_name}.mp4")

            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                continue

            try:
                logger.info(f"Processing video: {video_name}")

                # Get model prediction
                result = self.detector.predict(
                    video_path=video_path,
                    confidence_threshold=self.confidence_threshold,
                )

                predicted_label = int(result["has_multiple_people"])

                # Save results
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

                logger.info(
                    f"{video_name}: True={true_label}, Pred={predicted_label}, "
                    f"Correct={video_result['correct']}"
                )

            except Exception as e:
                logger.error(f"Error processing video {video_name}: {e}")
                continue

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
