"""
Tests for dataset evaluator module.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src.utils.dataset_evaluator import DatasetEvaluator


class TestDatasetEvaluator(unittest.TestCase):
    """Test cases for DatasetEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = self.temp_dir
        self.labels_file = os.path.join(self.temp_dir, "labels.txt")

        # Create test labels
        labels_df = pd.DataFrame(
            {
                "video": ["test1", "test2", "test3", "test4"],
                "label": [0, 0, 1, 1],
            }
        )
        labels_df.to_csv(self.labels_file, sep="\t", index=False)

        # Create mock video files
        for video_name in ["test1", "test2", "test3", "test4"]:
            video_path = os.path.join(self.dataset_path, f"{video_name}.mp4")
            with open(video_path, "w") as f:
                f.write("mock video content")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test DatasetEvaluator initialization."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
                model_size="n",
                confidence_threshold=0.5,
                num_workers=1,
            )

            self.assertEqual(evaluator.dataset_path, self.dataset_path)
            self.assertEqual(evaluator.labels_file, self.labels_file)
            self.assertEqual(evaluator.model_size, "n")
            self.assertEqual(evaluator.confidence_threshold, 0.5)
            self.assertEqual(len(evaluator.ground_truth), 4)

    def test_load_labels(self):
        """Test label loading functionality."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
                num_workers=1,
            )

            expected_labels = {"test1": 0, "test2": 0, "test3": 1, "test4": 1}
            self.assertEqual(evaluator.ground_truth, expected_labels)

    def test_load_labels_error(self):
        """Test label loading with error."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            with self.assertRaises(Exception):
                DatasetEvaluator(
                    dataset_path=self.dataset_path,
                    labels_file="nonexistent.txt",
                    num_workers=1,
                )

    def test_load_labels_missing_columns(self):
        """Labels file missing required columns should raise ValueError."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            bad_labels = os.path.join(self.dataset_path, "bad_labels.txt")
            # Missing required columns 'video' and/or 'label'
            df = pd.DataFrame({"foo": ["a"], "bar": [1]})
            df.to_csv(bad_labels, sep="\t", index=False)

            with self.assertRaises(ValueError):
                DatasetEvaluator(
                    dataset_path=self.dataset_path,
                    labels_file=bad_labels,
                    num_workers=1,
                )

    def test_load_labels_invalid_label_values(self):
        """Labels outside {0,1} should raise ValueError."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            bad_labels = os.path.join(self.dataset_path, "bad_values.txt")
            df = pd.DataFrame({"video": ["v1", "v2"], "label": [2, -1]})
            df.to_csv(bad_labels, sep="\t", index=False)

            with self.assertRaises(ValueError):
                DatasetEvaluator(
                    dataset_path=self.dataset_path,
                    labels_file=bad_labels,
                    num_workers=1,
                )

    def test_load_labels_duplicate_videos(self):
        """Duplicate video entries should raise ValueError."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            dup_labels = os.path.join(self.dataset_path, "dup_labels.txt")
            df = pd.DataFrame({"video": ["v1", "v1"], "label": [0, 1]})
            df.to_csv(dup_labels, sep="\t", index=False)

            with self.assertRaises(ValueError):
                DatasetEvaluator(
                    dataset_path=self.dataset_path,
                    labels_file=dup_labels,
                    num_workers=1,
                )

    def test_evaluate_success(self):
        """Test successful evaluation."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.predict.return_value = {
                "has_multiple_people": True,
                "num_people": 2,
                "max_people": 2,
                "multiple_people_ratio": 1.0,
                "frames_with_multiple": 100,
                "total_frames": 100,
            }
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
                num_workers=1,
            )

            results = evaluator.evaluate()

            self.assertEqual(results["total_videos"], 4)
            self.assertIn("accuracy", results)
            self.assertIn("precision", results)
            self.assertIn("recall", results)
            self.assertIn("f1_score", results)
            self.assertIn("confusion_matrix", results)
            self.assertIn("classification_report", results)
            self.assertEqual(len(results["video_results"]), 4)

    def test_evaluate_with_missing_video(self):
        """Test evaluation with missing video file."""
        # Remove one video file
        os.remove(os.path.join(self.dataset_path, "test1.mp4"))

        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.predict.return_value = {
                "has_multiple_people": True,
                "num_people": 2,
                "max_people": 2,
                "multiple_people_ratio": 1.0,
                "frames_with_multiple": 100,
                "total_frames": 100,
            }
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
                num_workers=1,
            )

            results = evaluator.evaluate()

            # Should process only 3 videos (test1.mp4 is missing)
            self.assertEqual(results["total_videos"], 3)

    def test_evaluate_with_prediction_error(self):
        """Test evaluation when prediction fails for one video."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            # Make prediction fail for test1
            mock_detector.predict.side_effect = lambda **kwargs: (
                Exception("Prediction error")
                if "test1.mp4" in kwargs.get("video_path", "")
                else {
                    "has_multiple_people": True,
                    "num_people": 2,
                    "max_people": 2,
                    "multiple_people_ratio": 1.0,
                    "frames_with_multiple": 100,
                    "total_frames": 100,
                }
            )
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
                num_workers=1,
            )

            results = evaluator.evaluate()

            # Should process only 3 videos (test1 failed)
            self.assertEqual(results["total_videos"], 3)

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
            )

            true_labels = [0, 0, 1, 1]
            predictions = [0, 1, 1, 0]  # 2 correct, 2 incorrect

            metrics = evaluator._calculate_metrics(true_labels, predictions)

            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1_score", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("classification_report", metrics)

            # With 2 correct out of 4, accuracy should be 0.5
            self.assertEqual(metrics["accuracy"], 0.5)

    def test_calculate_metrics_with_errors(self):
        """Test metrics calculation with all errors."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
            )

            true_labels = [0, 0, 1, 1]
            predictions = [1, 1, 0, 0]  # All incorrect

            metrics = evaluator._calculate_metrics(true_labels, predictions)

            # With 0 correct out of 4, accuracy should be 0.0
            self.assertEqual(metrics["accuracy"], 0.0)

    def test_save_results(self):
        """Test saving results to file."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
            )

            results = {
                "total_videos": 4,
                "accuracy": 0.75,
                "precision": 0.8,
                "recall": 0.7,
                "f1_score": 0.75,
                "confusion_matrix": [[2, 1], [0, 1]],
                "classification_report": {"test": "report"},
                "video_results": [],
                "true_labels": [0, 0, 1, 1],
                "predictions": [0, 1, 1, 1],
            }

            output_file = os.path.join(self.temp_dir, "results.json")
            evaluator.save_results(results, output_file)

            # Check that file was created
            self.assertTrue(os.path.exists(output_file))

            # Check file content
            with open(output_file, "r") as f:
                saved_results = json.load(f)

            self.assertEqual(saved_results["total_videos"], 4)
            self.assertEqual(saved_results["accuracy"], 0.75)

    def test_save_results_error(self):
        """Test saving results with error."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
            )

            results = {"test": "data"}

            # Try to save to a directory that doesn't exist
            with self.assertRaises(Exception):
                evaluator.save_results(results, "/nonexistent/path/results.json")

    def test_get_error_analysis(self):
        """Test error analysis functionality."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
            )

            results = {
                "video_results": [
                    {
                        "video_name": "test1",
                        "true_label": 0,
                        "predicted_label": 1,
                        "correct": False,
                    },
                    {
                        "video_name": "test2",
                        "true_label": 1,
                        "predicted_label": 0,
                        "correct": False,
                    },
                    {
                        "video_name": "test3",
                        "true_label": 1,
                        "predicted_label": 1,
                        "correct": True,
                    },
                ]
            }

            error_analysis = evaluator.get_error_analysis(results)

            self.assertEqual(error_analysis["total_errors"], 2)
            self.assertEqual(error_analysis["false_positives"], 1)
            self.assertEqual(error_analysis["false_negatives"], 1)
            self.assertEqual(error_analysis["false_positive_videos"], ["test1"])
            self.assertEqual(error_analysis["false_negative_videos"], ["test2"])

    def test_get_error_analysis_no_errors(self):
        """Test error analysis with no errors."""
        with patch("src.utils.dataset_evaluator.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            evaluator = DatasetEvaluator(
                dataset_path=self.dataset_path,
                labels_file=self.labels_file,
            )

            results = {
                "video_results": [
                    {
                        "video_name": "test1",
                        "true_label": 0,
                        "predicted_label": 0,
                        "correct": True,
                    },
                    {
                        "video_name": "test2",
                        "true_label": 1,
                        "predicted_label": 1,
                        "correct": True,
                    },
                ]
            }

            error_analysis = evaluator.get_error_analysis(results)

            self.assertEqual(error_analysis["total_errors"], 0)
            self.assertEqual(error_analysis["false_positives"], 0)
            self.assertEqual(error_analysis["false_negatives"], 0)
            self.assertEqual(error_analysis["false_positive_videos"], [])
            self.assertEqual(error_analysis["false_negative_videos"], [])


if __name__ == "__main__":
    unittest.main()
