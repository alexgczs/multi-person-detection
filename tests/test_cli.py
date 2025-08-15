"""
Tests for CLI interface.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import click.testing
import pandas as pd

from src.main import cli


class TestCLI(unittest.TestCase):
    """Test cases for CLI commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = click.testing.CliRunner()

    def test_cli_predict_label_one(self):
        """Test predict command with label 1."""
        with patch("src.main.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.predict.return_value = {"has_multiple_people": True}
            mock_detector_class.return_value = mock_detector

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name

            try:
                result = self.runner.invoke(cli, ["predict", "-i", video_path])
                self.assertEqual(result.exit_code, 0)
                self.assertIn("label predicted: 1", result.output)
            finally:
                os.unlink(video_path)

    def test_cli_predict_label_zero(self):
        """Test predict command with label 0."""
        with patch("src.main.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.predict.return_value = {"has_multiple_people": False}
            mock_detector_class.return_value = mock_detector

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name

            try:
                result = self.runner.invoke(cli, ["predict", "-i", video_path])
                self.assertEqual(result.exit_code, 0)
                self.assertIn("label predicted: 0", result.output)
            finally:
                os.unlink(video_path)

    def test_cli_predict_file_not_found(self):
        """Test predict command with non-existent file."""
        result = self.runner.invoke(cli, ["predict", "-i", "nonexistent.mp4"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Video file not found", result.output)

    def test_cli_predict_with_threshold_and_verbose(self):
        """Test predict command with threshold and verbose options."""
        with patch("src.main.PersonDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.predict.return_value = {"has_multiple_people": False}
            mock_detector_class.return_value = mock_detector

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name

            try:
                result = self.runner.invoke(
                    cli, ["-v", "predict", "-i", video_path, "-t", "0.7"]
                )
                self.assertEqual(result.exit_code, 0)
                self.assertIn("label predicted: 0", result.output)
            finally:
                os.unlink(video_path)

    def test_cli_predict_detector_init_error(self):
        """Test predict command when detector initialization fails."""
        with patch("src.main.PersonDetector", side_effect=Exception("Init error")):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name

            try:
                result = self.runner.invoke(cli, ["predict", "-i", video_path])
                self.assertEqual(result.exit_code, 1)
                self.assertIn("Error processing video", result.output)
            finally:
                os.unlink(video_path)

    def test_cli_evaluate_success(self):
        """Test evaluate command with successful execution."""
        # Create temporary dataset and labels
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock video file
            video_path = os.path.join(temp_dir, "test_video.mp4")
            with open(video_path, "w") as f:
                f.write("mock video content")

            # Create labels file
            labels_file = os.path.join(temp_dir, "labels.txt")
            labels_df = pd.DataFrame({"video": ["test_video"], "label": [1]})
            labels_df.to_csv(labels_file, sep="\t", index=False)

            with patch("src.main.DatasetEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.evaluate.return_value = {
                    "total_videos": 1,
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1_score": 1.0,
                    "confusion_matrix": [[1, 0], [0, 0]],
                }
                mock_evaluator.save_results.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                with patch("src.main.ReportGenerator") as mock_report_class:
                    mock_report = Mock()
                    mock_report.generate_report.return_value = "Mock report"
                    mock_report_class.return_value = mock_report

                    result = self.runner.invoke(
                        cli, ["evaluate", "-d", temp_dir, "-l", labels_file]
                    )

                    self.assertEqual(result.exit_code, 0)
                    self.assertIn("DATASET EVALUATION RESULTS", result.output)
                    self.assertIn("Accuracy: 1.000", result.output)

    def test_cli_evaluate_dataset_not_found(self):
        """Test evaluate command with non-existent dataset."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            labels_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["evaluate", "-d", "nonexistent", "-l", labels_file]
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Dataset directory not found", result.output)
        finally:
            os.unlink(labels_file)

    def test_cli_evaluate_labels_not_found(self):
        """Test evaluate command with non-existent labels file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(
                cli, ["evaluate", "-d", temp_dir, "-l", "nonexistent.txt"]
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Labels file not found", result.output)

    def test_cli_evaluate_with_custom_output_dir(self):
        """Test evaluate command with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock video file
            video_path = os.path.join(temp_dir, "test_video.mp4")
            with open(video_path, "w") as f:
                f.write("mock video content")

            # Create labels file
            labels_file = os.path.join(temp_dir, "labels.txt")
            labels_df = pd.DataFrame({"video": ["test_video"], "label": [1]})
            labels_df.to_csv(labels_file, sep="\t", index=False)

            with patch("src.main.DatasetEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.evaluate.return_value = {
                    "total_videos": 1,
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1_score": 1.0,
                    "confusion_matrix": [[1, 0], [0, 0]],
                }
                mock_evaluator.save_results.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                with patch("src.main.ReportGenerator") as mock_report_class:
                    mock_report = Mock()
                    mock_report.generate_report.return_value = "Mock report"
                    mock_report_class.return_value = mock_report

                    result = self.runner.invoke(
                        cli,
                        [
                            "evaluate",
                            "-d",
                            temp_dir,
                            "-l",
                            labels_file,
                            "-o",
                            "custom_output",
                        ],
                    )

                    self.assertEqual(result.exit_code, 0)
                    self.assertIn("DATASET EVALUATION RESULTS", result.output)

    def test_cli_evaluate_no_report(self):
        """Test evaluate command with --no-report flag."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock video file
            video_path = os.path.join(temp_dir, "test_video.mp4")
            with open(video_path, "w") as f:
                f.write("mock video content")

            # Create labels file
            labels_file = os.path.join(temp_dir, "labels.txt")
            labels_df = pd.DataFrame({"video": ["test_video"], "label": [1]})
            labels_df.to_csv(labels_file, sep="\t", index=False)

            with patch("src.main.DatasetEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.evaluate.return_value = {
                    "total_videos": 1,
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1_score": 1.0,
                    "confusion_matrix": [[1, 0], [0, 0]],
                }
                mock_evaluator.save_results.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                result = self.runner.invoke(
                    cli,
                    [
                        "evaluate",
                        "-d",
                        temp_dir,
                        "-l",
                        labels_file,
                        "--no-report",
                    ],
                )

                self.assertEqual(result.exit_code, 0)
                self.assertIn("Report generation skipped", result.output)

    def test_cli_report_success(self):
        """Test report command with successful execution."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            results_file = f.name
            # Write mock results
            import json

            mock_results = {
                "total_videos": 1,
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "confusion_matrix": [[1, 0], [0, 0]],
                "true_labels": [1],
                "video_results": [
                    {
                        "video_name": "test",
                        "true_label": 1,
                        "predicted_label": 1,
                        "correct": True,
                        "multiple_people_ratio": 0.5,
                    }
                ],
            }
            json.dump(mock_results, f)

        try:
            with patch("src.main.ReportGenerator") as mock_report_class:
                mock_report = Mock()
                mock_report.generate_report.return_value = "Mock report content"
                mock_report_class.return_value = mock_report

                result = self.runner.invoke(cli, ["report", "-r", results_file])
                self.assertEqual(result.exit_code, 0)
                self.assertIn("Mock report content", result.output)
        finally:
            os.unlink(results_file)

    def test_cli_report_file_not_found(self):
        """Test report command with non-existent results file."""
        result = self.runner.invoke(cli, ["report", "-r", "nonexistent.json"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Results file not found", result.output)

    def test_cli_report_with_output_file(self):
        """Test report command with output file."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            results_file = f.name
            # Write mock results
            import json

            mock_results = {
                "total_videos": 1,
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "confusion_matrix": [[1, 0], [0, 0]],
                "true_labels": [1],
                "video_results": [
                    {
                        "video_name": "test",
                        "true_label": 1,
                        "predicted_label": 1,
                        "correct": True,
                        "multiple_people_ratio": 0.5,
                    }
                ],
            }
            json.dump(mock_results, f)

        try:
            with patch("src.main.ReportGenerator") as mock_report_class:
                mock_report = Mock()
                mock_report.generate_report.return_value = "Mock report content"
                mock_report_class.return_value = mock_report

                result = self.runner.invoke(
                    cli, ["report", "-r", results_file, "-o", "output.md"]
                )
                self.assertEqual(result.exit_code, 0)
                self.assertIn("Mock report content", result.output)
                self.assertIn("Report saved to: output.md", result.output)
        finally:
            os.unlink(results_file)

    def test_cli_verbose_mode(self):
        """Test CLI with verbose mode."""
        result = self.runner.invoke(cli, ["--verbose", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_cli_log_level(self):
        """Test CLI with custom log level."""
        result = self.runner.invoke(cli, ["--log-level", "DEBUG", "--help"])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
