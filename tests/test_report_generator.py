"""
Tests for report generator module.
"""

import json
import os
import tempfile
import unittest

from src.utils.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class."""

    def setUp(self):
        """Set up test data."""
        self.test_results = {
            "total_videos": 4,
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.5,
            "confusion_matrix": [[1, 1], [1, 1]],
            "true_labels": [0, 0, 1, 1],
            "video_results": [
                {
                    "video_name": "test1",
                    "true_label": 0,
                    "predicted_label": 0,
                    "correct": True,
                    "multiple_people_ratio": 0.0,
                },
                {
                    "video_name": "test2",
                    "true_label": 0,
                    "predicted_label": 0,
                    "correct": True,
                    "multiple_people_ratio": 0.8,
                },
                {
                    "video_name": "test3",
                    "true_label": 0,
                    "predicted_label": 1,
                    "correct": False,
                    "multiple_people_ratio": 0.2,
                },
                {
                    "video_name": "test4",
                    "true_label": 1,
                    "predicted_label": 0,
                    "correct": False,
                    "multiple_people_ratio": 0.0,
                },
            ],
        }

    def test_initialization(self):
        """Test ReportGenerator initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_results, f)
            temp_file = f.name

        try:
            generator = ReportGenerator(temp_file)
            self.assertEqual(generator.results_file, temp_file)
            self.assertEqual(generator.results, self.test_results)
        finally:
            os.unlink(temp_file)

    def test_load_results_error(self):
        """Test error handling when loading results fails."""
        with self.assertRaises(ValueError):
            ReportGenerator("nonexistent_file.json")

    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_results, f)
            temp_file = f.name

        try:
            generator = ReportGenerator(temp_file)
            distribution = generator._get_class_distribution()
            self.assertEqual(distribution, [2, 2])
        finally:
            os.unlink(temp_file)

    def test_analyze_errors(self):
        """Test error analysis."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_results, f)
            temp_file = f.name

        try:
            generator = ReportGenerator(temp_file)
            error_analysis = generator._analyze_errors()

            self.assertEqual(len(error_analysis["false_positives"]), 1)
            self.assertEqual(len(error_analysis["false_negatives"]), 1)
            self.assertEqual(error_analysis["false_positives"][0], "test3")
            self.assertEqual(error_analysis["false_negatives"][0], "test4")
        finally:
            os.unlink(temp_file)

    def test_generate_report(self):
        """Test report generation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_results, f)
            temp_file = f.name

        try:
            generator = ReportGenerator(temp_file)
            report = generator.generate_report()

            self.assertIn(
                "# Multi-person detection - Technical evaluation report", report
            )
            self.assertIn("## Executive summary", report)
            self.assertIn("## Performance metrics", report)
            self.assertIn("## Confusion matrix", report)
            self.assertIn("## Error analysis", report)
            self.assertIn("**Total errors**: 2", report)
            self.assertIn("**False positives**: 1", report)
            self.assertIn("**False negatives**: 1", report)
        finally:
            os.unlink(temp_file)

    def test_generate_report_with_output_file(self):
        """Test report generation with output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_results, f)
            temp_file = f.name

        output_file = tempfile.mktemp(suffix=".md")

        try:
            generator = ReportGenerator(temp_file)
            report = generator.generate_report(output_file)

            # Check that file was created
            self.assertTrue(os.path.exists(output_file))

            # Check file content
            with open(output_file, "r") as f:
                saved_report = f.read()

            self.assertIn(
                "# Multi-person detection - Technical evaluation report", saved_report
            )
            self.assertEqual(report, saved_report)
        finally:
            os.unlink(temp_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_report_content_structure(self):
        """Test that report contains all expected sections."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_results, f)
            temp_file = f.name

        try:
            generator = ReportGenerator(temp_file)
            report = generator.generate_report()

            self.assertIn("**Total errors**: 2", report)
            self.assertIn("**False positives**: 1", report)
            self.assertIn("**False negatives**: 1", report)
            self.assertIn("### False positive videos", report)
            self.assertIn("### False negative videos", report)
            self.assertIn("## Detailed error analysis", report)
            self.assertIn("### Error patterns analysis", report)
            self.assertIn("#### False positives analysis", report)
            self.assertIn("#### False negatives analysis", report)
            self.assertIn("## Model behavior analysis", report)
            self.assertIn("### Multiple people ratio statistics", report)
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
