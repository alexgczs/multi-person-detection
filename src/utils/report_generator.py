"""
Report generator for evaluation results.

Creates technical reports from evaluation results.
"""

import json
import os
from typing import Any, Dict, List


class ReportGenerator:
    """Generates technical reports from evaluation results."""

    def __init__(self, results_file: str):
        """Initialize with path to results file."""
        self.results_file = results_file
        self.results = self._load_results()

    def _load_results(self) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        try:
            with open(self.results_file, "r") as f:
                results = json.load(f)
            return results
        except Exception as e:
            raise ValueError(f"Error loading results from {self.results_file}: {e}")

    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive technical report."""
        report = []

        # Title
        report.append("# Multi-person detection - Technical evaluation report")
        report.append("")
        report.append("*Report generated automatically from evaluation results*")
        report.append("")

        # Executive summary
        report.append("## Executive summary")
        report.append("")
        report.append(
            f"- **Model performance**: {self.results['accuracy']:.1%} accuracy"
        )
        report.append(f"- **Dataset size**: {self.results['total_videos']} videos")

        class_dist = self._get_class_distribution()
        report.append(
            f"- **Class distribution**: {class_dist[0]} single-person, "
            f"{class_dist[1]} multiple-person"
        )
        report.append("")

        # Performance metrics
        report.append("## Performance metrics")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Accuracy | {self.results['accuracy']:.3f} |")
        report.append(f"| Precision | {self.results['precision']:.3f} |")
        report.append(f"| Recall | {self.results['recall']:.3f} |")
        report.append(f"| F1-Score | {self.results['f1_score']:.3f} |")
        report.append("")

        # Confusion matrix
        report.append("## Confusion matrix")
        report.append("")
        report.append("| | Predicted 0 | Predicted 1 |")
        report.append("|---------|-------------|-------------|")
        cm = self.results["confusion_matrix"]
        report.append(f"| **Actual 0** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |")
        report.append(f"| **Actual 1** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |")
        report.append("")

        # Error analysis
        report.append("## Error analysis")
        report.append("")
        total_errors = cm[0][1] + cm[1][0]  # FP + FN
        report.append(f"- **Total errors**: {total_errors}")
        report.append(f"- **False positives**: {cm[0][1]}")
        report.append(f"- **False negatives**: {cm[1][0]}")
        report.append("")

        # Detailed error analysis
        error_analysis = self._analyze_errors()
        if error_analysis["false_positives"]:
            report.append("### False positive videos (predicted 1, true 0)")
            for video in error_analysis["false_positives"]:
                report.append(f"- `{video}`")
            report.append("")

        if error_analysis["false_negatives"]:
            report.append("### False negative videos (predicted 0, true 1)")
            for video in error_analysis["false_negatives"]:
                report.append(f"- `{video}`")
            report.append("")

        # Detailed error patterns
        report.append("## Detailed error analysis")
        report.append("")
        report.append("### Error patterns analysis")
        report.append("")

        if error_analysis["false_positives"]:
            report.append("#### False positives analysis")
            avg_fp_ratio = sum(
                vr["multiple_people_ratio"]
                for vr in error_analysis["false_positive_details"]
            ) / len(error_analysis["false_positive_details"])
            report.append(f"- **Average multiple people ratio**: {avg_fp_ratio:.3f}")
            report.append("- Videos with low ratios but classified as multiple people")
            report.append("")

        if error_analysis["false_negatives"]:
            report.append("#### False negatives analysis")
            avg_fn_ratio = sum(
                vr["multiple_people_ratio"]
                for vr in error_analysis["false_negative_details"]
            ) / len(error_analysis["false_negative_details"])
            report.append(f"- **Average multiple people ratio**: {avg_fn_ratio:.3f}")
            report.append("- Videos with multiple people but not detected")
            report.append("")

        # Model behavior analysis
        report.append("## Model behavior analysis")
        report.append("")
        report.append("### Multiple people ratio statistics")
        report.append("")

        ratios = [vr["multiple_people_ratio"] for vr in self.results["video_results"]]
        report.append(f"- **Mean**: {sum(ratios) / len(ratios):.3f}")
        report.append(f"- **Min**: {min(ratios):.3f}")
        report.append(f"- **Max**: {max(ratios):.3f}")
        report.append("")

        # Correct vs incorrect predictions
        correct_predictions = [
            vr for vr in self.results["video_results"] if vr["correct"]
        ]
        incorrect_predictions = [
            vr for vr in self.results["video_results"] if not vr["correct"]
        ]

        if correct_predictions:
            avg_correct_ratio = sum(
                vr["multiple_people_ratio"] for vr in correct_predictions
            ) / len(correct_predictions)
            report.append(
                f"- **Average ratio for correct predictions**: "
                f"{avg_correct_ratio:.3f}"
            )

        if incorrect_predictions:
            avg_incorrect_ratio = sum(
                vr["multiple_people_ratio"] for vr in incorrect_predictions
            ) / len(incorrect_predictions)
            report.append(
                f"- **Average ratio for incorrect predictions**: "
                f"{avg_incorrect_ratio:.3f}"
            )

        report.append("")

        report_text = "\n".join(report)

        # Save to file if specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)

        return report_text

    def _get_class_distribution(self) -> List[int]:
        """Get distribution of true labels."""
        true_labels = self.results["true_labels"]
        class_0 = sum(1 for label in true_labels if label == 0)
        class_1 = sum(1 for label in true_labels if label == 1)
        return [class_0, class_1]

    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze false positives and false negatives."""
        false_positives = []
        false_negatives = []
        false_positive_details = []
        false_negative_details = []

        for video_result in self.results["video_results"]:
            if not video_result["correct"]:
                if (
                    video_result["true_label"] == 0
                    and video_result["predicted_label"] == 1
                ):
                    false_positives.append(video_result["video_name"])
                    false_positive_details.append(video_result)
                elif (
                    video_result["true_label"] == 1
                    and video_result["predicted_label"] == 0
                ):
                    false_negatives.append(video_result["video_name"])
                    false_negative_details.append(video_result)

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "false_positive_details": false_positive_details,
            "false_negative_details": false_negative_details,
        }
