"""
CLI tests (to ensure CLI interface works).

Covers stdout label output and exit codes.
"""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from src.main import cli


@patch("src.main.PersonDetector")
def test_cli_predict_label_one(mock_detector_cls):
    runner = CliRunner()

    mock_detector = Mock()
    mock_detector.predict.return_value = {
        "has_multiple_people": True,
        "confidence": 0.9,
        "num_people": 2,
        "multiple_people_ratio": 1.0,
    }
    mock_detector_cls.return_value = mock_detector

    with runner.isolated_filesystem():
        # Create a dummy file to satisfy os.path.exists check
        with open("dummy.mp4", "wb") as f:
            f.write(b"")

        result = runner.invoke(
            cli,
            [
                "predict",
                "-i",
                "dummy.mp4",
            ],
        )

    # Click catches SystemExit; exit_code should be 0
    assert result.exit_code == 0
    # Check that output contains prediction format
    assert "label predicted" in result.output
    assert "1" in result.output


@patch("src.main.PersonDetector")
def test_cli_predict_label_zero(mock_detector_cls):
    runner = CliRunner()

    mock_detector = Mock()
    mock_detector.predict.return_value = {
        "has_multiple_people": False,
        "confidence": 0.1,
        "num_people": 1,
        "multiple_people_ratio": 0.0,
    }
    mock_detector_cls.return_value = mock_detector

    with runner.isolated_filesystem():
        with open("dummy.mp4", "wb") as f:
            f.write(b"")

        result = runner.invoke(
            cli,
            [
                "predict",
                "-i",
                "dummy.mp4",
            ],
        )

    assert result.exit_code == 0
    assert "label predicted" in result.output


@patch("src.main.PersonDetector")
def test_cli_predict_file_not_found(mock_detector_cls):
    runner = CliRunner()

    # Ensure detector is not even constructed due to file check
    result = runner.invoke(
        cli,
        [
            "predict",
            "-i",
            "nonexistent.mp4",
        ],
    )

    assert result.exit_code == 1
    # Should include error message
    assert "Video file not found" in result.output


@patch("src.main.PersonDetector")
def test_cli_predict_with_threshold_and_verbose(mock_detector_cls):
    runner = CliRunner()

    mock_detector = Mock()
    mock_detector.predict.return_value = {
        "has_multiple_people": False,
        "confidence": 0.4,
        "num_people": 1,
        "multiple_people_ratio": 0.0,
    }
    mock_detector_cls.return_value = mock_detector

    with runner.isolated_filesystem():
        with open("v.mp4", "wb") as f:
            f.write(b"")

        result = runner.invoke(
            cli,
            [
                "-v",
                "predict",
                "-i",
                "v.mp4",
                "-t",
                "0.7",
            ],
        )

    assert result.exit_code == 0
    # Check that output contains prediction format
    assert "label predicted" in result.output
    assert "0" in result.output
    # Verbose mode should log this line
    assert "Verbose mode enabled" in result.output


@patch("src.main.PersonDetector", side_effect=Exception("init error"))
def test_cli_predict_detector_init_error(_):
    runner = CliRunner()

    with runner.isolated_filesystem():
        with open("v.mp4", "wb") as f:
            f.write(b"")

        result = runner.invoke(
            cli,
            [
                "predict",
                "-i",
                "v.mp4",
            ],
        )

    assert result.exit_code == 1
    assert "Error processing video" in result.output
