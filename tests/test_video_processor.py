"""
Tests for VideoProcessor utilities (frames extraction and resizing).
"""

from unittest.mock import Mock, patch

import numpy as np

from src.utils.video_processor import VideoProcessor


@patch("src.utils.video_processor.cv2.VideoCapture")
def test_extract_frames_sampling_and_max(mock_capture_cls):
    """Test that extract_frames samples frames and respects max_frames."""
    # Mock VideoCapture behaviour
    mock_cap = Mock()
    # Simulate 10 reads then end
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
    reads = [(True, f) for f in frames] + [(False, None)]
    mock_cap.read.side_effect = reads
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {1: len(frames), 5: 30.0}.get(
        prop, 0
    )  # minimal
    mock_capture_cls.return_value = mock_cap

    processor = VideoProcessor()
    # sample_rate=2 -> expect 5 frames, max_frames=3 -> expect 3
    out_frames = processor.extract_frames("dummy.mp4", sample_rate=2, max_frames=3)
    assert len(out_frames) == 3
    # Each frame should be numpy array
    assert all(isinstance(f, np.ndarray) for f in out_frames)


@patch("src.utils.video_processor.cv2.VideoCapture")
def test_extract_frames_resize_path(mock_capture_cls):
    """Test that extract_frames resizes frames to the correct size."""
    mock_cap = Mock()
    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
    reads = [(True, f) for f in frames] + [(False, None)]
    mock_cap.read.side_effect = reads
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {1: len(frames), 5: 30.0}.get(prop, 0)
    mock_capture_cls.return_value = mock_cap

    processor = VideoProcessor()
    out_frames = processor.extract_frames("dummy.mp4", sample_rate=1, max_frames=2)
    assert len(out_frames) == 2
    assert out_frames[0].shape[0] == processor.config.FRAME_HEIGHT
    assert out_frames[0].shape[1] == processor.config.FRAME_WIDTH


def test_resize_frame_preserves_aspect_and_pads():
    """Test that _resize_frame preserves aspect ratio and pads."""
    processor = VideoProcessor()
    # Create non-matching frame 100x300; target 640x480 -> scale and pad
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    resized = processor._resize_frame(frame)
    assert resized.shape == (
        processor.config.FRAME_HEIGHT,
        processor.config.FRAME_WIDTH,
        3,
    )
