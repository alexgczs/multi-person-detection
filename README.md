# Multi-person detection for identity verification

This project addresses the fraud detection challenge of detecting when multiple people are present during a verification session (could indicate coercion or fraud).

## Current status

This is the initial implementation focusing on the core inference pipeline. The project currently provides:

- A working CLI that processes videos and outputs binary labels (0/1)
- YOLOv8-based person detection (using pre-trained COCO weights)
- Basic video processing with frame extraction

## Quick start

### Installation

```bash
# Clone and setup
git clone https://github.com/alexgczs/multi-person-detection.git
cd multi-person-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

Process a video and get the prediction:

```bash
python -m src.main predict -i path/to/video.mp4
```

The output will be printed to stdout as `label predicted: 0` or `label predicted: 1`.

You can also adjust the detection confidence threshold:

```bash
python -m src.main predict -i path/to/video.mp4 --threshold 0.7
```

## How it works

1. **Frame extraction**: Videos are processed frame by frame (currently sampling every frame up to 100 frames)
2. **Person detection**: Each frame is analyzed using YOLOv8 to detect people
3. **Aggregation**: The system counts how many frames contain multiple people
4. **Decision**: If the ratio of frames with multiple people exceeds a threshold, the video is classified as containing multiple people

## Technical details

- **Model**: YOLOv8n (nano) by default, but supports all sizes (n, s, m, l, x)
- **Detection**: COCO-trained model detecting "person" class (class 0)
- **Threshold**: Currently set to 0.0 (any frame with >1 person triggers classification)
- **Frame processing**: 640x480 resolution, maintaining aspect ratio

## Project structure

```
src/
├── models/
│   └── person_detector.py    # Main detection logic
├── utils/
│   ├── config.py            # Configuration management
│   └── video_processor.py   # Video frame extraction
└── main.py                  # CLI interface

tests/                       # Unit and integration tests
data/                        # Dataset and results storage
```

## Development

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=src
```

## Next steps

- [x] Inference pipeline (logic of detect multiple persons)
- [ ] Performance evaluation on the provided dataset
- [ ] Generalization of models
- [ ] Training pipeline for fit models to the task
- [ ] Analysis
- [ ] Extras (depending on the time)
- [ ] Report
- [ ] Next steps with more time and data


## Notes

- GPU acceleration is supported but not required