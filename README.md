# Multi-person detection for identity verification

[![Build](https://github.com/alexgczs/multi-person-detection/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/alexgczs/multi-person-detection/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alexgczs/multi-person-detection/branch/master/graph/badge.svg)](https://codecov.io/gh/alexgczs/multi-person-detection)

This project addresses the fraud detection challenge of detecting when multiple people are present during a verification session (could indicate coercion or fraud).

## Current status

This is the initial implementation focusing on the core inference pipeline. The project currently provides:

- A working CLI that processes videos and outputs binary labels (0/1)
- YOLOv8-based person detection (using pre-trained COCO weights)
- Basic video processing with frame extraction
- Dataset evaluation with performance metrics
- Technical report generation

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

### Unique prediction

Get a prediction over one unique video:

```bash
python -m src.main predict -i path/to/video.mp4 \
  --threshold 0.5 \
  --model-size n \
  --device cuda \
  --sample-rate 1 \
  --max-frames 100 \
  --people-threshold 0.2
```

The output will be printed to stdout as `label predicted: 0` or `label predicted: 1`.

You can also adjust the detection confidence threshold:

```bash
python -m src.main predict -i path/to/video.mp4 --threshold 0.7
```

#### Options

- **--video, -i**: Path to the input video file. Required.
- **--threshold, -t**: Minimum confidence for person detections. Default: 0.5.
- **--model-size**: YOLO model size to use (`n`, `s`, `m`, `l`, `x`). Default: `n`.
- **--device**: Compute device (`cpu` or `cuda`). Default: auto-detect.
- **--sample-rate**: Process every Nth frame (1 = every frame). Default: 1.
- **--max-frames**: Maximum number of frames to process per video. Default: 100.
- **--people-threshold**: Ratio of frames with >1 person to classify as multi-person. Default: 0.0.

### Dataset evaluation

Evaluate the model on a complete dataset:

```bash
python -m src.main evaluate -d path/to/dataset -l path/to/labels.txt \
  --threshold 0.5 --model-size n --device cuda \
  --sample-rate 1 --max-frames 100 --people-threshold 0.2
```

This will:
- Process all videos in the dataset
- Compare predictions with ground truth labels
- Calculate performance metrics (accuracy, precision, recall, F1-score)
- Generate a technical report
- Save results to a timestamped directory

You can also specify a custom output directory:

```bash
python -m src.main evaluate -d path/to/dataset -l path/to/labels.txt -o my_evaluation
```

Or skip automatic report generation:

```bash
python -m src.main evaluate -d path/to/dataset -l path/to/labels.txt --no-report
```

#### Arguments

- **--dataset-path, -d**: Path to directory with `.mp4` videos (names must match labels). Required.
- **--labels-file, -l**: TSV file with columns `video` and `label` (0/1). Required.

#### Options

- **--threshold, -t**: Minimum confidence for person detections. Default: 0.5.
- **--model-size**: YOLO model size to use (`n`, `s`, `m`, `l`, `x`). Default: `n`.
- **--device**: Compute device (`cpu` or `cuda`). Default: auto-detect.
- **--sample-rate**: Process every Nth frame (1 = every frame). Default: 1.
- **--max-frames**: Maximum number of frames to process per video. Default: 100.
- **--people-threshold**: Ratio of frames with >1 person to classify as multi-person. Default: 0.0.
- **--num-workers**: Parallel workers for evaluation (per-video). Default: 1.
- **--progress/--no-progress**: Show or hide progress bar. Default: show.
- **--output-dir, -o**: Name for results directory under `data/results/`. Default: timestamp.
- **--no-report**: Skip automatic report generation. Default: off.

### Report generation

Generate a technical report from existing evaluation results:

```bash
python -m src.main report -r path/to/evaluation_results.json
```

This is useful for:
- Regenerating reports with different parameters
- Creating reports from partial evaluation results
- Debugging report generation

#### Arguments

- **--results-file, -r**: Path to a JSON produced by `evaluate`. Required.
- **--output-file, -o**: Where to save the generated markdown. If omitted, prints to stdout.

## How it works

1. **Frame extraction**: Videos are processed frame by frame
2. **Person detection**: Each frame is analyzed using YOLOv8 to detect people
3. **Aggregation**: The system counts how many frames contain multiple people
4. **Decision**: If the ratio of frames with multiple people exceeds a threshold, the video is classified as containing multiple people

## Technical details

- **Model**: YOLOv8n (nano) by default, but supports all sizes (n, s, m, l, x)
- **Detection**: COCO-trained model detecting "person" class (class 0)
- **Frame processing**: 640x480 resolution, maintaining aspect ratio

## Project structure

```
src/
├── models/
│   └── person_detector.py    # Main detection logic
├── utils/
│   ├── config.py            # Configuration management
│   ├── dataset_evaluator.py # Dataset evaluation utilities
│   ├── report_generator.py  # Technical report generation
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
- [x] Performance evaluation on the provided dataset
- [x] Technical report generation
- [ ] Generalization of models
- [ ] Training pipeline for fit models to the task
- [ ] Analysis
- [ ] Extras (depending on the time)
- [ ] Console script (CLI, entry point) -> toml
- [ ] MakeFile
- [ ] Report
- [ ] Next steps with more time and data


## Notes

- GPU acceleration is supported but not required