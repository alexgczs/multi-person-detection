# Multi-person detection for identity verification

[![Build](https://github.com/alexgczs/multi-person-detection/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/alexgczs/multi-person-detection/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alexgczs/multi-person-detection/graph/badge.svg)](https://codecov.io/gh/alexgczs/multi-person-detection)

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

# Install in development mode (recommended)
pip install -e .

# Install with development dependencies (for testing, linting, etc.)
pip install -e ".[dev]"

# Or install dependencies only (alternative)
pip install -r requirements.txt # only runtime dependencies
pip install -r requirements-dev.txt  # includes runtime + dev tools
```

**Note**: Installing with `pip install -e .` enables console scripts (`multi-predict`, `multi-evaluate`, etc.) for easier usage.


### Usage

The project provides both console scripts and Python module interfaces. Console scripts are recommended for easier usage.

#### Console Scripts (Recommended)

After installing with `pip install -e .`, you can use the following commands:

- `multi-predict` - Predict multi-person detection on a single video
- `multi-evaluate` - Evaluate model performance on a dataset
- `multi-report` - Generate technical reports from evaluation results
- `multi-demo` - Run real-time webcam demo

#### Python Module Interface (Alternative)

You can also use the Python module directly:

```bash
python -m src.main <command> [options]
```

### Single prediction

Get a prediction over one single video:

```bash
# Using console script (recommended)
multi-predict -i path/to/video.mp4 \
  --threshold 0.5 \
  --model-size n \
  --device cuda \
  --sample-rate 1 \
  --max-frames 100 \
  --people-threshold 0.2 \
  --solution counting

# Or using Python module (alternative)
python -m src.main predict -i path/to/video.mp4 \
  --threshold 0.5 \
  --model-size n \
  --device cuda \
  --sample-rate 1 \
  --max-frames 100 \
  --people-threshold 0.2 \
  --solution counting
```

The output will be printed to stdout as `label predicted: 0` or `label predicted: 1`.

You can also adjust the detection confidence threshold:

```bash
python -m src.main predict -i path/to/video.mp4 --threshold 0.7
```

#### Options

- **--video, -i**: Path to the input video file. Required.
- **--threshold, -t**: Minimum confidence for person detections. Default: 0.5.
- **--model-size**: Model size (only applied if backend is based on YOLO). Sizes: `n`, `s`, `m`, `l`, `x`. Default: `n`.
- **--backend**: Detection backend. Supported: `yolov8`, `torchvision_frcnn`, `torchvision_ssd`, `torchvision_retinanet`, `opencv_hog`. Default: `yolov8`.
- **--solution**: Video-level solution strategy. Supported: `counting`, `temporal` (hysteresis), `temporal_cardaware` (hysteresis + ID-card suppression). Default: `counting`.
- **--device**: Compute device (`cpu` or `cuda`). Default: auto-detect.
- **--sample-rate**: Process every Nth frame (1 = every frame). Default: 1.
- **--max-frames**: Maximum number of frames to process per video (None = all frames). Default: None.
- **--people-threshold**: Ratio of frames with >1 person to classify as multi-person. Default: 0.0.

### Dataset evaluation

Evaluate the model on a complete dataset:

```bash
# Using console script (recommended)
multi-evaluate -d path/to/dataset -l path/to/labels.txt \
  --threshold 0.5 --model-size n --device cuda \
  --sample-rate 1 --max-frames 100 --people-threshold 0.2 \
  --solution temporal

# Or using Python module (alternative)
python -m src.main evaluate -d path/to/dataset -l path/to/labels.txt \
  --threshold 0.5 --model-size n --device cuda \
  --sample-rate 1 --max-frames 100 --people-threshold 0.2 \
  --solution temporal
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
- **--model-size**: Model size for the selected backend (YOLO sizes: `n`, `s`, `m`, `l`, `x`). Default: `n`.
- **--backend**: Detection backend. Supported: `yolov8`, `torchvision_frcnn`, `torchvision_ssd`, `torchvision_retinanet`, `opencv_hog`. Default: `yolov8`.
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
# Using console script (recommended)
multi-report -r path/to/evaluation_results.json

# Or using Python module (alternative)
python -m src.main report -r path/to/evaluation_results.json
```

The report is done automatically using the above evaluation command, but can be useful for:
- Regenerating reports with different parameters
- Creating reports from partial evaluation results
- Debugging report generation

#### Arguments

- **--results-file, -r**: Path to a JSON produced by `evaluate`. Required.
- **--output-file, -o**: Where to save the generated markdown. If omitted, prints to stdout.

## How system works

1. **Frame extraction**: Videos are processed frame by frame
2. **Person detection**: Each frame is analyzed using the model selected to detect people
3. **Aggregation**: The system aggregates the results of each frame into a unique result, using the ``solution`` strategy decided.
Check ``solution`` strategies integrated at next section.

## Solution strategies

You can choose the video-level decision strategy with `--solution`:

- **counting** (default):
  - Computes the ratio of frames where more than one person is detected.
  - Final decision: `has_multiple_people = (ratio > MULTIPLE_PEOPLE_THRESHOLD)`.
  - Config parameter: `Config.MULTIPLE_PEOPLE_THRESHOLD` (default: 0.0). A value of 0.2–0.3 can be more robust.

- **temporal** (hysteresis):
  - Looks at the temporal sequence of per-frame decisions and activates multi‑person when there are at least `TEMPORAL_MIN_CONSECUTIVE` consecutive frames with multi‑person.
  - Sticky behavior: once activated, the video is considered multi‑person for the rest of its duration.
  - Config parameters: `Config.TEMPORAL_MIN_CONSECUTIVE` (default: 3), `Config.TEMPORAL_WINDOW` (reserved for future use).

- **temporal_cardaware** (temporal with filtering of ID cards):
  - Same as ``temporal`` strategy, but with filtering of faces in ID cards
  - Use of the relation between height/width to detect ID cards
  - Use of the difference of areas between bounding boxes to detect ID cards

Examples:

```bash
# Counting (default)
python -m src.main predict -i video.mp4 --solution counting --people-threshold 0.2

# Temporal (hysteresis) solution
python -m src.main predict -i video.mp4 --solution temporal

# Evaluate a dataset with the temporal strategy
python -m src.main evaluate -d path/to/dataset -l labels.txt --solution temporal
```


### Real-time demo

Run a realtime webcam demo that draws person detections and overlays the number of people and FPS:

```bash
# Using console script (recommended)
multi-demo \
  --backend yolov8 \
  --model-size n \
  --device cpu \
  --threshold 0.5 \
  --sample-rate 1

# Or using Python module (alternative)
python -m src.main demo \
  --backend yolov8 \
  --model-size n \
  --device cpu \
  --threshold 0.5 \
  --sample-rate 1
```

Note: This demo is only useful to check the base performance of the models frame by frame.
The solutions strategies are not applied to aggregate results, so this demo only shows the people detected by the model in each frame.

#### Options

- **--camera-index**: Webcam index to open. Default: 0.
- **--backend**: Detection backend. Supported: `yolov8`, `torchvision_frcnn`, `torchvision_ssd`, `torchvision_retinanet`, `opencv_hog`. Default: `yolov8`.
- **--model-size**: Model size (only applied for YOLO). Sizes: `n`, `s`, `m`, `l`, `x`. Default: `n`.
- **--device**: `cpu` or `cuda`. Default: auto-detect.
- **--threshold**: Detection confidence threshold. Default: 0.5.
- **--sample-rate**: Process every Nth frame (1 = every frame). Default: 1.
- **--show-confidence/--no-show-confidence**: Toggle confidence labels on boxes. Default: show.

## Project structure

```
src/
├── models/
│   └── person_detector.py    # Main detection logic
├── solutions/                # Video-level aggregation strategies
│   ├── base.py               # Base solution interface
│   ├── counting.py               # Default behavior
│   ├── temporal.py               # Temporal hysteresis solution
│   └── temporal_cardaware.py     # Temporal + ID-card suppression
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

### Using Makefile

```bash
make help          # Show available commands
make install       # Install in development mode
make install-dev   # Install with development dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Run linting
make clean         # Clean build artifacts
make build         # Build the package (sdist and wheel)
```

### Using Python directly

```bash
# First ensure you have dev dependencies installed
pip install -e ".[dev]"

# Then run development commands
pytest             # Run tests
pytest --cov=src   # Run tests with coverage
flake8 src/ tests/ # Run linting
```

## Next steps

- [x] Inference pipeline (logic of detect multiple persons)
- [x] Performance evaluation on the provided dataset
- [x] Technical report generation
- [x] Generalization of models
- [x] Demo in real time
- [x] Console scripts (CLI, entry point) -> pyproject.toml
- [x] Makefile for development automation
- [ ] Add strategies for the multi-detection logic
- [ ] Analysis
- [ ] Extras (depending on the time)
- [ ] Report
- [ ] Next steps with more time and data


## Notes

- GPU acceleration is supported but not required