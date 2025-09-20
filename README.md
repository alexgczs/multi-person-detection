# Multi-person detection for identity verification

[![Build](https://github.com/alexgczs/multi-person-detection/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/alexgczs/multi-person-detection/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alexgczs/multi-person-detection/graph/badge.svg)](https://codecov.io/gh/alexgczs/multi-person-detection)

A system for detecting multiple people during identity verification sessions.

## Table of contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [Console scripts](#console-scripts-recommended)
  - [Python module interface](#python-module-interface-alternative)
  - [Single video prediction](#single-video-prediction)
  - [Dataset evaluation](#dataset-evaluation)
  - [Real-time demo](#real-time-demo)
- [Solution strategies](#solution-strategies)
  - [Counting strategy](#counting-strategy-default)
  - [Temporal strategy](#temporal-strategy-hysteresis)
  - [Card-aware strategy](#card-aware-strategy-temporal--id-filtering)
  - [Text-aware strategy](#text-aware-strategy-temporal--text-detection)
- [Configuration](#configuration)
  - [CLI parameters](#cli-parameters)
  - [Environment variables](#environment-variables)
  - [Default configuration](#default-configuration)
- [Project structure](#project-structure)
- [Development](#development)
  - [Using Makefile](#using-makefile-recommended)
  - [Using Python directly](#using-python-directly)
- [Next steps](#next-steps)
- [Notes](#notes)

## Overview

This project addresses a fraud detection challenge: identifying when multiple people are present during a verification session, which could indicate coercion or fraudulent behavior. The system processes video streams and outputs binary labels (0 = single person, 1 = multiple people).

### Key features

- **Multiple detection backends**: YOLOv8, TorchVision models, OpenCV HOG
- **Flexible aggregation strategies**: Counting, temporal hysteresis, card-aware detection, text-aware detection
- **CLI**: Command-line interface
- **Evaluation**: Dataset evaluation with metrics and reports
- **Real-time demo**: Webcam-based demonstration of backends
- **Extensible architecture**: Modular design to add backends, solutions...

## Quick start

### Installation

```bash
# Clone and setup
git clone https://github.com/alexgczs/multi-person-detection.git
cd multi-person-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode (recommended)
pip install -e .

# Install with development dependencies (for testing, linting, etc.)
pip install -e ".[dev]"

# Or install dependencies only (alternative)
pip install -r requirements.txt # only runtime dependencies
pip install -r requirements-dev.txt  # includes runtime + dev tools
```

**Note**: Installing with `pip install -e .` enables console scripts (`multi-predict`, `multi-evaluate`, etc.) for easier usage.

### Basic usage

```bash
# Predict on a single video
multi-predict -i video.mp4

# Evaluate on a dataset
multi-evaluate -d dataset/ -l labels.txt

# Run real-time demo
multi-demo
```

## Usage

The project provides both console scripts and Python module interfaces. Console scripts are recommended for easier usage.

### Console scripts (recommended)

After installing with `pip install -e .`, you can use:

- `multi-predict` - Predict multi-person detection on a single video
- `multi-evaluate` - Evaluate model performance on a dataset
- `multi-report` - Generate technical reports from evaluation results
- `multi-demo` - Run real-time webcam demo

### Python module interface (alternative)

```bash
python -m src.main <command> [options]
```

### Complete use
<details>
<summary><strong>Single video prediction</strong></summary>

Get a prediction for a single video:

```bash
# Basic prediction
multi-predict -i path/to/video.mp4

# With custom parameters
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

**Key options:**
- `--threshold`: Detection confidence threshold (default: 0.5)
- `--solution`: Aggregation strategy (counting, temporal, temporal_cardaware, temporal_textaware)
- `--people-threshold`: Ratio threshold for multi-person detection
- `--max-frames`: Maximum frames to process
- `--backend`: Detection model (yolov8, torchvision_frcnn, opencv_hog)

</details>

<details>
<summary><strong>Dataset evaluation</strong></summary>

Evaluate the model on a complete dataset:

```bash
# Basic evaluation
multi-evaluate -d path/to/dataset -l path/to/labels.txt

# With custom parameters
multi-evaluate -d path/to/dataset -l path/to/labels.txt \
  --threshold 0.5 --model-size n --device cuda \
  --sample-rate 1 --max-frames 100 --people-threshold 0.2 \
  --solution temporal

# With text-aware solution for document filtering
multi-evaluate -d path/to/dataset -l path/to/labels.txt \
  --solution temporal_textaware \
  --text-confidence-threshold 0.6 \
  --text-proximity-threshold 80

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

**Generated output:**
The evaluation automatically creates a timestamped directory (e.g., `data/results/2025-08-17_21-41-15/`) containing:

1. **`evaluation_results.json`** - Detailed results in JSON format:
   - Per-video predictions and ground truth
   - Performance metrics (accuracy, precision, recall, F1-score)
   - Classification report with class-wise statistics

2. **`technical_report.md`** - Markdown report including:
   - Executive summary with key performance metrics
   - Confusion matrix with TP/TN/FP/FN breakdown
   - Error analysis identifying specific failed videos

**Dataset format:**
- Videos: `.mp4` files in the dataset directory
- Labels: TSV file with columns `video` and `label` (0/1)

</details>

<details>
<summary><strong>Real-time demo</strong></summary>

Run a real-time webcam demo:

```bash
# Basic demo
multi-demo

# With custom parameters
multi-demo --backend yolov8 --model-size n --threshold 0.5
```

The demo shows:
- Person detections with bounding boxes
- Number of people detected
- FPS counter
- Confidence scores

**Note**: This demo shows frame-by-frame detections only. Solution strategies are not applied. The unique use of this demo is to understand the behavior of each model backend.

</details>

## Solution strategies

The system offers four different approaches for aggregating frame-level detections into video-level decisions:

<details>
<summary><strong>Counting strategy (default)</strong></summary>

**How it works:**
- Computes the ratio of frames where more than one person is detected
- Final decision: `has_multiple_people = (ratio > people_threshold)`

**Best for:**
- Simple scenarios with clear detections
- When you want fast processing
- Default performance evaluation

**Configuration:**
```bash
--solution counting --people-threshold 0.2
```

**Default threshold:** 0.0 (any frame with multiple people triggers detection)

</details>

<details>
<summary><strong>Temporal strategy (hysteresis)</strong></summary>

**How it works:**
- Looks at consecutive frames to avoid false positives from brief detections
- Activates when there are at least `temporal_min_consecutive` consecutive frames with multiple people
- Once activated, remains active for the rest of the video (sticky behavior)

**Best for:**
- Noisy detections or brief false positives
- Scenarios where sustained presence of multiple people is required
- More robust to detection artifacts

**Configuration:**
```bash
--solution temporal --temporal-min-consecutive 3
```

**Default:** 20 consecutive frames required

</details>

<details>
<summary><strong>Card-aware strategy (temporal + ID filtering)</strong></summary>

**How it works:**
- Same as temporal strategy but filters out faces in ID cards
- Uses geometric properties to identify ID card faces:
  - Area ratio compared to largest detection
  - Square aspect ratio tolerance
- Prevents false positives from ID card photos

**Best for:**
- Identity verification scenarios
- When users show ID cards during verification
- Most robust for real-world applications

**Configuration:**
```bash
--solution temporal_cardaware \
  --temporal-min-consecutive 3 \
  --card-min-area-ratio 0.85 \
  --card-square-tolerance 0.25
```

**Default parameters:**
- `card_min_area_ratio`: 0.9 (90% of largest detection)
- `card_square_tolerance`: 0.35 (35% tolerance for square aspect ratio)

</details>

<details>
<summary><strong>Text-aware strategy (temporal + text detection)</strong></summary>

**How it works:**
- Combines temporal hysteresis with text detection using EasyOCR
- Filters out faces that are near detected text regions (likely ID documents)
- Uses proximity threshold to determine if a face is near text

**Best for:**
- Identity verification with documents containing text
- Reducing false positives from document photos

**Configuration:**
```bash
--solution temporal_textaware \
  --temporal-min-consecutive 3 \
  --text-proximity-threshold 100 \
  --text-confidence-threshold 0.5
```

**Default parameters:**
- `text_proximity_threshold`: 100 pixels (distance between face and text centers)
- `text_confidence_threshold`: 0.5 (minimum OCR confidence for text detection)

**Note:** This solution requires EasyOCR for text detection and may be slower than other approaches. GPU acceleration is automatically enabled when CUDA is available for faster OCR processing.

</details>

## Configuration

The system supports three ways to configure parameters, in order of precedence:

1. **CLI parameters** (highest priority)
2. **Environment variables** (medium priority)  
3. **Default values** (lowest priority)

<details>
<summary><strong>CLI parameters</strong></summary>

All configuration parameters can be set via command-line arguments:

```bash
# Video processing
--sample-rate 2              # Process every 2nd frame
--max-frames 100             # Maximum frames per video
--threshold 0.7              # Detection confidence threshold

# Decision making
--people-threshold 0.2       # Ratio threshold for multi-person detection
--solution temporal          # Aggregation strategy

# Model selection
--backend yolov8             # Detection backend
--model-size n               # Model size (for YOLO models)
--device cuda                # Computation device

# Advanced parameters
--temporal-min-consecutive 3 # Minimum consecutive frames for temporal activation
--card-min-area-ratio 0.85   # Minimum area ratio for card detection
--card-square-tolerance 0.25 # Square tolerance for card detection
--text-proximity-threshold 100 # Text proximity threshold for text-aware detection
--text-confidence-threshold 0.5 # OCR confidence threshold for text detection
```

</details>

<details>
<summary><strong>Environment variables</strong></summary>

Set environment variables for consistent configuration across runs:

```bash
# Video processing
export MULTI_DETECT_FRAME_SAMPLE_RATE=2
export MULTI_DETECT_MAX_FRAMES=100

# Decision making
export MULTI_DETECT_MULTIPLE_PEOPLE_THRESHOLD=0.2

# Temporal solution parameters
export MULTI_DETECT_TEMPORAL_MIN_CONSECUTIVE=3

# Card detection parameters
export MULTI_DETECT_CARD_MIN_AREA_RATIO=0.85
export MULTI_DETECT_CARD_SQUARE_TOLERANCE=0.25

# Text detection parameters
export MULTI_DETECT_TEXT_PROXIMITY_THRESHOLD=100
export MULTI_DETECT_TEXT_CONFIDENCE_THRESHOLD=0.5
```

</details>

<details>
<summary><strong>Default configuration</strong></summary>

Default values in `src/utils/config.py`:

```python
# Video processing
FRAME_SAMPLE_RATE: int = 1            # Process every frame
MAX_FRAMES: int = None                # No limit
FRAME_WIDTH: int = 640                # Resize width
FRAME_HEIGHT: int = 480               # Resize height

# Decision making
MULTIPLE_PEOPLE_THRESHOLD: float = 0.0  # Any frame triggers detection

# Temporal solution parameters
TEMPORAL_MIN_CONSECUTIVE: int = 20      # Min consecutive frames for activation

# Card detection parameters
CARD_MIN_AREA_RATIO_TO_LARGEST: float = 0.9    # 90% of largest detection
CARD_SQUARE_TOLERANCE: float = 0.35            # 35% tolerance for square aspect ratio

# Text detection parameters
TEXT_PROXIMITY_THRESHOLD: int = 100            # Proximity threshold in pixels
TEXT_CONFIDENCE_THRESHOLD: float = 0.5         # OCR confidence threshold
```

</details>

## Project structure

```
src/
├── models/
│   ├── person_detector.py    # Main detection logic
│   └── backends.py          # Detection model implementations
├── solutions/               # Video-level aggregation strategies
│   ├── base.py              # Base solution interface
│   ├── counting.py          # Default counting strategy
│   ├── temporal.py          # Temporal hysteresis solution
│   ├── temporal_cardaware.py # Temporal + ID-card suppression
│   └── temporal_textaware.py # Temporal + text detection
├── utils/
│   ├── config.py            # Configuration management
│   ├── dataset_evaluator.py # Dataset evaluation utilities
│   ├── report_generator.py  # Technical report generation
│   └── video_processor.py   # Video frame extraction
├── demo/
│   └── realtime.py          # Real-time webcam demo
└── main.py                  # CLI interface

tests/                       # Unit and integration tests
data/                        # Results storage
resuts_report.md             # Report of the results obtained with each strategy
```

## Development

<details>
<summary><strong>Using Makefile (recommended)</strong></summary>

```bash
make help          # Show available commands
make install       # Install in development mode
make install-dev   # Install with development dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Run linting
make clean         # Clean build artifacts
make build         # Build the package
```

</details>

<details>
<summary><strong>Using Python directly</strong></summary>

```bash
# First ensure you have dev dependencies installed
pip install -e ".[dev]"

# Then run development commands
pytest             # Run tests
pytest --cov=src   # Run tests with coverage
flake8 src/ tests/ # Run linting
```

</details>

## Next steps

- [x] Core inference pipeline
- [x] Multiple detection backends
- [x] Flexible aggregation strategies
- [x] Dataset evaluation and reporting
- [x] Real-time demo
- [x] CLI interface (console scripting)
- [x] Advanced temporal analysis

## Notes

- GPU acceleration is supported but not required
