"""Configuration for HeAR Event Detection Evaluation."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Dataset_CLEAN.csv"
ENSEMBLE_CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"

# HeAR model paths (using local snapshot)
HEAR_SNAPSHOT_PATH = PROJECT_ROOT / "hear"
EVENT_DETECTOR_VARIANT = "event_detector_small"  # or "event_detector_large"
EVENT_DETECTOR_PATH = HEAR_SNAPSHOT_PATH / "event_detector" / EVENT_DETECTOR_VARIANT
FRONTEND_PATH = HEAR_SNAPSHOT_PATH / "event_detector" / "spectrogram_frontend"

# Audio processing
SAMPLE_RATE = 16000
CLIP_DURATION = 2  # seconds
CLIP_OVERLAP_PERCENT = 10  # 10% overlap between clips

# Event detection
DETECTION_THRESHOLD = 0.3  # Lower threshold to find more events (was 0.5)
# Pipeline: PyTorch Model 2 (Binary) finds segments → PyTorch Model 1 classifies types (optional)

# Evaluation parameters
IOU_THRESHOLDS = [0.3, 0.5, 0.7]
BOUNDARY_TOLERANCES = [0.05, 0.1, 0.25]  # seconds

# Output directories
EVENT_DETECT_DIR = PROJECT_ROOT / "EventDetect"
RESULTS_DIR = EVENT_DETECT_DIR / "results"
SAMPLES_DIR = EVENT_DETECT_DIR / "samples"
SELECTED_SAMPLES_JSON = SAMPLES_DIR / "selected_samples.json"
