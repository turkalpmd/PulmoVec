"""
Configuration for HeAR + BiLSTM temporal segmentation training.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# EventDetect directory
EVENT_DETECT_DIR = PROJECT_ROOT / "EventDetect"

# Data paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Dataset_CLEAN.csv"

# Model paths
MODEL_SAVE_DIR = PROJECT_ROOT / "EventDetect" / "models" / "hear_bilstm_checkpoints"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "last.pth"
TRAINING_HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"

# Embedding cache directory
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "EventDetect" / "models" / "hear_embeddings_cache"
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# HeAR encoder settings
HEAR_MODEL_NAME = "google/hear-pytorch"  # HuggingFace model name
HEAR_ENCODER_FROZEN = True  # Default: frozen encoder, only train BiLSTM
HEAR_FINE_TUNE_LAST_N_LAYERS = 0  # If > 0, unfreeze last N layers (only if HEAR_ENCODER_FROZEN=False)
HEAR_EMBEDDING_DIM = 512  # HeAR output embedding dimension

# Window parameters
WINDOW_SEC = 2.0  # HeAR requires 2s windows
HOP_SEC = 0.25  # Default hop size (0.25s or 0.5s)
SAMPLE_RATE = 16000  # Audio sample rate

# Label creation
OVERLAP_RATIO_THRESHOLD = 0.3  # Window overlap ratio threshold for positive label
# New strategy: Window is positive if:
# 1. Window center is inside an event, OR
# 2. Overlap ratio >= threshold, OR  
# 3. Total event duration in window >= 0.1s

# BiLSTM hyperparameters
BILSTM_HIDDEN_DIM = 256
BILSTM_NUM_LAYERS = 2
BILSTM_DROPOUT = 0.2  # Only active if num_layers > 1

# Training parameters
BATCH_SIZE = 16  # Batch size for embedding sequences
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Loss function
POS_WEIGHT_MODE = "auto"  # "auto" or "manual"
POS_WEIGHT_MANUAL = None  # If manual, specify value (e.g., 2.0)

# Best model selection
BEST_BY = "val_auprc"  # "val_auprc" or "val_f1"

# Train/Val split
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Device
DEVICE = "cuda"  # Will be set to "cpu" if CUDA not available

# Evaluation threshold
EVAL_THRESHOLD = 0.5  # Default threshold for binary classification

# Post-processing
MIN_DURATION_SEC = 0.05  # Minimum segment duration after merging (reduced for short events)
SMOOTHING_ENABLED = True  # Enable median filter smoothing to reduce noise
SMOOTHING_WINDOW_SIZE = 3  # Median filter window size
USE_HYSTERESIS = True  # Use hysteresis thresholding (on/off thresholds)
HYSTERESIS_ON_THRESHOLD = 0.68  # Threshold to turn on event (higher = fewer false positives)
HYSTERESIS_OFF_THRESHOLD = 0.58  # Threshold to turn off event (higher to break long segments)
MAX_GAP_SEC = 0.15  # Maximum gap between windows to merge (very small to prevent over-merging)

print(f"HeAR+BiLSTM Config Loaded")
print(f"  Model Save Dir: {MODEL_SAVE_DIR}")
print(f"  Embedding Cache Dir: {EMBEDDING_CACHE_DIR}")
print(f"  HeAR Model: {HEAR_MODEL_NAME}")
print(f"  Encoder Frozen: {HEAR_ENCODER_FROZEN}")
print(f"  Window: {WINDOW_SEC}s, Hop: {HOP_SEC}s")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
