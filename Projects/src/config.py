"""
Configuration file for HeAR SPRSound training.
All hyperparameters and paths are centralized here.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root (parent of src/ directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Dataset_CLEAN.csv"
WAV_ROOT_DIR = PROJECT_ROOT / "SPRSound-main"

# Model save directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Environment file for HuggingFace token
ENV_FILE = PROJECT_ROOT / ".env"

# ============================================================================
# AUDIO PARAMETERS
# ============================================================================

SAMPLE_RATE = 16000  # Hz - Required by HeAR model
CLIP_DURATION = 2  # seconds - Required by HeAR model
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32,000 samples

# Event clipping parameters
OVERLAP_PERCENT = 10  # ±10% overlap for event extraction

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# HeAR model
HEAR_MODEL_NAME = "google/hear-pytorch"
EMBEDDING_DIM = 512  # HeAR embedding dimension

# Classification head
HIDDEN_DIM = 256
DROPOUT = 0.3

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Batch sizes
BATCH_SIZE = 32  # Reduced for Phase 2 (full model training needs more VRAM)
NUM_WORKERS = 8  # For DataLoader (increased for faster data loading)

# Epochs and learning rates
PHASE1_EPOCHS = 10  # Freeze HeAR encoder, train only head
PHASE2_EPOCHS = 40  # Fine-tune all layers
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

PHASE1_LR = 1e-4  # Learning rate for classification head
PHASE2_LR = 5e-7  # Learning rate for fine-tuning entire model (ultra-conservative)

# Optimizer
WEIGHT_DECAY = 1e-4

# Gradient clipping
MAX_GRAD_NORM = 0.5  # Clip gradients to prevent NaN (more aggressive)

# Gradient accumulation (for even larger effective batch size)
ACCUMULATION_STEPS = 2  # Effective batch = 32 * 2 = 64 (maintains performance)

# Learning rate scheduler
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# ============================================================================
# CLASS DEFINITIONS AND WEIGHTS
# ============================================================================

# All 7 event types in SPRSound dataset
CLASS_NAMES = [
    "Normal",
    "Fine Crackle",
    "Wheeze",
    "Rhonchi",
    "Coarse Crackle",
    "Wheeze+Crackle",
    "Stridor"
]

NUM_CLASSES = len(CLASS_NAMES)

# Class distribution from SPRSound_Event_Level_Dataset_CLEAN.csv
# These will be used to compute class weights
CLASS_COUNTS = {
    "Normal": 18772,
    "Fine Crackle": 3530,
    "Wheeze": 1505,
    "Wheeze+Crackle": 303,
    "Rhonchi": 217,
    "Coarse Crackle": 177,
    "Stridor": 74
}

# ============================================================================
# DATA SPLIT
# ============================================================================

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Detection threshold for HeAR event detector
DETECTION_THRESHOLD = 0.5

# HeAR event detector labels
HEAR_DETECTOR_LABELS = [
    'Cough', 'Snore', 'Baby Cough', 'Breathe', 
    'Sneeze', 'Throat Clear', 'Laugh', 'Speech'
]

# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

# Checkpoint naming
BEST_MODEL_PATH = MODEL_DIR / "hear_sprsound_best.pth"
LAST_MODEL_PATH = MODEL_DIR / "hear_sprsound_last.pth"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.json"

# Visualization outputs
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = MODEL_DIR / "training_curves.png"

# Evaluation report
DETECTOR_EVAL_REPORT = PROJECT_ROOT / "hear_detector_evaluation.md"

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False  # Automatic Mixed Precision (disabled - causes NaN with large batches)

print(f"Configuration loaded. Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
