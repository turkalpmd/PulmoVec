"""
Configuration for U-Net temporal segmentation training.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# EventDetect directory
EVENT_DETECT_DIR = PROJECT_ROOT / "EventDetect"

# Data paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Dataset_CLEAN.csv"

# Model paths
MODEL_SAVE_DIR = PROJECT_ROOT / "EventDetect" / "models" / "unet_checkpoints"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "last.pth"
TRAINING_HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"

# Audio parameters
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Training parameters
BATCH_SIZE = 8  # Smaller batch size for full audio files
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Loss function weights
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

# Train/Val split
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Device
DEVICE = "cuda"  # Will be set to "cpu" if CUDA not available

print(f"U-Net Config Loaded")
print(f"  Model Save Dir: {MODEL_SAVE_DIR}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")


# Train - Loss: 0.9927, IoU: 0.1999, F1: 0.2720
#   Val   - Loss: 1.0348, IoU: 0.2031, F1: 0.2723