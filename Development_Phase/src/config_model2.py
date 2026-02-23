"""
Configuration for Model 2: Binary Abnormality Detection (2 classes)

Classes:
    0: Normal (normal respiratory sounds)
    1: Abnormal (any adventitious sounds)
"""

from config import *

# Model identification
MODEL_NAME = "model2_binary"

# Classification parameters
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Abnormal"]
LABEL_COLUMN = "model2_label"

# Class distribution (from ensemble dataset)
CLASS_COUNTS = {
    "Normal": 18772,
    "Abnormal": 5732
}

# Paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "model2_binary"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoints
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "last.pth"
TRAINING_HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"

# Visualization outputs
CONFUSION_MATRIX_PATH = MODEL_SAVE_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = MODEL_SAVE_DIR / "training_curves.png"

print(f"Model 2 Configuration Loaded: {MODEL_NAME}")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Save Dir: {MODEL_SAVE_DIR}")
