"""
Configuration for Model 1: Event Type Classification (3 classes)

Classes:
    0: Normal
    1: Crackles (Fine Crackle + Coarse Crackle + Wheeze+Crackle)
    2: Wheeze/Rhonchi (Wheeze + Rhonchi)
"""

from config import *

# Model identification
MODEL_NAME = "model1_event_type"

# Classification parameters
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Crackles", "Wheeze/Rhonchi"]
LABEL_COLUMN = "model1_label"

# Class distribution (from ensemble dataset)
CLASS_COUNTS = {
    "Normal": 18772,
    "Crackles": 4010,
    "Wheeze/Rhonchi": 1722
}

# Paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "model1_event_type"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoints
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "last.pth"
TRAINING_HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"

# Visualization outputs
CONFUSION_MATRIX_PATH = MODEL_SAVE_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = MODEL_SAVE_DIR / "training_curves.png"

print(f"Model 1 Configuration Loaded: {MODEL_NAME}")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Save Dir: {MODEL_SAVE_DIR}")
