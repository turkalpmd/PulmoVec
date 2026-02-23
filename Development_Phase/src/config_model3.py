"""
Configuration for Model 3: Disease Group Classification (3 classes)

Classes:
    0: Pneumonia (severe + non-severe)
    1: Bronchitis/Asthma/Bronchiolitis
    2: Normal/Other (Control Group + other diseases)
"""

from config import *

# Model identification
MODEL_NAME = "model3_disease"

# Classification parameters
NUM_CLASSES = 3
CLASS_NAMES = ["Pneumonia", "Bronchitis-Asthma-Bronchiolitis", "Normal/Other"]
LABEL_COLUMN = "model3_label"

# Class distribution (from ensemble dataset)
CLASS_COUNTS = {
    "Pneumonia": 12181,
    "Bronchitis-Asthma-Bronchiolitis": 4176,
    "Normal/Other": 8147
}

# Paths
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Ensemble_Dataset.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "model3_disease"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoints
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "last.pth"
TRAINING_HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"

# Visualization outputs
CONFUSION_MATRIX_PATH = MODEL_SAVE_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = MODEL_SAVE_DIR / "training_curves.png"

print(f"Model 3 Configuration Loaded: {MODEL_NAME}")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Save Dir: {MODEL_SAVE_DIR}")
