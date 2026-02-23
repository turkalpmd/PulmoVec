# HeAR SPRSound Training Guide

Complete implementation for fine-tuning Google's HeAR (Health Acoustic Representations) model on the SPRSound respiratory sound dataset.

## 📁 Project Structure

```
PulmoVec/
├── config.py                           # Central configuration
├── sprsound_dataset.py                 # PyTorch Dataset for SPRSound
├── models.py                           # HeAR classifier model
├── utils.py                            # Training utilities
├── train_hear_classifier.py            # Main training script
├── evaluate_hear_detector.py           # HeAR detector evaluation
├── requirements_hear.txt               # Python dependencies
├── .env                                # HuggingFace token (HF_TOKEN=xxx)
├── SPRSound_Event_Level_Dataset_CLEAN.csv  # Event annotations
├── SPRSound-main/                      # WAV files
└── models/                             # Saved models (created during training)
    ├── hear_sprsound_best.pth
    ├── hear_sprsound_last.pth
    ├── training_history.json
    ├── confusion_matrix.png
    └── training_curves.png
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements_hear.txt
```

### 2. Setup HuggingFace Token

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Verify Data

Ensure you have:
- `SPRSound_Event_Level_Dataset_CLEAN.csv` (24,808 events)
- WAV files accessible via paths in CSV (verified with `wav_exists='yes'`)

### 4. Train the Model

```bash
python train_hear_classifier.py
```

Training will run for up to 50 epochs (10 epochs Phase 1 + 40 epochs Phase 2) with early stopping.

**Expected training time**: 
- ~2-4 hours on GPU (NVIDIA RTX 3090)
- ~10-20 hours on CPU (not recommended)

### 5. Monitor Training

The script will display:
- Progress bars for each epoch
- Train/Val metrics (Loss, Accuracy, F1-macro)
- Learning rate updates
- Best model checkpoints

Example output:
```
======================================================================
Phase 1 - Epoch 1/10
======================================================================
Epoch 1 [Train]: 100%|████████| 619/619 [02:15<00:00, loss=1.2345, lr=1e-4]
Epoch 1 [Val]:   100%|████████| 155/155 [00:30<00:00, loss=1.1234]

  Train - Loss: 1.2345, Acc: 0.6789, F1-macro: 0.4567
  Val   - Loss: 1.1234, Acc: 0.7123, F1-macro: 0.5234
  ✓ Best model saved (F1-macro: 0.5234)
```

## 📊 Model Architecture

```
Input: Raw audio (2s, 16kHz, mono) → [32000 samples]
   ↓
HeAR Preprocessing (PCEN normalization)
   ↓
HeAR Encoder (Transformer-based, pre-trained)
   ↓
Embeddings [512-dimensional]
   ↓
Classification Head:
   - Dropout (0.3)
   - Linear (512 → 256)
   - ReLU + BatchNorm
   - Dropout (0.3)
   - Linear (256 → 7)
   ↓
Output: Class logits [7 classes]
```

## 🎯 Training Strategy

### Phase 1: Classification Head Training (10 epochs)
- **Freeze**: HeAR encoder parameters
- **Train**: Only classification head
- **Learning Rate**: 1e-4
- **Goal**: Quick adaptation to SPRSound classes

### Phase 2: End-to-End Fine-tuning (40 epochs)
- **Unfreeze**: All model parameters
- **Train**: Entire model
- **Learning Rate**: 1e-5 (lower to avoid catastrophic forgetting)
- **Goal**: Fine-tune HeAR representations for respiratory sounds

## 📈 Expected Performance

Based on dataset statistics and class imbalance:

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | >70% | Baseline expectation |
| F1-macro | >0.60 | Accounts for class imbalance |
| Normal (76% of data) | >0.85 F1 | Majority class |
| Fine Crackle (14%) | >0.65 F1 | Common pathological |
| Wheeze (6%) | >0.55 F1 | Moderate frequency |
| Minority classes | >0.30 F1 | Rhonchi, Stridor, etc. |

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Training
BATCH_SIZE = 32          # Increase if you have more GPU memory
PHASE1_EPOCHS = 10       # Classification head training
PHASE2_EPOCHS = 40       # Full model fine-tuning
PHASE1_LR = 1e-4         # Phase 1 learning rate
PHASE2_LR = 1e-5         # Phase 2 learning rate

# Early stopping
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement

# Data
OVERLAP_PERCENT = 10     # ±10% temporal overlap for event extraction

# Device
USE_AMP = True           # Automatic Mixed Precision (faster on modern GPUs)
```

## 🎓 Class Distribution & Weights

The dataset is highly imbalanced. Class weights are automatically calculated:

| Class | Count | Percentage | Weight |
|-------|-------|------------|--------|
| Normal | 18,772 | 76.38% | 0.13 |
| Fine Crackle | 3,530 | 14.36% | 0.70 |
| Wheeze | 1,505 | 6.12% | 1.65 |
| Wheeze+Crackle | 303 | 1.23% | 8.19 |
| Rhonchi | 217 | 0.88% | 11.43 |
| Coarse Crackle | 177 | 0.72% | 14.01 |
| Stridor | 74 | 0.30% | 33.51 |

Weights are used in `CrossEntropyLoss` to balance training.

## 📝 Output Files

After training, you'll find:

1. **hear_sprsound_best.pth** - Best model based on validation F1-macro
2. **hear_sprsound_last.pth** - Final model after all epochs
3. **training_history.json** - Loss and metrics per epoch
4. **confusion_matrix.png** - Visualization of predictions vs true labels
5. **training_curves.png** - Loss, accuracy, F1, and LR over time

## 🔍 Evaluation

### Load and Use Trained Model

```python
import torch
from models import load_model_checkpoint
import config

# Load best model
model = load_model_checkpoint('models/hear_sprsound_best.pth')

# Predict on new audio
audio = torch.randn(1, 32000).to(config.DEVICE)  # Your preprocessed audio
with torch.no_grad():
    logits = model(audio)
    pred_class = torch.argmax(logits, dim=1).item()
    
print(f"Predicted: {config.CLASS_NAMES[pred_class]}")
```

### HeAR Detector Evaluation (Optional)

Evaluate HeAR's pre-trained event detector on SPRSound:

```bash
python evaluate_hear_detector.py
```

This generates `hear_detector_evaluation.md` showing how well HeAR's general-purpose detector (trained on Cough, Breathe, etc.) performs on respiratory pathologies.

## 💡 Tips for Better Performance

### 1. Data Augmentation
Consider adding to `sprsound_dataset.py`:
- Time stretching
- Pitch shifting
- Background noise
- SpecAugment

### 2. Hyperparameter Tuning
- Adjust learning rates
- Try different batch sizes
- Experiment with dropout rates
- Change hidden layer dimensions

### 3. Advanced Techniques
- Label smoothing
- Mixup augmentation
- Focal loss (for extreme imbalance)
- Ensemble multiple checkpoints

### 4. Monitor Overfitting
- Watch train vs val loss divergence
- Use validation set for hyperparameter selection
- Consider adding more dropout if overfitting

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or 8
```

### HuggingFace Authentication Error
```bash
# Verify your token
huggingface-cli login
```

### LibriSoundFile Error
```bash
pip install soundfile --upgrade
```

### Slow Training
```bash
# Reduce number of workers if CPU bottleneck
NUM_WORKERS = 2  # in config.py

# Disable AMP if causing issues
USE_AMP = False
```

## 📚 References

1. **HeAR Model**: [google-health/hear](https://github.com/google-health/hear)
2. **SPRSound Dataset**: Published respiratory sound database
3. **Paper**: "Health Acoustic Representations" (Google Health AI)

## 🤝 Citation

If you use this code, please cite:

```bibtex
@software{hear_sprsound_training,
  title={HeAR Fine-tuning for SPRSound Respiratory Classification},
  year={2026},
  url={https://github.com/google-health/hear}
}
```

## 📧 Support

For issues:
1. Check configuration in `config.py`
2. Verify data paths and CSV integrity
3. Review error messages in console output
4. Ensure CUDA is properly installed (for GPU training)

## ✅ Success Criteria Checklist

Before considering training complete:

- [ ] No CUDA errors during training
- [ ] Training loss decreases smoothly
- [ ] Validation F1-macro > 0.60
- [ ] Minority class F1 > 0.30
- [ ] Best model checkpoint saved successfully
- [ ] Confusion matrix shows reasonable predictions
- [ ] Training curves indicate convergence

---

**Ready to train?** Run: `python train_hear_classifier.py`

Good luck! 🎉
