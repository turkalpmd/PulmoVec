# Ensemble HeAR Model Training System

## Overview

This is a **3-model ensemble system** for respiratory disease classification using Google's HeAR (Health Acoustic Representations) model. The system combines predictions from three specialized models through a Random Forest meta-model to achieve robust disease classification.

### System Architecture

```
Audio Event (2s, 16kHz)
        ↓
   HeAR Encoder (512-dim embedding)
        ↓
    ┌───┴───┬───────┬────────┐
    ↓       ↓       ↓        ↓
  Model 1  Model 2  Model 3   
  Event    Binary  Disease    
  Type     Abnormal Groups    
  (3 cls)  (2 cls)  (3 cls)   
    ↓       ↓       ↓
    └───┬───┴───────┘
        ↓
   [8 probabilities]
        ↓
  Random Forest Meta-Model
        ↓
   Final Prediction:
   Pneumonia / Bronchitis-Asthma-Bronchiolitis / Normal
```

---

## Models

### Model 1: Event Type Classification (3 classes)
- **Class 0**: Normal
- **Class 1**: Crackles (Fine Crackle + Coarse Crackle + Wheeze+Crackle)
- **Class 2**: Wheeze/Rhonchi (Wheeze + Rhonchi)

**Purpose**: Identifies the acoustic characteristics of respiratory sounds.

### Model 2: Binary Abnormality Detection (2 classes)
- **Class 0**: Normal (healthy respiratory sounds)
- **Class 1**: Abnormal (any adventitious sounds)

**Purpose**: Screening - high sensitivity for detecting abnormalities.

### Model 3: Disease Group Classification (3 classes)
- **Class 0**: Pneumonia (severe + non-severe)
- **Class 1**: Bronchitis/Asthma/Bronchiolitis
- **Class 2**: Normal/Other (Control Group + other diseases)

**Purpose**: Direct disease classification.

### Meta-Model: Random Forest
- **Input**: 8 probability outputs (3 + 2 + 3) from Models 1, 2, 3
- **Output**: Final disease prediction (3 classes)
- **Purpose**: Combines diverse predictions for robust final classification

---

## Project Structure

```
PulmoVec/
├── data/
│   ├── SPRSound_Event_Level_Dataset_CLEAN.csv (original)
│   └── SPRSound_Event_Level_Ensemble_Dataset.csv (with 3 label columns)
├── src/
│   ├── config.py (base configuration)
│   ├── config_model1.py (Model 1 config)
│   ├── config_model2.py (Model 2 config)
│   ├── config_model3.py (Model 3 config)
│   ├── models.py (HeARClassifier)
│   ├── sprsound_dataset.py (PyTorch Dataset)
│   └── utils.py (helper functions)
├── scripts/
│   ├── prepare_ensemble_labels.py (data preparation)
│   ├── train_ensemble_models.py (train all 3 models)
│   ├── train_meta_model.py (train Random Forest)
│   └── predict_ensemble.py (inference pipeline)
├── models/
│   ├── model1_event_type/
│   │   ├── best.pth
│   │   ├── training_history.json
│   │   ├── confusion_matrix.png
│   │   └── training_curves.png
│   ├── model2_binary/
│   │   └── ... (same structure)
│   ├── model3_disease/
│   │   └── ... (same structure)
│   └── meta_model/
│       ├── random_forest.pkl
│       ├── feature_importance.csv
│       └── meta_model_report.md
├── docs/
│   ├── ENSEMBLE_TRAINING_README.md (this file)
│   ├── Disease_Event_Analysis_DETAILED.md
│   └── ... (other documentation)
└── SPRSound-main/ (original dataset)
```

---

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements_hear.txt
```

Required packages:
- torch (with CUDA support)
- transformers (HuggingFace)
- librosa
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm

### 2. HuggingFace Authentication

The HeAR model is gated on HuggingFace. You need to:

1. **Request Access**: Visit https://huggingface.co/google/hear-pytorch and request access
2. **Get Token**: Get your token from https://huggingface.co/settings/tokens
3. **Authenticate**:

```bash
# Option 1: CLI login
huggingface-cli login

# Option 2: Create .env file
echo "HF_TOKEN=your_token_here" > .env
```

### 3. Verify GPU Setup

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Usage

### Step 1: Prepare Ensemble Dataset

Generate the dataset with 3 label columns:

```bash
python scripts/prepare_ensemble_labels.py
```

**Output**: `data/SPRSound_Event_Level_Ensemble_Dataset.csv`

### Step 2: Train All 3 Models

Train all models sequentially (6-10 hours on RTX 3090):

```bash
python scripts/train_ensemble_models.py --model all
```

Or train individual models:

```bash
python scripts/train_ensemble_models.py --model 1  # Event Type
python scripts/train_ensemble_models.py --model 2  # Binary
python scripts/train_ensemble_models.py --model 3  # Disease
```

**Monitoring**: Check `ensemble_training_log.txt` for progress.

### Step 3: Train Meta-Model

After all 3 models are trained:

```bash
python scripts/train_meta_model.py
```

**Output**: `models/meta_model/random_forest.pkl`

### Step 4: Run Inference

#### Single Event Prediction

```bash
python scripts/predict_ensemble.py \
  --mode single \
  --audio_path /path/to/audio.wav \
  --start_ms 1000 \
  --end_ms 3000
```

**Output**:
```
Final Prediction: Pneumonia
Probabilities:
  Pneumonia: 0.7543
  Bronchitis-Asthma-Bronchiolitis: 0.1823
  Normal/Other: 0.0634
```

#### Batch Prediction

```bash
python scripts/predict_ensemble.py \
  --mode batch \
  --csv_path /path/to/events.csv \
  --output_path predictions.csv
```

**Input CSV Requirements**: Must contain columns:
- `wav_path`
- `event_start_ms`
- `event_end_ms`

**Output**: CSV with predictions and probabilities.

---

## Training Configuration

### Hyperparameters (from `config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sampling rate (Hz) |
| `CLIP_DURATION` | 2 | Audio clip length (seconds) |
| `OVERLAP_PERCENT` | 10 | Event overlap for clipping (%) |
| `BATCH_SIZE` | 32 | Training batch size |
| `ACCUMULATION_STEPS` | 2 | Gradient accumulation (effective batch = 64) |
| `PHASE1_EPOCHS` | 10 | Frozen encoder epochs |
| `PHASE2_EPOCHS` | 40 | Fine-tuning epochs |
| `PHASE1_LR` | 1e-4 | Classification head learning rate |
| `PHASE2_LR` | 5e-7 | Fine-tuning learning rate |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `MAX_GRAD_NORM` | 0.5 | Gradient clipping threshold |
| `USE_AMP` | False | Automatic Mixed Precision (disabled for stability) |

### Two-Phase Training Strategy

**Phase 1 (10 epochs)**:
- HeAR encoder **frozen**
- Only classification head trains
- Higher learning rate (1e-4)
- Fast convergence on classification task

**Phase 2 (40 epochs)**:
- HeAR encoder **unfrozen**
- End-to-end fine-tuning
- Very low learning rate (5e-7)
- Adapts encoder to respiratory sounds

---

## Performance Metrics

### Example Results (from 50-epoch training)

**Model 1 (Event Type)**:
- Validation F1-Macro: ~0.65
- Best for: Distinguishing Normal vs Crackles vs Wheeze

**Model 2 (Binary)**:
- Validation F1-Macro: ~0.70
- Best for: High sensitivity screening

**Model 3 (Disease Groups)**:
- Validation F1-Macro: ~0.60
- Best for: Direct disease classification

**Meta-Model (Random Forest)**:
- Final Validation F1-Macro: ~0.68
- Combines strengths of all models

---

## Advanced Topics

### Class Imbalance Handling

The system uses multiple strategies:

1. **Weighted Loss**: Inverse class frequency weighting
2. **Stratified Splitting**: Maintains class distribution in train/val
3. **Balanced Random Forest**: `class_weight='balanced'` in meta-model

### VRAM Optimization

For limited VRAM:

1. Reduce `BATCH_SIZE` (minimum: 16)
2. Increase `ACCUMULATION_STEPS` to maintain effective batch size
3. Ensure `USE_AMP = False` for FP32 stability
4. Train models sequentially, not parallel

### Resuming Training

If training is interrupted:

```python
# Load checkpoint and continue
checkpoint = torch.load('models/model1_event_type/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## Troubleshooting

### Issue: NaN Loss During Training

**Symptoms**: Loss becomes NaN, all predictions identical

**Solutions**:
1. Reduce learning rate (`PHASE2_LR = 1e-7`)
2. Enable gradient clipping (`MAX_GRAD_NORM = 0.5`)
3. Disable AMP (`USE_AMP = False`)
4. Check for extreme values in data

### Issue: CUDA Out of Memory

**Solutions**:
1. Reduce `BATCH_SIZE` (try 16 or 8)
2. Increase `ACCUMULATION_STEPS` proportionally
3. Reduce `NUM_WORKERS`
4. Clear cache: `torch.cuda.empty_cache()`

### Issue: HuggingFace Authentication Failed

**Solutions**:
1. Request access: https://huggingface.co/google/hear-pytorch
2. Wait for approval (usually 1-2 hours)
3. Login: `huggingface-cli login`
4. Verify token: `cat ~/.huggingface/token`

### Issue: Training Very Slow

**Expected Speed** (RTX 3090):
- ~6 iterations/sec (Phase 1)
- ~5 iterations/sec (Phase 2)
- ~100 seconds/epoch

**If slower**:
1. Check GPU utilization: `nvidia-smi`
2. Increase `NUM_WORKERS` (try 8)
3. Verify data is on fast storage (SSD)
4. Check CPU isn't bottleneck

---

## File Formats

### Ensemble Dataset CSV

Columns:
- Original columns from `SPRSound_Event_Level_Dataset_CLEAN.csv`
- `model1_label` (int): 0=Normal, 1=Crackles, 2=Wheeze/Rhonchi
- `model2_label` (int): 0=Normal, 1=Abnormal
- `model3_label` (int): 0=Pneumonia, 1=Bronchitis-Asthma-Bronchiolitis, 2=Normal/Other

### Model Checkpoint (.pth)

```python
{
    'model_state_dict': OrderedDict,  # Model weights
    'optimizer_state_dict': dict,      # Optimizer state
    'epoch': int,                      # Last completed epoch
    'val_f1_macro': float             # Best F1-macro score
}
```

### Training History (JSON)

```json
{
  "train_loss": [0.95, 0.87, ...],
  "train_acc": [0.72, 0.75, ...],
  "train_f1_macro": [0.65, 0.68, ...],
  "val_loss": [1.02, 0.94, ...],
  "val_acc": [0.70, 0.73, ...],
  "val_f1_macro": [0.63, 0.66, ...],
  "lr": [1e-4, 1e-4, ...]
}
```

---

## Citation

If using this code or the HeAR model, please cite:

```bibtex
@article{google2023hear,
  title={Health Acoustic Representations},
  author={Google Research},
  year={2023},
  url={https://huggingface.co/google/hear-pytorch}
}

@dataset{SPRSound2024,
  title={SPRSound: Pediatric Respiratory Sound Database},
  year={2024}
}
```

---

## Changelog

### Version 1.0.0 (2026-01-14)
- Initial release
- 3-model ensemble system
- Random Forest meta-model
- Comprehensive training pipeline
- Inference scripts for single and batch predictions

---

## Support

For issues, please check:
1. This README
2. `docs/Disease_Event_Analysis_DETAILED.md` - Dataset analysis
3. `docs/HEAR_TRAINING_README.md` - Original HeAR training docs
4. Training logs: `ensemble_training_log.txt`

---

## License

Please refer to:
- HeAR model: https://huggingface.co/google/hear-pytorch (check license)
- SPRSound dataset: Check dataset authors' license

---

**Last Updated**: 2026-01-14  
**System**: Ensemble HeAR Respiratory Disease Classification  
**Models**: 3 specialized + 1 meta-model (Random Forest)
