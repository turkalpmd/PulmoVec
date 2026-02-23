# PulmoVec

**Pediatric Lung Sounds at Home — A HeAR-Based Pipeline for Pediatric Home Lung Sound Monitoring**

PulmoVec is an end-to-end respiratory-sound classification pipeline designed for real-world pediatric home monitoring. It combines Google's Health Acoustic Representations (HeAR) foundation encoder with a stacking meta-learner (LightGBM) to simultaneously predict acoustic phenotypes, abnormality status, and clinical disease groups from 2-second lung-sound clips.

---

## Table of Contents

1. [Clinical Motivation](#clinical-motivation)
2. [Team](#team)
3. [Dataset](#dataset)
4. [Architecture Overview](#architecture-overview)
5. [Folder Structure](#folder-structure)
6. [Pipeline Steps](#pipeline-steps)
7. [Results](#results)
8. [Key Configuration](#key-configuration)
9. [Data Flow Diagram](#data-flow-diagram)
10. [Label Governance](#label-governance)
11. [Mobile Workflow and Safety Gating](#mobile-workflow-and-safety-gating)
12. [Validation Roadmap](#validation-roadmap)
13. [Limitations](#limitations)
14. [Troubleshooting](#troubleshooting)
15. [References](#references)

---

## Clinical Motivation

Pediatric respiratory illness can change course rapidly. Infants discharged from the emergency department with bronchiolitis frequently return for reassessment within 48–72 hours. During that window, families and clinicians face the same questions: Has the lung-sound pattern changed in a meaningful way? Is there a new finding that could indicate secondary pneumonia? Does the child need same-day re-evaluation, or is home monitoring still safe?

Today there is no practical tool that bridges this gap. Home recordings are noisy, hardware varies from phone to phone, and caregiver technique is inconsistent. Even in controlled clinical settings, high-quality labeled pediatric datasets remain scarce, inter-observer variability in auscultation is high, and no existing tool has demonstrated clinical value in real-world home deployment.

**What PulmoVec aims to do**: provide families and their care teams with structured acoustic signals that track change over time, so follow-up decisions can be made with more confidence and less guesswork — not to replace clinicians, but to give them a more objective signal between visits.

---

## Team

| Name | Role |
|---|---|
| Izzet Turkalp Akbasli, MD | Pediatric Emergency Medicine / Clinical Data Scientist. Clinical review, label governance design, model evaluation. |
| Baris Ozturk, MD | Emergency Medicine / Mobile App Developer. Mobile capture workflow, quality gate, safety gating logic. |
| Oguzhan Serin, MD | Pediatric Emergency Medicine / Digital Health Specialist. Change tracking, caregiver-facing outputs, stakeholder engagement. |
| Volkan Dogan, MSc | Computer Vision Engineer / AI Architect. HeAR fine-tuning pipeline, model architecture, stacking meta-model. |

---

## Dataset

### Source
- **SPRSound** (IEEE TBioCAS 2022, Grand Challenge 2023–2024)
- 1,181 recordings — SPRSound main dataset (292), Grand Challenge 2023 (94), Grand Challenge 2024 (795)
- 725+ unique patients, ages 0.1–16.8 years
- Collected at Shanghai Children's Medical Center (SCMC) with electronic stethoscope
- 4 auscultation locations per recording: p1–p4 (anterior/posterior, left/right)
- 57,123 annotated events in total (raw corpus)

### Curation steps
1. Deduplicate recordings appearing across multiple dataset splits.
2. Exclude "Poor Quality" recordings using record-level quality labels.
3. Include records without record-level labels if event-level annotation is present (some test sets provide only event-level annotations).
4. Remove events with no annotation.

### Curated event-level cohort

| CSV | Rows | Description |
|---|---|---|
| `data/SPRSound_Event_Level_Dataset_CLEAN.csv` | 20,007 | Deduplicated, Poor Quality removed, wav-verified |
| `data/SPRSound_Event_Level_Ensemble_Dataset.csv` | 19,806 | + model1/2/3 derived label columns |
| `data/ensemble_probabilities_train.csv` | ~15,800 | Base-model predict_proba (train split) |
| `data/ensemble_probabilities_val.csv` | ~3,960 | Base-model predict_proba (val split) |
| `data/disease_event_pivot_counts.csv` | 17 rows | Disease × event type pivot (counts) |
| `data/disease_event_pivot_percentages.csv` | 17 rows | Disease × event type pivot (%) |

> **Note on the "24,808" figure**: The full curated cohort before local WAV verification was 24,808 events (1,167 unique patients). After confirming that WAV files are locally accessible on this machine, 20,007 rows remain in the clean CSV. 1,307 WAV files reside on the original collection machine only.

> **Audio files**: `data/wavfiles/` — 5,184 WAV files; relative paths used throughout (`data/wavfiles/<name>.wav`).

### Train / validation split
- 80 / 20 stratified split; training set: **n = 19,846** events.
- Stratification is by label column to preserve class distribution.

### Disease distribution (record level)

| Disease | Recordings | % | Dominant sound pattern |
|---|---|---|---|
| Pneumonia (non-severe) | 641 | 54.3 | Fine Crackle (12.1 % of events) |
| Bronchitis | 144 | 12.2 | Wheeze (7.9 %) + Rhonchi (2.3 %) |
| Control Group | 120 | 10.2 | Normal (94.5 %) |
| Asthma | 114 | 9.7 | Wheeze (10.2 %) |
| Pneumonia (severe) | 66 | 5.6 | Fine Crackle (9.2 %) + Wheeze (11.5 %) |
| Other respiratory | 49 | 4.1 | Heterogeneous |
| Bronchiolitis | 9 | 0.8 | **Wheeze (43.0 %)** ← highest wheeze rate |
| Bronchiectasis | 9 | 0.8 | Fine Crackle (13.0 %) |

### Event-level class distribution (raw corpus, 57,123 events)

| Event type | Count | % |
|---|---|---|
| Normal | 44,714 | 78.9 |
| Fine Crackle | 6,651 | 11.7 |
| Wheeze | 3,764 | 6.6 |
| Rhonchi | 575 | 1.0 |
| Coarse Crackle | 389 | 0.7 |
| Wheeze+Crackle | 381 | 0.7 |
| Stridor | 189 | 0.3 |

### Key disease–sound associations (from pivot analysis)

- **Bronchiolitis** (likely RSV in infants < 2 y): 43 % Wheeze events — highest of any disease.
- **Severe vs non-severe pneumonia**: Severe shows 2× more Coarse Crackle (2.5 % vs 0.5 %) and more Wheeze (11.5 % vs 5.1 %), consistent with greater airway involvement.
- **Asthma**: Only 10.2 % Wheeze events despite being the archetypal wheeze disease — most recordings capture stable inter-attack periods.
- **Control group**: 94.5 % Normal; residual 5.5 % is physiological variation, cardiac-sound interference, and annotation sensitivity.
- **"Normal-dominant" paradox in severe pneumonia (74.9 % Normal events)**: Pneumonia is focal — 4-location recordings capture unaffected lung regions. With 1–2 lobes affected, 2–3 recording locations are acoustically normal. This is clinically expected, not a labeling error.

### Column reference (Ensemble CSV)

| Column | Values |
|---|---|
| `wav_path` | `data/wavfiles/<name>.wav` (relative to `Projects/`) |
| `event_start_ms`, `event_end_ms` | Millisecond timestamps |
| `disease` | 17 SPRSound disease labels |
| `event_type` | 7 raw SPRSound event labels |
| `model1_label` | 0 = Normal, 1 = Crackles, 2 = Wheeze/Rhonchi |
| `model2_label` | 0 = Normal, 1 = Abnormal |
| `model3_label` | 0 = Normal/Other, 1 = Pneumonia, 2 = Bronchitis-Asthma-Bronchiolitis |

---

## Architecture Overview

```
Raw audio (WAV, any sample rate)
        │
        ▼
  Preprocessing
  • Mono conversion
  • Resample → 16 kHz
  • Event-centric clip (±10 % overlap around annotation boundaries)
  • Pad / trim → exactly 2 s (32,000 samples)
        │
        ▼
 HeAR Encoder  (google/hear-pytorch, 303 M params)
 Transformer/ViT on PCEN log-mel patches
 Pretrained via MAE on 313 M health-audio clips (YT-NS corpus)
 ~75 % masking ratio, reconstruction objective
        │
        ▼ 512-dim embedding
        │
   ┌────┴────┬──────────┐
   ▼         ▼          ▼
Model 1   Model 2    Model 3
3-class   2-class    3-class
Event     Binary     Disease
Type      Screening  Group
   │         │          │
   ▼         ▼          ▼
 p×3       p×2        p×3
   └────┬────┴──────────┘
        │  8 probabilities
        │  + age + gender + recording location (= 11 features)
        ▼
 LightGBM Meta-Models  (×5 outcomes, Optuna-tuned, 100 trials)
 Nonparametric bootstrap CI (1,000 iterations)
        │
        ▼
  Final Predictions across 5 clinical outcomes
```

### Base classifiers (HeAR + classification head)

| Model | Task | Classes | Head architecture |
|---|---|---|---|
| Model 1 | Acoustic phenotype | Normal / Crackles / Wheeze-Rhonchi | 512 → 256 → 3 |
| Model 2 | Binary screening | Normal / Abnormal | 512 → 256 → 2 |
| Model 3 | Disease group | Pneumonia / Bronchitis-Asthma-Bronchiolitis / Normal-Other | 512 → 256 → 3 |
| Model 4 | Extended disease group | Pneumonia / Bronchoconstriction / Normal / Others | 512 → 256 → 4 |

Classification head: `Linear(512→256)` + ReLU + `Dropout(0.3)` + `Linear(256→K)`.

All four share the same HeAR backbone (frozen in Phase 1, unfrozen in Phase 2).

### Meta-feature matrix (11 columns per event)

| Source | Features |
|---|---|
| Model 1 | `p_Normal`, `p_Crackles`, `p_Rhonchi` |
| Model 2 | `p_Normal`, `p_Abnormal` |
| Model 3 | `p_Normal`, `p_Pneumonia`, `p_Bronchiolitis` |
| Demographics | `age`, `gender_encoded`, `recording_location` |

### LightGBM meta-model outcomes (5 classifiers)

| Outcome | Type | Classes |
|---|---|---|
| `model2_label` | Binary | Normal / Abnormal |
| `model1_label` | Multi-class | Normal / Crackles / Wheeze-Rhonchi |
| `model3_label` | Multi-class | Normal-Other / Pneumonia / Bronchitis-Asthma-Bronchiolitis |
| `event_type` | Multi-class | 7 raw SPRSound event types |
| `disease` | Multi-class | 17 raw SPRSound disease labels |

---

## Folder Structure

```
Projects/
├── README.md                          ← This file
│
├── scripts/                           ← Run in order (see Pipeline Steps)
│   ├── analyze_disease_event_pivot.py   Step 0 – exploratory analysis
│   ├── prepare_ensemble_labels.py       Step 1 – derive label columns
│   ├── train_hear_classifier.py         Step 2a – single HeAR model
│   ├── train_ensemble_models.py         Step 2b – all 3/4 base models
│   ├── resume_training.py               Step 2c – resume single model
│   ├── resume_ensemble_training.py      Step 2c – resume ensemble
│   ├── extract_ensemble_probabilities.py Step 3 – predict_proba CSVs
│   ├── train_meta_model.py              Step 4 – LightGBM stacking
│   └── predict_ensemble.py              Step 5 – inference
│
├── src/                               ← Importable modules
│   ├── config.py                        Global hyperparameters & paths
│   ├── config_model1.py                 Model 1 overrides
│   ├── config_model2.py                 Model 2 overrides
│   ├── config_model3.py                 Model 3 overrides
│   ├── models.py                        HeARClassifier (backbone + head)
│   ├── sprsound_dataset.py              PyTorch Dataset + stratified split
│   └── utils.py                         Metrics, checkpointing, early stopping
│
└── data/                              ← All artefacts
    ├── wavfiles/                        5,184 WAV files (relative paths)
    ├── SPRSound_Event_Level_Dataset_CLEAN.csv
    ├── SPRSound_Event_Level_Ensemble_Dataset.csv
    ├── disease_event_pivot_counts.csv
    ├── disease_event_pivot_percentages.csv
    ├── ensemble_probabilities_train.csv
    └── ensemble_probabilities_val.csv
```

> Trained model checkpoints live outside this folder at the repository root under `models/`.

---

## Pipeline Steps

### Step 0 — Exploratory disease × event pivot

```bash
cd Projects/
python scripts/analyze_disease_event_pivot.py
```

Outputs `disease_event_pivot_counts.csv`, `disease_event_pivot_percentages.csv`, and a heatmap PNG.

---

### Step 1 — Prepare ensemble labels

```bash
python scripts/prepare_ensemble_labels.py
```

Reads `data/SPRSound_Event_Level_Dataset_CLEAN.csv`, appends `model1_label`, `model2_label`, `model3_label`, writes `data/SPRSound_Event_Level_Ensemble_Dataset.csv`.

---

### Step 2 — Train base models (two-phase fine-tuning)

```bash
# All three models sequentially (~6–10 h on RTX 3090)
python scripts/train_ensemble_models.py --model all

# Individual models
python scripts/train_ensemble_models.py --model 1   # Event Type
python scripts/train_ensemble_models.py --model 2   # Binary
python scripts/train_ensemble_models.py --model 3   # Disease

# Resume from checkpoint
python scripts/resume_ensemble_training.py --model1_epochs 10
```

**Two-phase protocol**

| Phase | Encoder | Epochs | LR | Purpose |
|---|---|---|---|---|
| 1 | Frozen | 10 | 1 × 10⁻⁴ | Fast head adaptation, preserve pretrained representations |
| 2 | Unfrozen | 40 | 5 × 10⁻⁷ | Controlled end-to-end fine-tuning, prevent catastrophic forgetting |

Common settings: `batch = 32`, `gradient_accumulation = 2` (effective 64), `weight_decay = 1e-4`, `max_grad_norm = 0.5`, class-weighted CrossEntropy, `ReduceLROnPlateau`, `EarlyStopping(patience = 10)`.

> **HuggingFace access**: HeAR is a gated model. Request access at  
> https://huggingface.co/google/hear-pytorch then add `HF_TOKEN=<token>` to a `.env` file.

---

### Step 3 — Extract base-model probabilities

```bash
python scripts/extract_ensemble_probabilities.py
```

Runs all three base models over train and val splits; saves softmax outputs to `data/ensemble_probabilities_train.csv` and `data/ensemble_probabilities_val.csv`.

---

### Step 4 — Train LightGBM stacking meta-models

```bash
python scripts/train_meta_model.py --use_csv --n_trials 100
```

For each of the 5 outcomes: Optuna (TPE, 100 trials) searches over `learning_rate`, `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, and regularization terms → final LightGBM refit → 1,000-iteration bootstrap 95 % CI evaluation → saves `model.pkl`, `metrics.json`, `report.md`, `confusion_matrix.png`.

---

### Step 5 — Inference

```bash
# Single event
python scripts/predict_ensemble.py --mode single \
    --audio_path data/wavfiles/example.wav \
    --start_ms 1200 --end_ms 2800

# Batch
python scripts/predict_ensemble.py --mode batch \
    --csv_path data/SPRSound_Event_Level_Ensemble_Dataset.csv \
    --output_path predictions.csv
```

---

## Results

All metrics are on the held-out validation split (20 % stratified). Bootstrap CI = 1,000-iteration nonparametric resampling.

### Base classifier performance (HeAR + classification head)

| Model | Accuracy (95 % CI) | Weighted-F1 (95 % CI) | ROC-AUC (95 % CI) |
|---|---|---|---|
| **Model 2** — Binary screening | 0.92 [0.92–0.93] | 0.92 [0.91–0.93] | 0.96 [0.95–0.97] |
| **Model 1** — Acoustic phenotype (3-class) | 0.91 [0.90–0.92] | 0.91 [0.90–0.92] | 0.96 [0.95–0.96] OvR weighted |
| **Event type** (7-class, raw) | 0.90 [0.89–0.91] | 0.89 [0.88–0.90] | 0.96 [0.95–0.96] OvR weighted |
| **Model 3** — Disease group (3-class) | 0.81 [0.80–0.82] | 0.81 [0.80–0.82] | 0.93 [0.93–0.94] OvR weighted |
| **Model 4** — Extended disease group (4-class) | 0.80 [0.78–0.81] | 0.79 [0.78–0.80] | 0.94 [0.93–0.94] OvR weighted |
| **Disease** (17-class, raw) | 0.74 [0.73–0.76] | 0.73 [0.72–0.75] | 0.98 [0.97–0.98] macro |

Model 4 macro-AUC = 0.94, macro-AUPRC = 0.82.  
Disease 17-class weighted AUC = 0.94 [0.93–0.94].

---

### Table 1 — Model 1 per-class (Acoustic phenotype, 3-class)

Validation set: Normal n = 3,743 · Crackles n = 813 · Rhonchi n = 345

| Class | PPV | Sensitivity | F1-Score | Specificity | NPV | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|
| Normal | 0.93 [0.92–0.94] | 0.97 [0.96–0.97] | 0.95 [0.94–0.95] | 0.76 [0.74–0.79] | 0.88 [0.86–0.89] | 0.96 [0.95–0.96] |
| Crackles | 0.82 [0.79–0.84] | 0.68 [0.65–0.72] | 0.74 [0.72–0.77] | 0.97 [0.96–0.97] | 0.94 [0.93–0.95] | 0.95 [0.94–0.96] |
| Rhonchi | 0.88 [0.84–0.91] | 0.83 [0.80–0.87] | 0.86 [0.83–0.88] | 0.99 [0.99–0.99] | 0.99 [0.98–0.99] | 0.98 [0.97–0.99] |

**Clinical interpretation**: Crackle specificity 0.97 (high confidence when positive) but sensitivity 0.68 (misses ~32 % of crackle events). This is clinically coherent: crackles are transient, focal, and inspiratory-phase dominant; a 2-second clip recorded over a non-affected lung region will be acoustically silent even in a patient with active crackles elsewhere. Rhonchi/Wheeze shows the best AUC (0.98), consistent with the distinctive continuous acoustic character of these sounds.

---

### Table 2 — Model 4 per-class (Extended disease group, 4-class)

Validation set: Pneumonia n = 2,436 · Bronchoconstriction n = 920 · Normal n = 405 · Others n = 1,140

| Class | PPV | Sensitivity | F1-Score | Specificity | NPV | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|
| Pneumonia | 0.82 [0.81–0.84] | 0.89 [0.88–0.90] | 0.85 [0.84–0.86] | 0.81 [0.79–0.82] | 0.88 [0.87–0.89] | 0.93 [0.92–0.94] |
| Bronchoconstriction | 0.75 [0.72–0.78] | 0.63 [0.59–0.66] | 0.68 [0.66–0.70] | 0.95 [0.94–0.96] | 0.92 [0.91–0.93] | 0.92 [0.91–0.93] |
| Normal | 0.71 [0.66–0.76] | 0.54 [0.49–0.59] | 0.61 [0.57–0.65] | 0.98 [0.98–0.98] | 0.96 [0.95–0.97] | 0.94 [0.93–0.95] |
| Others | 0.79 [0.77–0.82] | 0.82 [0.80–0.84] | 0.81 [0.79–0.82] | 0.94 [0.93–0.94] | 0.95 [0.94–0.95] | 0.96 [0.96–0.97] |

---

### LightGBM meta-model summary (stacking layer)

| Outcome | Accuracy (95 % CI) | Macro-F1 (95 % CI) | Weighted-F1 (95 % CI) | ROC-AUC |
|---|---|---|---|---|
| `model2_label` — Binary | 0.921 [0.913–0.928] | 0.885 [0.874–0.896] | 0.919 [0.911–0.926] | 0.959 [0.952–0.965] |
| `model1_label` — 3-class acoustic | 0.916 [0.907–0.924] | 0.854 [0.838–0.868] | 0.913 [0.904–0.921] | 0.965 macro |
| `model3_label` — 3-class disease | 0.811 [0.800–0.822] | 0.776 [0.762–0.789] | 0.808 [0.796–0.819] | 0.933 macro |
| `event_type` — 7-class | 0.899 [0.890–0.909] | 0.651 [0.609–0.694] | 0.892 [0.882–0.903] | 0.963 macro |
| `disease` — 17-class | 0.763 [0.751–0.775] | 0.605 [0.556–0.662] | 0.754 [0.741–0.768] | 0.979 macro |

### LightGBM meta-model per-class detail

**Binary (`model2_label`)**

| Class | n | PPV | Sensitivity | Specificity | NPV | AUC |
|---|---|---|---|---|---|---|
| Normal | 3,743 | 0.930 | 0.969 | 0.766 | 0.883 | 0.959 |
| Abnormal | 1,158 | 0.883 | 0.766 | 0.969 | 0.930 | 0.959 |

**Acoustic phenotype (`model1_label`)**

| Class | n | PPV | Sensitivity | Specificity | NPV | AUC (OvR) |
|---|---|---|---|---|---|---|
| Normal | 3,743 | 0.932 | 0.972 | 0.770 | 0.896 | 0.959 [0.953–0.966] |
| Crackles | 813 | 0.839 | 0.695 | 0.973 | 0.941 | 0.951 [0.944–0.959] |
| Rhonchi/Wheeze | 345 | 0.879 | 0.822 | 0.991 | 0.987 | 0.984 [0.975–0.990] |

**Disease group (`model3_label`)**

| Class | n | PPV | Sensitivity | Specificity | NPV | AUC (OvR) |
|---|---|---|---|---|---|---|
| Normal/Other | 2,436 | 0.834 | 0.883 | 0.826 | 0.878 | 0.934 [0.927–0.940] |
| Pneumonia | 835 | 0.737 | 0.600 | 0.956 | 0.921 | 0.924 [0.914–0.932] |
| Bronchitis-Asthma-Bronch. | 1,630 | 0.806 | 0.811 | 0.903 | 0.905 | 0.942 [0.936–0.949] |

### Key qualitative finding — "The model hears well; the labels fall short"

When randomly sampled recordings were listened to by the pediatric emergency physicians on the team, the model's acoustic predictions frequently made more sense than the dataset labels. A clip labeled "Rhonchi" clearly contained short, explosive, discontinuous bursts consistent with crackles; the model predicted "Crackles" with 0.86 probability, confirmed on auscultation — yet this counted as a misclassification. In other cases, recordings contained multiple audible adventitious events, but only one type was annotated; the model detected the unannotated sounds and was penalized.

These are not isolated cases. **The bottleneck is no longer model capacity; it is label validity.**

This observation motivates the label governance protocol and Phase 1 of the validation roadmap.

---

## Key Configuration

All hyperparameters live in `src/config.py`:

| Parameter | Value | Note |
|---|---|---|
| `SAMPLE_RATE` | 16,000 Hz | HeAR requirement |
| `CLIP_DURATION` | 2 s | HeAR requirement |
| `CLIP_LENGTH` | 32,000 samples | |
| `OVERLAP_PERCENT` | 10 % | Jitter around annotation boundaries |
| `EMBEDDING_DIM` | 512 | HeAR pooler output |
| `HIDDEN_DIM` | 256 | Classification head intermediate layer |
| `DROPOUT` | 0.3 | Applied in classification head |
| `BATCH_SIZE` | 32 | Effective 64 with accumulation |
| `ACCUMULATION_STEPS` | 2 | Gradient accumulation |
| `PHASE1_LR` | 1 × 10⁻⁴ | Frozen-encoder phase |
| `PHASE2_LR` | 5 × 10⁻⁷ | Fine-tuning phase |
| `WEIGHT_DECAY` | 1 × 10⁻⁴ | AdamW |
| `MAX_GRAD_NORM` | 0.5 | Gradient clipping |
| `TRAIN_SPLIT` | 0.80 | Stratified |
| `RANDOM_SEED` | 42 | |
| `USE_AMP` | False | Disabled for numerical stability |

**Hardware used**: NVIDIA RTX 3090 (24 GB VRAM), CUDA 12.8.  
**Estimated training time**: ~6–10 h for all base models; ~5 min per meta-model.

---

## Data Flow Diagram

```
SPRSound WAV files  (data/wavfiles/)
           │
           ▼
SPRSound_Event_Level_Dataset_CLEAN.csv   (20,007 rows, wav-verified)
           │
           ├──► analyze_disease_event_pivot.py ──► pivot CSVs + heatmap
           │
           ▼
prepare_ensemble_labels.py
           │
           ▼
SPRSound_Event_Level_Ensemble_Dataset.csv  (19,806 rows + model1/2/3 labels)
           │
           ▼
train_ensemble_models.py   (HeAR + 2-phase fine-tuning × 3 base models)
    ├── model1_event_type/best.pth    wF1=0.91
    ├── model2_binary/best.pth        wF1=0.92
    └── model3_disease/best.pth       wF1=0.81
           │
           ▼
extract_ensemble_probabilities.py
    ├── ensemble_probabilities_train.csv   (8 proba + age + gender + location)
    └── ensemble_probabilities_val.csv
           │
           ▼
train_meta_model.py   (LightGBM × 5 outcomes, Optuna 100 trials, 1000-iter bootstrap CI)
    └── meta_model/
        ├── model2_label/  ← Binary          AUC 0.959
        ├── model1_label/  ← 3-class         AUC 0.965 macro
        ├── model3_label/  ← 3-class disease AUC 0.933 macro
        ├── event_type/    ← 7-class         AUC 0.963 macro
        └── disease/       ← 17-class        AUC 0.979 macro
           │
           ▼
predict_ensemble.py   (single-event or batch CSV inference)
```

---

## Label Governance

High metrics on an open dataset do not guarantee bedside utility when labels are misaligned with clinical reality. The PulmoVec label governance protocol:

1. **Labeling manual**: Clear definitions for crackles, rhonchi, wheeze, mixed patterns (Wheeze+Crackle), and disease group mappings. Specifies inspiratory vs expiratory phase, duration criteria, and recording-quality thresholds.

2. **Multi-expert annotation**: At least two independent clinician annotators per clip, with inter-rater agreement analysis (Cohen's κ).

3. **Discordance-driven review**: The model itself is used to flag clips where predictions conflict with existing labels. This focuses expert review time on the most ambiguous cases — precisely where label noise is highest.

4. **Panel adjudication**: Discordant cases resolved by a third clinician reviewer, building a high-confidence reference set incrementally.

This protocol is the primary objective of Phase 1 of the validation roadmap.

---

## Mobile Workflow and Safety Gating

Because HeAR was pretrained on a massive, heterogeneous audio corpus, it produces useful representations even from noisy smartphone recordings. Families do not need a professional stethoscope.

### Recording workflow
The app guides caregivers through short recordings at each auscultation location, stores a baseline from the discharge day, and tracks probability shifts over 48–72 hours post-discharge.

### Three-level safety architecture

| Level | Mechanism | Action |
|---|---|---|
| **Quality gate** | Duration check, clipping detection, SNR, non-respiratory dominance | Rejects noisy recordings with actionable feedback ("move to quieter room") |
| **Abstention** | Low model confidence or base-learner disagreement | Returns "insufficient certainty — repeat or contact your care team" |
| **Change tracking** | Probability shift relative to patient baseline | Distinguishes true acoustic change from recording variability |

The output is positioned as a **triage support signal** for clinicians — not a standalone diagnosis.

---

## Validation Roadmap

### Phase 1 (Months 1–4) — Clinically adjudicated reference set
- Re-annotate a curated SPRSound subset using the governance protocol above.
- Quantify inter-rater agreement; resolve discordant cases by panel adjudication.
- Retrain PulmoVec on cleaned labels with **strict patient-level split** (no event-level split).
- Primary objective: establish a reference set that addresses the label-validity bottleneck.

### Phase 2 (Months 3–8) — Cross-dataset generalization
- Incorporate additional publicly available respiratory sound datasets (different devices, populations, recording conditions).
- Evaluate cross-dataset generalization; fine-tune quality gate for device diversity.
- Edge-device microphone calibration experiments.

### Phase 3 (Months 6–12) — Prospective clinical pilot
- Prospective validation at the participating pediatric emergency department with fresh recordings.
- Head-to-head comparison against bedside clinical auscultation assessment.
- Multi-center generalizability evaluation.
- Pre-deployment safety and performance threshold definition.

---

## Limitations

1. **Event-level split and data leakage**: Current results are based on an event-level train/validation split. In physiological acoustic datasets, event-level splitting carries an inherent risk of data leakage — models may learn patient-specific acoustic signatures rather than generalized disease features. These metrics demonstrate the representation power of the HeAR backbone; they should not be interpreted as a finalized clinical performance threshold. Resolving this is the primary objective of Phase 1 (patient-level split, adjudicated reference set).

2. **Label validity**: SPRSound annotations were produced under standard academic labeling conditions. As documented in the "Key finding" section above, clinician review identified multiple cases where model predictions were acoustically more correct than the original labels. Model performance is bounded by label quality.

3. **Dataset scope**: All results are from a single pediatric dataset (SPRSound, SCMC, Shanghai). Multi-center generalizability and device heterogeneity have not been prospectively evaluated.

4. **Missing WAV files**: 1,307 audio files (out of 24,808 curated events) are absent from this machine. These rows were removed from both CSVs. Results may shift slightly when the full cohort is available.

5. **Home-deployment readiness**: Edge-device microphone calibration, real-world caregiver technique variability, and prospective clinical pilot results are prerequisites before deployment.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: models` | Run from `Projects/` root: `cd Projects && python scripts/...` |
| NaN loss during Phase 2 | Already mitigated: `MAX_GRAD_NORM = 0.5`, `USE_AMP = False`, `PHASE2_LR = 5e-7` |
| CUDA out of memory | Reduce `BATCH_SIZE` to 16, raise `ACCUMULATION_STEPS` to 4 |
| HuggingFace auth error | `huggingface-cli login` or add `HF_TOKEN=<token>` to `.env` |
| `wav_path` not found | Ensure you run from `Projects/`; relative paths resolve via `config.PROJECT_ROOT` |
| Missing WAV files (`wav_exists = no`) | 1,307 files absent from this machine; those rows were removed from both CSVs |
| Training very slow | Expected: ~100 s/epoch on RTX 3090. Verify GPU utilization with `nvidia-smi`. |

---

## References

1. **SPRSound**: Zhang et al., *IEEE Transactions on Biomedical Circuits and Systems*, 2022.
2. **HeAR**: Google Research, *Health Acoustic Representations*, 2023. https://huggingface.co/google/hear-pytorch
3. **Pediatric Pneumonia**: WHO Guidelines, 2014.
4. **Bronchiolitis**: AAP Clinical Practice Guideline, 2014.
5. **RSV Epidemiology**: Hall et al., *NEJM*, 2009.
6. **Pediatric Asthma**: GINA Guidelines, 2023.
7. **LightGBM**: Ke et al., *NeurIPS*, 2017.
8. **Optuna**: Akiba et al., *KDD*, 2019.
9. **Bootstrap CI**: Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993.

---

*Branch: `turkalpmdmbp` — Last updated: 2026-02-23*
