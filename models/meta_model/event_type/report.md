# LightGBM Meta-Model Report: event_type

## Overview

This meta-model predicts **event_type** using ensemble model probabilities and demographic features.

**Input Features (10 total):**
- Model 1 probabilities (3): Normal, Crackles, Rhonchi
- Model 2 probabilities (2): Normal, Abnormal
- Model 3 probabilities (3): Normal, Pneumonia, Bronchiolitis
- Demographics (2): age, gender

**Output Classes:** 6
- Coarse Crackle, Fine Crackle, Normal, Rhonchi, Wheeze, Wheeze+Crackle

---

## Performance Metrics (with 95% Confidence Intervals)

### Basic Metrics

#### Accuracy
- **Value**: 0.8994
- **CI95**: [0.8902, 0.9086]

#### Macro F1
- **Value**: 0.6512
- **CI95**: [0.6087, 0.6938]

#### Weighted F1
- **Value**: 0.8924
- **CI95**: [0.8823, 0.9026]

#### Matthews Correlation Coefficient (MCC)
- **Value**: 0.7288
- **CI95**: [0.7067, 0.7519]

### Probabilistic Metrics

#### Log-Loss
- **Value**: 0.3012
- **CI95**: [0.2788, 0.3239]

#### ROC-AUC (One-vs-Rest)

**Macro Average:**
- **Value**: 0.9633
- **CI95**: [0.9533, 0.9705]

**Weighted Average:**
- **Value**: 0.9574
- **CI95**: [0.9512, 0.9634]

### Per-Class Metrics

| Class | Precision (PPV) | Recall (Sensitivity) | F1-Score | Specificity | NPV | Support | ROC-AUC (OvR) |
|-------|------------------|----------------------|----------|-------------|-----|---------|---------------|
| Coarse Crackle | 0.8291 [0.4000, 1.0000] | 0.1312 [0.0322, 0.2500] | 0.2326 | 0.9998 [0.9994, 1.0000] | 0.9934 [0.9912, 0.9955] | 37 | 0.9474 [0.9239, 0.9661] |
| Fine Crackle | 0.7889 [0.7548, 0.8213] | 0.6631 [0.6287, 0.6989] | 0.7199 | 0.9703 [0.9647, 0.9755] | 0.9451 [0.9382, 0.9521] | 700 | 0.9452 [0.9366, 0.9537] |
| Normal | 0.9231 [0.9149, 0.9313] | 0.9734 [0.9683, 0.9785] | 0.9476 | 0.7385 [0.7144, 0.7641] | 0.8960 [0.8776, 0.9155] | 3743 | 0.9571 [0.9508, 0.9637] |
| Rhonchi | 0.8446 [0.7142, 0.9615] | 0.5526 [0.4130, 0.6945] | 0.6667 | 0.9990 [0.9981, 0.9998] | 0.9955 [0.9934, 0.9973] | 49 | 0.9697 [0.9181, 0.9940] |
| Wheeze | 0.8313 [0.7862, 0.8741] | 0.8008 [0.7532, 0.8457] | 0.8158 | 0.9896 [0.9865, 0.9924] | 0.9872 [0.9839, 0.9904] | 296 | 0.9842 [0.9780, 0.9894] |
| Wheeze+Crackle | 0.7442 [0.6000, 0.8637] | 0.4217 [0.3056, 0.5410] | 0.5378 | 0.9977 [0.9963, 0.9990] | 0.9909 [0.9883, 0.9936] | 76 | 0.9762 [0.9604, 0.9868] |

---

## Confusion Matrix

See `confusion_matrix.png` for detailed confusion matrix visualization.

---

**Report Generated**: 2026-01-17 00:08:45
