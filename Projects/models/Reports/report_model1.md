# LightGBM Meta-Model Report: model1_label

## Overview

This meta-model predicts **model1_label** using ensemble model probabilities and demographic features.

**Input Features (10 total):**
- Model 1 probabilities (3): Normal, Crackles, Rhonchi
- Model 2 probabilities (2): Normal, Abnormal
- Model 3 probabilities (3): Normal, Pneumonia, Bronchiolitis
- Demographics (2): age, gender

**Output Classes:** 3
- Normal, Crackles, Rhonchi

---

## Performance Metrics (with 95% Confidence Intervals)

### Basic Metrics

#### Accuracy
- **Value**: 0.9155
- **CI95**: [0.9074, 0.9237]

#### Macro F1
- **Value**: 0.8538
- **CI95**: [0.8383, 0.8682]

#### Weighted F1
- **Value**: 0.9125
- **CI95**: [0.9040, 0.9212]

#### Matthews Correlation Coefficient (MCC)
- **Value**: 0.7715
- **CI95**: [0.7509, 0.7921]

### Probabilistic Metrics

#### Log-Loss
- **Value**: 0.2422
- **CI95**: [0.2219, 0.2628]

#### ROC-AUC (One-vs-Rest)

**Macro Average:**
- **Value**: 0.9648
- **CI95**: [0.9590, 0.9701]

**Weighted Average:**
- **Value**: 0.9596
- **CI95**: [0.9535, 0.9657]

### Per-Class Metrics

| Class | Precision (PPV) | Recall (Sensitivity) | F1-Score | Specificity | NPV | Support | ROC-AUC (OvR) |
|-------|------------------|----------------------|----------|-------------|-----|---------|---------------|
| Normal | 0.9317 [0.9236, 0.9395] | 0.9721 [0.9671, 0.9775] | 0.9515 | 0.7701 [0.7462, 0.7932] | 0.8956 [0.8784, 0.9148] | 3743 | 0.9592 [0.9529, 0.9656] |
| Crackles | 0.8392 [0.8106, 0.8671] | 0.6950 [0.6635, 0.7275] | 0.7599 | 0.9734 [0.9685, 0.9784] | 0.9411 [0.9331, 0.9484] | 813 | 0.9510 [0.9436, 0.9590] |
| Rhonchi | 0.8793 [0.8433, 0.9133] | 0.8224 [0.7822, 0.8601] | 0.8503 | 0.9915 [0.9886, 0.9939] | 0.9866 [0.9832, 0.9897] | 345 | 0.9844 [0.9753, 0.9904] |

---

## Confusion Matrix

See `confusion_matrix.png` for detailed confusion matrix visualization.

---

**Report Generated**: 2026-01-17 00:10:09
