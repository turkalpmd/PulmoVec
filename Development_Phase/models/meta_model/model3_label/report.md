# LightGBM Meta-Model Report: model3_label

## Overview

This meta-model predicts **model3_label** using ensemble model probabilities and demographic features.

**Input Features (10 total):**
- Model 1 probabilities (3): Normal, Crackles, Rhonchi
- Model 2 probabilities (2): Normal, Abnormal
- Model 3 probabilities (3): Normal, Pneumonia, Bronchiolitis
- Demographics (2): age, gender

**Output Classes:** 3
- Normal, Pneumonia, Bronchiolitis

---

## Performance Metrics (with 95% Confidence Intervals)

### Basic Metrics

#### Accuracy
- **Value**: 0.8110
- **CI95**: [0.7996, 0.8217]

#### Macro F1
- **Value**: 0.7758
- **CI95**: [0.7624, 0.7886]

#### Weighted F1
- **Value**: 0.8079
- **CI95**: [0.7961, 0.8189]

#### Matthews Correlation Coefficient (MCC)
- **Value**: 0.6879
- **CI95**: [0.6692, 0.7057]

### Probabilistic Metrics

#### Log-Loss
- **Value**: 0.4646
- **CI95**: [0.4455, 0.4840]

#### ROC-AUC (One-vs-Rest)

**Macro Average:**
- **Value**: 0.9332
- **CI95**: [0.9274, 0.9387]

**Weighted Average:**
- **Value**: 0.9349
- **CI95**: [0.9291, 0.9402]

### Per-Class Metrics

| Class | Precision (PPV) | Recall (Sensitivity) | F1-Score | Specificity | NPV | Support | ROC-AUC (OvR) |
|-------|------------------|----------------------|----------|-------------|-----|---------|---------------|
| Normal | 0.8336 [0.8186, 0.8482] | 0.8834 [0.8709, 0.8953] | 0.8579 | 0.8258 [0.8098, 0.8419] | 0.8775 [0.8635, 0.8903] | 2436 | 0.9337 [0.9270, 0.9396] |
| Pneumonia | 0.7370 [0.7036, 0.7709] | 0.5997 [0.5659, 0.6312] | 0.6618 | 0.9561 [0.9497, 0.9621] | 0.9209 [0.9125, 0.9297] | 835 | 0.9236 [0.9139, 0.9323] |
| Bronchiolitis | 0.8061 [0.7849, 0.8256] | 0.8110 [0.7904, 0.8315] | 0.8083 | 0.9027 [0.8918, 0.9132] | 0.9054 [0.8943, 0.9156] | 1630 | 0.9424 [0.9358, 0.9485] |

---

## Confusion Matrix

See `confusion_matrix.png` for detailed confusion matrix visualization.

---

**Report Generated**: 2026-01-17 00:12:10
