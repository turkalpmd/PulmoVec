# LightGBM Meta-Model Report: model2_label

## Overview

This meta-model predicts **model2_label** using ensemble model probabilities and demographic features.

**Input Features (10 total):**
- Model 1 probabilities (3): Normal, Crackles, Rhonchi
- Model 2 probabilities (2): Normal, Abnormal
- Model 3 probabilities (3): Normal, Pneumonia, Bronchiolitis
- Demographics (2): age, gender

**Output Classes:** 2
- Normal, Abnormal

---

## Performance Metrics (with 95% Confidence Intervals)

### Basic Metrics

#### Accuracy
- **Value**: 0.9206
- **CI95**: [0.9131, 0.9280]

#### Macro F1
- **Value**: 0.8847
- **CI95**: [0.8737, 0.8956]

#### Weighted F1
- **Value**: 0.9186
- **CI95**: [0.9106, 0.9264]

#### Matthews Correlation Coefficient (MCC)
- **Value**: 0.7730
- **CI95**: [0.7514, 0.7942]

### Probabilistic Metrics

#### Log-Loss
- **Value**: 0.2057
- **CI95**: [0.1875, 0.2227]

#### ROC-AUC (Binary)
- **Value**: 0.9588
- **CI95**: [0.9523, 0.9652]

### Per-Class Metrics

| Class | Precision (PPV) | Recall (Sensitivity) | F1-Score | Specificity | NPV | Support | ROC-AUC (OvR) |
|-------|------------------|----------------------|----------|-------------|-----|---------|---------------|
| Normal | 0.9303 [0.9221, 0.9379] | 0.9686 [0.9633, 0.9741] | 0.9491 | 0.7659 [0.7415, 0.7900] | 0.8834 [0.8647, 0.9032] | 3743 | N/A |
| Abnormal | 0.8834 [0.8647, 0.9032] | 0.7659 [0.7415, 0.7900] | 0.8202 | 0.9686 [0.9633, 0.9741] | 0.9303 [0.9221, 0.9379] | 1158 | N/A |

---

## Confusion Matrix

See `confusion_matrix.png` for detailed confusion matrix visualization.

---

**Report Generated**: 2026-01-17 00:10:37
