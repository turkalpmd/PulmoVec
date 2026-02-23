# LightGBM Meta-Model Report: disease

## Overview

This meta-model predicts **disease** using ensemble model probabilities and demographic features.

**Input Features (10 total):**
- Model 1 probabilities (3): Normal, Crackles, Rhonchi
- Model 2 probabilities (2): Normal, Abnormal
- Model 3 probabilities (3): Normal, Pneumonia, Bronchiolitis
- Demographics (2): age, gender

**Output Classes:** 16
- Acute upper respiratory infection, Airway foreign body, Asthma, Bronchiectasia, Bronchiolitis, Bronchitis, Chronic cough, Control Group, Hemoptysis, Kawasaki disease, Other respiratory diseases, Pneumonia (non-severe), Pneumonia (severe), Protracted bacterial bronchitis, Pulmonary hemosiderosis, Unknown

---

## Performance Metrics (with 95% Confidence Intervals)

### Basic Metrics

#### Accuracy
- **Value**: 0.7628
- **CI95**: [0.7505, 0.7749]

#### Macro F1
- **Value**: 0.6049
- **CI95**: [0.5564, 0.6619]

#### Weighted F1
- **Value**: 0.7544
- **CI95**: [0.7408, 0.7676]

#### Matthews Correlation Coefficient (MCC)
- **Value**: 0.6725
- **CI95**: [0.6564, 0.6887]

### Probabilistic Metrics

#### Log-Loss
- **Value**: 0.6563
- **CI95**: [0.6286, 0.6832]

#### ROC-AUC (One-vs-Rest)

**Macro Average:**
- **Value**: 0.9787
- **CI95**: [0.9765, 0.9806]

**Weighted Average:**
- **Value**: 0.9455
- **CI95**: [0.9410, 0.9500]

### Per-Class Metrics

| Class | Precision (PPV) | Recall (Sensitivity) | F1-Score | Specificity | NPV | Support | ROC-AUC (OvR) |
|-------|------------------|----------------------|----------|-------------|-----|---------|---------------|
| Acute upper respiratory infection | 0.7785 [0.6285, 0.9118] | 0.5604 [0.4166, 0.7001] | 0.6512 | 0.9984 [0.9971, 0.9994] | 0.9955 [0.9934, 0.9973] | 50 | 0.9790 [0.9560, 0.9940] |
| Airway foreign body | 0.9950 [1.0000, 1.0000] | 0.6261 [0.2500, 1.0000] | 0.7692 | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9986, 1.0000] | 8 | 0.9947 [0.9866, 1.0000] |
| Asthma | 0.7070 [0.6470, 0.7667] | 0.5143 [0.4589, 0.5714] | 0.5957 | 0.9851 [0.9817, 0.9885] | 0.9667 [0.9612, 0.9719] | 321 | 0.9547 [0.9447, 0.9638] |
| Bronchiectasia | 0.7254 [0.6000, 0.8373] | 0.4818 [0.3797, 0.5859] | 0.5778 | 0.9969 [0.9952, 0.9983] | 0.9913 [0.9888, 0.9938] | 81 | 0.9934 [0.9902, 0.9961] |
| Bronchiolitis | 0.7280 [0.6071, 0.8364] | 0.8596 [0.7567, 0.9460] | 0.7890 | 0.9967 [0.9950, 0.9981] | 0.9985 [0.9975, 0.9996] | 50 | 0.9975 [0.9949, 0.9993] |
| Bronchitis | 0.6572 [0.6088, 0.7035] | 0.5373 [0.4901, 0.5856] | 0.5924 | 0.9707 [0.9659, 0.9755] | 0.9525 [0.9461, 0.9585] | 464 | 0.9341 [0.9232, 0.9438] |
| Chronic cough | 0.3367 [0.0000, 0.7783] | 0.1720 [0.0000, 0.4286] | 0.2222 | 0.9992 [0.9984, 0.9998] | 0.9980 [0.9967, 0.9992] | 12 | 0.9977 [0.9960, 0.9991] |
| Control Group | 0.6560 [0.6042, 0.7086] | 0.5212 [0.4716, 0.5725] | 0.5805 | 0.9754 [0.9709, 0.9799] | 0.9576 [0.9514, 0.9634] | 405 | 0.9406 [0.9297, 0.9510] |
| Hemoptysis | 0.6998 [0.5833, 0.8200] | 0.9071 [0.8163, 0.9792] | 0.7879 | 0.9966 [0.9949, 0.9981] | 0.9992 [0.9983, 0.9998] | 43 | 0.9990 [0.9983, 0.9996] |
| Kawasaki disease | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9986, 1.0000] | 3 | 0.9981 [0.9949, 1.0000] |
| Other respiratory diseases | 0.7302 [0.6190, 0.8333] | 0.6557 [0.5400, 0.7561] | 0.6917 | 0.9965 [0.9948, 0.9979] | 0.9950 [0.9930, 0.9969] | 70 | 0.9927 [0.9881, 0.9961] |
| Pneumonia (non-severe) | 0.7881 [0.7715, 0.8037] | 0.8777 [0.8642, 0.8903] | 0.8307 | 0.8101 [0.7956, 0.8246] | 0.8917 [0.8797, 0.9039] | 2185 | 0.9238 [0.9164, 0.9309] |
| Pneumonia (severe) | 0.7608 [0.7005, 0.8182] | 0.6858 [0.6290, 0.7396] | 0.7212 | 0.9884 [0.9850, 0.9912] | 0.9831 [0.9795, 0.9867] | 251 | 0.9807 [0.9735, 0.9864] |
| Protracted bacterial bronchitis | 0.6120 [0.0000, 1.0000] | 0.2383 [0.0000, 1.0000] | 0.4000 | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9986, 1.0000] | 4 | 0.9996 [0.9990, 1.0000] |
| Pulmonary hemosiderosis | 0.7047 [0.4000, 1.0000] | 0.6928 [0.3750, 1.0000] | 0.7000 | 0.9994 [0.9986, 1.0000] | 0.9994 [0.9986, 1.0000] | 10 | 0.9994 [0.9987, 0.9999] |
| Unknown | 0.7994 [0.7766, 0.8236] | 0.8612 [0.8379, 0.8838] | 0.8292 | 0.9483 [0.9417, 0.9550] | 0.9662 [0.9600, 0.9719] | 944 | 0.9748 [0.9705, 0.9789] |

---

## Confusion Matrix

See `confusion_matrix.png` for detailed confusion matrix visualization.

---

**Report Generated**: 2026-01-17 00:07:21
