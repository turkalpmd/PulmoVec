#!/usr/bin/env python3
"""
scripts/benchmark_meta_models.py

Comprehensive Multi-Model Benchmark for PulmoVec's Stacking Meta-Learner:
Addresses Reviewer comments by rigorously benchmarking LightGBM against:
1. XGBoost (Gradient Boosting)
2. CatBoost / HistGradientBoosting (Histogram GBDT)
3. Random Forest (Bagging Ensemble of Deep Trees)
4. Extra Trees (Extremely Randomized Trees)
5. Multi-Layer Perceptron (MLP Neural Network)
6. Regularized Logistic Regression (Softmax Baseline)
7. Soft-Voting Ensemble (LightGBM + XGBoost + Random Forest)

Evaluates on the held-out Zero-Leakage Test Set across all clinical outcomes:
- Model 1 (Sound Event Type: Normal vs. Crackles vs. Wheeze/Rhonchi)
- Model 2 (Binary Abnormality: Normal vs. Abnormal)
- Model 3 (Disease Groups: Pneumonia vs. Bronchitis/Asthma vs. Normal/Other)
- Disease (16-Class Fine-Grained Diagnosis)

Generates:
- Manuscript/Submission/Supplementary_Table_Model_Benchmark.csv
- data/meta_model_benchmark_results.json
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelBenchmark")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MS_DIR = PROJECT_ROOT / "Manuscript" / "Submission"
MS_DIR.mkdir(parents=True, exist_ok=True)

PROB_COLS = [
    'Model1_Normalpp', 'Model1_Cracklespp', 'Model1_Rhonchipp',
    'Model2_Normalpp', 'Model2_Abnormalpp',
    'Model3_Normalpp', 'Model3_Pneumoniapp', 'Model3_Bronchiolitispp'
]

def load_and_prep_datasets():
    train_df = pd.read_csv(DATA_DIR / "ensemble_probabilities_train.csv")
    val_df = pd.read_csv(DATA_DIR / "ensemble_probabilities_val.csv")
    test_df = pd.read_csv(DATA_DIR / "ensemble_probabilities_test.csv")
    
    for df in [train_df, val_df, test_df]:
        df['gender_encoded'] = df['gender'].map(
            {'Female': 0, 'Male': 1, 'F': 0, 'M': 1, 0: 0, 1: 1}
        ).fillna(0).astype(int)
        if df['age'].isna().any():
            df['age'] = df['age'].fillna(train_df['age'].median())
            
    feature_cols = PROB_COLS + ['age', 'gender_encoded']
    return train_df, val_df, test_df, feature_cols


def get_candidate_models(n_classes: int) -> Dict[str, Any]:
    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=4
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=4
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=4
        ),
        "Hist-GBDT": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            alpha=1e-3,
            early_stopping=True,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            random_state=42
        ),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            n_jobs=4
        )
        
    if HAS_CATBOOST:
        models["CatBoost"] = cb.CatBoostClassifier(
            iterations=150,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_seed=42
        )
        
    # Voting ensemble of top tree models
    estimators = [
        ('lgb', models['LightGBM']),
        ('rf', models['Random Forest']),
    ]
    if HAS_XGBOOST:
        estimators.append(('xgb', models['XGBoost']))
        
    models["Voting Ensemble"] = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    
    return models


def evaluate_predictions(y_true, y_pred, y_prob, n_classes: int) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = np.nan
        
    return {
        "Accuracy": float(acc),
        "Macro_F1": float(f1_macro),
        "Weighted_F1": float(f1_weighted),
        "AUROC": float(auc)
    }


def main():
    logger.info("="*75)
    logger.info("PULMOVEC: EXPERIMENTAL MULTI-MODEL BENCHMARK FOR STACKING META-LEARNER")
    logger.info("="*75)
    
    train_df, val_df, test_df, feature_cols = load_and_prep_datasets()
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # Standardize for MLP & Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    tasks = {
        "Model 1 (Event Type)": "model1_label",
        "Model 2 (Binary Abnormality)": "model2_label",
        "Model 3 (Disease Groups)": "model3_label",
        "Clinical Diagnosis (Disease)": "disease"
    }
    
    all_benchmark_rows = []
    benchmark_dict = {}
    
    for task_name, target_col in tasks.items():
        logger.info(f"\n" + "-"*65)
        logger.info(f"Evaluating Task: {task_name} (Target: {target_col})")
        logger.info("-"*65)
        
        y_train_raw = train_df[target_col].values
        y_val_raw = val_df[target_col].values
        y_test_raw = test_df[target_col].values
        
        if target_col in ['disease', 'event_type']:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            # Filter test classes present in train
            mask_val = np.isin(y_val_raw, le.classes_)
            mask_test = np.isin(y_test_raw, le.classes_)
            
            y_val = le.transform(y_val_raw[mask_val])
            y_test = le.transform(y_test_raw[mask_test])
            
            X_tr = X_train.values
            X_vl = X_val.values[mask_val]
            X_ts = X_test.values[mask_test]
            
            X_tr_sc = X_train_scaled
            X_vl_sc = X_val_scaled[mask_val]
            X_ts_sc = X_test_scaled[mask_test]
        else:
            y_train = y_train_raw.astype(int)
            y_val = y_val_raw.astype(int)
            y_test = y_test_raw.astype(int)
            
            X_tr = X_train.values
            X_vl = X_val.values
            X_ts = X_test.values
            
            X_tr_sc = X_train_scaled
            X_vl_sc = X_val_scaled
            X_ts_sc = X_test_scaled
            
        n_classes = len(np.unique(y_train))
        models = get_candidate_models(n_classes)
        
        benchmark_dict[task_name] = {}
        
        for model_name, model in models.items():
            is_linear_or_nn = "MLP" in model_name or "Logistic" in model_name
            cur_X_tr = X_tr_sc if is_linear_or_nn else X_tr
            cur_X_ts = X_ts_sc if is_linear_or_nn else X_ts
            
            # Train
            model.fit(cur_X_tr, y_train)
            
            # Predict
            y_pred = model.predict(cur_X_ts)
            y_prob = model.predict_proba(cur_X_ts)
            
            metrics = evaluate_predictions(y_test, y_pred, y_prob, n_classes)
            benchmark_dict[task_name][model_name] = metrics
            
            all_benchmark_rows.append({
                "Clinical Task": task_name,
                "Meta-Model Architecture": model_name,
                "Test Accuracy (%)": round(metrics["Accuracy"] * 100, 2),
                "Test Macro F1 (%)": round(metrics["Macro_F1"] * 100, 2),
                "Test Weighted F1 (%)": round(metrics["Weighted_F1"] * 100, 2),
                "Test AUROC": round(metrics["AUROC"], 4) if not np.isnan(metrics["AUROC"]) else "N/A"
            })
            
            logger.info(
                f"  [{model_name:20s}] Test Acc: {metrics['Accuracy']*100:.2f}% | "
                f"Macro F1: {metrics['Macro_F1']*100:.2f}% | "
                f"Weighted F1: {metrics['Weighted_F1']*100:.2f}% | "
                f"AUROC: {metrics['AUROC']:.4f}"
            )
            
    # Save CSV
    df_benchmark = pd.DataFrame(all_benchmark_rows)
    csv_path = MS_DIR / "Supplementary_Table_Model_Benchmark.csv"
    df_benchmark.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Saved Benchmark Table to: {csv_path}")
    
    # Save JSON
    json_path = DATA_DIR / "meta_model_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_dict, f, indent=2)
    logger.info(f"✓ Saved Detailed Metrics JSON to: {json_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENTAL META-MODEL BENCHMARK SUMMARY (HELD-OUT TEST SET)")
    print("="*80)
    print(df_benchmark.to_markdown(index=False))


if __name__ == "__main__":
    main()
