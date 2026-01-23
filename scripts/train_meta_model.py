#!/usr/bin/env python3
"""
Train LightGBM meta-models using predictions from all 3 ensemble models.

The meta-models take as input the probability outputs from:
    - Model 1: Event Type (3 probabilities)
    - Model 2: Binary Abnormal (2 probabilities)
    - Model 3: Disease Groups (3 probabilities)
    - Demographics: age, gender
    Total: 10 features

Output: 5 separate models predicting:
    1. disease - Multi-class classification
    2. event_type - Multi-class classification
    3. model1_label - Multi-class classification (0, 1, 2)
    4. model2_label - Binary classification (0, 1)
    5. model3_label - Multi-class classification (0, 1, 2)

Usage:
    # Extract probabilities first (run once)
    python scripts/extract_ensemble_probabilities.py
    
    # Train all 5 meta-models with Optuna optimization
    python scripts/train_meta_model.py --use_csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, roc_auc_score, matthews_corrcoef
)
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
# import shap  # Optional - SHAP analysis disabled

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Meta-model save directory
META_MODEL_DIR = Path(__file__).parent.parent / "models" / "meta_model"
META_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Probability CSV paths
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_PROB_CSV = DATA_DIR / "ensemble_probabilities_train.csv"
VAL_PROB_CSV = DATA_DIR / "ensemble_probabilities_val.csv"


def load_probabilities_from_csv(train_csv_path, val_csv_path):
    """
    Load probabilities from pre-extracted CSV files.
    
    Args:
        train_csv_path: Path to training probabilities CSV
        val_csv_path: Path to validation probabilities CSV
        
    Returns:
        train_df, val_df: DataFrames with all columns
    """
    logging.info("Loading probabilities from CSV files...")
    
    # Load training data
    train_df = pd.read_csv(train_csv_path)
    logging.info(f"Loaded {len(train_df)} training samples")
    
    # Load validation data
    val_df = pd.read_csv(val_csv_path)
    logging.info(f"Loaded {len(val_df)} validation samples")
    
    # Verify probabilities sum to ~1.0
    for model_name in ['Model1', 'Model2', 'Model3']:
        if model_name == 'Model1':
            cols = ['Model1_Normalpp', 'Model1_Cracklespp', 'Model1_Rhonchipp']
        elif model_name == 'Model2':
            cols = ['Model2_Normalpp', 'Model2_Abnormalpp']
        else:
            cols = ['Model3_Normalpp', 'Model3_Pneumoniapp', 'Model3_Bronchiolitispp']
        
        train_sum = train_df[cols].sum(axis=1)
        val_sum = val_df[cols].sum(axis=1)
        logging.info(f"{model_name} probability sum - Train: min={train_sum.min():.4f}, max={train_sum.max():.4f}, "
                    f"Val: min={val_sum.min():.4f}, max={val_sum.max():.4f}")
    
    return train_df, val_df


def prepare_features_and_labels(df, outcome_name):
    """
    Prepare features (10 total) and labels for a specific outcome.
    
    Args:
        df: DataFrame with probabilities and metadata
        outcome_name: Name of outcome column (disease, event_type, model1_label, etc.)
        
    Returns:
        X: Feature DataFrame [N, 10] (with column names for LightGBM)
        y: Label array [N]
        label_encoder: LabelEncoder if outcome is categorical
        feature_names: List of feature names
    """
    # Extract probability features (8)
    prob_cols = [
        'Model1_Normalpp', 'Model1_Cracklespp', 'Model1_Rhonchipp',
        'Model2_Normalpp', 'Model2_Abnormalpp',
        'Model3_Normalpp', 'Model3_Pneumoniapp', 'Model3_Bronchiolitispp'
    ]
    
    # Extract demographic features (2)
    # Encode gender: Female=0, Male=1
    df_encoded = df.copy()
    df_encoded['gender_encoded'] = df_encoded['gender'].map({'Female': 0, 'Male': 1})
    
    # Handle missing age (fill with median)
    if df_encoded['age'].isna().any():
        median_age = df_encoded['age'].median()
        df_encoded['age'] = df_encoded['age'].fillna(median_age)
        logging.warning(f"Filled {df_encoded['age'].isna().sum()} missing age values with median: {median_age}")
    
    # Combine all features (10 total) - Keep as DataFrame for feature names
    feature_cols = prob_cols + ['age', 'gender_encoded']
    X = df_encoded[feature_cols].copy()  # Keep as DataFrame
    feature_names = feature_cols
    
    # Extract labels
    y = df_encoded[outcome_name].values
    
    # Encode labels if they are strings (disease, event_type)
    label_encoder = None
    if outcome_name in ['disease', 'event_type']:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        logging.info(f"Encoded {outcome_name} labels: {len(label_encoder.classes_)} classes")
    else:
        # model1_label, model2_label, model3_label are already numeric
        y = y.astype(int)
    
    return X, y, label_encoder, feature_names


def bootstrap_metric(y_true, y_pred, y_proba, metric_func, n_bootstrap=1000):
    """
    Calculate metric with 95% confidence interval using bootstrap resampling.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (can be None)
        metric_func: Function to calculate metric: f(y_true, y_pred, y_proba) -> float
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        dict with 'value', 'ci95_lower', 'ci95_upper'
    """
    n_samples = len(y_true)
    metric_values = []
    
    rng = np.random.RandomState(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_proba_boot = y_proba[indices] if y_proba is not None else None
        
        # Calculate metric
        try:
            if y_proba_boot is not None:
                metric_val = metric_func(y_true_boot, y_pred_boot, y_proba_boot)
            else:
                metric_val = metric_func(y_true_boot, y_pred_boot)
            # Skip NaN values
            if not np.isnan(metric_val):
                metric_values.append(metric_val)
        except Exception:
            # Skip if metric calculation fails (e.g., single class in bootstrap)
            continue
    
    if len(metric_values) == 0:
        return {'value': np.nan, 'ci95_lower': np.nan, 'ci95_upper': np.nan}
    
    # Calculate CI95
    mean_val = np.mean(metric_values)
    ci95_lower = np.percentile(metric_values, 2.5)
    ci95_upper = np.percentile(metric_values, 97.5)
    
    return {'value': mean_val, 'ci95_lower': ci95_lower, 'ci95_upper': ci95_upper}


def evaluate_model_detailed_with_ci95(model, X_val, y_val, outcome_name, label_encoder=None, n_bootstrap=1000):
    """
    Evaluate model with detailed metrics and CI95.
    
    Args:
        model: Trained LightGBM model
        X_val: Validation features
        y_val: Validation labels
        outcome_name: Name of outcome
        label_encoder: LabelEncoder for class names (if applicable)
        n_bootstrap: Number of bootstrap iterations for CI95
        
    Returns:
        metrics: Dictionary with all metrics and CI95
    """
    logging.info(f"\nEvaluating model for {outcome_name}...")
    
    # Keep as DataFrame if possible (for feature names), LightGBM handles both
    # Predictions (LightGBM can handle DataFrame directly, no need to convert)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    
    # Get class names
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        # Map numeric labels to meaningful class names
        n_classes = len(np.unique(y_val))
        if outcome_name == 'model1_label':
            # Model 1: Normal (0), Crackles (1), Rhonchi (2)
            class_names = ['Normal', 'Crackles', 'Rhonchi'][:n_classes]
        elif outcome_name == 'model2_label':
            # Model 2: Normal (0), Abnormal (1)
            class_names = ['Normal', 'Abnormal'][:n_classes]
        elif outcome_name == 'model3_label':
            # Model 3: Normal (0), Pneumonia (1), Bronchiolitis (2)
            class_names = ['Normal', 'Pneumonia', 'Bronchiolitis'][:n_classes]
        else:
            # For other outcomes, use generic class names
            class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Calculate metrics with CI95
    metrics = {}
    
    # Accuracy
    def acc_func(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    metrics['accuracy'] = bootstrap_metric(y_val, y_pred, None, acc_func, n_bootstrap)
    
    # Macro F1
    def f1_macro_func(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = bootstrap_metric(y_val, y_pred, None, f1_macro_func, n_bootstrap)
    
    # Weighted F1
    def f1_weighted_func(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = bootstrap_metric(y_val, y_pred, None, f1_weighted_func, n_bootstrap)
    
    # MCC
    def mcc_func(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)
    metrics['mcc'] = bootstrap_metric(y_val, y_pred, None, mcc_func, n_bootstrap)
    
    # Log-loss
    def logloss_func(y_true, y_pred, y_proba):
        return log_loss(y_true, y_proba)
    metrics['log_loss'] = bootstrap_metric(y_val, y_pred, y_proba, logloss_func, n_bootstrap)
    
    # Per-class precision and recall
    from sklearn.metrics import precision_recall_fscore_support
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_val, y_pred, zero_division=0
    )
    
    metrics['precision_per_class'] = {}
    metrics['recall_per_class'] = {}
    metrics['f1_per_class'] = {}
    metrics['support_per_class'] = {}
    
    for i, class_name in enumerate(class_names):
        def prec_func(y_true, y_pred):
            p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[i], zero_division=0)
            return p[0] if len(p) > 0 else 0.0
        metrics['precision_per_class'][class_name] = bootstrap_metric(y_val, y_pred, None, prec_func, n_bootstrap)
        
        def rec_func(y_true, y_pred):
            _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[i], zero_division=0)
            return r[0] if len(r) > 0 else 0.0
        metrics['recall_per_class'][class_name] = bootstrap_metric(y_val, y_pred, None, rec_func, n_bootstrap)
        
        metrics['f1_per_class'][class_name] = f1_per_class[i]
        metrics['support_per_class'][class_name] = int(support[i])
    
    # Sensitivity, Specificity, NPV, PPV (per-class, one-vs-rest)
    metrics['sensitivity_per_class'] = {}  # Sensitivity = Recall = TP/(TP+FN)
    metrics['specificity_per_class'] = {}   # Specificity = TN/(TN+FP)
    metrics['npv_per_class'] = {}           # NPV = TN/(TN+FN)
    metrics['ppv_per_class'] = {}           # PPV = Precision = TP/(TP+FP)
    
    for i, class_name in enumerate(class_names):
        # One-vs-rest: class i vs all others
        def sensitivity_func(y_true, y_pred):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            if tp + fn == 0:
                return 0.0
            return tp / (tp + fn)
        
        def specificity_func(y_true, y_pred):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            if tn + fp == 0:
                return 0.0
            return tn / (tn + fp)
        
        def npv_func(y_true, y_pred):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            if tn + fn == 0:
                return 0.0
            return tn / (tn + fn)
        
        def ppv_func(y_true, y_pred):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)
        
        metrics['sensitivity_per_class'][class_name] = bootstrap_metric(y_val, y_pred, None, sensitivity_func, n_bootstrap)
        metrics['specificity_per_class'][class_name] = bootstrap_metric(y_val, y_pred, None, specificity_func, n_bootstrap)
        metrics['npv_per_class'][class_name] = bootstrap_metric(y_val, y_pred, None, npv_func, n_bootstrap)
        metrics['ppv_per_class'][class_name] = bootstrap_metric(y_val, y_pred, None, ppv_func, n_bootstrap)
    
    # ROC-AUC (one-vs-rest)
    n_classes = len(class_names)
    if n_classes == 2:
        # Binary: single ROC-AUC
        def roc_auc_func(y_true, y_pred, y_proba):
            try:
                # Check if both classes are present
                if len(np.unique(y_true)) < 2:
                    return np.nan
                return roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                return np.nan
        metrics['roc_auc'] = bootstrap_metric(y_val, y_pred, y_proba, roc_auc_func, n_bootstrap)
    else:
        # Multi-class: one-vs-rest
        metrics['roc_auc_macro'] = {}
        metrics['roc_auc_weighted'] = {}
        metrics['roc_auc_per_class'] = {}
        
        # Macro and weighted
        def roc_auc_macro_func(y_true, y_pred, y_proba):
            try:
                # Check if at least 2 classes are present
                if len(np.unique(y_true)) < 2:
                    return np.nan
                return roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except Exception:
                return np.nan
        metrics['roc_auc_macro'] = bootstrap_metric(y_val, y_pred, y_proba, roc_auc_macro_func, n_bootstrap)
        
        def roc_auc_weighted_func(y_true, y_pred, y_proba):
            try:
                if len(np.unique(y_true)) < 2:
                    return np.nan
                return roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except Exception:
                return np.nan
        metrics['roc_auc_weighted'] = bootstrap_metric(y_val, y_pred, y_proba, roc_auc_weighted_func, n_bootstrap)
        
        # Per-class (one-vs-rest)
        for i, class_name in enumerate(class_names):
            def roc_auc_class_func(y_true, y_pred, y_proba):
                try:
                    y_true_binary = (y_true == i).astype(int)
                    # Check if both classes (0 and 1) are present in binary classification
                    if len(np.unique(y_true_binary)) < 2:
                        return np.nan
                    return roc_auc_score(y_true_binary, y_proba[:, i])
                except Exception:
                    return np.nan
            metrics['roc_auc_per_class'][class_name] = bootstrap_metric(
                y_val, y_pred, y_proba, roc_auc_class_func, n_bootstrap
            )
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # Return class_names along with metrics so they match
    metrics['_class_names'] = class_names
    
    return metrics


def optimize_lightgbm_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, early_stopping_rounds=20, n_jobs=-1):
    """
    Optimize LightGBM hyperparameters using Optuna with TPE sampler.
    
    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training labels
        X_val: Validation features (DataFrame or array)
        y_val: Validation labels
        n_trials: Number of optimization trials
        early_stopping_rounds: Early stopping rounds (fixed, not optimized)
        n_jobs: Number of parallel jobs for Optuna (-1 for all CPUs)
        
    Returns:
        best_params: Dictionary of best hyperparameters
    """
    import os
    n_cpus = os.cpu_count() if n_jobs == -1 else n_jobs
    logging.info(f"\nOptimizing LightGBM hyperparameters with Optuna (TPE sampler)...")
    logging.info(f"  Trials: {n_trials}")
    logging.info(f"  Early stopping rounds: {early_stopping_rounds} (fixed)")
    logging.info(f"  Parallel jobs: {n_cpus} (using all available CPUs)")
    
    def objective(trial):
        # Suggest hyperparameters (early_stopping is NOT optimized, it's fixed)
        params = {
            'objective': 'multiclass' if len(np.unique(y_train)) > 2 else 'binary',
            'metric': 'multi_logloss' if len(np.unique(y_train)) > 2 else 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_class': len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else None,
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        if params['num_class'] is None:
            del params['num_class']
        
        # Train model with fixed early stopping
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        
        return f1_macro
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Optimize (parallel execution)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'multiclass' if len(np.unique(y_train)) > 2 else 'binary',
        'metric': 'multi_logloss' if len(np.unique(y_train)) > 2 else 'binary_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    })
    
    if len(np.unique(y_train)) > 2:
        best_params['num_class'] = len(np.unique(y_train))
    
    logging.info(f"\n✓ Optimization completed!")
    logging.info(f"  Best F1-Macro: {study.best_value:.4f}")
    logging.info(f"  Best parameters: {best_params}")
    
    return best_params


def train_lightgbm_model(X_train, y_train, X_val, y_val, best_params, early_stopping_rounds=20):
    """
    Train final LightGBM model with best hyperparameters.
    
    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training labels
        X_val: Validation features (DataFrame or array)
        y_val: Validation labels
        best_params: Best hyperparameters from Optuna
        early_stopping_rounds: Early stopping rounds (fixed)
        
    Returns:
        model: Trained LightGBM model
    """
    logging.info("\nTraining final LightGBM model...")
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    )
    
    logging.info("✓ Model trained!")
    
    return model


# SHAP analysis function (disabled - removed due to multi-class compatibility issues)
# def generate_shap_analysis(model, X_val, feature_names, outcome_name, save_dir, max_samples=1000):
#     """SHAP analysis disabled."""
#     pass


def plot_confusion_matrix_with_ci95(cm, class_names, outcome_name, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix: {outcome_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved confusion matrix: {save_path.name}")


def save_metrics_report(metrics, outcome_name, class_names, save_path):
    """Generate detailed markdown report with CI95."""
    report = f"""# LightGBM Meta-Model Report: {outcome_name}

## Overview

This meta-model predicts **{outcome_name}** using ensemble model probabilities and demographic features.

**Input Features (10 total):**
- Model 1 probabilities (3): Normal, Crackles, Rhonchi
- Model 2 probabilities (2): Normal, Abnormal
- Model 3 probabilities (3): Normal, Pneumonia, Bronchiolitis
- Demographics (2): age, gender

**Output Classes:** {len(class_names)}
- {', '.join([str(c) for c in class_names])}

---

## Performance Metrics (with 95% Confidence Intervals)

### Basic Metrics

#### Accuracy
- **Value**: {metrics['accuracy']['value']:.4f}
- **CI95**: [{metrics['accuracy']['ci95_lower']:.4f}, {metrics['accuracy']['ci95_upper']:.4f}]

#### Macro F1
- **Value**: {metrics['f1_macro']['value']:.4f}
- **CI95**: [{metrics['f1_macro']['ci95_lower']:.4f}, {metrics['f1_macro']['ci95_upper']:.4f}]

#### Weighted F1
- **Value**: {metrics['f1_weighted']['value']:.4f}
- **CI95**: [{metrics['f1_weighted']['ci95_lower']:.4f}, {metrics['f1_weighted']['ci95_upper']:.4f}]

#### Matthews Correlation Coefficient (MCC)
- **Value**: {metrics['mcc']['value']:.4f}
- **CI95**: [{metrics['mcc']['ci95_lower']:.4f}, {metrics['mcc']['ci95_upper']:.4f}]

### Probabilistic Metrics

#### Log-Loss
- **Value**: {metrics['log_loss']['value']:.4f}
- **CI95**: [{metrics['log_loss']['ci95_lower']:.4f}, {metrics['log_loss']['ci95_upper']:.4f}]

"""
    
    # ROC-AUC
    if 'roc_auc' in metrics:
        report += f"""#### ROC-AUC (Binary)
- **Value**: {metrics['roc_auc']['value']:.4f}
- **CI95**: [{metrics['roc_auc']['ci95_lower']:.4f}, {metrics['roc_auc']['ci95_upper']:.4f}]

"""
    else:
        report += f"""#### ROC-AUC (One-vs-Rest)

**Macro Average:**
- **Value**: {metrics['roc_auc_macro']['value']:.4f}
- **CI95**: [{metrics['roc_auc_macro']['ci95_lower']:.4f}, {metrics['roc_auc_macro']['ci95_upper']:.4f}]

**Weighted Average:**
- **Value**: {metrics['roc_auc_weighted']['value']:.4f}
- **CI95**: [{metrics['roc_auc_weighted']['ci95_lower']:.4f}, {metrics['roc_auc_weighted']['ci95_upper']:.4f}]

"""
    
    # Per-class metrics
    report += """### Per-Class Metrics

| Class | Precision (PPV) | Recall (Sensitivity) | F1-Score | Specificity | NPV | Support | ROC-AUC (OvR) |
|-------|------------------|----------------------|----------|-------------|-----|---------|---------------|
"""
    
    for class_name in class_names:
        prec = metrics['precision_per_class'][class_name]
        rec = metrics['recall_per_class'][class_name]
        f1 = metrics['f1_per_class'][class_name]
        support = metrics['support_per_class'][class_name]
        sens = metrics['sensitivity_per_class'][class_name]
        spec = metrics['specificity_per_class'][class_name]
        npv = metrics['npv_per_class'][class_name]
        ppv = metrics['ppv_per_class'][class_name]
        
        if 'roc_auc_per_class' in metrics and class_name in metrics['roc_auc_per_class']:
            roc_auc = metrics['roc_auc_per_class'][class_name]
            roc_auc_str = f"{roc_auc['value']:.4f} [{roc_auc['ci95_lower']:.4f}, {roc_auc['ci95_upper']:.4f}]"
        else:
            roc_auc_str = "N/A"
        
        report += f"| {class_name} | {ppv['value']:.4f} [{ppv['ci95_lower']:.4f}, {ppv['ci95_upper']:.4f}] | {sens['value']:.4f} [{sens['ci95_lower']:.4f}, {sens['ci95_upper']:.4f}] | {f1:.4f} | {spec['value']:.4f} [{spec['ci95_lower']:.4f}, {spec['ci95_upper']:.4f}] | {npv['value']:.4f} [{npv['ci95_lower']:.4f}, {npv['ci95_upper']:.4f}] | {support} | {roc_auc_str} |\n"
    
    report += f"""
---

## Confusion Matrix

See `confusion_matrix.png` for detailed confusion matrix visualization.

---

**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    logging.info(f"  ✓ Saved report: {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM meta-models with Optuna optimization")
    parser.add_argument('--use_csv', action='store_true',
                        help='Load probabilities from pre-extracted CSV files (faster)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of Optuna optimization trials (default: 100)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for Optuna (-1 for all CPUs, default: -1)')
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("LIGHTGBM META-MODEL TRAINING WITH OPTUNA OPTIMIZATION")
    logging.info("=" * 80)
    
    # Check if CSV files exist
    if not TRAIN_PROB_CSV.exists() or not VAL_PROB_CSV.exists():
        logging.error(f"\n❌ CSV files not found!")
        logging.error(f"  Training CSV: {TRAIN_PROB_CSV}")
        logging.error(f"  Validation CSV: {VAL_PROB_CSV}")
        logging.error(f"\n  Please run: python scripts/extract_ensemble_probabilities.py")
        return
    
    # Load probabilities from CSV
    train_df, val_df = load_probabilities_from_csv(TRAIN_PROB_CSV, VAL_PROB_CSV)
    
    # Define 5 outcomes
    outcomes = {
        'disease': {'type': 'multiclass'},
        'event_type': {'type': 'multiclass'},
        'model1_label': {'type': 'multiclass'},
        'model2_label': {'type': 'binary'},
        'model3_label': {'type': 'multiclass'}
    }
    
    # Train model for each outcome
    for outcome_name, outcome_info in outcomes.items():
        logging.info("\n" + "=" * 80)
        logging.info(f"TRAINING MODEL FOR: {outcome_name}")
        logging.info("=" * 80)
        
        # Create output directory
        outcome_dir = META_MODEL_DIR / outcome_name
        outcome_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare features and labels
        X_train, y_train, label_encoder_train, feature_names = prepare_features_and_labels(train_df, outcome_name)
        X_val, y_val, label_encoder_val, _ = prepare_features_and_labels(val_df, outcome_name)
        
        # Use training label encoder for validation (to ensure same class mapping)
        if label_encoder_train is not None:
            label_encoder = label_encoder_train
            # Re-encode validation labels using training encoder
            val_df_temp = val_df.copy()
            y_val = label_encoder.transform(val_df_temp[outcome_name])
        else:
            label_encoder = None
        
        logging.info(f"  Training samples: {len(X_train)}")
        logging.info(f"  Validation samples: {len(X_val)}")
        logging.info(f"  Number of classes: {len(np.unique(y_train))}")
        
        # Get class names
        if label_encoder is not None:
            class_names = label_encoder.classes_
        else:
            unique_classes = sorted(np.unique(y_train))
            class_names = [str(c) for c in unique_classes]
        
        # Optimize hyperparameters (early stopping is fixed at 20 rounds, use all CPUs)
        best_params = optimize_lightgbm_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=args.n_trials, 
            early_stopping_rounds=20, n_jobs=args.n_jobs
        )
        
        # Train final model (early stopping is fixed at 20 rounds)
        model = train_lightgbm_model(X_train, y_train, X_val, y_val, best_params, early_stopping_rounds=20)
        
        # Evaluate with CI95
        metrics = evaluate_model_detailed_with_ci95(
            model, X_val, y_val, outcome_name, label_encoder, n_bootstrap=1000
        )
        
        # Get class_names from metrics (they match the keys in precision_per_class, etc.)
        class_names = metrics['_class_names']
        del metrics['_class_names']  # Remove temporary key
        
        # SHAP analysis (skipped - optional)
        # shap_importance = generate_shap_analysis(
        #     model, X_val, feature_names, outcome_name, outcome_dir, max_samples=1000
        # )
        logging.info(f"  ⏭️  SHAP analysis skipped (optional)")
        
        # Plot confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix_with_ci95(cm, class_names, outcome_name, outcome_dir / "confusion_matrix.png")
        
        # Save model
        model_path = outcome_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"  ✓ Saved model: {model_path.name}")
        
        # Save label encoder if exists
        if label_encoder is not None:
            encoder_path = outcome_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
        
        # Save metrics (convert numpy types to native Python types for JSON)
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        metrics_serializable = convert_to_serializable(metrics)
        metrics_path = outcome_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        logging.info(f"  ✓ Saved metrics: {metrics_path.name}")
        
        # Save report
        report_path = outcome_dir / "report.md"
        save_metrics_report(metrics, outcome_name, class_names, report_path)
        
        # Print summary
        logging.info(f"\n✓ Model for {outcome_name} completed!")
        logging.info(f"  Accuracy: {metrics['accuracy']['value']:.4f} "
                    f"(CI95: [{metrics['accuracy']['ci95_lower']:.4f}, {metrics['accuracy']['ci95_upper']:.4f}])")
        logging.info(f"  F1-Macro: {metrics['f1_macro']['value']:.4f} "
                    f"(CI95: [{metrics['f1_macro']['ci95_lower']:.4f}, {metrics['f1_macro']['ci95_upper']:.4f}])")
        logging.info(f"  All outputs saved to: {outcome_dir}")
    
    logging.info("\n" + "=" * 80)
    logging.info("ALL META-MODELS TRAINING COMPLETED!")
    logging.info("=" * 80)
    logging.info(f"All outputs saved to: {META_MODEL_DIR}")


if __name__ == "__main__":
    main()
