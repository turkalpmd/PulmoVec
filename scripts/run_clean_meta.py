#!/usr/bin/env python3
"""
scripts/run_clean_meta.py

Stage 4-5 of the leakage-free rebuild.

Consumes the out-of-fold meta-feature tables written by run_clean_pipeline.py and
produces the numbers that should replace the current manuscript results:

  * LightGBM stacking meta-model, hyperparameters searched on VAL only,
    final metrics reported once on the untouched TEST partition.
  * Bootstrap 95% CIs resampled at the PATIENT level (events of one patient are
    not independent, so event-level bootstrap understates the interval).
  * Calibration: Brier score, expected calibration error, reliability curves.
  * Ablation ladder, including the demographics-only baseline that establishes how
    much the acoustic models actually add over age and sex.
  * Patient-level aggregation on TEST.

Outputs to results_clean/meta/.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, brier_score_loss,
                             confusion_matrix, classification_report)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results_clean'
PROB_DIR = RESULTS / 'probabilities'
META_DIR = RESULTS / 'meta'

TARGETS = {
    'model2_label': ['Normal', 'Abnormal'],
    'model1_label': ['Normal', 'Crackles', 'Rhonchi'],
    'model3_label': ['Pneumonia', 'Bronchial', 'Normal_Other'],
}

PROB_COLS = ([f'model1_label_p{i}' for i in range(3)] +
             [f'model2_label_p{i}' for i in range(2)] +
             [f'model3_label_p{i}' for i in range(3)])
DEMO_COLS = ['age', 'gender_code', 'recording_location']

FEATURE_SETS = {
    'demographics_only': DEMO_COLS,
    'acoustic_only': PROB_COLS,
    'full_stack': PROB_COLS + DEMO_COLS,
}


def log(m):
    print(m, flush=True)


def prep(df):
    df = df.copy()
    if df['recording_location'].dtype == object:
        df['recording_location'] = (df['recording_location'].astype(str)
                                    .str.extract(r'(\d+)').astype(float))
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender_code'] = pd.to_numeric(df['gender_code'], errors='coerce')
    return df


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """Multi-class ECE on the top-1 confidence."""
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true, y_proba):
    n_classes = y_proba.shape[1]
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def core_metrics(y_true, y_proba, n_classes):
    pred = y_proba.argmax(axis=1)
    out = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, pred, average='weighted', zero_division=0)),
        'brier': multiclass_brier(y_true, y_proba),
        'ece': expected_calibration_error(y_true, y_proba),
    }
    try:
        if n_classes == 2:
            out['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            out['macro_roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr',
                                                       average='macro'))
    except ValueError:
        out['roc_auc'] = None
    return out


def patient_bootstrap_ci(y_true, y_proba, groups, n_classes, n_boot=1000, seed=42):
    """Resample PATIENTS, not events - events within a patient are correlated."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    acc = {}
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in picked])
        if len(np.unique(y_true[rows])) < 2:
            continue
        for k, v in core_metrics(y_true[rows], y_proba[rows], n_classes).items():
            if v is not None:
                acc.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in acc.items()}


def tune_and_fit(Xtr, ytr, Xva, yva, n_classes, n_trials, seed=42):
    def objective(trial):
        params = {
            'objective': 'binary' if n_classes == 2 else 'multiclass',
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'class_weight': 'balanced',
            'verbose': -1,
            'random_state': seed,
        }
        if n_classes > 2:
            params['num_class'] = n_classes
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        return f1_score(yva, m.predict(Xva), average='macro', zero_division=0)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    best.update({'objective': 'binary' if n_classes == 2 else 'multiclass',
                 'class_weight': 'balanced', 'verbose': -1, 'random_state': seed})
    if n_classes > 2:
        best['num_class'] = n_classes
    model = lgb.LGBMClassifier(**best)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    return model, best, float(study.best_value)


def reliability_plot(y_true, y_proba, title, path, n_bins=10):
    conf = y_proba.max(axis=1)
    correct = (y_proba.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Perfect calibration')
    ax.plot(xs, ys, 'o-', color='#c0392b', lw=1.8, label='Model')
    ax.set_xlabel('Mean predicted confidence')
    ax.set_ylabel('Observed accuracy')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def aggregate_to_patient(df, y_proba, n_classes):
    """Confidence-weighted soft voting within each patient."""
    out_true, out_proba, pids = [], [], []
    for pid, g in df.groupby('_group_key'):
        rows = g.index.values
        p = y_proba[df.index.get_indexer(rows)]
        w = p.max(axis=1)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        out_proba.append((p * w[:, None]).sum(axis=0))
        out_true.append(g['patient_target'].iloc[0])
        pids.append(pid)
    return np.array(out_true), np.vstack(out_proba), np.array(pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    META_DIR.mkdir(parents=True, exist_ok=True)

    tr = prep(pd.read_csv(PROB_DIR / 'meta_train_oof.csv'))
    va = prep(pd.read_csv(PROB_DIR / 'meta_val.csv'))
    te = prep(pd.read_csv(PROB_DIR / 'meta_test.csv'))
    log(f"meta rows  train={len(tr)}  val={len(va)}  test={len(te)}")

    for name, a, b in [('train/val', tr, va), ('train/test', tr, te), ('val/test', va, te)]:
        shared = set(a['_group_key']) & set(b['_group_key'])
        assert not shared, f"patient leakage in {name}: {len(shared)}"
    log("patient-level leakage check across meta tables: PASSED")

    report = {}

    for target, class_names in TARGETS.items():
        n_classes = len(class_names)
        log("=" * 70)
        log(f"META TARGET {target} ({n_classes} classes)")
        ytr = tr[target].values.astype(int)
        yva = va[target].values.astype(int)
        yte = te[target].values.astype(int)

        target_report = {'classes': class_names, 'ablation': {}}

        for set_name, cols in FEATURE_SETS.items():
            model, params, val_f1 = tune_and_fit(
                tr[cols], ytr, va[cols], yva, n_classes, args.trials)
            proba_te = model.predict_proba(te[cols])
            m = core_metrics(yte, proba_te, n_classes)
            m['val_macro_f1'] = val_f1
            target_report['ablation'][set_name] = m
            log(f"  {set_name:18s} val_f1={val_f1:.4f}  test_macro_f1={m['macro_f1']:.4f}  "
                f"acc={m['accuracy']:.4f}  brier={m['brier']:.3f}  ece={m['ece']:.3f}")

            if set_name == 'full_stack':
                ci = patient_bootstrap_ci(yte, proba_te, te['_group_key'].values,
                                          n_classes, n_boot=args.boot)
                target_report['test_metrics'] = m
                target_report['test_ci95'] = ci
                target_report['best_params'] = params
                target_report['confusion_matrix'] = confusion_matrix(
                    yte, proba_te.argmax(axis=1)).tolist()
                target_report['classification_report'] = classification_report(
                    yte, proba_te.argmax(axis=1), target_names=class_names,
                    output_dict=True, zero_division=0)
                reliability_plot(yte, proba_te, f'{target} - test calibration',
                                 META_DIR / f'reliability_{target}.png')
                np.save(META_DIR / f'test_proba_{target}.npy', proba_te)

                if target == 'model3_label':
                    te_idx = te.reset_index(drop=True)
                    te_idx['patient_target'] = yte
                    yp_true, yp_proba, _ = aggregate_to_patient(te_idx, proba_te, n_classes)
                    pm = core_metrics(yp_true, yp_proba, n_classes)
                    target_report['patient_level'] = {
                        'n_patients': int(len(yp_true)), 'metrics': pm,
                        'confusion_matrix': confusion_matrix(
                            yp_true, yp_proba.argmax(axis=1)).tolist()}
                    log(f"  patient-level ({len(yp_true)} patients): "
                        f"acc={pm['accuracy']:.4f} macro_f1={pm['macro_f1']:.4f}")

        report[target] = target_report
        (META_DIR / 'meta_report.json').write_text(json.dumps(report, indent=2))

    log("=" * 70)
    log(f"Written to {META_DIR}")


if __name__ == '__main__':
    main()
