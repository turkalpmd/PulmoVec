"""
scripts/clean_stats.py

Shared statistics for the run_clean_* analysis scripts (library, not a runner).

All uncertainty is estimated by resampling PATIENTS: events of one child are not
independent, so event-level bootstrap or DeLong intervals would be too narrow.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score,
                             confusion_matrix)

from run_clean_meta import expected_calibration_error, multiclass_brier


# ----------------------------------------------------------------------------- metrics
def auc_macro(y, p):
    if len(np.unique(y)) < p.shape[1]:
        return np.nan
    if p.shape[1] == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class='ovr', average='macro')


def auprc_macro(y, p):
    k = p.shape[1]
    if len(np.unique(y)) < k:
        return np.nan
    if k == 2:
        return average_precision_score(y, p[:, 1])
    return float(np.mean([average_precision_score((y == c).astype(int), p[:, c])
                          for c in range(k)]))


def overall_metrics(y, p):
    pred = p.argmax(axis=1)
    return {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y, pred),
        'auc': auc_macro(y, p),
        'auprc': auprc_macro(y, p),
        'brier': multiclass_brier(y, p),
        'ece': expected_calibration_error(y, p),
    }


def per_class_metrics(y, p, class_names):
    """One-vs-rest sensitivity, specificity, PPV, NPV, F1, MCC, AUC for every class."""
    pred = p.argmax(axis=1)
    out = {}
    for c, name in enumerate(class_names):
        yt, yp = (y == c).astype(int), (pred == c).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        div = lambda a, b: a / b if b else np.nan  # noqa: E731
        out[name] = {
            'n': int(yt.sum()),
            'sensitivity': div(tp, tp + fn), 'specificity': div(tn, tn + fp),
            'ppv': div(tp, tp + fp), 'npv': div(tn, tn + fn),
            'f1': div(2 * tp, 2 * tp + fp + fn),
            'mcc': matthews_corrcoef(yt, yp) if yt.min() != yt.max() else np.nan,
            'auc': roc_auc_score(yt, p[:, c]) if yt.min() != yt.max() else np.nan,
        }
    return out


def calibration_slope_intercept(y, p, positive=1):
    """Logistic recalibration of the positive-class probability (binary tasks)."""
    import statsmodels.api as sm
    eps = 1e-6
    q = np.clip(p[:, positive], eps, 1 - eps)
    lp = np.log(q / (1 - q))
    yt = (y == positive).astype(int)
    slope = sm.Logit(yt, sm.add_constant(lp)).fit(disp=0).params[1]
    intercept = sm.GLM(yt, np.ones_like(lp), family=sm.families.Binomial(),
                       offset=lp).fit().params[0]
    return {'slope': float(slope), 'intercept': float(intercept)}


# --------------------------------------------------------------------------- bootstrap
def _group_index(groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    return uniq, [np.where(inv == i)[0] for i in range(len(uniq))]


def bootstrap_rows(groups, n_boot=2000, seed=42):
    """Yield row-index arrays for patient-level bootstrap resamples."""
    rng = np.random.default_rng(seed)
    uniq, rows = _group_index(np.asarray(groups))
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([rows[i] for i in pick])


def cluster_bootstrap(fn, groups, n_boot=2000, seed=42):
    """fn(rows) -> dict of scalars. Returns {metric: (lo, hi)} and the raw draws."""
    draws = {}
    for rows in bootstrap_rows(groups, n_boot, seed):
        try:
            res = fn(rows)
        except ValueError:
            continue
        for k, v in res.items():
            draws.setdefault(k, []).append(v)
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
          for k, v in draws.items()}
    return ci, {k: np.asarray(v, dtype=float) for k, v in draws.items()}


def paired_cluster_bootstrap_delta(metric_fn, y, p_a, p_b, groups, n_boot=2000, seed=42):
    """
    Difference metric(A) - metric(B) with both models scored on the SAME resampled
    patients. Two-sided p = 2 * min(P(delta<=0), P(delta>=0)), floored at 1/(B+1).
    """
    point = metric_fn(y, p_a) - metric_fn(y, p_b)
    deltas = []
    for rows in bootstrap_rows(groups, n_boot, seed):
        try:
            deltas.append(metric_fn(y[rows], p_a[rows]) - metric_fn(y[rows], p_b[rows]))
        except ValueError:
            continue
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {'delta': float(point), 'ci': [float(np.percentile(d, 2.5)),
                                          float(np.percentile(d, 97.5))],
            'p': float(max(min(p, 1.0), 1.0 / (len(d) + 1))), 'n_boot': int(len(d))}


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


# ------------------------------------------------------------------------- aggregation
def aggregate(df, proba, level, rule='confweighted', target=None, derived_any_positive=False):
    """
    Pool event probabilities within `level` ('_group_key' = patient, 'filename' = recording).

    rule: 'confweighted' (weights = top-1 confidence) or 'mean'.
    Truth: first event label of the unit, or - for the acoustic tasks, where the unit has
    no annotated label of its own - a DERIVED label:
      binary task      -> 1 if any event is abnormal
      multi-class task -> most frequent non-normal class if any, else 0
    """
    df = df.reset_index(drop=True)
    y_out, p_out, ids = [], [], []
    for uid, g in df.groupby(level, sort=False):
        p = proba[g.index.values]
        w = p.max(axis=1) if rule == 'confweighted' else np.ones(len(p))
        p_out.append((p * (w / w.sum())[:, None]).sum(axis=0))
        lab = g[target].values.astype(int)
        if derived_any_positive:
            pos = lab[lab > 0]
            y_out.append(int(np.bincount(pos).argmax()) if len(pos) else 0)
        else:
            y_out.append(int(lab[0]))
        ids.append(uid)
    return np.asarray(y_out), np.vstack(p_out), np.asarray(ids)


def fmt_ci(v, ci, d=3):
    return f"{v:.{d}f} ({ci[0]:.{d}f}–{ci[1]:.{d}f})"
