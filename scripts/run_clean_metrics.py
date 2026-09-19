#!/usr/bin/env python3
"""
scripts/run_clean_metrics.py

Full metric set for one evaluation design, from the outputs of run_clean_meta_v2.py.

  --meta-dirs A            hold-out (one directory)
  --meta-dirs F1 F2 ...    nested CV: test predictions of all outer folds are pooled;
                           per-fold metrics are reported alongside.

For every task: overall + per-class metrics with patient-bootstrap CIs for every rung of
the ablation ladder, majority-class reference, pre-specified paired contrasts
(Holm-adjusted), event / recording / patient-level aggregation, calibration, and ROC / PR /
reliability curve coordinates for plotting.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from run_clean_meta import TARGETS  # noqa: E402
import clean_stats as cs  # noqa: E402

LADDER = ['duration_only', 'demographics_only', 'duration_demographics', 'own_task_base',
          'acoustic_only', 'full_stack']
CONTRASTS = [('full_stack', 'acoustic_only'), ('full_stack', 'demographics_only'),
             ('acoustic_only', 'demographics_only'), ('full_stack', 'own_task_base'),
             ('acoustic_only', 'duration_only')]
PRIMARY = 'full_stack'


def load(meta_dirs):
    idx, proba = [], {}
    for d in map(Path, meta_dirs):
        t = pd.read_csv(d / 'test_index.csv')
        t['_fold'] = d.parent.name if len(meta_dirs) > 1 else 'holdout'
        idx.append(t)
        for target in TARGETS:
            for s in LADDER:
                proba.setdefault((target, s), []).append(
                    np.load(d / f'test_proba_{target}_{s}.npy'))
    index = pd.concat(idx, ignore_index=True)
    assert not index.duplicated(['filename', 'event_index']).any(), "event scored twice"
    return index, {k: np.vstack(v) for k, v in proba.items()}


def with_ci(y, p, groups, n_boot, names=None):
    point = cs.overall_metrics(y, p)
    ci, _ = cs.cluster_bootstrap(lambda r: cs.overall_metrics(y[r], p[r]), groups, n_boot)
    out = {k: {'value': float(v), 'ci': ci.get(k)} for k, v in point.items()}
    if names is not None:
        pc = cs.per_class_metrics(y, p, names)

        def flat(r):
            return {f'{c}|{m}': v for c, d in cs.per_class_metrics(y[r], p[r], names).items()
                    for m, v in d.items() if m != 'n'}
        pci, _ = cs.cluster_bootstrap(flat, groups, n_boot)
        out['per_class'] = {c: {m: ({'value': float(v), 'ci': pci.get(f'{c}|{m}')}
                                    if m != 'n' else v) for m, v in d.items()}
                            for c, d in pc.items()}
    return out


def curves(y, p, names):
    out = {}
    for c, name in enumerate(names):
        if len(names) == 2 and c == 0:
            continue
        yt = (y == c).astype(int)
        fpr, tpr, _ = roc_curve(yt, p[:, c])
        prec, rec, _ = precision_recall_curve(yt, p[:, c])
        keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
        keep2 = np.linspace(0, len(rec) - 1, min(len(rec), 400)).astype(int)
        out[name] = {'fpr': fpr[keep].tolist(), 'tpr': tpr[keep].tolist(),
                     'recall': rec[keep2].tolist(), 'precision': prec[keep2].tolist(),
                     'prevalence': float(yt.mean())}
    conf, correct = p.max(axis=1), (p.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, 11)
    rel = [{'conf': float(conf[m].mean()), 'acc': float(correct[m].mean()), 'n': int(m.sum())}
           for lo, hi in zip(bins[:-1], bins[1:]) if (m := (conf > lo) & (conf <= hi)).sum()]
    out['_reliability'] = rel
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta-dirs', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index, proba = load(args.meta_dirs)
    groups = index['_group_key'].values
    report = {'design': 'nested_cv' if len(args.meta_dirs) > 1 else 'holdout',
              'n_events': int(len(index)), 'n_patients': int(index['_group_key'].nunique()),
              'n_recordings': int(index['filename'].nunique()), 'n_boot': args.boot,
              'tasks': {}}

    for target, names in TARGETS.items():
        y = index[target].values.astype(int)
        acoustic = target != 'model3_label'
        t = {'classes': names,
             'class_counts_events': np.bincount(y, minlength=len(names)).tolist(),
             'majority_class_accuracy': float(np.bincount(y).max() / len(y)),
             'ladder': {}, 'contrasts': {}, 'aggregation': {}}

        for s in LADDER:
            t['ladder'][s] = with_ci(y, proba[(target, s)], groups, args.boot,
                                     names if s == PRIMARY else None)
        p_main = proba[(target, PRIMARY)]
        t['confusion_matrix'] = confusion_matrix(y, p_main.argmax(axis=1)).tolist()
        t['curves'] = curves(y, p_main, names)
        if len(names) == 2:
            t['calibration'] = cs.calibration_slope_intercept(y, p_main)

        for metric, fn in [('auc', cs.auc_macro),
                           ('accuracy', lambda yy, pp: (pp.argmax(1) == yy).mean())]:
            res = {f'{a} - {b}': cs.paired_cluster_bootstrap_delta(
                fn, y, proba[(target, a)], proba[(target, b)], groups, args.boot)
                for a, b in CONTRASTS}
            for (k, r), padj in zip(res.items(), cs.holm([r['p'] for r in res.values()])):
                r['p_holm'] = padj
            t['contrasts'][metric] = res

        for level, label in [('filename', 'recording'), ('_group_key', 'patient')]:
            for rule in ('confweighted', 'mean'):
                ya, pa, ids = cs.aggregate(index, p_main, level, rule, target,
                                           derived_any_positive=acoustic)
                g = (ids if level == '_group_key' else
                     index.drop_duplicates('filename').set_index('filename')
                     .loc[ids, '_group_key'].values)
                a = with_ci(ya, pa, g, args.boot, names if rule == 'confweighted' else None)
                a['n_units'] = int(len(ya))
                a['class_counts'] = np.bincount(ya, minlength=len(names)).tolist()
                a['majority_class_accuracy'] = float(np.bincount(ya).max() / len(ya))
                a['label_is_derived'] = acoustic
                a['confusion_matrix'] = confusion_matrix(ya, pa.argmax(axis=1)).tolist()
                t['aggregation'][f'{label}|{rule}'] = a

        if report['design'] == 'nested_cv':
            per_fold = {}
            for f, rows in index.groupby('_fold').indices.items():
                per_fold[f] = {k: float(v) for k, v in
                               cs.overall_metrics(y[rows], p_main[rows]).items()}
            t['per_fold'] = per_fold
            t['per_fold_summary'] = {
                m: {'mean': float(np.mean([v[m] for v in per_fold.values()])),
                    'sd': float(np.std([v[m] for v in per_fold.values()], ddof=1)),
                    'min': float(np.min([v[m] for v in per_fold.values()])),
                    'max': float(np.max([v[m] for v in per_fold.values()]))}
                for m in ('accuracy', 'macro_f1', 'auc', 'mcc', 'brier', 'ece')}

        report['tasks'][target] = t
        m = t['ladder'][PRIMARY]
        print(f"{target}: acc {cs.fmt_ci(m['accuracy']['value'], m['accuracy']['ci'])}  "
              f"auc {cs.fmt_ci(m['auc']['value'], m['auc']['ci'])}  "
              f"(majority {t['majority_class_accuracy']:.3f})", flush=True)
        (out_dir / 'metrics.json').write_text(json.dumps(report, indent=2))

    print(f"Written to {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
