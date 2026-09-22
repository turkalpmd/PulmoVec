#!/usr/bin/env python3
"""
scripts/run_clean_patient_rules.py

Child-level detection of adventitious sounds under different aggregation rules
(post hoc; not in the pre-specified plan).

The pre-specified rule averages a child's event probabilities with confidence weights.
Because most of a child's events are normal, that average rarely favours "adventitious",
whereas the derived child-level label is "any adventitious event". This compares the
pre-specified rule with count rules that match the label: a child is called positive if at
least k of its events are predicted adventitious (k = 1, 2), and - for sound pattern - is
assigned the most frequent predicted adventitious pattern.

Input: pooled out-of-fold predictions of the nested cross-validation (full stack).
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from run_clean_metrics import load, PRIMARY  # noqa: E402

RC = ROOT / 'results_clean'
NAMES = {'model2_label': ['Normal', 'Adventitious'],
         'model1_label': ['Normal', 'Crackles', 'Wheeze/rhonchi']}


def child_table(index, proba, target):
    d = index[['_group_key', target]].copy()
    d['pred'] = proba.argmax(1)
    d['conf'] = proba.max(1)
    rows = []
    for g, e in d.groupby('_group_key', sort=False):
        lab = e[target].values.astype(int)
        pos = lab[lab > 0]
        truth = int(np.bincount(pos).argmax()) if len(pos) else 0
        p = proba[e.index.values]
        w = p.max(1) / p.max(1).sum()
        cw = int((p * w[:, None]).sum(0).argmax())
        pp = e['pred'].values
        adv = pp[pp > 0]
        rows.append({'child': g, 'truth': truth, 'confweighted': cw,
                     'any_1': int(np.bincount(adv).argmax()) if len(adv) >= 1 else 0,
                     'any_2': int(np.bincount(adv).argmax()) if len(adv) >= 2 else 0,
                     'n_events': len(e)})
    return pd.DataFrame(rows)


def metrics(t, rule, k):
    y, p = t['truth'].values, t[rule].values
    out = {}
    for c in range(1, k):
        m = y == c
        out[f'sens_{c}'] = float((p[m] == c).mean()) if m.any() else np.nan
    yb, pb = y > 0, p > 0
    out['sensitivity_any'] = float(pb[yb].mean())
    out['specificity'] = float((~pb[~yb]).mean())
    per = [(p[y == c] == c).mean() for c in range(k) if (y == c).any()]
    out['balanced_accuracy'] = float(np.mean(per))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', default=str(RC / 'metrics' / 'patient_rules' / 'patient_rules.json'))
    a = ap.parse_args()
    index, proba = load([RC / 'nested_cv' / f'fold{k}' / 'meta_v2' for k in range(1, 6)])
    rng = np.random.default_rng(42)
    report = {'source': 'nested cross-validation, pooled out-of-fold full-stack predictions',
              'pre_specified': False, 'n_boot': a.boot, 'tasks': {}}
    for target, names in NAMES.items():
        t = child_table(index, proba[(target, PRIMARY)], target)
        k = len(names)
        res = {'n_children': int(len(t)),
               'children_by_derived_label': np.bincount(t['truth'], minlength=k).tolist(),
               'rules': {}}
        for rule in ('confweighted', 'any_1', 'any_2'):
            point = metrics(t, rule, k)
            draws = {m: [] for m in point}
            for _ in range(a.boot):
                b = t.iloc[rng.integers(0, len(t), len(t))]
                for m, v in metrics(b, rule, k).items():
                    draws[m].append(v)
            res['rules'][rule] = {m: {'value': v, 'ci': [float(np.nanpercentile(draws[m], 2.5)),
                                                          float(np.nanpercentile(draws[m], 97.5))]}
                                  for m, v in point.items()}
            print(f"{target:13s} {rule:13s} " + "  ".join(
                f"{m} {v:.3f}" for m, v in point.items()), flush=True)
        report['tasks'][target] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print('->', a.out)


if __name__ == '__main__':
    main()
