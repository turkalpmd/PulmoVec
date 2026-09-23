#!/usr/bin/env python3
"""
scripts/run_clean_duration_operating_point.py

Post hoc, not pre-specified: does the operating point of the two acoustic outcomes depend on the
annotated event duration? For the full-stack model of the nested cross-validation, per
duration tertile (same cut points as run_clean_subgroups.py): sensitivity for each adventitious
class and specificity for normal events at the model's own decision (argmax), and, to separate
ranking from threshold, sensitivity at the threshold that gives the overall specificity within
each tertile. Patient-cluster bootstrap, B = 2000, seed 42.
Output: results_clean/duration_operating_point/.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from run_clean_metrics import load, PRIMARY  # noqa: E402
from run_clean_subgroups import subgroup_columns  # noqa: E402
import clean_stats as cs  # noqa: E402

OUT = ROOT / 'results_clean' / 'duration_operating_point'
CLASSES = {'model2_label': {1: 'Adventitious'},
           'model1_label': {1: 'Crackles', 2: 'Wheeze/rhonchi'}}


def cell_metrics(y, p, cls, spec_target):
    pred = p.argmax(1)
    out = {'sensitivity': float((pred[y == cls] == cls).mean()),
           'specificity': float((pred[y == 0] == 0).mean())}
    # sensitivity at the threshold on P(abnormal) that reproduces spec_target in this cell
    s = 1.0 - p[:, 0]
    thr = np.quantile(s[y == 0], spec_target)
    out['sensitivity_at_overall_specificity'] = float((s[y == cls] > thr).mean())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = [ROOT / 'results_clean' / 'nested_cv' / f'fold{k}' / 'meta_v2' for k in range(1, 6)]
    index, proba = load([str(d) for d in dirs])
    tert = subgroup_columns(index)['duration_tertile'].values
    groups = index['_group_key'].values
    bounds = pd.qcut(index['event_duration_ms'], 3, retbins=True)[1]
    rows = []
    for target, classes in CLASSES.items():
        y, p = index[target].values.astype(int), proba[(target, PRIMARY)]
        spec_all = float((p.argmax(1)[y == 0] == 0).mean())
        for cls, name in classes.items():
            for t in ['short', 'medium', 'long', 'all']:
                r = np.arange(len(y)) if t == 'all' else np.where(tert == t)[0]
                yy, pp, gg = y[r], p[r], groups[r]
                point = cell_metrics(yy, pp, cls, spec_all)
                ci, _ = cs.cluster_bootstrap(lambda b: cell_metrics(yy[b], pp[b], cls, spec_all),
                                             gg, n_boot=2000, seed=42)
                row = {'target': target, 'class': name, 'tertile': t,
                       'n_class_events': int((yy == cls).sum()), 'n_normal_events': int((yy == 0).sum()),
                       'n_patients': int(pd.Series(gg).nunique())}
                for k, v in point.items():
                    row[k] = round(v, 3)
                    row[k + '_ci'] = f'{ci[k][0]:.3f}-{ci[k][1]:.3f}'
                rows.append(row)
                print(row, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'duration_operating_point.csv', index=False)
    json.dump({'tertile_bounds_ms': [float(b) for b in bounds], 'n_boot': 2000, 'seed': 42,
               'rows': rows}, open(OUT / 'duration_operating_point.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
