#!/usr/bin/env python3
"""
scripts/run_clean_subgroups.py

Exploratory subgroup performance of the full-stack model (no hypothesis tests):
age band, sex, auscultation site, and annotated-event-duration tertile (shortcut check).
Cells with fewer than MIN_PATIENTS patients or a missing class are suppressed.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from run_clean_meta import TARGETS  # noqa: E402
from run_clean_metrics import load, PRIMARY  # noqa: E402
import clean_stats as cs  # noqa: E402

MIN_PATIENTS = 10
AGE_BINS = [0, 3, 6, 12, 200]
AGE_LABELS = ['<3 y', '3-<6 y', '6-<12 y', '>=12 y']


def subgroup_columns(index):
    out = pd.DataFrame(index=index.index)
    out['age_band'] = pd.cut(index['age'], AGE_BINS, right=False, labels=AGE_LABELS).astype(str)
    # SPRSound README: Male 0, Female 1
    out['sex'] = index['gender_code'].map({0: 'male', 1: 'female'}).fillna('unknown')
    out['site'] = 'p' + index['recording_location'].astype('Int64').astype(str)
    out['duration_tertile'] = pd.qcut(index['event_duration_ms'], 3,
                                      labels=['short', 'medium', 'long']).astype(str)
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
    sg = subgroup_columns(index)
    report, rows = {}, []

    for target, names in TARGETS.items():
        y, p = index[target].values.astype(int), proba[(target, PRIMARY)]
        report[target] = {}
        for col in sg.columns:
            report[target][col] = {}
            for level, r in sg.groupby(col).indices.items():
                n_pat = index['_group_key'].iloc[r].nunique()
                entry = {'n_events': int(len(r)), 'n_patients': int(n_pat)}
                if n_pat < MIN_PATIENTS or len(np.unique(y[r])) < len(names):
                    entry['suppressed'] = True
                else:
                    g = index['_group_key'].values[r]
                    yy, pp = y[r], p[r]
                    point = cs.overall_metrics(yy, pp)
                    ci, _ = cs.cluster_bootstrap(
                        lambda b, yy=yy, pp=pp: cs.overall_metrics(yy[b], pp[b]), g, args.boot)
                    for m in ('accuracy', 'macro_f1', 'auc'):
                        entry[m] = {'value': float(point[m]), 'ci': ci[m]}
                    if len(names) == 2:
                        pc = cs.per_class_metrics(yy, pp, names)[names[1]]
                        entry['sensitivity'], entry['specificity'] = (pc['sensitivity'],
                                                                      pc['specificity'])
                    rows.append({'task': target, 'subgroup': col, 'level': level,
                                 'n_patients': n_pat, 'n_events': len(r),
                                 'auc': cs.fmt_ci(point['auc'], ci['auc']),
                                 'accuracy': cs.fmt_ci(point['accuracy'], ci['accuracy'])})
                report[target][col][level] = entry

    (out_dir / 'subgroups.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(out_dir / 'subgroups.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
