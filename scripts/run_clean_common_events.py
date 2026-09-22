#!/usr/bin/env python3
"""
scripts/run_clean_common_events.py

Isolates PATIENT-IDENTITY leakage from event overlap.

Arms L0 and L2 are otherwise scored on different test partitions, so their difference mixes
leakage with sampling. This script finds the events that BOTH arms held out of training and
validation - the locked hold-out test events that fall in the event-level arm's test
partition - and scores both arms' full stacked models on exactly those events.

On that common set:
  * arm L0 has seen neither the event nor the patient;
  * arm L2 has not seen the event either, but has trained on other events of the same patient.
The difference is therefore attributable to patient identity alone.
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from run_clean_meta import TARGETS, prep  # noqa: E402
import clean_stats as cs  # noqa: E402

RC = ROOT / 'results_clean'
KEY = ['filename', 'event_index']
ARMS = {'L0': RC / 'arm_L0_clean', 'L2': RC / 'arm_L2_event_split'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(RC / 'metrics' / 'common_events'))
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = {a: prep(pd.read_csv(p / 'probabilities' / 'meta_test.csv')) for a, p in ARMS.items()}
    seen_by_l2 = set()
    for part in ('train', 'val'):
        t = pd.read_csv(ARMS['L2'] / f'split_{part}.csv', usecols=KEY)
        seen_by_l2 |= set(map(tuple, t.values))

    common = (set(map(tuple, idx['L0'][KEY].values)) & set(map(tuple, idx['L2'][KEY].values))
              - seen_by_l2)
    assert common, 'no events are held out by both arms'
    rows = {a: np.array([tuple(r) in common for r in idx[a][KEY].values]) for a in ARMS}
    order = {a: idx[a][rows[a]].reset_index(drop=True) for a in ARMS}
    # align L2 rows to L0 row order
    pos = {tuple(r): i for i, r in enumerate(order['L2'][KEY].values)}
    take = np.array([pos[tuple(r)] for r in order['L0'][KEY].values])

    base = order['L0']
    groups = base['_group_key'].values
    l2_train_patients = set(pd.read_csv(ARMS['L2'] / 'split_train.csv',
                                        usecols=['_group_key'])['_group_key'])
    l2_train = pd.read_csv(ARMS['L2'] / 'split_train.csv', usecols=KEY + ['_group_key'])
    rec_in_train = set(l2_train['filename'])
    # which recordings of each child did L2 train on, excluding the evaluated events' own recordings
    other_rec = (l2_train.groupby('_group_key')['filename'].apply(set).to_dict())
    same_rec = base['filename'].isin(rec_in_train)
    child_other = {g: bool(other_rec.get(g, set()) - set(base.loc[base['_group_key'] == g,
                                                                    'filename']))
                   for g in base['_group_key'].unique()}
    report = {
        'events_whose_recording_was_in_L2_training': int(same_rec.sum()),
        'patients_with_another_recording_in_L2_training': int(sum(child_other.values())),
        'n_events': int(len(base)), 'n_patients': int(base['_group_key'].nunique()),
        'n_recordings': int(base['filename'].nunique()), 'n_boot': args.boot,
        'patients_also_in_L2_training': int(len(set(groups) & l2_train_patients)),
        'definition': ('locked hold-out test events that the event-level arm also kept out of '
                       'training and validation; both arms score the identical events, and the '
                       'event-level arm has trained on other events of the same patients'),
        'tasks': {},
    }

    for target, names in TARGETS.items():
        y = base[target].values.astype(int)
        proba = {}
        for a in ARMS:
            with open(ARMS[a] / 'meta_v2' / f'model_{target}_full_stack.pkl', 'rb') as f:
                bundle = pickle.load(f)
            rows_a = order[a] if a == 'L0' else order[a].iloc[take].reset_index(drop=True)
            assert (rows_a[KEY].values == base[KEY].values).all()
            assert (rows_a[target].values.astype(int) == y).all()
            proba[a] = bundle['model'].predict_proba(rows_a[bundle['features']])

        pat_y = base.groupby('_group_key')[target].agg(lambda v: int(v.max())
                                                      if target != 'model3_label'
                                                      else int(v.iloc[0]))
        t = {'classes': names,
             'class_counts': np.bincount(y, minlength=len(names)).tolist(),
             'patient_class_counts': np.bincount(pat_y.values,
                                                 minlength=len(names)).tolist(),
             'majority_class_accuracy': float(np.bincount(y).max() / len(y))}
        for a in ARMS:
            point = cs.overall_metrics(y, proba[a])
            ci, _ = cs.cluster_bootstrap(
                lambda r, p=proba[a]: cs.overall_metrics(y[r], p[r]), groups, args.boot)
            t[a] = {k: {'value': float(v), 'ci': ci.get(k)} for k, v in point.items()}
        t['delta_auc_L2_minus_L0'] = cs.paired_cluster_bootstrap_delta(
            cs.auc_macro, y, proba['L2'], proba['L0'], groups, args.boot)
        t['delta_accuracy_L2_minus_L0'] = cs.paired_cluster_bootstrap_delta(
            lambda yy, pp: (pp.argmax(1) == yy).mean(), y, proba['L2'], proba['L0'],
            groups, args.boot)
        report['tasks'][target] = t
        d = t['delta_auc_L2_minus_L0']
        print(f"{target:13s} L0 {t['L0']['auc']['value']:.3f}  L2 {t['L2']['auc']['value']:.3f}  "
              f"delta {d['delta']:+.3f} ({d['ci'][0]:+.3f},{d['ci'][1]:+.3f}) p={d['p']:.3f}",
              flush=True)

    (out / 'common_events.json').write_text(json.dumps(report, indent=2))
    print(f"{report['n_events']} events / {report['n_patients']} patients -> {out}")


if __name__ == '__main__':
    main()
