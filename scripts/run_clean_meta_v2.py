#!/usr/bin/env python3
"""
scripts/run_clean_meta_v2.py

Stacking meta-learner with the full ablation ladder. Same tuning protocol as
run_clean_meta.py (Optuna on VAL, scored once on TEST) but

  * two extra rungs: duration_only (annotated event length - a potential shortcut) and
    own_task_base (the task's own base-model probabilities, no stacking);
  * TEST probabilities, fitted model and best parameters are saved for EVERY feature
    set, so paired contrasts can be computed downstream;
  * --prob-dir / --out-dir, so it serves the hold-out, every nested-CV fold and the
    leakage arms.

Metrics and CIs are produced by run_clean_metrics.py, not here.
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
sys.path.insert(0, str(ROOT / 'src'))

from run_clean_meta import (TARGETS, PROB_COLS, DEMO_COLS, prep, tune_and_fit,  # noqa: E402
                            core_metrics)

KEY = ['filename', 'event_index']
DURATION_COLS = ['event_duration_ms']


def feature_sets(target):
    own = [c for c in PROB_COLS if c.startswith(target)]
    return {
        'duration_only': DURATION_COLS,
        'demographics_only': DEMO_COLS,
        'duration_demographics': DURATION_COLS + DEMO_COLS,
        'own_task_base': own,
        'acoustic_only': PROB_COLS,
        'full_stack': PROB_COLS + DEMO_COLS,
    }


def attach_duration(meta, split_frames):
    """event_duration_ms lives in the split tables, keyed by (filename, event_index)."""
    if 'event_duration_ms' in meta.columns:
        return meta
    dur = pd.concat(split_frames)[KEY + ['event_duration_ms']].drop_duplicates(KEY)
    out = meta.merge(dur, on=KEY, how='left', validate='one_to_one')
    assert out['event_duration_ms'].notna().all(), "event duration missing for some rows"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--split-dir', default=str(ROOT / 'results_clean'),
                    help='directory holding split_{train,val,test}.csv (for event duration)')
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--class-weight', choices=['balanced', 'none'], default='balanced',
                    help="'none' = unweighted second stage (post hoc sensitivity analysis)")
    ap.add_argument('--sets', nargs='+', default=None, help='restrict to these feature sets')
    ap.add_argument('--allow-patient-overlap', action='store_true',
                    help='only for the deliberately leaky event-level arm')
    args = ap.parse_args()

    prob_dir, out_dir, split_dir = Path(args.prob_dir), Path(args.out_dir), Path(args.split_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [pd.read_csv(split_dir / f'split_{n}.csv', usecols=KEY + ['event_duration_ms'])
              for n in ('train', 'val', 'test')]
    tr, va, te = (attach_duration(prep(pd.read_csv(prob_dir / f)), splits)
                  for f in ('meta_train_oof.csv', 'meta_val.csv', 'meta_test.csv'))
    print(f"meta rows train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    for name, a, b in [('train/val', tr, va), ('train/test', tr, te), ('val/test', va, te)]:
        shared = set(a['_group_key']) & set(b['_group_key'])
        if shared and not args.allow_patient_overlap:
            raise AssertionError(f"patient leakage in {name}: {len(shared)}")

    te[KEY + ['_group_key', 'age', 'gender_code', 'recording_location', 'event_duration_ms',
              'event_type', 'disease'] + list(TARGETS)].to_csv(out_dir / 'test_index.csv',
                                                               index=False)
    summary = {}
    for target, class_names in TARGETS.items():
        n_classes = len(class_names)
        ytr, yva, yte = (d[target].values.astype(int) for d in (tr, va, te))
        summary[target] = {}
        for set_name, cols in feature_sets(target).items():
            if args.sets and set_name not in args.sets:
                continue
            model, params, val_f1 = tune_and_fit(tr[cols], ytr, va[cols], yva, n_classes,
                                                 args.trials, seed=args.seed,
                                                 class_weight=None if args.class_weight == 'none' else 'balanced')
            proba = model.predict_proba(te[cols])
            np.save(out_dir / f'test_proba_{target}_{set_name}.npy', proba)
            with open(out_dir / f'model_{target}_{set_name}.pkl', 'wb') as f:
                pickle.dump({'model': model, 'features': cols, 'classes': class_names}, f)
            m = core_metrics(yte, proba, n_classes)
            summary[target][set_name] = {'val_macro_f1': val_f1, 'best_params': params,
                                         'features': cols, 'test_quicklook': m}
            auc = m.get('roc_auc', m.get('macro_roc_auc'))
            print(f"  {target:13s} {set_name:22s} val_f1={val_f1:.4f} "
                  f"test_acc={m['accuracy']:.4f} test_auc={auc:.4f}", flush=True)
            (out_dir / 'meta_v2_summary.json').write_text(json.dumps(summary, indent=2))

    print(f"Written to {out_dir}")


if __name__ == '__main__':
    main()
