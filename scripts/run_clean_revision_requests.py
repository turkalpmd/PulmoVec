#!/usr/bin/env python3
"""
scripts/run_clean_revision_requests.py

Post hoc analyses requested during revision (none pre-specified):
  A1  child-level confusion counts of the ">= 2 events predicted adventitious" screening rule,
      unrounded sensitivity/specificity, PPV and NPV at prevalences 0.10, 0.20, 0.327;
  A2  inventory of the 42 post hoc disease-group configurations;
  C1  child-level decision curve (net benefit) for the pre-specified and the >= 2-event rule;
  B1  release inventory for a cross-release validation (feasibility counts, event-type and
      duration distributions per release group) - no model is trained here;
  B2  unweighted second stage in the nested CV (summary of meta_v2_unweighted/, --b2 only).
Output: results_clean/revision_requests/.
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
from run_clean_patient_rules import child_table  # noqa: E402
import clean_stats as cs  # noqa: E402

RC = ROOT / 'results_clean'
OUT = RC / 'revision_requests'
FOLDS = [RC / 'nested_cv' / f'fold{k}' / 'meta_v2' for k in range(1, 6)]
PREVS = [0.10, 0.20, 0.327]


def a1_c1():
    index, proba = load(FOLDS)
    t = child_table(index, proba[('model2_label', PRIMARY)], 'model2_label')
    y = t['truth'].values > 0
    out = {}
    for rule in ('confweighted', 'any_2'):
        p = t[rule].values > 0
        tp, fn = int((p & y).sum()), int((~p & y).sum())
        tn, fp = int((~p & ~y).sum()), int((p & ~y).sum())
        se, sp = tp / (tp + fn), tn / (tn + fp)
        out[rule] = {'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp, 'sensitivity': se, 'specificity': sp,
                     'at_prevalence': {str(pr): {'PPV': se * pr / (se * pr + (1 - sp) * (1 - pr)),
                                                 'NPV': sp * (1 - pr) / (sp * (1 - pr) + (1 - se) * pr)}
                                       for pr in PREVS}}
    assert out['any_2']['TP'] + out['any_2']['FN'] == 241 and out['any_2']['TN'] + out['any_2']['FP'] == 495
    # C1: net benefit of binary decisions at threshold probabilities 0.05-0.40
    n, prev = len(t), y.mean()
    curve = []
    for pt in np.round(np.arange(0.05, 0.401, 0.05), 2):
        w = pt / (1 - pt)
        row = {'threshold': float(pt), 'treat_none': 0.0, 'treat_all': float(prev - (1 - prev) * w)}
        for rule in ('confweighted', 'any_2'):
            row[rule] = (out[rule]['TP'] - out[rule]['FP'] * w) / n
        curve.append(row)
    return out, pd.DataFrame(curve)


def a2():
    r = pd.read_csv(RC / 'posthoc_disease' / 'results.csv')
    tgt = {'3class': 'three groups', 'pneumonia_vs_rest': 'pneumonia vs other',
           'pneumonia_vs_control': 'pneumonia vs healthy controls'}
    feat = {'demo': 'age and sex', 'acoustic': 'acoustic aggregates',
            'acoustic+demo': 'acoustic aggregates, age and sex'}
    lrn = {'logreg': 'logistic regression', 'lgbm': 'LightGBM', '-': 'none'}
    inv = pd.DataFrame({
        'target': r.target.map(tgt), 'level': r.level,
        'features': [('unfitted mean predicted probability' if f.startswith('ref') else feat[f])
                     for f in r.features],
        'learner': r.learner.map(lrn), 'n_units': r.n, 'majority_rate': r.majority,
        'auc': r.auc, 'auc_ci': r.auc_ci, 'accuracy': r.acc, 'accuracy_ci': r.acc_ci,
        'balanced_accuracy': r.bacc})
    assert len(inv) == 42 and (inv.features == 'unfitted mean predicted probability').sum() == 6
    return inv


def b1_inventory():
    raw = pd.read_csv(ROOT / 'data' / 'SPRSound_Event_Level_Dataset_CLEAN.csv')
    coh = pd.concat([pd.read_csv(RC / f'split_{p}.csv') for p in ('train', 'val', 'test')])
    grp = {'Classification-Train': '2022', 'Classification-Valid-2022-Intra': '2022',
           'Classification-Valid-2022-Inter': '2022', 'Classification-Valid-2023': '2023',
           'Detection-Test-2024': '2024', 'BioCAS2025-Test': '2025'}
    for d in (raw, coh):
        d['release'] = d['dataset'].map(grp)
        d['pid'] = d['patient_number']
    rows = []
    for rel, d in raw.groupby('release'):
        c = coh[coh.release == rel]
        rows.append({'release': rel, 'raw_events': len(d), 'raw_recordings': d.filename.nunique(),
                     'raw_events_without_id': int(d.pid.isna().sum()),
                     'raw_children_with_id': int(d.pid.nunique()),
                     'children_with_documented_diagnosis': int(d.loc[d.disease != 'Unknown', 'pid'].nunique()),
                     'cohort_events': len(c), 'cohort_recordings': c.filename.nunique(),
                     'cohort_children': c._group_key.nunique()})
    by_child = coh.groupby('_group_key').release.nunique()
    multi = int((by_child > 1).sum())
    et = coh.groupby('release').event_type.value_counts(normalize=True).round(3).unstack(fill_value=0)
    dur = coh.groupby('release').event_duration_ms.describe(percentiles=[.25, .5, .75])[['25%', '50%', '75%']]
    adv = coh.groupby('release').model2_label.mean().round(3)
    dxs = coh.drop_duplicates('_group_key').groupby('release').model3_label.value_counts().unstack(fill_value=0)
    return pd.DataFrame(rows), multi, et, dur, adv, dxs


def b2():
    import clean_stats as cs
    tasks = {'model2_label': 2, 'model1_label': 3, 'model3_label': 3}
    idx, P = [], {}
    for k in range(1, 6):
        d = RC / 'nested_cv' / f'fold{k}'
        idx.append(pd.read_csv(d / 'meta_v2_unweighted' / 'test_index.csv'))
        for t in tasks:
            for s in ('full_stack', 'demographics_only', 'duration_only'):
                P.setdefault((t, s), []).append(np.load(d / 'meta_v2_unweighted' / f'test_proba_{t}_{s}.npy'))
    index = pd.concat(idx, ignore_index=True)
    P = {k: np.vstack(v) for k, v in P.items()}
    g = index._group_key.values
    out = {}
    for t, nc in tasks.items():
        y = index[t].values.astype(int)
        res = {}
        for level in ('event', 'patient'):
            if level == 'event':
                yy, pp, gg = y, P[(t, 'full_stack')], g
            else:
                yy, pp, gg = cs.aggregate(index, P[(t, 'full_stack')], '_group_key', 'confweighted',
                                          t, derived_any_positive=(t != 'model3_label'))
            pt = cs.overall_metrics(yy, pp)
            ci, _ = cs.cluster_bootstrap(lambda r: {'auc': cs.overall_metrics(yy[r], pp[r])['auc']},
                                         gg, n_boot=2000, seed=42)
            res[level] = {'auc': pt['auc'], 'auc_ci': ci['auc'], 'accuracy': pt['accuracy'],
                          'majority_rate': float(np.bincount(yy).max() / len(yy)),
                          'balanced_accuracy': pt['balanced_accuracy'], 'mcc': pt['mcc']}
        # wording rules
        if t == 'model3_label':
            d = cs.paired_cluster_bootstrap_delta(lambda a, b: cs.auc_macro(a, b), y,
                                                  P[(t, 'full_stack')], P[(t, 'demographics_only')], g)
            res['rule_full_vs_demographics'] = d
        else:
            res['duration_only_auc'] = cs.auc_macro(y, P[(t, 'duration_only')])
        out[t] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b2', action='store_true', help='only summarise the unweighted re-run')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.b2:
        r = b2()
        (OUT / 'b2_unweighted.json').write_text(json.dumps(r, indent=1, default=float))
        print(json.dumps(r, indent=1, default=float))
        return
    a1, dc = a1_c1()
    (OUT / 'a1_child_confusion.json').write_text(json.dumps(a1, indent=1))
    dc.to_csv(OUT / 'c1_decision_curve.csv', index=False)
    inv = a2()
    inv.to_csv(OUT / 'a2_posthoc_inventory.csv', index=False)
    rel, multi, et, dur, adv, dxs = b1_inventory()
    rel.to_csv(OUT / 'b1_release_inventory.csv', index=False)
    et.to_csv(OUT / 'b1_event_types_by_release.csv')
    dur.to_csv(OUT / 'b1_duration_by_release.csv')
    dxs.to_csv(OUT / 'b1_disease_group_children_by_release.csv')
    print(json.dumps(a1, indent=1)); print(dc.round(4).to_string())
    print(inv.to_string()); print(rel.to_string()); print('children in >1 release group:', multi)
    print(et.to_string()); print(dur.to_string()); print(adv.to_string()); print(dxs.to_string())


if __name__ == '__main__':
    main()
