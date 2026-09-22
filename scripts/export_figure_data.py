#!/usr/bin/env python3
"""
scripts/export_figure_data.py

Flattens the result JSONs into tidy CSVs for the R figure script, so that the plotting
code contains no numbers of its own.  Output: results_clean/figure_data/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / 'results_clean'
TASK = {'model2_label': 'Screening', 'model1_label': 'Sound pattern',
        'model3_label': 'Disease group'}
CLASS = {'Abnormal': 'Adventitious', 'Rhonchi': 'Wheeze/rhonchi', 'Wheeze/Rhonchi': 'Wheeze/rhonchi',
         'Normal_Other': 'Normal/other', 'Bronchial': 'Bronchial disease'}
RUNG = {'duration_only': 'Event duration only', 'demographics_only': 'Demographics only',
        'own_task_base': 'Own-task base model', 'acoustic_only': 'All acoustic probabilities',
        'full_stack': 'Full stack'}
ARMS = {'L0': 'arm_L0_clean', 'L1': 'arm_L1_backbone_leak', 'L2': 'arm_L2_event_split'}


def load(p):
    return json.loads(Path(p).read_text())


def ci(block, key='auc'):
    b = block[key]
    return b['value'], b['ci'][0], b['ci'][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(RC / 'figure_data'))
    out = Path(ap.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ncv = load(RC / 'metrics' / 'nested_cv' / 'metrics.json')

    # ---- Fig 2: cohort flow ----------------------------------------------------------
    fl = load(RC / 'cohort_flow.json')
    sp = load(RC / 'split_summary.json')
    fold = pd.read_csv(RC / 'nested_cv' / 'patient_fold_assignment.csv')['outer_fold']
    rows = [{'stage': k, **{f: fl[s][f] for f in
                            ('events', 'recordings', 'patients_with_id')}}
            for k, s in (('raw', '1_annotated_events_with_audio'),
                         ('labelled', '2_after_label_mapping'),
                         ('cohort', '4_analysis_cohort'))]
    pd.DataFrame(rows).to_csv(out / 'flow_main.csv', index=False)
    pd.DataFrame([
        {'after': 'raw', 'reason': "No usable task label (stridor, 'no event')",
         'events': fl['2_excluded_no_task_label'], 'detail': ''},
        {'after': 'labelled', 'reason': 'Diagnosis not documented',
         'events': fl['3_excluded_undocumented_diagnosis']['events'],
         'detail': 'includes all {:,} events without a participant identifier'.format(
             fl['3_excluded_undocumented_diagnosis']['events_without_patient_id'])},
        {'after': 'labelled', 'reason': 'Implausible encoded age (>18 y)',
         'events': fl['3b_excluded_implausible_age']['events'], 'detail': 'one child'},
    ]).to_csv(out / 'flow_excluded.csv', index=False)
    pd.DataFrame([{'partition': k.capitalize(), 'patients': v['patients'],
                   'events': v['events']} for k, v in sp.items()]).to_csv(
        out / 'flow_holdout.csv', index=False)
    pd.DataFrame({'fold': sorted(fold.unique()),
                  'patients': [int((fold == f).sum()) for f in sorted(fold.unique())]}).to_csv(
        out / 'flow_folds.csv', index=False)

    # why an event-level split leaks: how much material each child contributes
    ev = pd.read_csv(ROOT / 'results_clean' / 'split_train.csv', usecols=['_group_key'])
    for part in ('val', 'test'):
        ev = pd.concat([ev, pd.read_csv(RC / f'split_{part}.csv', usecols=['_group_key'])])
    per = ev.groupby('_group_key').size().rename('events').reset_index()
    per.to_csv(out / 'events_per_patient.csv', index=False)

    # ---- Fig 3: curves ---------------------------------------------------------------
    roc, pr, rel, head = [], [], [], []
    for t, task in TASK.items():
        cur = ncv['tasks'][t]['curves']
        for c, d in cur.items():
            if c.startswith('_'):
                continue
            lab = CLASS.get(c, c)
            roc.append(pd.DataFrame({'task': task, 'class': lab, 'fpr': d['fpr'],
                                     'tpr': d['tpr']}))
            pr.append(pd.DataFrame({'task': task, 'class': lab, 'recall': d['recall'],
                                    'precision': d['precision'],
                                    'prevalence': d['prevalence']}))
        rel.append(pd.DataFrame(cur['_reliability']).assign(task=task))
        v, lo, hi = ci(ncv['tasks'][t]['ladder']['full_stack'])
        head.append({'task': task, 'auc': v, 'lo': lo, 'hi': hi})
    pd.concat(roc).to_csv(out / 'curves_roc.csv', index=False)
    pd.concat(pr).to_csv(out / 'curves_pr.csv', index=False)
    pd.concat(rel).to_csv(out / 'curves_reliability.csv', index=False)
    pd.DataFrame(head).to_csv(out / 'curves_headline.csv', index=False)

    # ---- Fig 4: comparator ladder ----------------------------------------------------
    rows = []
    for t, task in TASK.items():
        lad = ncv['tasks'][t]['ladder']
        for r, lab in RUNG.items():
            v, lo, hi = ci(lad[r])
            rows.append({'task': task, 'model': lab, 'auc': v, 'lo': lo, 'hi': hi,
                         'is_full': r == 'full_stack'})
    pd.DataFrame(rows).to_csv(out / 'ladder.csv', index=False)

    # ---- Fig 5: leakage arms ---------------------------------------------------------
    rows = []
    for arm, d in ARMS.items():
        m = load(RC / d / 'metrics' / 'metrics.json')
        for t, task in TASK.items():
            for level, blk in (('Event level', m['tasks'][t]['ladder']['full_stack']),
                               ('Patient level',
                                m['tasks'][t]['aggregation']['patient|confweighted'])):
                v, lo, hi = ci(blk)
                rows.append({'arm': arm, 'task': task, 'level': level, 'auc': v,
                             'lo': lo, 'hi': hi})
    l3 = load(RC / 'preprint_reported.json')
    short = {v: k for k, v in {'screen': 'Screening', 'pattern': 'Sound pattern',
                               'disease': 'Disease group'}.items()}
    for task, s in short.items():
        for level, suffix in (('Event level', 'event'), ('Patient level', 'patient')):
            key = f'{s}.{suffix}.auc'
            if key in l3:
                rows.append({'arm': 'L3', 'task': task, 'level': level, 'auc': l3[key],
                             'lo': np.nan, 'hi': np.nan})
    pd.DataFrame(rows).to_csv(out / 'arms.csv', index=False)

    ce = load(RC / 'metrics' / 'common_events' / 'common_events.json')
    rows = []
    for t, task in TASK.items():
        b = ce['tasks'][t]
        for arm in ('L0', 'L2'):
            v, lo, hi = ci(b[arm])
            rows.append({'task': task, 'arm': arm, 'auc': v, 'lo': lo, 'hi': hi})
        d = b['delta_auc_L2_minus_L0']
        rows.append({'task': task, 'arm': 'delta', 'auc': d['delta'], 'lo': d['ci'][0],
                     'hi': d['ci'][1]})
    df = pd.DataFrame(rows)
    df.attrs = {}
    df.to_csv(out / 'common_events.csv', index=False)
    pd.DataFrame([{k: ce[k] for k in ('n_events', 'n_patients',
                                      'patients_also_in_L2_training')}]).to_csv(
        out / 'common_events_meta.csv', index=False)

    # ---- Fig 6: attribution ----------------------------------------------------------
    sal = RC / 'arm_L0_clean' / 'saliency'
    rep = load(sal / 'saliency_report.json')['tasks']
    edges = [100] + load(sal / 'mel_band_edges.json')['patch_band_upper_edge_hz'] + [8000]
    pd.DataFrame({'band': range(1, 9), 'low_hz': edges[:-1], 'high_hz': edges[1:]}).to_csv(
        out / 'mel_bands.csv', index=False)
    maps, prof, dele = [], [], []
    panels = [('model2_label', 0, 'Normal'), ('model2_label', 1, 'Adventitious'),
              ('model1_label', 1, 'Crackles'), ('model1_label', 2, 'Wheeze/rhonchi')]
    for t, c, lab in panels:
        z = np.load(sal / f'saliency_maps_{t}.npz')
        ok = (z['true'] == c) & (z['pred'] == c)
        m = np.clip(z['occlusion'][ok], 0, None).mean(0).reshape(12, 8)
        m = m / m.max()
        g = pd.DataFrame([{'task': TASK[t], 'class': lab, 'n': int(ok.sum()),
                           'time': (i + 0.5) / 6, 'band': j + 1, 'value': m[i, j]}
                          for i in range(12) for j in range(8)])
        maps.append(g)
    for t in ('model2_label', 'model1_label'):
        for c, p in rep[t]['mel_band_profile_correct'].items():
            p = np.array(p) / np.sum(p)
            prof.append(pd.DataFrame({'task': TASK[t], 'class': CLASS.get(c, c),
                                      'band': range(1, 9), 'share': p}))
        r = pd.read_csv(sal / f'saliency_records_{t}.csv')
        r = r[r['true'] == r['pred']]
        for col, lab in (('deletion_auc_attr', 'Attribution-ordered'),
                         ('deletion_auc_rand', 'Random order')):
            q = r[col].quantile([0.25, 0.5, 0.75]).values
            dele.append({'task': TASK[t], 'order': lab, 'median': q[1], 'lo': q[0],
                         'hi': q[2], 'n': int(len(r))})
    pd.concat(maps).to_csv(out / 'saliency_maps.csv', index=False)
    pd.concat(prof).to_csv(out / 'saliency_profile.csv', index=False)
    pd.DataFrame(dele).to_csv(out / 'saliency_deletion.csv', index=False)

    print('figure data ->', out, f'({len(list(out.glob("*.csv")))} files)')


if __name__ == '__main__':
    main()
