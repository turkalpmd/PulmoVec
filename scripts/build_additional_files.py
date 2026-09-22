#!/usr/bin/env python3
"""
scripts/build_additional_files.py

Machine-readable Additional files for the submission, assembled from result files only.
  AF2  subgroup performance (csv)          AF3  second-stage learner benchmark (csv)
  AF6  complete metrics workbook (xlsx)    AF8  participant partition assignment (csv)
"""

import json
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / 'results_clean'
NAME = {'model2_label': 'Screening', 'model1_label': 'Sound pattern',
        'model3_label': 'Disease group'}
ARMS = {'Nested CV (primary)': RC / 'metrics' / 'nested_cv' / 'metrics.json',
        'L0 hold-out': RC / 'arm_L0_clean' / 'metrics' / 'metrics.json',
        'L1 fine-tuned encoder': RC / 'arm_L1_backbone_leak' / 'metrics' / 'metrics.json',
        'L2 event-level split': RC / 'arm_L2_event_split' / 'metrics' / 'metrics.json'}


def flat(block):
    out = {}
    for m, v in block.items():
        if isinstance(v, dict) and 'value' in v:
            out[m] = v['value']
            if v.get('ci'):
                out[m + '_lo'], out[m + '_hi'] = v['ci']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    out = Path(ap.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for design, d in (('Nested CV', RC / 'metrics' / 'nested_cv_subgroups'),
                      ('L0 hold-out', RC / 'arm_L0_clean' / 'subgroups')):
        rep = json.loads((d / 'subgroups.json').read_text())
        for task, cols in rep.items():
            if task not in NAME:
                continue
            for col, levels in cols.items():
                for level, e in levels.items():
                    r = {'design': design, 'outcome': NAME[task], 'subgroup': col,
                         'level': level, 'n_patients': e['n_patients'],
                         'n_events': e['n_events'], 'suppressed': bool(e.get('suppressed'))}
                    for m in ('auc', 'accuracy', 'macro_f1'):
                        if m in e:
                            r[m], (r[m + '_lo'], r[m + '_hi']) = e[m]['value'], e[m]['ci']
                    for m in ('sensitivity', 'specificity'):
                        if m in e:
                            r[m] = e[m]
                    rows.append(r)
    sub = pd.DataFrame(rows)
    sub.to_csv(out / 'Additional_file_2.csv', index=False)

    b = pd.read_csv(RC / 'arm_L0_clean' / 'meta_benchmark' / 'meta_benchmark.csv')
    b['task'] = b['task'].map(NAME)
    b.to_csv(out / 'Additional_file_3.csv', index=False)

    overall, perclass, contrasts, agg, folds, cms = [], [], [], [], [], []
    for arm, path in ARMS.items():
        rep = json.loads(path.read_text())
        for task, t in rep['tasks'].items():
            key = {'arm': arm, 'outcome': NAME[task]}
            for rung, blk in t['ladder'].items():
                overall.append({**key, 'model': rung,
                                'majority_class_accuracy': t['majority_class_accuracy'],
                                **flat(blk)})
                for c, pc in blk.get('per_class', {}).items():
                    perclass.append({**key, 'model': rung, 'class': c, 'n_events': pc['n'],
                                     **flat(pc)})
            for metric, res in t['contrasts'].items():
                for name, r in res.items():
                    contrasts.append({**key, 'metric': metric, 'contrast': name,
                                      'delta': r['delta'], 'ci_lo': r['ci'][0],
                                      'ci_hi': r['ci'][1], 'p': r['p'], 'p_holm': r['p_holm']})
            for name, a in t['aggregation'].items():
                lvl, rule = name.split('|')
                agg.append({**key, 'level': lvl, 'rule': rule, 'n_units': a['n_units'],
                            'label_is_derived': a['label_is_derived'],
                            'majority_class_accuracy': a['majority_class_accuracy'], **flat(a)})
            for f, m in t.get('per_fold', {}).items():
                folds.append({**key, 'outer_fold': f, **m})
            for i, row in enumerate(t['confusion_matrix']):
                cms.append({**key, 'true_class': t['classes'][i],
                            **{f'pred_{c}': v for c, v in zip(t['classes'], row)}})
    with pd.ExcelWriter(out / 'Additional_file_6.xlsx') as xl:
        for name, rows in [('overall', overall), ('per_class', perclass),
                           ('paired_contrasts', contrasts), ('aggregation', agg),
                           ('outer_folds', folds), ('confusion_matrices', cms)]:
            pd.DataFrame(rows).to_excel(xl, sheet_name=name, index=False)

    parts = []
    for name in ('train', 'val', 'test'):
        d = pd.read_csv(RC / f'split_{name}.csv', usecols=['_group_key'])
        parts.append(pd.DataFrame({'participant': d['_group_key'].unique(), 'holdout': name}))
    parts = pd.concat(parts).merge(
        pd.read_csv(RC / 'nested_cv' / 'patient_fold_assignment.csv')
        .rename(columns={'_group_key': 'participant'}), on='participant', how='outer')
    parts['participant'] = parts['participant'].str.replace('pid:', '', regex=False)
    parts.sort_values('participant').to_csv(out / 'Additional_file_8.csv', index=False)
    print({'AF2 rows': len(sub), 'AF3 rows': len(b), 'AF6 overall rows': len(overall),
           'AF8 participants': len(parts), 'AF8 missing fold': int(parts['outer_fold'].isna().sum()),
           'AF8 missing holdout': int(parts['holdout'].isna().sum())})


if __name__ == '__main__':
    main()
