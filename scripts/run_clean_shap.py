#!/usr/bin/env python3
"""
scripts/run_clean_shap.py

TreeExplainer attribution for the full-stack LightGBM meta-learners of all three tasks,
computed on the TEST partition of a patient-grouped run (outputs of run_clean_meta_v2.py).

The question it answers: how much of each meta-learner's decision rests on the task's own
acoustic probabilities, on the other tasks' probabilities, and on demographics?
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from run_clean_meta import TARGETS, prep  # noqa: E402

TASK_TITLE = {'model2_label': 'Screening', 'model1_label': 'Sound pattern',
              'model3_label': 'Disease group'}
TASK_CLASSES = {'model1_label': ['Normal', 'Crackles', 'Wheeze/Rhonchi'],
                'model2_label': ['Normal', 'Abnormal'],
                'model3_label': ['Pneumonia', 'Bronchial', 'Normal/Other']}
GROUP_COLOURS = {'own-task acoustic': '#0072B2', 'other-task acoustic': '#56B4E9',
                 'age': '#D55E00', 'sex': '#E69F00', 'site': '#999999'}


def pretty(col):
    for t, classes in TASK_CLASSES.items():
        if col.startswith(t + '_p'):
            return f"{TASK_TITLE[t]}: P({classes[int(col.rsplit('p', 1)[1])]})"
    return {'age': 'Age (years)', 'gender_code': 'Sex',
            'recording_location': 'Auscultation site'}[col]


def group_of(col, target):
    if col.startswith(target):
        return 'own-task acoustic'
    if col.endswith(tuple(f'_p{i}' for i in range(4))):
        return 'other-task acoustic'
    return {'age': 'age', 'gender_code': 'sex', 'recording_location': 'site'}[col]


def shap_array(explainer, X, n_classes):
    """-> [n, features, classes] regardless of shap version / binary collapsing."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    if sv.ndim == 2:                       # binary: log-odds of class 1 only
        sv = np.stack([-sv, sv], axis=-1)
    assert sv.shape[-1] == n_classes
    return sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob-dir', required=True)
    ap.add_argument('--meta-dir', required=True, help='run_clean_meta_v2.py output')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    te = prep(pd.read_csv(Path(args.prob_dir) / 'meta_test.csv'))
    report, table = {}, []
    fig, axes = plt.subplots(1, 3, figsize=(170 / 25.4, 78 / 25.4), constrained_layout=True)

    for ax, (target, names) in zip(axes, TARGETS.items()):
        with open(Path(args.meta_dir) / f'model_{target}_full_stack.pkl', 'rb') as f:
            bundle = pickle.load(f)
        cols = bundle['features']
        X = te[cols]
        sv = shap_array(shap.TreeExplainer(bundle['model']), X, len(names))
        mean_abs = np.abs(sv).mean(axis=(0, 2))                 # per feature
        share = mean_abs / mean_abs.sum()
        groups = [group_of(c, target) for c in cols]
        gshare = pd.Series(share, index=groups).groupby(level=0).sum()
        report[target] = {
            'n_test_events': int(len(X)),
            'feature_share': {c: float(s) for c, s in zip(cols, share)},
            'group_share': {g: float(v) for g, v in gshare.items()},
            'demographic_share': float(gshare.reindex(['age', 'sex', 'site']).fillna(0).sum()),
            'per_class_mean_abs': {TASK_CLASSES[target][k]: {c: float(v) for c, v in zip(
                cols, np.abs(sv[:, :, k]).mean(axis=0))} for k in range(len(names))},
        }
        for c, s, g in zip(cols, share, groups):
            table.append({'task': TASK_TITLE[target], 'feature': pretty(c), 'group': g,
                          'mean_abs_shap': float(mean_abs[cols.index(c)]), 'share': float(s)})

        order = np.argsort(share)
        ax.barh([pretty(cols[i]) for i in order], share[order] * 100,
                color=[GROUP_COLOURS[groups[i]] for i in order], edgecolor='none')
        ax.set_xlabel('Share of mean |SHAP| (%)', fontsize=7)
        ax.set_title(TASK_TITLE[target], fontsize=8, loc='left')
        ax.tick_params(labelsize=6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

        plt.figure(figsize=(5, 3.6))
        k = 0 if target == 'model3_label' else len(names) - 1   # pneumonia / abnormal class
        shap.summary_plot(sv[:, :, k], X.rename(columns=pretty), show=False, max_display=11)
        plt.title(f"{TASK_TITLE[target]} - SHAP for class '{TASK_CLASSES[target][k]}'",
                  fontsize=9)
        plt.savefig(out_dir / f'beeswarm_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOURS.values()]
    fig.legend(handles, GROUP_COLOURS.keys(), loc='outside lower center', ncol=5,
               fontsize=6, frameon=False)
    fig.savefig(out_dir / 'shap_group_share.pdf')
    fig.savefig(out_dir / 'shap_group_share.png', dpi=300)
    plt.close(fig)

    (out_dir / 'shap_report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(table).to_csv(out_dir / 'shap_feature_table.csv', index=False)
    for t, r in report.items():
        print(f"{TASK_TITLE[t]:14s} demographic share of attribution = "
              f"{r['demographic_share'] * 100:.1f}%  groups={ {k: round(v, 3) for k, v in r['group_share'].items()} }")


if __name__ == '__main__':
    main()
