#!/usr/bin/env python3
"""
scripts/generate_cohort_table.py

Builds Table 1 (cohort descriptives) directly from the curated event table and the
patient-grouped split, so the manuscript never again carries hand-typed counts.

Continuous variables: Mann-Whitney U (train vs test).
Categorical variables: chi-square, or Fisher's exact where any expected cell < 5.

Patient-level rows are counted over unique patients; event-level rows over events.
Writes results_clean/table1_cohort.md and .csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
RESULTS = ROOT / 'results_clean'


def cat_pvalue(a_counts, b_counts):
    """Chi-square, falling back to Fisher for sparse 2xN tables."""
    keys = sorted(set(a_counts.index) | set(b_counts.index))
    table = np.array([[a_counts.get(k, 0) for k in keys],
                      [b_counts.get(k, 0) for k in keys]], dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return np.nan, 'n/a'
    chi2, p, dof, expected = stats.chi2_contingency(table)
    if (expected < 5).any():
        if table.shape[1] == 2:
            _, p = stats.fisher_exact(table)
            return p, "Fisher's exact"
        return p, 'chi-square (sparse cells - interpret with caution)'
    return p, 'chi-square'


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '—'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def main():
    train = pd.read_csv(RESULTS / 'split_train.csv')
    val = pd.read_csv(RESULTS / 'split_val.csv')
    test = pd.read_csv(RESULTS / 'split_test.csv')
    full = pd.concat([train, val, test], ignore_index=True)

    rows = []

    def add(label, all_v='', tr='', va='', te='', p='', note=''):
        rows.append({'Variable': label, 'All': all_v, 'Train': tr,
                     'Validation': va, 'Test': te, 'p (train vs test)': p,
                     'Test used': note})

    # --- patient-level ---
    pats = {k: d.drop_duplicates('_group_key') for k, d in
            [('all', full), ('train', train), ('val', val), ('test', test)]}

    add('**Patients**', f"{len(pats['all']):,}", f"{len(pats['train']):,}",
        f"{len(pats['val']):,}", f"{len(pats['test']):,}")

    idless = full['_group_key'].str.startswith('unknown:')
    add('  of which id-less recordings (training-only)',
        f"{full.loc[idless, 'filename'].nunique():,}",
        f"{train.loc[train['_group_key'].str.startswith('unknown:'), 'filename'].nunique():,}",
        '0', '0')

    # Age (patient level, known patients only)
    def ages(d):
        return pd.to_numeric(d.loc[~d['_group_key'].str.startswith('unknown:'), 'age'],
                             errors='coerce').dropna()
    a_tr, a_te, a_va, a_all = ages(pats['train']), ages(pats['test']), ages(pats['val']), ages(pats['all'])
    u_p = stats.mannwhitneyu(a_tr, a_te, alternative='two-sided')[1] if len(a_te) else np.nan

    def med(x):
        return f"{x.median():.1f} ({x.quantile(.25):.1f}–{x.quantile(.75):.1f})" if len(x) else '—'
    add('Age (years), median (IQR)', med(a_all), med(a_tr), med(a_va), med(a_te),
        fmt_p(u_p), 'Mann-Whitney U')

    # Sex
    p, tst = cat_pvalue(pats['train']['gender'].value_counts(), pats['test']['gender'].value_counts())
    add('Sex, n (%)', '', '', '', '', fmt_p(p), tst)
    for lvl in sorted(pats['all']['gender'].dropna().unique()):
        def pct(d):
            n = int((d['gender'] == lvl).sum())
            return f"{n:,} ({n/len(d)*100:.1f})"
        add(f'  {lvl}', f"{int((pats['all']['gender']==lvl).sum()):,}",
            pct(pats['train']), pct(pats['val']), pct(pats['test']))

    # Disease group (patient level)
    p, tst = cat_pvalue(pats['train']['model3_label'].value_counts(),
                        pats['test']['model3_label'].value_counts())
    add('Disease group (patient-level), n (%)', '', '', '', '', fmt_p(p), tst)
    names = {0: 'Pneumonia', 1: 'Bronchial diseases', 2: 'Normal / Other'}
    for lvl, nm in names.items():
        def pct(d):
            n = int((d['model3_label'] == lvl).sum())
            return f"{n:,} ({n/len(d)*100:.1f})"
        add(f'  {nm}', f"{int((pats['all']['model3_label']==lvl).sum()):,}",
            pct(pats['train']), pct(pats['val']), pct(pats['test']))

    # --- event-level ---
    add('**Events**', f"{len(full):,}", f"{len(train):,}", f"{len(val):,}", f"{len(test):,}")

    p, tst = cat_pvalue(train['event_type'].value_counts(), test['event_type'].value_counts())
    add('Event type, n (%)', '', '', '', '', fmt_p(p), tst)
    for lvl in full['event_type'].value_counts().index:
        def pct(d):
            n = int((d['event_type'] == lvl).sum())
            return f"{n:,} ({n/len(d)*100:.1f})"
        add(f'  {lvl}', f"{int((full['event_type']==lvl).sum()):,}",
            pct(train), pct(val), pct(test))

    p, tst = cat_pvalue(train['recording_location'].value_counts(),
                        test['recording_location'].value_counts())
    add('Recording location, n (%)', '', '', '', '', fmt_p(p), tst)
    for lvl in sorted(full['recording_location'].dropna().unique()):
        def pct(d):
            n = int((d['recording_location'] == lvl).sum())
            return f"{n:,} ({n/len(d)*100:.1f})"
        add(f'  {lvl}', f"{int((full['recording_location']==lvl).sum()):,}",
            pct(train), pct(val), pct(test))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'table1_cohort.csv', index=False)

    md = ["# Table 1. Cohort characteristics (patient-grouped split)", "",
          "Generated directly from the curated event table; do not edit by hand.", "",
          out.to_markdown(index=False), "",
          "Patients are grouped so that every event of one patient sits in a single "
          "partition. Recordings whose SPRSound filename carries no patient identifier "
          "are training-only and are therefore excluded from the validation and test "
          "columns and from all patient-level statistics.", ""]
    (RESULTS / 'table1_cohort.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
