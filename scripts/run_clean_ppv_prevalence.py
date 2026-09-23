#!/usr/bin/env python3
"""
scripts/run_clean_ppv_prevalence.py

Post hoc, not pre-specified: positive predictive value of the screening outcome at other
prevalences, from the observed sensitivity and specificity (Bayes' rule), for adventitious
events (full stack, model's own decision) and for children (post hoc rule: at least two events
predicted adventitious). Output: results_clean/ppv_prevalence/ppv_prevalence.csv.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results_clean' / 'ppv_prevalence'


def ppv(se, sp, p):
    return se * p / (se * p + (1 - sp) * (1 - p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    op = pd.read_csv(ROOT / 'results_clean' / 'duration_operating_point' / 'duration_operating_point.csv')
    ev = op[(op['class'] == 'Adventitious') & (op.tertile == 'all')].iloc[0]
    rules = json.load(open(ROOT / 'results_clean' / 'metrics' / 'patient_rules' / 'patient_rules.json'))
    t = rules['tasks']['model2_label']
    ch = t['rules']['any_2']
    n_pos, n_all = t['children_by_derived_label'][1], t['n_children']
    rows = []
    for unit, se, sp, prevs in [
            ('event', ev.sensitivity, ev.specificity, [0.05, 0.10, 3297 / 19693]),
            ('child', ch['sensitivity_any']['value'], ch['specificity']['value'], [0.10, 0.20, n_pos / n_all])]:
        for p in prevs:
            rows.append({'unit': unit, 'sensitivity': round(se, 3), 'specificity': round(sp, 3),
                         'prevalence': round(p, 3), 'ppv': round(ppv(se, sp, p), 3),
                         'npv': round(sp * (1 - p) / (sp * (1 - p) + (1 - se) * p), 3)})
    pd.DataFrame(rows).to_csv(OUT / 'ppv_prevalence.csv', index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == '__main__':
    main()
