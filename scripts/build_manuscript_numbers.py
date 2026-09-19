#!/usr/bin/env python3
"""
scripts/build_manuscript_numbers.py

Single source of truth for every number printed in the manuscript.

Flattens the JSON outputs of the run_clean_* scripts into
results_clean/manuscript_numbers.json:

    "<arm>.<task>.<rung>.<metric>": {"value": .., "ci": [lo, hi], "source": "<file>"}

arm   l0 | l1 | l2 | ncv          task  screen | pattern | disease
The manuscript never contains a typed number: manuscript.md uses {{key}} placeholders that
render_manuscript.py fills from this registry, and check_manuscript_numbers.py fails the
build if a stray numeric literal is found.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / 'results_clean'
TASK = {'model2_label': 'screen', 'model1_label': 'pattern', 'model3_label': 'disease'}

ARMS = {  # arm -> metrics.json produced by run_clean_metrics.py
    'l0': RC / 'arm_L0_clean' / 'metrics' / 'metrics.json',
    'ncv': RC / 'metrics' / 'nested_cv' / 'metrics.json',
    'l1': RC / 'arm_L1_backbone_leak' / 'metrics' / 'metrics.json',
    'l2': RC / 'arm_L2_event_split' / 'metrics' / 'metrics.json',
}


def slug(s):
    return (s.lower().replace('/', '_').replace(' ', '_').replace('-', '_')
            .replace('|', '.'))


def put(reg, key, value, source, ci=None):
    if value is None:
        return
    assert key not in reg, f"duplicate registry key {key}"
    reg[key] = {'value': value, 'source': str(source.relative_to(ROOT))}
    if ci:
        reg[key]['ci'] = list(ci)


def add_metric_block(reg, prefix, block, source):
    for m, v in block.items():
        if isinstance(v, dict) and 'value' in v:
            put(reg, f'{prefix}.{m}', v['value'], source, v.get('ci'))


def main():
    reg = {}
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"[skip] {arm}: {path.relative_to(ROOT)} not found")
            continue
        rep = json.loads(path.read_text())
        for k in ('n_events', 'n_patients', 'n_recordings', 'n_boot'):
            put(reg, f'{arm}.{k}', rep[k], path)
        for target, t in rep['tasks'].items():
            base = f'{arm}.{TASK[target]}'
            put(reg, f'{base}.majority_accuracy', t['majority_class_accuracy'], path)
            for c, n in zip(t['classes'], t['class_counts_events']):
                put(reg, f'{base}.n_events.{slug(c)}', n, path)
            for rung, block in t['ladder'].items():
                add_metric_block(reg, f'{base}.{rung}', block, path)
                for c, pc in block.get('per_class', {}).items():
                    add_metric_block(reg, f'{base}.{rung}.{slug(c)}', pc, path)
            for metric, res in t['contrasts'].items():
                for name, r in res.items():
                    a, b = name.split(' - ')
                    k = f'{base}.delta_{metric}.{a}_vs_{b}'
                    put(reg, k, r['delta'], path, r['ci'])
                    put(reg, k + '.p_holm', r['p_holm'], path)
            for name, a in t['aggregation'].items():
                k = f'{base}.agg.{slug(name)}'
                put(reg, f'{k}.n_units', a['n_units'], path)
                put(reg, f'{k}.majority_accuracy', a['majority_class_accuracy'], path)
                add_metric_block(reg, k, a, path)
            if 'calibration' in t:
                put(reg, f'{base}.cal_slope', t['calibration']['slope'], path)
                put(reg, f'{base}.cal_intercept', t['calibration']['intercept'], path)
            for m, s in t.get('per_fold_summary', {}).items():
                for stat, v in s.items():
                    put(reg, f'{base}.fold_{stat}.{m}', v, path)

    split = RC / 'split_summary.json'
    for part, d in json.loads(split.read_text()).items():
        for k, v in d.items():
            put(reg, f'cohort.{part}.{k}', v, split)
    for k in ('events', 'patients'):
        put(reg, f'cohort.total.{k}', sum(reg[f'cohort.{p}.{k}']['value']
                                          for p in ('train', 'val', 'test')), split)

    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=ROOT).decode().strip()
    except Exception:
        commit = 'unknown'
    out = RC / 'manuscript_numbers.json'
    out.write_text(json.dumps({'_git_commit': commit, 'numbers': reg}, indent=1))
    print(f"{len(reg)} numbers -> {out.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
