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
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=ROOT).decode().strip()
    except Exception:
        commit = 'unknown'
    put(reg, 'repo.commit', commit, RC / 'manuscript_numbers.json')
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

    # ---- derived and auxiliary numbers -------------------------------------------------
    def val(k):
        return reg[k]['value']

    for task in TASK.values():                      # leakage inflation relative to arm L0
        for arm in ('l1', 'l2'):
            for lvl, key in (('event', 'full_stack'), ('patient', 'agg.patient.confweighted')):
                for m in ('auc', 'accuracy'):
                    a_, b_ = f'{arm}.{task}.{key}.{m}', f'l0.{task}.{key}.{m}'
                    if a_ in reg and b_ in reg:
                        put(reg, f'infl.{arm}.{task}.{lvl}.{m}', val(a_) - val(b_),
                            ARMS[arm])

    flow_p = RC / 'cohort_flow.json'
    if flow_p.exists():
        fl = json.loads(flow_p.read_text())
        for stage, short in (('1_annotated_events_with_audio', 'raw'),
                             ('2_after_label_mapping', 'labelled'),
                             ('3_excluded_undocumented_diagnosis', 'excl_unknown'),
                             ('3b_excluded_implausible_age', 'excl_age'),
                             ('4_analysis_cohort', 'cohort')):
            for k2, v in fl[stage].items():
                put(reg, f'flow.{short}.{k2}', v, flow_p)
        put(reg, 'flow.excl_no_label.events', fl['2_excluded_no_task_label'], flow_p)
        for q, v in fl['4_events_per_patient'].items():
            put(reg, f"flow.events_per_patient.{q.strip('%')}", v, flow_p)
        for q, v in fl['4_recordings_per_patient'].items():
            put(reg, f"flow.recordings_per_patient.{q.strip('%')}", v, flow_p)

    ov_p = RC / 'arm_L2_event_split' / 'overlap.json'
    if ov_p.exists():
        for part, d in json.loads(ov_p.read_text()).items():
            for k2, v in d.items():
                put(reg, f'l2.overlap.{part}.{k2}', v, ov_p)

    for arm, d in (('l0', 'arm_L0_clean'), ('l1', 'arm_L1_backbone_leak')):
        sp = RC / d / 'shap' / 'shap_report.json'
        if sp.exists():
            for target, r in json.loads(sp.read_text()).items():
                put(reg, f'shap.{arm}.{TASK[target]}.demographic_share',
                    r['demographic_share'], sp)
                for g, v in r['group_share'].items():
                    put(reg, f'shap.{arm}.{TASK[target]}.share.{slug(g)}', v, sp)
        bp = RC / d / 'meta_benchmark' / 'meta_benchmark.json'
        if bp.exists():
            for target, models in json.loads(bp.read_text()).items():
                aucs = [m['auc']['value'] for m in models.values()]
                put(reg, f'bench.{arm}.{TASK[target]}.auc_min', min(aucs), bp)
                put(reg, f'bench.{arm}.{TASK[target]}.auc_max', max(aucs), bp)
                put(reg, f'bench.{arm}.{TASK[target]}.n_learners', len(aucs), bp)

    sal_p = RC / 'arm_L0_clean' / 'saliency' / 'saliency_report.json'
    if sal_p.exists():
        for target, r in json.loads(sal_p.read_text())['tasks'].items():
            for k2, v in r.items():
                if isinstance(v, (int, float)):
                    put(reg, f'sal.{TASK[target]}.{k2}', v, sal_p)
            for c, prof in r['mel_band_profile_correct'].items():
                tot = sum(prof)
                put(reg, f'sal.{TASK[target]}.band_peak_share.{slug(c)}', max(prof) / tot, sal_p)
                put(reg, f'sal.{TASK[target]}.band_peak_index.{slug(c)}',
                    prof.index(max(prof)) + 1, sal_p)
                # bands 5-8 lie above the 1800 Hz band-pass cut-off (mel_band_edges.json)
                put(reg, f'sal.{TASK[target]}.above_passband_share.{slug(c)}',
                    sum(prof[4:]) / tot, sal_p)
        ed_p = sal_p.with_name('mel_band_edges.json')
        if ed_p.exists():
            ed = json.loads(ed_p.read_text())['patch_band_upper_edge_hz']
            put(reg, 'sal.band2_low_hz', ed[0], ed_p)
            put(reg, 'sal.band2_high_hz', ed[1], ed_p)
            put(reg, 'sal.band4_high_hz', ed[3], ed_p)

    l3_p = RC / 'preprint_reported.json'
    if l3_p.exists():
        for k2, v in json.loads(l3_p.read_text()).items():
            if not k2.startswith('_'):
                put(reg, f'l3.{k2}', v, l3_p)

    for arm, sp in (('ncv', RC / 'metrics' / 'nested_cv_subgroups' / 'subgroups.json'),
                    ('l0', RC / 'arm_L0_clean' / 'subgroups' / 'subgroups.json')):
        if sp.exists():
            for target, cols in json.loads(sp.read_text()).items():
                if target not in TASK:
                    continue
                for col, levels in cols.items():
                    aucs = [e['auc']['value'] for e in levels.values() if 'auc' in e]
                    if aucs:
                        put(reg, f'sub.{arm}.{TASK[target]}.{col}.auc_min', min(aucs), sp)
                        put(reg, f'sub.{arm}.{TASK[target]}.{col}.auc_max', max(aucs), sp)
                demo = [e['auc']['value'] for c in ('age_band', 'sex', 'site')
                        for e in cols.get(c, {}).values() if 'auc' in e]
                if demo:   # range across all three participant/recording characteristics
                    put(reg, f'sub.{arm}.{TASK[target]}.demographic.auc_min', min(demo), sp)
                    put(reg, f'sub.{arm}.{TASK[target]}.demographic.auc_max', max(demo), sp)
                sa = [e.get('sensitivity') for e in cols.get('age_band', {}).values()
                      if isinstance(e.get('sensitivity'), (int, float))]
                if sa:
                    put(reg, f'sub.{arm}.{TASK[target]}.age_band.sens_min', min(sa), sp)
                    put(reg, f'sub.{arm}.{TASK[target]}.age_band.sens_max', max(sa), sp)
                sens = [e.get('sensitivity') for e in cols.get('duration_tertile', {}).values()
                        if isinstance(e.get('sensitivity'), (int, float))]
                if sens:
                    put(reg, f'sub.{arm}.{TASK[target]}.duration_tertile.sens_min', min(sens), sp)
                    put(reg, f'sub.{arm}.{TASK[target]}.duration_tertile.sens_max', max(sens), sp)

    pr_p = RC / 'metrics' / 'patient_rules' / 'patient_rules.json'
    if pr_p.exists():
        for target, t in json.loads(pr_p.read_text())['tasks'].items():
            b = f'prule.{TASK[target]}'
            put(reg, f'{b}.n_children', t['n_children'], pr_p)
            for i, n in enumerate(t['children_by_derived_label']):
                put(reg, f'{b}.children.class{i}', n, pr_p)
            for rule, blk in t['rules'].items():
                add_metric_block(reg, f'{b}.{rule}', blk, pr_p)

    ab_p = RC / 'ablations' / 'ablations.json'
    if ab_p.exists():
        ab = json.loads(ab_p.read_text())
        put(reg, 'abl.n_test_events', ab['n_test_events'], ab_p)
        for target, t in ab['tasks'].items():
            b = f'abl.{TASK[target]}'
            for mode in ('reference', 'band', 'context'):
                if mode in t:
                    add_metric_block(reg, f'{b}.{mode}', t[mode], ab_p)
            for mode in ('band', 'context'):
                k = f'delta_auc_{mode}_minus_reference'
                if k in t:
                    put(reg, f'{b}.delta_auc.{mode}', t[k]['delta'], ab_p, t[k]['ci'])
                    put(reg, f'{b}.delta_auc.{mode}.p', t[k]['p'], ab_p)

    ce_p = RC / 'metrics' / 'common_events' / 'common_events.json'
    if ce_p.exists():
        ce = json.loads(ce_p.read_text())
        for k in ('n_events', 'n_patients', 'n_recordings', 'patients_also_in_L2_training',
                  'events_whose_recording_was_in_L2_training',
                  'patients_with_another_recording_in_L2_training'):
            put(reg, f'common.{k}', ce[k], ce_p)
        for target, t in ce['tasks'].items():
            b = f'common.{TASK[target]}'
            put(reg, f'{b}.majority_accuracy', t['majority_class_accuracy'], ce_p)
            for i, n in enumerate(t.get('patient_class_counts', [])):
                put(reg, f'{b}.children.class{i}', n, ce_p)
            for i, n in enumerate(t['class_counts']):
                put(reg, f'{b}.events.class{i}', n, ce_p)
            for arm in ('L0', 'L2'):
                add_metric_block(reg, f'{b}.{arm.lower()}', t[arm], ce_p)
            for m in ('auc', 'accuracy'):
                d = t[f'delta_{m}_L2_minus_L0']
                put(reg, f'{b}.delta_{m}', d['delta'], ce_p, d['ci'])
                put(reg, f'{b}.delta_{m}.p', d['p'], ce_p)

    conf_p = ROOT / 'data' / 'SPRSound_Event_Level_Dataset_CLEAN.diagnosis_conflicts.csv'
    if conf_p.exists():
        import csv
        with open(conf_p) as f:
            put(reg, 'data.participants_conflicting_diagnosis',
                len({row['pid'] for row in csv.DictReader(f)}), conf_p)

    au_p = RC / 'backbone_audit.json'
    if au_p.exists():
        au = json.loads(au_p.read_text())
        for k2 in ('n_tensors_common', 'n_tensors_changed', 'max_abs_diff'):
            if k2 in au:
                put(reg, f'audit.{k2}', au[k2], au_p)

    split = RC / 'split_summary.json'
    for part, d in json.loads(split.read_text()).items():
        for k, v in d.items():
            put(reg, f'cohort.{part}.{k}', v, split)
    for k in ('events', 'patients'):
        put(reg, f'cohort.total.{k}', sum(reg[f'cohort.{p}.{k}']['value']
                                          for p in ('train', 'val', 'test')), split)

    out = RC / 'manuscript_numbers.json'
    out.write_text(json.dumps({'_git_commit': commit, 'numbers': reg}, indent=1))
    print(f"{len(reg)} numbers -> {out.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
