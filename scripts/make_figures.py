#!/usr/bin/env python3
"""
scripts/make_figures.py

Main-text figures, drawn only from result files (no number is typed here).
Output: PDF (vector) + 300 dpi LZW TIFF, 170 mm wide, no in-graphic titles (BMC keeps the
title and legend in the manuscript text).

Colour encodes the OUTCOME everywhere (palette validated for colour-vision deficiency with
the dataviz validator); arms and classes use marker shape / line style, never a new hue.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / 'results_clean'
MM = 1 / 25.4
W = 170 * MM

TASKS = ['model2_label', 'model1_label', 'model3_label']
NAME = {'model2_label': 'Screening', 'model1_label': 'Sound pattern',
        'model3_label': 'Disease group'}
COL = {'model2_label': '#0072B2', 'model1_label': '#D55E00', 'model3_label': '#009E73'}
CLASS_LABEL = {'Abnormal': 'Adventitious', 'Rhonchi': 'Wheeze/rhonchi',
               'Normal_Other': 'Normal/other', 'Bronchial': 'Bronchial disease'}
INK, MUTED, GRID = '#1a1a1a', '#6b6b6b', '#e4e4e4'
STYLES = ['-', '--', ':']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.5, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'pdf.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False,
})


def load(path):
    return json.loads((RC / path).read_text())


def save(fig, out_dir, name):
    fig.savefig(out_dir / f'{name}.pdf')
    fig.savefig(out_dir / f'{name}.tiff', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(out_dir / f'{name}.png', dpi=150)
    plt.close(fig)
    print(f"  {name}: {(out_dir / (name + '.tiff')).stat().st_size / 1e6:.1f} MB tiff")


def box(ax, xy, wh, text, fc='#f4f4f2', ec=MUTED, fs=6.5, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch(xy, *wh, boxstyle='round,pad=0.006,rounding_size=0.012',
                                fc=fc, ec=ec, lw=0.6))
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=tc, linespacing=1.25)


def arrow(ax, a, b, color=MUTED):
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', lw=0.6, color=color,
                                                    shrinkA=0, shrinkB=0, mutation_scale=6))


# --------------------------------------------------------------------------- Fig. 1
def fig1(out):
    fig, ax = plt.subplots(figsize=(W, 92 * MM))
    fig.subplots_adjust(0.005, 0.01, 0.995, 0.99)
    ax.set_xlim(-0.035, 1.01); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(-0.03, 0.985, 'Every partition is drawn on patients; an assertion verifies it before '
            'each fit and each evaluation', fontsize=7.2, fontweight='bold', va='top')
    y, h = 0.54, 0.22
    steps = [(0.000, 0.135, 'Annotated event\n(onset, offset,\ntype)'),
             (0.165, 0.155, '2 s clip: centred,\ntapered, band-pass\n100–1800 Hz,\nnoise-padded'),
             (0.350, 0.155, 'HeAR ViT-L encoder\n(frozen) + LoRA\nin last 6 blocks'),
             (0.535, 0.135, 'Pooled + patch\nembedding\n(1536-D)')]
    for x, w, t in steps:
        box(ax, (x, y), (w, h), t, fs=6.3)
    for (x, w, _), (x2, _, _) in zip(steps, steps[1:]):
        arrow(ax, (x + w + 0.004, y + h / 2), (x2 - 0.004, y + h / 2))
    heads = [('model2_label', 0.765, 'Screening\nnormal / adventitious'),
             ('model1_label', 0.585, 'Sound pattern\n3 classes'),
             ('model3_label', 0.405, 'Disease group\n3 classes')]
    for t, yy, lab in heads:
        box(ax, (0.705, yy), (0.135, 0.13), lab, fc='white', ec=COL[t], fs=6.3)
        arrow(ax, (0.674, y + h / 2), (0.701, yy + 0.065))
        arrow(ax, (0.844, yy + 0.065), (0.871, 0.65))
    box(ax, (0.875, 0.53), (0.125, 0.24), 'Second stage\nLightGBM:\n8 probabilities\n+ age, sex, site',
        fs=6.3)
    box(ax, (0.875, 0.24), (0.125, 0.19), 'Aggregation\nevent → recording\n→ patient', fs=6.3)
    arrow(ax, (0.9375, 0.526), (0.9375, 0.434))
    red = '#b3261e'
    leaks = [(0.0675, 0.49, 'Leak point a', 'Partitioning of events\n(arm L2: event-level split)'),
             (0.4275, 0.49, 'Leak point c', 'Provenance of encoder weights\n(arm L1: checkpoint '
                                            'fine-tuned on\nan event-level split)'),
             (0.7725, 0.355, 'Leak point b', 'Folds producing out-of-fold\nprobabilities for the '
                                             'second stage')]
    for x, yy, tag, txt in leaks:
        ax.plot([x], [yy], marker='^', ms=6, color=red, clip_on=False)
        ax.text(x, yy - 0.045, tag, ha='center', va='top', fontsize=6.5, fontweight='bold',
                color=red)
        ax.text(x, yy - 0.095, txt, ha='center', va='top', fontsize=6, linespacing=1.25)
    ax.text(-0.03, 0.115, 'Comparators sharing the second-stage protocol', fontsize=6.5,
            fontweight='bold', va='top')
    ax.text(-0.03, 0.065, 'majority class  ·  event duration only  ·  demographics only  ·  own-task '
            'base model  ·  all acoustic probabilities  ·  full stack', fontsize=6.3,
            color=MUTED, va='top')
    save(fig, out, 'Fig1')


# --------------------------------------------------------------------------- Fig. 2
def fig2(out):
    fl = load('cohort_flow.json')
    sp = load('split_summary.json')
    n = load('metrics/nested_cv/metrics.json')
    fold = np.bincount(np.loadtxt(RC / 'nested_cv' / 'patient_fold_assignment.csv',
                                  delimiter=',', skiprows=1, usecols=1, dtype=int))[1:]

    def cnt(d):
        return (f"{d['events']} events · {d['recordings']} recordings · "
                f"{d['patients_with_id']} participants")
    fig, ax = plt.subplots(figsize=(W, 92 * MM))
    fig.subplots_adjust(0.005, 0.01, 0.995, 0.99)
    ax.set_xlim(0.03, 1.0); ax.set_ylim(0.14, 1.0); ax.axis('off')
    main = [(0.86, 'Annotated events with audio\n' + cnt(fl['1_annotated_events_with_audio'])),
            (0.66, 'Events with a task label\n' + cnt(fl['2_after_label_mapping'])),
            (0.46, 'Analysis cohort\n' + cnt(fl['4_analysis_cohort']))]
    for yy, t in main:
        box(ax, (0.05, yy), (0.50, 0.11), t, bold=(yy == 0.46))
    arrow(ax, (0.30, 0.858), (0.30, 0.772)); arrow(ax, (0.30, 0.658), (0.30, 0.572))
    ex = [(0.785, f"Excluded: no usable task label (stridor, 'no event')\n"
                  f"{fl['2_excluded_no_task_label']} events"),
          (0.60, f"Excluded: diagnosis not documented\n"
                 f"{fl['3_excluded_undocumented_diagnosis']['events']} events, incl. all "
                 f"{fl['3_excluded_undocumented_diagnosis']['events_without_patient_id']} events "
                 f"without participant id"),
          (0.535, f"Excluded: implausible encoded age (>18 y)\n"
                  f"{fl['3b_excluded_implausible_age']['events']} events of one child")]
    for yy, t in ex:
        ax.text(0.61, yy + 0.015, t, fontsize=6.3, va='center', color=INK, linespacing=1.25)
        arrow(ax, (0.30, yy + 0.015), (0.60, yy + 0.015))
    ax.text(0.05, 0.40, 'Secondary analysis: locked hold-out', fontsize=6.8, fontweight='bold',
            va='center')
    for i, (k, lab) in enumerate([('train', 'Training'), ('val', 'Validation'), ('test', 'Test')]):
        box(ax, (0.05 + i * 0.17, 0.27), (0.16, 0.10),
            f"{lab}\n{sp[k]['patients']} participants\n{sp[k]['events']} events", fc='white')
    arrow(ax, (0.20, 0.458), (0.20, 0.375))
    ax.text(0.595, 0.40, f"Primary analysis: nested cross-validation\n({n['n_patients']} "
            f"participants, each tested once)", fontsize=6.8, fontweight='bold', va='center',
            linespacing=1.3)
    for i, c in enumerate(fold):
        box(ax, (0.595 + i * 0.08, 0.255), (0.072, 0.10), f"Fold {i + 1}\n{c}", fc='white', fs=6)
    arrow(ax, (0.42, 0.458), (0.70, 0.36))
    ax.text(0.595, 0.225, 'Each outer fold: remaining participants → validation\n(10% of cohort) + '
            'training with four-fold\nparticipant-grouped out-of-fold stacking', fontsize=6,
            color=MUTED, va='top', linespacing=1.3)
    save(fig, out, 'Fig2')


# --------------------------------------------------------------------------- Fig. 3
def fig3(out):
    n = load('metrics/nested_cv/metrics.json')['tasks']
    fig, axes = plt.subplots(3, 3, figsize=(W, 150 * MM), constrained_layout=True)
    for j, t in enumerate(TASKS):
        cur = n[t]['curves']
        classes = [c for c in cur if not c.startswith('_')]
        for k, c in enumerate(classes):
            d = cur[c]
            lab = CLASS_LABEL.get(c, c)
            axes[0, j].plot(d['fpr'], d['tpr'], STYLES[k % 3], color=COL[t], lw=1.2, label=lab)
            axes[1, j].plot(d['recall'], d['precision'], STYLES[k % 3], color=COL[t], lw=1.2,
                            label=lab)
            axes[1, j].axhline(d['prevalence'], color=MUTED, lw=0.5, ls=STYLES[k % 3])
        axes[0, j].plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=0)
        rel = cur['_reliability']
        axes[2, j].plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=0)
        axes[2, j].plot([r['conf'] for r in rel], [r['acc'] for r in rel], '-o', color=COL[t],
                        lw=1.2, ms=3.5, mec='white', mew=0.6)
        auc = n[t]['ladder']['full_stack']['auc']
        axes[0, j].set_title(f"{NAME[t]}\nAUC {auc['value']:.3f} ({auc['ci'][0]:.3f}–"
                             f"{auc['ci'][1]:.3f})", loc='left')
        axes[0, j].set_xlabel('1 − specificity'); axes[1, j].set_xlabel('Sensitivity (recall)')
        axes[2, j].set_xlabel('Mean predicted confidence')
        axes[0, j].legend(frameon=False, handlelength=2.2,
                          loc='upper left' if t == 'model3_label' else 'lower right')
        for i in range(3):
            axes[i, j].set_xlim(0, 1); axes[i, j].set_ylim(0, 1.005)
            axes[i, j].set_aspect('equal'); axes[i, j].grid(color=GRID, lw=0.4)
    for i, (lab, tag) in enumerate([('Sensitivity', 'a'), ('Positive predictive value', 'b'),
                                    ('Observed accuracy', 'c')]):
        axes[i, 0].set_ylabel(lab)
        axes[i, 0].text(-0.32, 1.06, tag, transform=axes[i, 0].transAxes, fontsize=10,
                        fontweight='bold')
    save(fig, out, 'Fig3')


# --------------------------------------------------------------------------- Fig. 4
def fig4(out):
    arms = {'L0': load('arm_L0_clean/metrics/metrics.json')['tasks'],
            'L1': load('arm_L1_backbone_leak/metrics/metrics.json')['tasks'],
            'L2': load('arm_L2_event_split/metrics/metrics.json')['tasks']}
    l3 = load('preprint_reported.json')
    short = {'model2_label': 'screen', 'model1_label': 'pattern', 'model3_label': 'disease'}
    mk = {'L0': 'o', 'L1': 's', 'L2': '^'}
    lab = {'L0': 'L0  released encoder, patient-level split',
           'L1': 'L1  fine-tuned encoder, patient-level split',
           'L2': 'L2  released encoder, event-level split'}
    fig, axes = plt.subplots(1, 3, figsize=(W, 62 * MM), sharey=True, constrained_layout=True)
    for ax, t in zip(axes, TASKS):
        for gi, (lvl, key) in enumerate([('Event level', None), ('Patient level', 'patient')]):
            for ai, a in enumerate(arms):
                blk = (arms[a][t]['ladder']['full_stack'] if key is None
                       else arms[a][t]['aggregation']['patient|confweighted'])['auc']
                y = gi * 4 + ai
                ax.plot(blk['ci'], [y, y], color=COL[t], lw=1.0, solid_capstyle='round')
                ax.plot(blk['value'], y, mk[a], color=COL[t], ms=5, mec='white', mew=0.7,
                        label=lab[a] if (gi == 0 and t == TASKS[0]) else None)
            v = l3.get(f"{short[t]}.{'event' if key is None else 'patient'}.auc")
            if v is not None:
                ax.plot(v, gi * 4 + 3, 'D', mfc='white', mec=COL[t], ms=4.5, mew=0.9, ls='',
                        label='L3  earlier preprint (uncontrolled)' if (gi == 0 and t == TASKS[0])
                        else None)
        ax.axvline(0.5, color=MUTED, lw=0.5, ls=':')
        ax.set_xlim(0.45, 1.0); ax.set_title(NAME[t], loc='left'); ax.set_xlabel('AUC')
        ax.grid(axis='x', color=GRID, lw=0.4)
        ax.set_yticks([1.5, 5.5]); ax.set_yticklabels(['Event\nlevel', 'Patient\nlevel'])
        ax.set_ylim(7.6, -0.8); ax.tick_params(axis='y', length=0)
    h, l_ = axes[0].get_legend_handles_labels()
    for hh in h:
        hh.set_color(INK) if hasattr(hh, 'set_color') else None
    fig.legend(h, l_, loc='outside lower center', ncol=2, frameon=False)
    save(fig, out, 'Fig4')


# --------------------------------------------------------------------------- Fig. 5
def fig5(out):
    n = load('metrics/nested_cv/metrics.json')['tasks']
    rungs = [('duration_only', 'Event duration only'), ('demographics_only', 'Demographics only'),
             ('own_task_base', 'Own-task base model'),
             ('acoustic_only', 'All acoustic probabilities'), ('full_stack', 'Full stack')]
    fig, axes = plt.subplots(1, 3, figsize=(W, 55 * MM), sharey=True, constrained_layout=True)
    for ax, t in zip(axes, TASKS):
        for y, (r, _) in enumerate(rungs):
            a = n[t]['ladder'][r]['auc']
            ax.plot(a['ci'], [y, y], color=COL[t], lw=1.0, solid_capstyle='round')
            ax.plot(a['value'], y, 'o', color=COL[t], ms=5, mec='white', mew=0.7)
            ax.text(a['ci'][1] + 0.012, y, f"{a['value']:.2f}", va='center', fontsize=6.3,
                    color=INK)
        ax.axvline(0.5, color=MUTED, lw=0.5, ls=':')
        ax.set_xlim(0.45, 1.06); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_title(NAME[t], loc='left'); ax.set_xlabel('AUC (event level)')
        ax.grid(axis='x', color=GRID, lw=0.4); ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(rungs))); axes[0].set_yticklabels([l for _, l in rungs])
    axes[0].set_ylim(len(rungs) - 0.5, -0.5)
    save(fig, out, 'Fig5')


# --------------------------------------------------------------------------- Fig. 6
def fig6(out):
    import pandas as pd
    sal = RC / 'arm_L0_clean' / 'saliency'
    rep = json.loads((sal / 'saliency_report.json').read_text())['tasks']
    edges = [100] + json.loads((sal / 'mel_band_edges.json').read_text())['patch_band_upper_edge_hz']
    panels = [('model2_label', 0, 'Normal'), ('model2_label', 1, 'Adventitious'),
              ('model1_label', 1, 'Crackles'), ('model1_label', 2, 'Wheeze/rhonchi')]
    fig = plt.figure(figsize=(W, 112 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9])
    map_axes = []
    for j, (t, c, lab) in enumerate(panels):
        z = np.load(sal / f'saliency_maps_{t}.npz')
        m = np.clip(z['occlusion'][(z['true'] == c) & (z['pred'] == c)], 0, None).mean(0)
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow((m / m.max()).reshape(12, 8).T, origin='lower', aspect='auto',
                       cmap='Blues', extent=[0, 2, 0, 8], vmin=0, vmax=1)
        ax.axhline(4, color='#b3261e', lw=0.7, ls='--')
        ax.set_title(f"{NAME[t]}\n{lab}", loc='left', color=INK)
        map_axes.append(ax)
        ax.set_xlabel('Time in clip (s)')
        ax.set_yticks(range(9))
        ax.set_yticklabels([f'{e / 1000:.1f}'.rstrip('0').rstrip('.') for e in edges + [8000]]
                           if j == 0 else [])
        if j == 0:
            ax.set_ylabel('Mel patch band edge (kHz)')
            ax.text(-0.42, 1.08, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold')
    cb = fig.colorbar(im, ax=map_axes, fraction=0.025, pad=0.015)
    cb.set_label('Mean probability drop\n(relative to map maximum)', fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_linewidth(0.3)
    axb = fig.add_subplot(gs[1, :2])
    for t, c, lab, ls in [('model2_label', 'Normal', 'Screening: normal', '--'),
                          ('model2_label', 'Abnormal', 'Screening: adventitious', '-'),
                          ('model1_label', 'Crackles', 'Sound pattern: crackles', '-'),
                          ('model1_label', 'Wheeze/Rhonchi', 'Sound pattern: wheeze/rhonchi', ':')]:
        p = np.array(rep[t]['mel_band_profile_correct'][c]); p = p / p.sum()
        axb.plot(range(1, 9), p, ls, color=COL[t], lw=1.3, marker='o', ms=3.5, mec='white',
                 mew=0.6, label=lab)
    axb.axvline(4.5, color='#b3261e', lw=0.7, ls='--', label='band-pass cut-off (1.8 kHz)')
    axb.set_xlabel('Mel patch band (1 = lowest frequency)')
    axb.set_ylabel('Share of positive attribution')
    axb.set_ylim(0, None); axb.grid(color=GRID, lw=0.4); axb.legend(frameon=False)
    axb.text(-0.14, 1.05, 'b', transform=axb.transAxes, fontsize=10, fontweight='bold')
    axc = fig.add_subplot(gs[1, 2:])
    pos = 0
    ticks = []
    for t in ('model2_label', 'model1_label'):
        r = pd.read_csv(sal / f'saliency_records_{t}.csv')
        r = r[r['true'] == r['pred']]
        for col, mk, off in (('deletion_auc_attr', 'o', -0.17), ('deletion_auc_rand', 's', 0.17)):
            q = r[col].quantile([0.25, 0.5, 0.75]).values
            axc.plot([pos + off, pos + off], [q[0], q[2]], color=COL[t], lw=1.2,
                     solid_capstyle='round')
            axc.plot(pos + off, q[1], mk, ms=5.5, mew=0.9, mec=COL[t],
                     mfc=COL[t] if mk == 'o' else 'white')
        ticks.append((pos, NAME[t])); pos += 1
    axc.plot([], [], 'o', color=INK, ms=5, label='attribution-ordered removal')
    axc.plot([], [], 's', mfc='white', mec=INK, ms=5, label='random removal')
    axc.set_xticks([p for p, _ in ticks]); axc.set_xticklabels([l for _, l in ticks])
    axc.set_xlim(-0.6, 1.6); axc.set_ylim(0, 1)
    axc.set_ylabel('Area under deletion curve\n(median, IQR; lower = more faithful)')
    axc.grid(axis='y', color=GRID, lw=0.4); axc.legend(frameon=False, loc='lower right')
    axc.text(-0.2, 1.05, 'c', transform=axc.transAxes, fontsize=10, fontweight='bold')
    save(fig, out, 'Fig6')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in [('1', fig1), ('2', fig2), ('3', fig3), ('4', fig4), ('5', fig5), ('6', fig6)]:
        if not a.only or name in a.only:
            fn(out)
