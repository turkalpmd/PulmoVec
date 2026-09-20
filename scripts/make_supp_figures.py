#!/usr/bin/env python3
"""
scripts/make_supp_figures.py

Supplementary attribution figures (Additional file 5), from the saved saliency outputs of
the hold-out arm: class-mean maps of three methods side by side, and the per-event
faithfulness statistics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from make_figures import RC, W, MM, COL, NAME, INK, MUTED, GRID  # noqa: F401  (sets rcParams)

SAL = RC / 'arm_L0_clean' / 'saliency'
PANELS = [('model2_label', 0, 'Normal'), ('model2_label', 1, 'Adventitious'),
          ('model1_label', 1, 'Crackles'), ('model1_label', 2, 'Wheeze/rhonchi')]
METHODS = [('occlusion', 'Occlusion'), ('ig', 'Integrated gradients'),
           ('attention', 'Attention pooling\n(descriptive only)')]


def figs1(out):
    fig, axes = plt.subplots(3, 4, figsize=(W, 125 * MM), constrained_layout=True)
    for j, (t, c, lab) in enumerate(PANELS):
        z = np.load(SAL / f'saliency_maps_{t}.npz')
        ok = (z['true'] == c) & (z['pred'] == c)
        for i, (key, mlab) in enumerate(METHODS):
            m = np.clip(z[key][ok], 0, None).mean(0)
            ax = axes[i, j]
            im = ax.imshow((m / m.max()).reshape(12, 8).T, origin='lower', aspect='auto',
                           cmap='Blues', extent=[0, 2, 0.5, 8.5], vmin=0, vmax=1)
            ax.axhline(4.5, color='#b3261e', lw=0.7, ls='--')
            if i == 0:
                ax.set_title(f"{NAME[t]}\n{lab} (n = {int(ok.sum())})", loc='left')
            if j == 0:
                ax.set_ylabel(f'{mlab}\nmel patch band')
            if i == 2:
                ax.set_xlabel('Time in clip (s)')
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label('Mean positive attribution (relative to map maximum)', fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.savefig(out / 'FigS1_attribution_methods.png', dpi=300)
    plt.close(fig)


def figs2(out):
    fig, axes = plt.subplots(1, 3, figsize=(W, 55 * MM), constrained_layout=True)
    for t in ('model2_label', 'model1_label'):
        r = pd.read_csv(SAL / f'saliency_records_{t}.csv')
        r = r[r['true'] == r['pred']]
        axes[0].scatter(r['event_time_share'], r['mass_in_event'], s=5, color=COL[t], alpha=0.35,
                        lw=0, label=NAME[t])
        axes[1].hist(r['occ_randhead_spearman'].dropna(), bins=25, histtype='step', color=COL[t],
                     lw=1.1, label=NAME[t])
        axes[2].hist(r['occ_ig_spearman'].dropna(), bins=25, histtype='step', color=COL[t],
                     lw=1.1, label=NAME[t])
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=':')
    axes[0].set_xlabel('Share of the clip occupied by the event')
    axes[0].set_ylabel('Share of positive attribution\ninside the event')
    axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.02)
    axes[1].set_xlabel('Spearman correlation: trained vs\nre-initialised classification head')
    axes[2].set_xlabel('Spearman correlation: occlusion vs\nintegrated gradients')
    for ax, tag in zip(axes, 'abc'):
        ax.grid(color=GRID, lw=0.4)
        ax.text(-0.18, 1.04, tag, transform=ax.transAxes, fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Events'); axes[0].legend(frameon=False, loc='lower right')
    fig.savefig(out / 'FigS2_faithfulness.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    out = Path(ap.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs1(out); figs2(out)
    print('supplementary figures ->', out)
