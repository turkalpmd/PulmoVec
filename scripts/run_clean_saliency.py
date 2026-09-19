#!/usr/bin/env python3
"""
scripts/run_clean_saliency.py

MODEL-DERIVED attribution for the HeAR+LoRA base models (replaces the illustrative
figures of visualize_sound_saliency.py, which never queried a model).

Attribution is computed on the 192 x 128 mel-PCEN input the encoder actually sees:

  occlusion   each of the 96 ViT patches (12 time x 8 mel) is replaced by the clip's median
              PCEN value; attribution = drop in the probability of the evaluated class
              (primary method - model-agnostic, no gradient pathologies under LoRA).
  int. grad.  integrated gradients, 32 steps, same baseline, pooled to the patch grid.
  attention   attention-pooling weights [4, 96] - descriptive only: the CLS branch of the
              classifier bypasses the pooler, so these are not a faithful explanation.

Faithfulness checks written next to the maps:
  * deletion curves (most-attributed patches removed first) against random order;
  * classifier-head randomisation test (Adebayo et al.) - maps must change;
  * share of positive attribution inside the annotated event interval. NOTE: clips centre
    the event and pad with low-level synthetic noise, so temporal localisation is partly
    guaranteed by construction - the mel-band profile and deletion curves are the
    informative checks;
  * mel-band occlusion profile per class.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'hear'))

from sprsound_dataset import SPRSoundDatasetFromDF  # noqa: E402
from models_hear_lora import HeARLoRAClassifier  # noqa: E402
from python.data_processing import audio_utils  # noqa: E402
from repro import seed_everything, sha256_file  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T_PATCH, F_PATCH, PATCH = 12, 8, 16          # 192/16 time x 128/16 mel
TASKS = {'model1_label': ['Normal', 'Crackles', 'Wheeze/Rhonchi'],
         'model2_label': ['Normal', 'Abnormal']}


def spec_of(audio):
    return audio_utils.preprocess_audio(audio).to(audio.device).float()      # [B,1,192,128]


def forward_spec(model, spec):
    out = model.hear_encoder(spec, return_dict=True)
    attended, attn = model.patch_pooler(out.last_hidden_state[:, 1:, :])
    logits = model.classifier(torch.cat([out.pooler_output, attended], dim=-1))
    return logits, attn


def check_time_axis():
    """Axis 2 (192) must be time: a late-only burst may only change late rows."""
    a = torch.zeros(1, 32000)
    b = a.clone()
    b[0, 24000:28000] = torch.randn(4000) * 0.3
    d = (spec_of(b) - spec_of(a)).abs()[0, 0]
    rows = d.mean(dim=1)
    assert rows[120:].sum() > 20 * rows[:96].sum() + 1e-9, "axis 2 of the PCEN input is not time"


def patch_mask(idx):
    """[len(idx),1,192,128] boolean masks, token order = row-major over (time, mel)."""
    m = torch.zeros(len(idx), 1, T_PATCH * PATCH, F_PATCH * PATCH, dtype=torch.bool)
    for n, i in enumerate(idx):
        t, f = divmod(int(i), F_PATCH)
        m[n, 0, t * PATCH:(t + 1) * PATCH, f * PATCH:(f + 1) * PATCH] = True
    return m


ALL_MASKS = None


@torch.no_grad()
def occlusion(model, spec, cls):
    """spec [1,1,192,128] -> [96] probability drop for class `cls`."""
    base = spec.median()
    p0 = torch.softmax(forward_spec(model, spec)[0], 1)[0, cls]
    occluded = torch.where(ALL_MASKS, base, spec.expand(96, -1, -1, -1))
    p = torch.cat([torch.softmax(forward_spec(model, occluded[i:i + 48])[0], 1)[:, cls]
                   for i in (0, 48)])
    return (p0 - p).cpu().numpy(), float(p0)


def integrated_gradients(model, spec, cls, steps=32):
    base = torch.full_like(spec, spec.median().item())
    total = torch.zeros_like(spec)
    for chunk in torch.linspace(0, 1, steps, device=spec.device).split(8):
        x = (base + chunk.view(-1, 1, 1, 1) * (spec - base)).requires_grad_(True)
        prob = torch.softmax(forward_spec(model, x)[0], 1)[:, cls].sum()
        total += torch.autograd.grad(prob, x)[0].sum(0, keepdim=True)
    ig = ((spec - base) * total / steps)[0, 0]
    return ig.reshape(T_PATCH, PATCH, F_PATCH, PATCH).sum(dim=(1, 3)).flatten().cpu().numpy()


@torch.no_grad()
def deletion_curve(model, spec, cls, order, steps=(0, 4, 8, 16, 24, 32, 48, 64, 96)):
    base = spec.median()
    xs = []
    for k in steps:
        m = patch_mask(order[:k]).any(dim=0, keepdim=True).to(spec.device) if k else \
            torch.zeros_like(spec, dtype=torch.bool)
        xs.append(torch.where(m, base, spec))
    p = torch.softmax(forward_spec(model, torch.cat(xs))[0], 1)[:, cls]
    return p.cpu().numpy()


def event_time_patches(duration_ms):
    """Time-patch indices (0..11) covered by the centred event in an eval clip."""
    dur = min(duration_ms / 1000.0, 2.0)
    lo, hi = 1.0 - dur / 2, 1.0 + dur / 2
    edges = np.linspace(0, 2.0, T_PATCH + 1)
    return [t for t in range(T_PATCH) if edges[t + 1] > lo and edges[t] < hi]


def main():
    global ALL_MASKS
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='directory with checkpoints/ and run_info.json')
    ap.add_argument('--split-csv', default=str(ROOT / 'results_clean' / 'split_test.csv'))
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--per-class', type=int, default=150)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_time_axis()
    ALL_MASKS = patch_mask(range(96)).to(DEVICE)
    test = pd.read_csv(args.split_csv)
    rng = np.random.default_rng(args.seed)
    report = {'backbone_sha256': sha256_file(args.backbone), 'tasks': {}}

    for task, names in TASKS.items():
        model = HeARLoRAClassifier(num_classes=len(names), weights_path=args.backbone).to(DEVICE)
        ckpt = torch.load(Path(args.run_dir) / 'checkpoints' / f'{task}_full.pth',
                          map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # fixed-rule sample: up to N events per true class, seeded, no hand-picking
        pick = np.concatenate([rng.permutation(np.where(test[task].values == c)[0])[:args.per_class]
                               for c in range(len(names))])
        ds = SPRSoundDatasetFromDF(test.iloc[pick].reset_index(drop=True), label_column=task,
                                   is_training=False)
        rand_head = HeARLoRAClassifier(num_classes=len(names), weights_path=args.backbone).to(DEVICE)
        rand_head.load_state_dict(ckpt['model_state_dict'])
        for layer in rand_head.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        rand_head.eval()

        recs, occ_maps, ig_maps, attn_maps = [], [], [], []
        for i in range(len(ds)):
            audio, y, meta = ds[i]
            spec = spec_of(audio.unsqueeze(0).to(DEVICE))
            with torch.no_grad():
                logits, attn = forward_spec(model, spec)
            pred = int(logits.argmax(1))
            occ, p0 = occlusion(model, spec, y)
            ig = integrated_gradients(model, spec, y)
            occ_r, _ = occlusion(rand_head, spec, y)
            order = np.argsort(-occ)
            d_attr = deletion_curve(model, spec, y, order)
            d_rand = deletion_curve(model, spec, y, rng.permutation(96))
            inside = event_time_patches(meta['event_duration_ms'])
            pos = np.clip(occ, 0, None).reshape(T_PATCH, F_PATCH)
            recs.append({
                'row': int(pick[i]), 'true': y, 'pred': pred, 'p_true': p0,
                'duration_ms': meta['event_duration_ms'],
                'mass_in_event': float(pos[inside].sum() / pos.sum()) if pos.sum() > 0 else np.nan,
                'event_time_share': len(inside) / T_PATCH,
                'deletion_auc_attr': float(np.trapz(d_attr, dx=1) / (len(d_attr) - 1)),
                'deletion_auc_rand': float(np.trapz(d_rand, dx=1) / (len(d_rand) - 1)),
                'occ_ig_spearman': float(pd.Series(occ).corr(pd.Series(ig), method='spearman')),
                'occ_randhead_spearman': float(pd.Series(occ).corr(pd.Series(occ_r),
                                                                   method='spearman')),
            })
            occ_maps.append(occ)
            ig_maps.append(ig)
            attn_maps.append(attn[0].mean(0).cpu().numpy())
            if (i + 1) % 50 == 0:
                print(f"  {task}: {i + 1}/{len(ds)}", flush=True)

        r = pd.DataFrame(recs)
        r.to_csv(out_dir / f'saliency_records_{task}.csv', index=False)
        np.savez_compressed(out_dir / f'saliency_maps_{task}.npz', occlusion=np.array(occ_maps),
                            ig=np.array(ig_maps), attention=np.array(attn_maps),
                            true=r['true'].values, pred=r['pred'].values)
        ok = r[r['true'] == r['pred']]
        occ_arr = np.array(occ_maps)
        report['tasks'][task] = {
            'classes': names, 'n_events': int(len(r)), 'n_correct': int(len(ok)),
            'deletion_auc_attr_mean': float(ok['deletion_auc_attr'].mean()),
            'deletion_auc_rand_mean': float(ok['deletion_auc_rand'].mean()),
            'deletion_gap_mean': float((ok['deletion_auc_rand'] - ok['deletion_auc_attr']).mean()),
            'mass_in_event_mean': float(ok['mass_in_event'].mean()),
            'event_time_share_mean': float(ok['event_time_share'].mean()),
            'occ_ig_spearman_median': float(ok['occ_ig_spearman'].median()),
            'occ_randhead_spearman_median': float(ok['occ_randhead_spearman'].median()),
            'mel_band_profile_correct': {
                names[c]: np.clip(occ_arr[(r['true'] == c) & (r['pred'] == c)], 0, None)
                .reshape(-1, T_PATCH, F_PATCH).sum(1).mean(0).tolist()
                for c in range(len(names)) if ((r['true'] == c) & (r['pred'] == c)).any()},
        }
        (out_dir / 'saliency_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report['tasks'][task].items()
                          if k != 'mel_band_profile_correct'}, indent=2), flush=True)
        del model, rand_head
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
