#!/usr/bin/env python3
"""
scripts/run_clean_ablations.py

Two controls for cues that are properties of the preprocessing rather than of the breath
sound. Both retrain a base model on the locked patient-level split and score the same test
partition, so the only difference from the reference model is the stated manipulation.

  band     the mel bands above the band-pass cut-off of our own filter are replaced by the
           clip's median value before the encoder sees them. Attribution analysis showed
           that most of the attribution for NORMAL events lies in that region, where only
           attenuated residual energy remains; this asks how much of the performance
           depends on it.

  context  the event is not isolated. A 2 s window centred on the midpoint of the event is
           taken from the recording as it is, so the margins contain real neighbouring
           audio instead of synthetic noise. In the reference pipeline the padded fraction
           is a monotone function of the annotated event duration, which the encoder could
           read off the clip geometry; this removes that cue (at the cost of letting
           neighbouring events into the clip).

Reference values come from the arm-L0 base models, re-scored here so that every number in
the comparison is produced by the same code path.
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'hear'))

import run_clean_pipeline as rcp  # noqa: E402
from run_clean_pipeline import TASKS, KEY, train_model, log  # noqa: E402
from sprsound_dataset import (SPRSoundDatasetFromDF, extract_isolated_centered_clip,  # noqa: E402
                              apply_age_exclusion)
from models_hear_lora import HeARLoRAClassifier  # noqa: E402
from python.data_processing import audio_utils  # noqa: E402
from repro import seed_everything, sha256_file, env_snapshot  # noqa: E402
import clean_stats as cs  # noqa: E402
import config  # noqa: E402
import librosa  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MEL_BANDS, BAND_ROWS = 8, 16          # the encoder input is 12 x 8 patches of 16 x 16
CUTOFF_BAND = 4                         # bands 5-8 lie above the 1800 Hz filter cut-off


# ----------------------------------------------------------------- band-masked encoder
class BandMasked(nn.Module):
    """Replace the mel bands above the filter cut-off with the clip's median value."""

    def __init__(self, base, keep_bands=CUTOFF_BAND):
        super().__init__()
        self.base = base
        self.keep = keep_bands * BAND_ROWS

    def forward(self, audio):
        spec = audio_utils.preprocess_audio(audio).to(audio.device)
        med = spec.median()
        spec = torch.cat([spec[:, :, :, :self.keep],
                          torch.full_like(spec[:, :, :, self.keep:], med)], dim=3)
        out = self.base.hear_encoder(spec, return_dict=True)
        attended, attn = self.base.patch_pooler(out.last_hidden_state[:, 1:, :])
        logits = self.base.classifier(torch.cat([out.pooler_output, attended], dim=-1))
        return logits, attn

    def get_class_weights(self, *a, **k):      # pragma: no cover - passthrough
        return self.base.get_class_weights(*a, **k)


# ------------------------------------------------------------------ real-context clips
class ContextDataset(SPRSoundDatasetFromDF):
    """2 s window centred on the event, taken from the recording without isolation."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(self._resolve_wav_path(str(row['wav_path'])),
                                sr=config.SAMPLE_RATE, mono=True)
        n = int(config.CLIP_DURATION * config.SAMPLE_RATE)
        mid = int((float(row['event_start_ms']) + float(row['event_end_ms'])) / 2
                  * config.SAMPLE_RATE / 1000.0)
        start = mid - n // 2
        if self.is_training and self.rng is not None:
            start += int(self.rng.integers(-3200, 3201))       # same +-200 ms jitter budget
        clip = np.zeros(n, dtype=np.float32)
        lo, hi = max(0, start), min(len(audio), start + n)
        if hi > lo:
            clip[lo - start:hi - start] = audio[lo:hi]
        pk = float(np.max(np.abs(clip)))
        if pk > 1e-5:
            clip = np.clip(clip / pk * 0.90, -0.99, 0.99)
        return (torch.from_numpy(clip).float(), int(row['label']),
                {'filename': row['filename'], 'event_index': int(row['event_index'])})


def make_loader_factory(mode):
    def make_loader(df, label_col, training, batch_size, workers):
        cls = ContextDataset if mode == 'context' else SPRSoundDatasetFromDF
        ds = cls(df, label_column=label_col, is_training=training)
        return ds, torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=training, num_workers=workers,
            pin_memory=True, prefetch_factor=2)
    return make_loader


@torch.no_grad()
def predict(model, loader, n_classes):
    model.eval()
    out = []
    for audio, _, _ in loader:
        audio = audio.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(audio)
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=['reference', 'band', 'context'])
    ap.add_argument('--tasks', nargs='+', default=['model2_label', 'model1_label'])
    ap.add_argument('--split-dir', default=str(ROOT / 'results_clean'))
    ap.add_argument('--out-dir', default=str(ROOT / 'results_clean' / 'ablations'))
    ap.add_argument('--backbone', default=str(ROOT / 'models' / 'hear_pristine_encoder.pth'))
    ap.add_argument('--reference-ckpt-dir',
                    default=str(ROOT / 'results_clean' / 'arm_L0_clean' / 'checkpoints'))
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rcp.configure(out, args.backbone, args.seed)
    seed_everything(args.seed)

    split = {n: apply_age_exclusion(pd.read_csv(Path(args.split_dir) / f'split_{n}.csv'))
             for n in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not set(split[a]['_group_key']) & set(split[b]['_group_key'])

    report_path = out / 'ablations.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'definition': {'band': f'mel bands {CUTOFF_BAND + 1}-{N_MEL_BANDS} (above the '
                               f'1800 Hz filter cut-off) replaced by the clip median',
                       'context': '2 s window centred on the event, real surrounding audio, '
                                  'no isolation and no synthetic padding',
                       'reference': 'the pipeline as reported'},
        'n_test_events': int(len(split['test'])),
        'n_test_patients': int(split['test']['_group_key'].nunique()),
        'env': env_snapshot(), 'backbone_sha256': sha256_file(args.backbone), 'tasks': {}}

    groups = split['test']['_group_key'].values
    for task in args.tasks:
        n_classes = TASKS[task]['n']
        y = split['test'][task].values.astype(int)
        report['tasks'].setdefault(task, {})
        for mode in args.modes:
            if mode in report['tasks'][task]:
                log(f"{task} / {mode}: already done, skipping")
                continue
            t0 = time.time()
            rcp.make_loader = make_loader_factory(mode)
            if mode == 'reference':
                model = HeARLoRAClassifier(num_classes=n_classes, lora_r=8, lora_alpha=16.0,
                                           lora_layers=6, dropout=0.3,
                                           weights_path=args.backbone).to(DEVICE)
                ckpt = torch.load(Path(args.reference_ckpt_dir) / f'{task}_full.pth',
                                  map_location=DEVICE)
                model.load_state_dict(ckpt['model_state_dict'])
                info = {'source': 'arm L0 checkpoint', 'best_epoch': ckpt.get('epoch')}
            else:
                log(f"{task} / {mode}: training")
                if mode == 'band':
                    original = HeARLoRAClassifier

                    def patched(*a, **k):
                        return BandMasked(original(*a, **k))
                    rcp.HeARLoRAClassifier = patched
                model, info = train_model(split['train'], split['val'], task, n_classes,
                                          args.epochs, args.patience, args.lr,
                                          args.batch_size, args.workers,
                                          tag=f'{task}_{mode}')
                if mode == 'band':
                    rcp.HeARLoRAClassifier = original
            _, loader = rcp.make_loader(split['test'], task, False, args.batch_size,
                                        args.workers)
            proba = predict(model, loader, n_classes)
            np.save(out / f'test_proba_{task}_{mode}.npy', proba)
            point = cs.overall_metrics(y, proba)
            ci, _ = cs.cluster_bootstrap(lambda r, p=proba: cs.overall_metrics(y[r], p[r]),
                                         groups, args.boot)
            report['tasks'][task][mode] = {
                **{k: {'value': float(v), 'ci': ci.get(k)} for k, v in point.items()},
                'training': info, 'minutes': round((time.time() - t0) / 60, 1)}
            log(f"{task} / {mode}: AUC {point['auc']:.4f} "
                f"({ci['auc'][0]:.4f}-{ci['auc'][1]:.4f})  acc {point['accuracy']:.4f}")
            report_path.write_text(json.dumps(report, indent=2))
            del model
            torch.cuda.empty_cache()

        ref = out / f'test_proba_{task}_reference.npy'
        if ref.exists():
            pr = np.load(ref)
            for mode in ('band', 'context'):
                f = out / f'test_proba_{task}_{mode}.npy'
                if f.exists():
                    report['tasks'][task][f'delta_auc_{mode}_minus_reference'] = \
                        cs.paired_cluster_bootstrap_delta(cs.auc_macro, y, np.load(f), pr,
                                                          groups, args.boot)
            report_path.write_text(json.dumps(report, indent=2))

    log(f"written to {report_path}")


if __name__ == '__main__':
    main()
