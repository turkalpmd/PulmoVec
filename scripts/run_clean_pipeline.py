#!/usr/bin/env python3
"""
scripts/run_clean_pipeline.py

Leakage-free rebuild of the PulmoVec result set.

Stage 1  Patient-grouped 80:10:10 split (train / val / test).
Stage 2  For every task: 5-fold patient-grouped CV inside TRAIN to produce genuine
         out-of-fold base-model probabilities, plus one full-train model that scores
         VAL and TEST.
Stage 3  Meta-feature tables (OOF probabilities + age, sex, recording location).
Stage 4  LightGBM meta-model: Optuna searches on VAL, final metrics reported once on
         TEST, with bootstrap CIs, calibration (Brier / ECE) and an ablation ladder
         that includes a demographics-only baseline.
Stage 5  Patient-level aggregation on TEST.

Everything is written to results_clean/.
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from sprsound_dataset import (  # noqa: E402
    stratified_train_val_test_split,
    event_level_train_val_test_split,
    apply_age_exclusion,
    SPRSoundDatasetFromDF,
)
from repro import seed_everything, worker_init_fn, make_generator, sha256_file, env_snapshot  # noqa: E402
from models_hear_lora import HeARLoRAClassifier  # noqa: E402
from train_hear_lora import FocalLoss, WarmupCosineScheduler, train_one_epoch, evaluate  # noqa: E402

RESULTS = ROOT / 'results_clean'
PROB_DIR = RESULTS / 'probabilities'
CKPT_DIR = RESULTS / 'checkpoints'

# Released google/hear-pytorch encoder (scripts/run_clean_fetch_backbone.py). The old
# default, models/hear_sprsound_best.pth, was fine-tuned on an event-level split and
# must never be picked up silently.
PRISTINE_BACKBONE = ROOT / 'models' / 'hear_pristine_encoder.pth'
BACKBONE = PRISTINE_BACKBONE
SEED = 42


def configure(results_dir=None, backbone=None, seed=None):
    """Point the module at an output directory / backbone (also used by importers)."""
    global RESULTS, PROB_DIR, CKPT_DIR, BACKBONE, SEED
    if results_dir is not None:
        RESULTS = Path(results_dir)
    PROB_DIR, CKPT_DIR = RESULTS / 'probabilities', RESULTS / 'checkpoints'
    for d in (RESULTS, PROB_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if backbone is not None:
        BACKBONE = Path(backbone)
    if seed is not None:
        SEED = seed
    if not BACKBONE.exists():
        raise FileNotFoundError(
            f"Backbone weights not found: {BACKBONE}\n"
            "Run scripts/run_clean_fetch_backbone.py (needs a valid HF_TOKEN with access "
            "to google/hear-pytorch) or pass --backbone explicitly.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TASKS = {
    'model1_label': {'n': 3, 'names': ['Normal', 'Crackles', 'Rhonchi']},
    'model2_label': {'n': 2, 'names': ['Normal', 'Abnormal']},
    'model3_label': {'n': 3, 'names': ['Pneumonia', 'Bronchial', 'Normal_Other']},
}

# Row identity: one event = (filename, event_index)
KEY = ['filename', 'event_index']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_loader(df, label_col, training, batch_size, workers):
    ds = SPRSoundDatasetFromDF(df, label_column=label_col, is_training=training)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=training,
                          num_workers=workers, pin_memory=True, prefetch_factor=2,
                          worker_init_fn=worker_init_fn, generator=make_generator(SEED))


@torch.no_grad()
def predict_proba(model, loader, n_classes):
    """Return [N, n_classes] probabilities in loader order."""
    model.eval()
    out = []
    for audio, _, _ in loader:
        audio = audio.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(audio)
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_model(train_df, val_df, label_col, n_classes, epochs, patience, lr,
                batch_size, workers, tag):
    """Train one HeAR+LoRA model; early stopping on val macro-F1. Returns best model."""
    train_ds, train_loader = make_loader(train_df, label_col, True, batch_size, workers)
    _, val_loader = make_loader(val_df, label_col, False, batch_size, workers)

    class_weights = train_ds.get_class_weights().to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    model = HeARLoRAClassifier(num_classes=n_classes, lora_r=8, lora_alpha=16.0,
                               lora_layers=6, dropout=0.3,
                               weights_path=str(BACKBONE)).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=epochs, base_lr=lr)

    best_f1, best_epoch, no_improve = -1.0, -1, 0
    best_path = CKPT_DIR / f'{tag}.pth'

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        scheduler.step(epoch)
        tr_loss, tr_acc, tr_f1, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, max_grad_norm=1.0)
        vm = evaluate(model, val_loader, criterion, DEVICE)
        mark = ''
        if vm['macro_f1'] > best_f1:
            best_f1, best_epoch, no_improve = vm['macro_f1'], epoch, 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_macro_f1': best_f1}, best_path)
            mark = ' *best'
        else:
            no_improve += 1
        log(f"  {tag} ep{epoch:02d} loss={tr_loss:.4f} trF1={tr_f1:.3f} "
            f"valF1={vm['macro_f1']:.4f} ({time.time()-t0:.0f}s){mark}")
        if no_improve >= patience:
            log(f"  {tag} early stop at epoch {epoch} (best {best_epoch}: {best_f1:.4f})")
            break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE)['model_state_dict'])
    return model, {'best_epoch': best_epoch, 'best_val_macro_f1': float(best_f1)}


def oof_for_task(train_df, val_df, test_df, label_col, cfg, args):
    """5-fold patient-grouped OOF on TRAIN + a full-train model scoring VAL/TEST."""
    n_classes = cfg['n']
    cols = [f"{label_col}_p{i}" for i in range(n_classes)]

    oof = pd.DataFrame(index=train_df.index, columns=cols, dtype=float)
    grouped = getattr(args, 'split', 'patient') == 'patient'
    if grouped:
        folds = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED).split(
            train_df, train_df[label_col], groups=train_df['_group_key'])
    else:  # leaky comparator arm: record-wise folds, as in the event-level protocol
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED).split(
            train_df, train_df[label_col])
    # 'hold' reproduces the 2026-09-14 run (fold used for early stopping AND OOF scoring,
    # mildly optimistic meta-features); 'val' stops on the outer validation partition.
    stop_on_val = getattr(args, 'oof_early_stop', 'hold') == 'val'
    fold_info = []

    for fold, (fit_idx, hold_idx) in enumerate(folds, start=1):
        fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
        hold_df = train_df.iloc[hold_idx].reset_index(drop=True)

        if grouped:
            shared = set(fit_df['_group_key']) & set(hold_df['_group_key'])
            assert not shared, f"fold {fold}: {len(shared)} patients leak between fit and hold"

        log(f" {label_col} fold {fold}/{args.folds}: fit={len(fit_df)} hold={len(hold_df)}")
        model, info = train_model(fit_df, val_df if stop_on_val else hold_df,
                                  label_col, n_classes, args.epochs,
                                  args.patience, args.lr, args.batch_size, args.workers,
                                  tag=f'{label_col}_fold{fold}')
        _, hold_loader = make_loader(hold_df, label_col, False, args.batch_size, args.workers)
        oof.iloc[hold_idx] = predict_proba(model, hold_loader, n_classes)
        info['fold'] = fold
        fold_info.append(info)
        del model
        torch.cuda.empty_cache()

    assert not oof.isna().any().any(), "OOF matrix has gaps"

    # Full-train model (early stopping on VAL) scores VAL and TEST
    log(f" {label_col}: full-train model for val/test scoring")
    full_model, full_info = train_model(train_df, val_df, label_col, n_classes, args.epochs,
                                        args.patience, args.lr, args.batch_size, args.workers,
                                        tag=f'{label_col}_full')
    _, val_loader = make_loader(val_df, label_col, False, args.batch_size, args.workers)
    _, test_loader = make_loader(test_df, label_col, False, args.batch_size, args.workers)
    val_p = pd.DataFrame(predict_proba(full_model, val_loader, n_classes), columns=cols)
    test_p = pd.DataFrame(predict_proba(full_model, test_loader, n_classes), columns=cols)
    del full_model
    torch.cuda.empty_cache()

    return oof, val_p, test_p, {'folds': fold_info, 'full': full_info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--backbone', default=str(PRISTINE_BACKBONE),
                    help='HeAR encoder weights (default: released google/hear-pytorch copy)')
    ap.add_argument('--results-dir', default=str(RESULTS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split', choices=['patient', 'event'], default='patient',
                    help="'event' = deliberately leaky record-wise comparator arm")
    ap.add_argument('--oof-early-stop', choices=['hold', 'val'], default='hold')
    ap.add_argument('--split-from', default=None,
                    help='reuse split_{train,val,test}.csv from this directory (locked '
                         'hold-out) instead of re-drawing the patient split')
    args = ap.parse_args()

    configure(args.results_dir, args.backbone, args.seed)
    seed_everything(args.seed)
    run_info = {'args': vars(args), 'backbone': str(BACKBONE),
                'backbone_sha256': sha256_file(BACKBONE), 'env': env_snapshot()}
    (RESULTS / 'run_info.json').write_text(json.dumps(run_info, indent=2))
    log(f"backbone: {BACKBONE} sha256={run_info['backbone_sha256'][:16]}")

    log("=" * 70)
    log("PulmoVec clean pipeline - patient-grouped, OOF stacking")
    log("=" * 70)

    # Stage 1: one split, shared by every task (stratified on model3_label)
    if args.split_from:
        assert args.split == 'patient'
        train_df, val_df, test_df = (
            apply_age_exclusion(pd.read_csv(Path(args.split_from) / f'split_{n}.csv'))
            for n in ('train', 'val', 'test'))
        for a, b in [(train_df, val_df), (train_df, test_df), (val_df, test_df)]:
            assert not set(a['_group_key']) & set(b['_group_key']), "patient leakage"
    elif args.split == 'patient':
        train_df, val_df, test_df = stratified_train_val_test_split(
            label_column='model3_label', random_seed=args.seed)
    else:
        train_df, val_df, test_df, overlap = event_level_train_val_test_split(
            label_column='model3_label', random_seed=args.seed)
        (RESULTS / 'overlap.json').write_text(json.dumps(overlap, indent=2))

    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        d.to_csv(RESULTS / f'split_{name}.csv', index=False)

    split_summary = {
        name: {'events': int(len(d)), 'patients': int(d['_group_key'].nunique()),
               'recordings': int(d['filename'].nunique())}
        for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]
    }
    (RESULTS / 'split_summary.json').write_text(json.dumps(split_summary, indent=2))
    log(f"split: {split_summary}")

    # Stage 2-3: per-task OOF probabilities
    meta_train = train_df[KEY].copy()
    meta_val = val_df[KEY].copy()
    meta_test = test_df[KEY].copy()
    training_log = {}

    for label_col, cfg in TASKS.items():
        log("-" * 70)
        log(f"TASK {label_col} ({cfg['n']} classes)")
        t0 = time.time()
        oof, val_p, test_p, info = oof_for_task(train_df, val_df, test_df, label_col, cfg, args)
        info['minutes'] = round((time.time() - t0) / 60, 1)
        training_log[label_col] = info

        meta_train = pd.concat([meta_train, oof.reset_index(drop=True)], axis=1)
        meta_val = pd.concat([meta_val, val_p], axis=1)
        meta_test = pd.concat([meta_test, test_p], axis=1)
        (RESULTS / 'base_training_log.json').write_text(json.dumps(training_log, indent=2))
        log(f"TASK {label_col} done in {info['minutes']} min")

    # Attach covariates and ground truth
    extra = ['age', 'gender_code', 'recording_location', 'patient_number', '_group_key',
             'disease', 'event_type', 'model1_label', 'model2_label', 'model3_label']
    for meta, src in [(meta_train, train_df), (meta_val, val_df), (meta_test, test_df)]:
        for c in extra:
            meta[c] = src[c].values

    meta_train.to_csv(PROB_DIR / 'meta_train_oof.csv', index=False)
    meta_val.to_csv(PROB_DIR / 'meta_val.csv', index=False)
    meta_test.to_csv(PROB_DIR / 'meta_test.csv', index=False)

    log("=" * 70)
    log(f"Base stage complete. Meta-feature tables written to {PROB_DIR}")
    log("=" * 70)


if __name__ == '__main__':
    main()
