#!/usr/bin/env python3
"""
scripts/run_clean_nested_cv.py

PRIMARY ANALYSIS - nested patient-grouped cross-validation over the whole cohort.

Outer loop  5-fold StratifiedGroupKFold on patients: every patient is a test patient
            exactly once.
Per fold    the non-test patients are split again (patient-level, stratified) into a
            validation partition (~10% of the cohort) and a training partition.
Inner loop  K-fold patient-grouped out-of-fold base-model probabilities on the training
            partition. Inner models early-stop on the fold's VALIDATION partition, never
            on the inner fold they go on to score (the 2026-09-14 run stopped on the scored
            fold, which makes the OOF meta-features mildly optimistic).
Full model  one model per task trained on the whole training partition scores VAL and TEST.

Writes results_clean/nested_cv/fold{k}/probabilities/meta_{train_oof,val,test}.csv in the
schema run_clean_meta_v2.py expects, plus split_{train,val,test}.csv per fold.
Resumable: a finished task/fold is skipped.
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_clean_pipeline as rcp  # noqa: E402
from run_clean_pipeline import TASKS, KEY, make_loader, predict_proba, train_model, log  # noqa: E402
from sprsound_dataset import (_load_curated_frame, _split_known_unknown,  # noqa: E402
                              _patient_level_split, _assert_no_patient_leakage)
from repro import seed_everything, sha256_file, env_snapshot  # noqa: E402

EXTRA = ['age', 'gender_code', 'recording_location', 'patient_number', '_group_key',
         'disease', 'event_type', 'event_duration_ms', 'model1_label', 'model2_label',
         'model3_label']
STRAT = 'model3_label'   # patient-level attribute, the scarcest strata


def task_tables(train_df, val_df, test_df, label_col, n_classes, args, fold_tag):
    cols = [f"{label_col}_p{i}" for i in range(n_classes)]
    oof = pd.DataFrame(index=train_df.index, columns=cols, dtype=float)
    inner = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True,
                                 random_state=args.seed)
    info = {'inner': []}
    for k, (fit_idx, hold_idx) in enumerate(
            inner.split(train_df, train_df[label_col], groups=train_df['_group_key']), start=1):
        fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
        hold_df = train_df.iloc[hold_idx].reset_index(drop=True)
        assert not set(fit_df['_group_key']) & set(hold_df['_group_key'])
        tag = f'{fold_tag}_{label_col}_inner{k}'
        model, m = train_model(fit_df, val_df, label_col, n_classes, args.epochs,
                               args.patience, args.lr, args.batch_size, args.workers, tag=tag)
        _, loader = make_loader(hold_df, label_col, False, args.batch_size, args.workers)
        oof.iloc[hold_idx] = predict_proba(model, loader, n_classes)
        info['inner'].append(m)
        del model
        torch.cuda.empty_cache()
        (rcp.CKPT_DIR / f'{tag}.pth').unlink(missing_ok=True)   # 1.2 GB each
    assert not oof.isna().any().any()

    tag = f'{fold_tag}_{label_col}_full'
    model, info['full'] = train_model(train_df, val_df, label_col, n_classes, args.epochs,
                                      args.patience, args.lr, args.batch_size, args.workers,
                                      tag=tag)
    out = []
    for d in (val_df, test_df):
        _, loader = make_loader(d, label_col, False, args.batch_size, args.workers)
        out.append(pd.DataFrame(predict_proba(model, loader, n_classes), columns=cols))
    del model
    torch.cuda.empty_cache()
    if not args.keep_full_checkpoints:
        (rcp.CKPT_DIR / f'{tag}.pth').unlink(missing_ok=True)
    return oof.reset_index(drop=True), out[0], out[1], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outer-folds', type=int, default=5)
    ap.add_argument('--inner-folds', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.10, help='fraction of the COHORT')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--backbone', default=str(rcp.PRISTINE_BACKBONE))
    ap.add_argument('--out-dir', default=str(ROOT / 'results_clean' / 'nested_cv'))
    ap.add_argument('--only-fold', type=int, default=None)
    ap.add_argument('--keep-full-checkpoints', action='store_true')
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    rcp.configure(out_root, args.backbone, args.seed)
    seed_everything(args.seed)
    (out_root / 'run_info.json').write_text(json.dumps(
        {'args': vars(args), 'backbone_sha256': sha256_file(rcp.BACKBONE),
         'env': env_snapshot()}, indent=2))

    df = _load_curated_frame(None, STRAT)
    known, unknown = _split_known_unknown(df)
    assert len(unknown) == 0, "id-less recordings survived curation; decide how to place them"

    pat = known.groupby('_group_key')[STRAT].first()
    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True,
                                 random_state=args.seed)
    fold_of = pd.Series(-1, index=pat.index)
    for k, (_, te_idx) in enumerate(outer.split(pat.index, pat.values, groups=pat.index), 1):
        fold_of.iloc[te_idx] = k
    fold_of.rename('outer_fold').to_csv(out_root / 'patient_fold_assignment.csv')

    for k in range(1, args.outer_folds + 1):
        if args.only_fold and k != args.only_fold:
            continue
        fold_dir = out_root / f'fold{k}'
        prob_dir = fold_dir / 'probabilities'
        prob_dir.mkdir(parents=True, exist_ok=True)
        log_path = fold_dir / 'training_log.json'
        done = json.loads(log_path.read_text()) if log_path.exists() else {}

        test_df = known[known['_group_key'].map(fold_of) == k].reset_index(drop=True)
        rest = known[known['_group_key'].map(fold_of) != k].reset_index(drop=True)
        val_share = args.val_frac / (1 - 1 / args.outer_folds)
        tr_pat, va_pat, _ = _patient_level_split(rest, STRAT, (1 - val_share, val_share, 0.0),
                                                 args.seed + k)
        train_df = rest[rest['_group_key'].isin(tr_pat)].reset_index(drop=True)
        val_df = rest[rest['_group_key'].isin(va_pat)].reset_index(drop=True)
        _assert_no_patient_leakage({'train': train_df, 'val': val_df, 'test': test_df},
                                   df.columns)
        for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
            d.to_csv(fold_dir / f'split_{name}.csv', index=False)
        log(f"OUTER FOLD {k}: train {train_df['_group_key'].nunique()} pts / {len(train_df)} ev | "
            f"val {val_df['_group_key'].nunique()} / {len(val_df)} | "
            f"test {test_df['_group_key'].nunique()} / {len(test_df)}")

        for label_col, cfg in TASKS.items():
            parts = [prob_dir / f'_{label_col}_{s}.csv' for s in ('train', 'val', 'test')]
            if label_col in done and all(p.exists() for p in parts):
                log(f"  fold {k} {label_col}: already done, skipping")
                continue
            t0 = time.time()
            oof, val_p, test_p, info = task_tables(train_df, val_df, test_df, label_col,
                                                   cfg['n'], args, f'fold{k}')
            for p, table in zip(parts, (oof, val_p, test_p)):
                table.to_csv(p, index=False)
            info['minutes'] = round((time.time() - t0) / 60, 1)
            done[label_col] = info
            log_path.write_text(json.dumps(done, indent=2))
            log(f"  fold {k} {label_col} done in {info['minutes']} min")

        if all((prob_dir / f'_{t}_test.csv').exists() for t in TASKS):
            for s, src, fname in [('train', train_df, 'meta_train_oof.csv'),
                                  ('val', val_df, 'meta_val.csv'),
                                  ('test', test_df, 'meta_test.csv')]:
                meta = pd.concat([src[KEY].reset_index(drop=True)] +
                                 [pd.read_csv(prob_dir / f'_{t}_{s}.csv') for t in TASKS], axis=1)
                for c in EXTRA:
                    meta[c] = src[c].values
                meta['outer_fold'] = k
                meta.to_csv(prob_dir / fname, index=False)
            log(f"OUTER FOLD {k} meta tables written")

    log("nested CV base stage complete")


if __name__ == '__main__':
    main()
