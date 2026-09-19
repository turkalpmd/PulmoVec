#!/usr/bin/env python3
"""
scripts/run_clean_meta_benchmark.py

Does the choice of meta-learner matter? Re-runs the candidate learners of
benchmark_meta_models.py on the patient-grouped tables: fit on TRAIN out-of-fold
probabilities (+ VAL is not used - fixed hyper-parameters), score TEST once, with
patient-bootstrap CIs and a paired contrast against LightGBM.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
warnings.filterwarnings('ignore')

from run_clean_meta import TARGETS, PROB_COLS, DEMO_COLS, prep  # noqa: E402
from benchmark_meta_models import get_candidate_models  # noqa: E402
import clean_stats as cs  # noqa: E402

NEEDS_SCALING = ('MLP', 'Logistic')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()
    prob_dir, out_dir = Path(args.prob_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = prep(pd.read_csv(prob_dir / 'meta_train_oof.csv'))
    te = prep(pd.read_csv(prob_dir / 'meta_test.csv'))
    assert not set(tr['_group_key']) & set(te['_group_key'])
    cols = PROB_COLS + DEMO_COLS
    groups = te['_group_key'].values
    report, rows = {}, []

    for target, names in TARGETS.items():
        ytr, yte = tr[target].values.astype(int), te[target].values.astype(int)
        probas = {}
        for name, model in get_candidate_models(len(names)).items():
            if name.startswith(NEEDS_SCALING):
                model = make_pipeline(StandardScaler(), model)
            model.fit(tr[cols].values, ytr)
            probas[name] = model.predict_proba(te[cols].values)
        report[target] = {}
        for name, p in probas.items():
            point = cs.overall_metrics(yte, p)
            ci, _ = cs.cluster_bootstrap(lambda r, p=p: cs.overall_metrics(yte[r], p[r]),
                                         groups, args.boot)
            entry = {k: {'value': float(v), 'ci': ci.get(k)} for k, v in point.items()}
            if name != 'LightGBM':
                entry['delta_auc_vs_lightgbm'] = cs.paired_cluster_bootstrap_delta(
                    cs.auc_macro, yte, p, probas['LightGBM'], groups, args.boot)
            report[target][name] = entry
            rows.append({'task': target, 'model': name,
                         **{k: round(float(v), 4) for k, v in point.items()},
                         'auc_lo': ci['auc'][0], 'auc_hi': ci['auc'][1]})
            print(f"{target:13s} {name:22s} acc={point['accuracy']:.4f} "
                  f"auc={cs.fmt_ci(point['auc'], ci['auc'])}", flush=True)
        (out_dir / 'meta_benchmark.json').write_text(json.dumps(report, indent=2))

    pd.DataFrame(rows).to_csv(out_dir / 'meta_benchmark.csv', index=False)
    print(f"Written to {out_dir}")


if __name__ == '__main__':
    main()
