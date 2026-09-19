#!/usr/bin/env bash
# Queue for the GPU arms after L0. Each stage is skipped if its final output exists.
# Usage: setsid nohup scripts/run_clean_all.sh > logs/queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
RC=results_clean; BB=models/hear_pristine_encoder.pth
say() { echo "[$(date '+%F %T')] $*"; }

cpu_chain() {   # $1 = prob dir, $2 = arm dir, extra meta_v2 flags in $3
  python3 scripts/run_clean_meta_v2.py --prob-dir "$1" --out-dir "$2/meta_v2" --split-dir "${4:-$RC}" $3 \
    && python3 scripts/run_clean_metrics.py --meta-dirs "$2/meta_v2" --out-dir "$2/metrics" \
    && python3 scripts/run_clean_subgroups.py --meta-dirs "$2/meta_v2" --out-dir "$2/subgroups" \
    && python3 scripts/run_clean_meta_benchmark.py --prob-dir "$1" --out-dir "$2/meta_benchmark" \
    && python3 scripts/run_clean_shap.py --prob-dir "$1" --meta-dir "$2/meta_v2" --out-dir "$2/shap"
}

say "waiting for arm L0 base stage"
until [ -f $RC/arm_L0_clean/probabilities/meta_test.csv ]; do
  pgrep -f "run_clean_pipeline.py.*arm_L0_clean" >/dev/null || { say "L0 process gone without output - abort"; exit 1; }
  sleep 120
done
say "L0 base stage done -> CPU chain (background) + saliency"
( cpu_chain $RC/arm_L0_clean/probabilities $RC/arm_L0_clean "" > logs/arm_L0_cpu.log 2>&1; say "L0 CPU chain exit $?" ) &

[ -f $RC/arm_L0_clean/saliency/saliency_report.json ] || \
  python3 scripts/run_clean_saliency.py --run-dir $RC/arm_L0_clean --backbone $BB \
    --out-dir $RC/arm_L0_clean/saliency > logs/saliency_L0.log 2>&1
say "saliency exit $?"

say "nested CV (primary analysis)"
python3 scripts/run_clean_nested_cv.py --backbone $BB > logs/nested_cv.log 2>&1
say "nested CV base stage exit $?"
( for k in 1 2 3 4 5; do
    python3 scripts/run_clean_meta_v2.py --prob-dir $RC/nested_cv/fold$k/probabilities \
      --out-dir $RC/nested_cv/fold$k/meta_v2 --split-dir $RC/nested_cv/fold$k --trials 50 || exit 1
  done
  python3 scripts/run_clean_metrics.py --meta-dirs $RC/nested_cv/fold{1,2,3,4,5}/meta_v2 --out-dir $RC/metrics/nested_cv \
  && python3 scripts/run_clean_subgroups.py --meta-dirs $RC/nested_cv/fold{1,2,3,4,5}/meta_v2 --out-dir $RC/metrics/nested_cv_subgroups
  say "nested CV CPU chain exit $?" ) > logs/nested_cv_cpu.log 2>&1 &

say "arm L2 (event-level split, deliberately leaky)"
[ -f $RC/arm_L2_event_split/probabilities/meta_test.csv ] || \
  python3 scripts/run_clean_pipeline.py --split event --results-dir $RC/arm_L2_event_split \
    --backbone $BB --seed 42 > logs/arm_L2.log 2>&1
say "L2 base stage exit $?"
cpu_chain $RC/arm_L2_event_split/probabilities $RC/arm_L2_event_split "--allow-patient-overlap" $RC/arm_L2_event_split > logs/arm_L2_cpu.log 2>&1
say "L2 CPU chain exit $?"
wait
say "QUEUE COMPLETE"
