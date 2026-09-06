#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_dir="logs/PU4D_0711_BN_OPEN_20260905"
mkdir -p "$run_dir"

# 0->1 was completed as the BN-open smoke/full task before this sweep.
tasks=(
  "0,2" "0,3"
  "1,0" "1,2" "1,3"
  "2,0" "2,1" "2,3"
  "3,0" "3,1" "3,2"
)

for task in "${tasks[@]}"; do
  task_tag="${task//,/to}"
  task_log="$run_dir/task_${task_tag}.log"
  echo "[PU4D BN-OPEN] starting ${task}" | tee -a "$run_dir/progress.log"
  /home/std04/miniconda3/bin/python main_tta_0711_pu4d_bn_open.py \
    "+only_task=[$task]" \
    2>&1 | tee "$task_log"
  echo "[PU4D BN-OPEN] finished ${task}" | tee -a "$run_dir/progress.log"
done

echo "[PU4D BN-OPEN] all remaining tasks finished" | tee -a "$run_dir/progress.log"
