#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_dir="logs/PU4D_0711_BN_OPEN_20260905"
mkdir -p "$run_dir"

wait_for_result() {
  local log_path="$1"
  while ! grep -q "Post-stream Full-Target Acc" "$log_path" 2>/dev/null; do
    sleep 60
  done
}

# The current pair 2->3 and 3->0 is already running in independent sessions.
wait_for_result "$run_dir/task_2to3.log"
wait_for_result "$run_dir/task_3to0.log"

for task in "3,1" "3,2"; do
  task_tag="${task//,/to}"
  task_log="$run_dir/task_${task_tag}.log"
  echo "[PU4D BN-OPEN SUPERVISOR] starting ${task}" >> "$run_dir/progress.log"
  /home/std04/miniconda3/bin/python -u main_tta_0711_pu4d_bn_open.py \
    "+only_task=[$task]" \
    2>&1 | tee "$task_log"
  echo "[PU4D BN-OPEN SUPERVISOR] finished ${task}" >> "$run_dir/progress.log"
done

echo "[PU4D BN-OPEN SUPERVISOR] all tasks finished" >> "$run_dir/progress.log"
