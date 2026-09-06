#!/usr/bin/env bash
set -euo pipefail

# PU4D ViT Protocol A
#   DtCC-ViT-Prompt: prompt-only TTA
#   0711-Full-ViT-Prompt: prompt + Spectral Adapter + F-Warp TTA
# Both source models train for src_epoch=60.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

SEED="${SEED:-2025}"
STREAM_SEED="${STREAM_SEED:-2025}"
DRY_RUN="${DRY_RUN:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-logs/vit_protocol_a_${TIMESTAMP}}"
mkdir -p logs "$RUN_DIR/source_logs" "$RUN_DIR/target_logs" "$RUN_DIR/audit"
echo "$RUN_DIR" > logs/latest_vit_protocol_a_run_dir.txt
PROGRESS="$RUN_DIR/progress.log"
: > "$PROGRESS"

echo "[RUN] project=$PROJECT_ROOT seed=$SEED stream_seed=$STREAM_SEED dry_run=$DRY_RUN" | tee -a "$PROGRESS"

run_fg() {
  local cmd="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY] $cmd" | tee -a "$PROGRESS"
  else
    bash -lc "$cmd"
  fi
}

run_two() {
  local label0="$1" cmd0="$2" log0="$3"
  local label1="$4" cmd1="$5" log1="$6"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY][$label0] $cmd0 > $log0 2>&1" | tee -a "$PROGRESS"
    echo "[DRY][$label1] $cmd1 > $log1 2>&1" | tee -a "$PROGRESS"
    return 0
  fi
  echo "[START] $label0" | tee -a "$PROGRESS"
  bash -lc "$cmd0" > "$log0" 2>&1 & local p0=$!
  echo "[START] $label1" | tee -a "$PROGRESS"
  bash -lc "$cmd1" > "$log1" 2>&1 & local p1=$!
  local rc=0
  wait "$p0" || rc=$?
  if [[ "$rc" != "0" ]]; then
    echo "[FAIL] $label0 rc=$rc log=$log0" | tee -a "$PROGRESS"; return "$rc"
  fi
  wait "$p1" || rc=$?
  if [[ "$rc" != "0" ]]; then
    echo "[FAIL] $label1 rc=$rc log=$log1" | tee -a "$PROGRESS"; return "$rc"
  fi
  echo "[DONE] $label0" | tee -a "$PROGRESS"
  echo "[DONE] $label1" | tee -a "$PROGRESS"
}

source_cmd() {
  local script="$1" src="$2" tar="$3" gpu="$4"
  printf "python %q Model=ViT1D Dataset=PU4D ++only_task='[%s,%s]' gpu_id=%s ++src_epoch=60 ++batch_size=64 ++num_workers=4 process_wandb=False" \
    "$script" "$src" "$tar" "$gpu"
}

target_cmd() {
  local script="$1" method="$2" src="$3" tar="$4" gpu="$5"
  local ckpt="TTA_Model_ViT_A/${method}/PU4D/source_${src}/seed_${SEED}"
  printf "python %q Model=ViT1D Dataset=PU4D ++only_task='[%s,%s]' gpu_id=%s ++seed_run=%s ++seed_runs='[%s]' ++stream_seed=%s ++batch_size=128 ++num_workers=4 ++source_ckpt_dir=%q process_wandb=False" \
    "$script" "$src" "$tar" "$gpu" "$SEED" "$SEED" "$STREAM_SEED" "$ckpt"
}

stage_source() {
  local method="$1" src="$2" tar="$3"
  run_fg "python stage_vit_protocol_a_checkpoint.py --method $method --source $src --target $tar --seed $SEED"
}

# ---------------------------------------------------------------------------
# 1) Four method-specific DtCC source models, two GPUs at a time.
# Representative tasks only provide the source-domain data loader.  The best
# checkpoint is selected by SOURCE accuracy in the existing source trainer.
# ---------------------------------------------------------------------------
echo "[PHASE] DtCC ViT source training (60 epochs)" | tee -a "$PROGRESS"
run_two "DTCC-src0" "$(source_cmd main_src_dtcc_vit_prompt.py 0 1 0)" "$RUN_DIR/source_logs/dtcc_source_0.log" \
        "DTCC-src1" "$(source_cmd main_src_dtcc_vit_prompt.py 1 0 1)" "$RUN_DIR/source_logs/dtcc_source_1.log"
stage_source DTCC_VIT 0 1
stage_source DTCC_VIT 1 0
run_two "DTCC-src2" "$(source_cmd main_src_dtcc_vit_prompt.py 2 0 0)" "$RUN_DIR/source_logs/dtcc_source_2.log" \
        "DTCC-src3" "$(source_cmd main_src_dtcc_vit_prompt.py 3 0 1)" "$RUN_DIR/source_logs/dtcc_source_3.log"
stage_source DTCC_VIT 2 0
stage_source DTCC_VIT 3 0
run_fg "python audit_vit_protocol_a_checkpoints.py --method DTCC_VIT --seed $SEED > $RUN_DIR/audit/dtcc_vit.log"

# ---------------------------------------------------------------------------
# 2) Four 0711-Full ViT source models, preserving the current SSP/SDE recipe.
# ---------------------------------------------------------------------------
echo "[PHASE] 0711-Full ViT source training (60 epochs)" | tee -a "$PROGRESS"
run_two "0711-src0" "$(source_cmd main_src_0711_full_vit_prompt.py 0 1 0)" "$RUN_DIR/source_logs/0711_source_0.log" \
        "0711-src1" "$(source_cmd main_src_0711_full_vit_prompt.py 1 0 1)" "$RUN_DIR/source_logs/0711_source_1.log"
stage_source 0711_FULL_VIT 0 1
stage_source 0711_FULL_VIT 1 0
run_two "0711-src2" "$(source_cmd main_src_0711_full_vit_prompt.py 2 0 0)" "$RUN_DIR/source_logs/0711_source_2.log" \
        "0711-src3" "$(source_cmd main_src_0711_full_vit_prompt.py 3 0 1)" "$RUN_DIR/source_logs/0711_source_3.log"
stage_source 0711_FULL_VIT 2 0
stage_source 0711_FULL_VIT 3 0
run_fg "python audit_vit_protocol_a_checkpoints.py --method 0711_FULL_VIT --seed $SEED > $RUN_DIR/audit/0711_full_vit.log"

# ---------------------------------------------------------------------------
# 3) Target TTA: all 12 directed transfer tasks for each method.
# ---------------------------------------------------------------------------
TASKS=("0 1" "0 2" "0 3" "1 0" "1 2" "1 3" "2 0" "2 1" "2 3" "3 0" "3 1" "3 2")

run_method_targets() {
  local method="$1" script="$2" prefix="$3"
  echo "[PHASE] target method=$method" | tee -a "$PROGRESS"
  local i=0
  while (( i < ${#TASKS[@]} )); do
    read -r s0 t0 <<< "${TASKS[$i]}"
    read -r s1 t1 <<< "${TASKS[$((i+1))]}"
    local log0="$RUN_DIR/target_logs/${prefix}_${s0}to${t0}.log"
    local log1="$RUN_DIR/target_logs/${prefix}_${s1}to${t1}.log"
    run_two "${prefix}-${s0}to${t0}" "$(target_cmd "$script" "$method" "$s0" "$t0" 0)" "$log0" \
            "${prefix}-${s1}to${t1}" "$(target_cmd "$script" "$method" "$s1" "$t1" 1)" "$log1"
    i=$((i+2))
  done
}

run_method_targets DTCC_VIT main_tta_dtcc_vit_prompt.py dtcc_vit
run_method_targets 0711_FULL_VIT main_tta_0711_full_vit_prompt.py 0711_full_vit

# ---------------------------------------------------------------------------
# 4) Summary.
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY] python summarize_vit_protocol_a.py $RUN_DIR" | tee -a "$PROGRESS"
else
  python summarize_vit_protocol_a.py "$RUN_DIR" | tee "$RUN_DIR/summary_console.log"
fi

echo "[COMPLETE] $RUN_DIR" | tee -a "$PROGRESS"
