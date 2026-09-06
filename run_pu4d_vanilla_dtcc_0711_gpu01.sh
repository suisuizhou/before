#!/usr/bin/env bash
set -uo pipefail

PROJECT="${PROJECT:-/home/std04/Projects/DtCC_FOA_ViT_v1/DtCC_FOA_V1}"
DATA_PATH="${PROJECT}/Dataset/PU4D_CACHE"
VANILLA_ROOT="${PROJECT}/TTA_Model_VANILLA"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SOURCE="${SKIP_SOURCE:-0}"

cd "${PROJECT}" || exit 1
mkdir -p logs

for required in \
  main_src_pu4d_vanilla_r18.py \
  main_tta_dtcc_resnet18_vanilla.py \
  main_tta_0711_resnet18_vanilla.py \
  main_tta_0711_strict_randomstream.py \
  main_tta_0711_full_tuned.py \
  Lib/pu4d_vanilla_protocol.py \
  Lib/pu4d_common_source.py \
  Lib/dtcc_resnet18_common.py \
  Lib/fixed_random_stream.py \
  Lib/adaptation_ema.py \
  Lib/physical_fault_evidence.py; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] missing required project file: ${required}"
    exit 1
  fi
done

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  "$@"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  PYTHONPATH=. pytest -q tests/test_pu4d_vanilla_protocol.py tests/test_pu4d_vanilla_runner_contracts.py || exit 1
  python -m py_compile \
    main_src_pu4d_vanilla_r18.py \
    main_tta_dtcc_resnet18_vanilla.py \
    main_tta_0711_resnet18_vanilla.py \
    audit_pu4d_vanilla_checkpoints.py \
    summarize_pu4d_vanilla_results.py || exit 1
fi

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="${PROJECT}/logs/PU4D_VANILLA_DTCC_0711_${RUN_ID}"
mkdir -p "${RUN_DIR}/source_logs"
echo "${RUN_DIR}" > logs/latest_pu4d_vanilla_run_dir.txt
echo "[RUN_DIR] ${RUN_DIR}" | tee "${RUN_DIR}/progress.log"

SOURCE_COMMON=(
  Model=ResNet18_1D_SDE Dataset=PU4D gpu_id=0 process_wandb=False
  "++Dataset.data_path=${DATA_PATH}"
  Model.model_name=ResNet18_1D_SDE Model.model_type=linear
  Model.use_spectral_adapter=True Model.band_num=256 Model.adapter_delta=0.1
  batch_size=128 num_workers=4
  "++seed_run=2025" "++vanilla_source_root=${VANILLA_ROOT}"
)

run_source() {
  local s="$1"; local g="$2"; local log="${RUN_DIR}/source_logs/source_${s}.log"
  echo "[START] VANILLA source=${s} GPU=${g}" | tee -a "${RUN_DIR}/progress.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_cmd env CUDA_VISIBLE_DEVICES="${g}" python main_src_pu4d_vanilla_r18.py "${SOURCE_COMMON[@]}" "++only_source=${s}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${g}" python main_src_pu4d_vanilla_r18.py \
    "${SOURCE_COMMON[@]}" "++only_source=${s}" >"${log}" 2>&1
  local status=$?
  echo "[END] VANILLA source=${s} status=${status}" | tee -a "${RUN_DIR}/progress.log"
  return "${status}"
}

if [[ "${SKIP_SOURCE}" != "1" ]]; then
  failed=0
  run_source 0 0 & p0=$!; run_source 1 1 & p1=$!
  wait "$p0" || failed=1; wait "$p1" || failed=1
  run_source 2 0 & p2=$!; run_source 3 1 & p3=$!
  wait "$p2" || failed=1; wait "$p3" || failed=1
  if [[ "$failed" != "0" ]]; then echo "[ERROR] vanilla source phase failed" | tee -a "${RUN_DIR}/progress.log"; exit 1; fi
else
  echo "[SKIP] vanilla source training" | tee -a "${RUN_DIR}/progress.log"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  run_cmd python audit_pu4d_vanilla_checkpoints.py --root "${VANILLA_ROOT}" --seed 2025
else
  python audit_pu4d_vanilla_checkpoints.py --root "${VANILLA_ROOT}" --seed 2025 | tee -a "${RUN_DIR}/progress.log" || exit 1
fi

TARGET_COMMON=(
  Model=ResNet18_1D_SDE Dataset=PU4D gpu_id=0 process_wandb=False
  "++Dataset.data_path=${DATA_PATH}" "++seed_runs=[2025]"
  Model.model_name=ResNet18_1D_SDE Model.model_type=linear
  Model.use_spectral_adapter=True Model.band_num=256 Model.adapter_delta=0.1
  Opt.lr_src=0.001 batch_size=128 num_workers=4
  "++stream_seed=2025" "++vanilla_source_root=${VANILLA_ROOT}"
)

run_dtcc() {
  local s="$1"; local t="$2"; local g="$3"; local log="${RUN_DIR}/dtcc_${s}to${t}.log"
  echo "[START] DTCC [${s},${t}] GPU=${g}" | tee -a "${RUN_DIR}/progress.log"
  local cmd=(env CUDA_VISIBLE_DEVICES="${g}" python main_tta_dtcc_resnet18_vanilla.py
    "${TARGET_COMMON[@]}" "++only_task=[${s},${t}]"
    ++DtCC.lr=0.01 ++DtCC.weight_decay=0.001 ++DtCC.optim_steps=2
    ++DtCC.filter_k=50 ++DtCC.neighbor_k=5 ++DtCC.alpha=2.0
    ++DtCC.ncl_temperature=1.0 ++DtCC.log_interval=25)
  if [[ "${DRY_RUN}" == "1" ]]; then run_cmd "${cmd[@]}"; return 0; fi
  "${cmd[@]}" >"${log}" 2>&1
  local status=$?; echo "[END] DTCC [${s},${t}] status=${status}" | tee -a "${RUN_DIR}/progress.log"; return "$status"
}

run_0711() {
  local s="$1"; local t="$2"; local g="$3"; local log="${RUN_DIR}/0711_${s}to${t}.log"
  echo "[START] 0711 [${s},${t}] GPU=${g}" | tee -a "${RUN_DIR}/progress.log"
  local cmd=(env CUDA_VISIBLE_DEVICES="${g}" python main_tta_0711_resnet18_vanilla.py
    "${TARGET_COMMON[@]}" "++only_task=[${s},${t}]"
    Opt.lr_tar=0.015 Opt.weight_decay_tar=0.001
    ++TTA0711.mode=full ++TTA0711.passes=1 ++TTA0711.stream_seed=2025
    ++TTA0711.warmup_batches=10 ++TTA0711.aux_ramp_batches=20
    ++TTA0711.alpha=2.0 ++TTA0711.eta=0.05 ++TTA0711.min_reliability=0.20
    ++TTA0711.ema_beta=0.995 ++TTA0711.teacher_temp=1.0
    ++TTA0711.lambda_mt=0.02 ++TTA0711.mt_warmup_scale=0.5
    ++TTA0711.view_style_strength=0.05 ++TTA0711.view_style_knots=8
    ++TTA0711.view_warp_max=0.5 ++TTA0711.view_warp_knots=8
    ++TTA0711.view_gain_strength=0.03 ++TTA0711.view_baseline_strength=0.02 ++TTA0711.view_noise_std=0.01
    ++TTA0711.view_gamma=5.0 ++TTA0711.evidence_interval=1 ++TTA0711.evidence_metric=margin
    ++TTA0711.sampling_rate_hz=64000 ++TTA0711.fft_size=1024 ++TTA0711.spectrum_length=512
    ++TTA0711.physical_harmonics=8 "++TTA0711.outer_sideband_orders=[0,1]" "++TTA0711.inner_sideband_orders=[0,1,2]"
    ++TTA0711.mask_sigma_bins=1.0 ++TTA0711.physical_background_width=7 ++TTA0711.max_mask_ratio=0.18
    ++TTA0711.mask_activity_threshold=0.10 ++TTA0711.exclude_dc=True
    ++TTA0711.memory_per_class=64 ++TTA0711.lambda_pcl=0.02 ++TTA0711.pcl_temperature=0.20 ++TTA0711.min_pcl_classes=8
    ++TTA0711.lambda_ncl=0.01 ++TTA0711.ncl_neighbors=3 ++TTA0711.ncl_temperature=0.20 ++TTA0711.min_ncl_classes=16 ++TTA0711.min_ncl_entries=64
    ++TTA0711.use_frequency_warp=True ++TTA0711.warp_knots=16 ++TTA0711.max_warp=2.0 ++TTA0711.warp_smooth_weight=2.0
    ++TTA0711.adapter_lr_scale=1.0 ++TTA0711.warp_lr_scale=0.1 ++TTA0711.lambda_adapter=0.001 ++TTA0711.lambda_warp=0.0002
    ++TTA0711.log_interval=25)
  if [[ "${DRY_RUN}" == "1" ]]; then run_cmd "${cmd[@]}"; return 0; fi
  "${cmd[@]}" >"${log}" 2>&1
  local status=$?; echo "[END] 0711 [${s},${t}] status=${status}" | tee -a "${RUN_DIR}/progress.log"; return "$status"
}

GPU0_TASKS=("0 1" "0 3" "1 2" "2 0" "2 3" "3 1")
GPU1_TASKS=("0 2" "1 0" "1 3" "2 1" "3 0" "3 2")
# Full explicit task contract: 0 1  0 2  0 3  1 0  1 2  1 3  2 0  2 1  2 3  3 0  3 1  3 2

run_queue() {
  local method="$1" gpu="$2"; shift 2; local failed=0
  for pair in "$@"; do read -r s t <<<"$pair"; if [[ "$method" == "dtcc" ]]; then run_dtcc "$s" "$t" "$gpu" || failed=1; else run_0711 "$s" "$t" "$gpu" || failed=1; fi; done
  return "$failed"
}

failed=0
run_queue dtcc 0 "${GPU0_TASKS[@]}" & p0=$!; run_queue dtcc 1 "${GPU1_TASKS[@]}" & p1=$!
wait "$p0" || failed=1; wait "$p1" || failed=1
run_queue 0711 0 "${GPU0_TASKS[@]}" & p2=$!; run_queue 0711 1 "${GPU1_TASKS[@]}" & p3=$!
wait "$p2" || failed=1; wait "$p3" || failed=1

if [[ "${DRY_RUN}" != "1" ]]; then
  python summarize_pu4d_vanilla_results.py "${RUN_DIR}" | tee -a "${RUN_DIR}/progress.log" || exit 1
fi

echo "[FINISHED] ${RUN_DIR}" | tee -a "${RUN_DIR}/progress.log"
exit "$failed"
