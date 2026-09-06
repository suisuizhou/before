#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:-all}
GPUS=${GPUS-auto}
DRY_RUN=${DRY_RUN:-0}

case "$STAGE" in
    cache|source|beginning|baseline|tune|final|load-audit|report|all) ;;
    *)
        printf 'unsupported STAGE: %s\n' "$STAGE" >&2
        exit 2
        ;;
esac

case "$DRY_RUN" in
    0|1) ;;
    *)
        printf 'DRY_RUN must be 0 or 1, got: %s\n' "$DRY_RUN" >&2
        exit 2
        ;;
esac

if [[ "$GPUS" != auto && ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'GPUS must be auto or a comma-separated list of nonnegative integers: %s\n' "$GPUS" >&2
    exit 2
fi

SCRIPT_PATH=${BASH_SOURCE[0]}
if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH=$PWD/$SCRIPT_PATH
fi
SCRIPT_NAME=${SCRIPT_PATH##*/}
SCRIPT_DIR=${SCRIPT_PATH%/*}
cd -P "$SCRIPT_DIR"
PROJECT_DIR=$PWD
SCRIPT_PATH=$PROJECT_DIR/$SCRIPT_NAME

GENERATED_RUN_DIR=0
if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_EPOCH=-1
    if [[ -n "${SOURCE_DATE_EPOCH:-}" ]]; then
        if [[ ! "$SOURCE_DATE_EPOCH" =~ ^[0-9]+$ ]]; then
            printf 'SOURCE_DATE_EPOCH must be a nonnegative integer\n' >&2
            exit 2
        fi
        RUN_EPOCH=$SOURCE_DATE_EPOCH
    fi
    TZ=UTC printf -v RUN_STAMP '%(%Y%m%d_%H%M%S)T' "$RUN_EPOCH"
    RUN_DIR="logs/HUST_DTCC_0711_STRICT_${RUN_STAMP}"
    GENERATED_RUN_DIR=1
fi
RUN_BASENAME=${RUN_DIR%/}
RUN_BASENAME=${RUN_BASENAME##*/}
if [[ ! "$RUN_BASENAME" =~ ^HUST_DTCC_0711_STRICT_[0-9]{8}_[0-9]{6}$ ]]; then
    printf 'RUN_DIR must end in HUST_DTCC_0711_STRICT_YYYYMMDD_HHMMSS: %s\n' "$RUN_DIR" >&2
    exit 2
fi

CONFIG=Configs/Experiments/HUST0711_strict_tuning.yaml
CONFIG_CANONICAL=$PROJECT_DIR/$CONFIG
ORCHESTRATOR=tools/tune_hust_0711_strict.py
CACHE_BUILDER=tools/build_hust_strict_cache.py
MARKER_NAME=.hust_dtcc_0711_launcher
printf -v EXPECTED_MARKER 'schema=1\nrepo=%s\nconfig=%s\nrun=%s' \
    "$PROJECT_DIR" "$CONFIG_CANONICAL" "$RUN_BASENAME"

print_command() {
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
}

orchestrator_command() {
    local requested_stage=$1
    ORCHESTRATOR_COMMAND=(
        python "$ORCHESTRATOR"
        --config "$CONFIG"
        --stage "$requested_stage"
        --run-dir "$RUN_DIR"
        --gpus "$GPUS"
    )
}

if [[ "$DRY_RUN" == 1 ]]; then
    if [[ "$STAGE" == cache ]]; then
        print_command python "$CACHE_BUILDER" \
            --raw-root Dataset/HUST \
            --output Dataset/HUST_STRICT_CACHE_V2 \
            --seed 2025 \
            --dry-run
    elif [[ "$STAGE" == report ]]; then
        orchestrator_command report
        print_command "${ORCHESTRATOR_COMMAND[@]}" --dry-run
    elif [[ "$STAGE" == all ]]; then
        orchestrator_command source
        print_command "${ORCHESTRATOR_COMMAND[@]}" --dry-run
        orchestrator_command all
        print_command "${ORCHESTRATOR_COMMAND[@]}" --dry-run
    else
        orchestrator_command "$STAGE"
        print_command "${ORCHESTRATOR_COMMAND[@]}" --dry-run
    fi
    exit 0
fi

RUN_DIR_EXISTS=0
if [[ -e "$RUN_DIR" ]]; then
    if [[ "$GENERATED_RUN_DIR" == 1 ]]; then
        printf 'generated RUN_DIR collision; refusing to reuse: %s\n' "$RUN_DIR" >&2
        exit 2
    fi
    if [[ ! -d "$RUN_DIR" || ! -f "$RUN_DIR/$MARKER_NAME" ]]; then
        printf 'RUN_DIR lacks valid launcher ownership: %s\n' "$RUN_DIR" >&2
        exit 2
    fi
    ACTUAL_MARKER=$(<"$RUN_DIR/$MARKER_NAME")
    if [[ "$ACTUAL_MARKER" != "$EXPECTED_MARKER" ]]; then
        printf 'RUN_DIR launcher ownership marker mismatch: %s\n' "$RUN_DIR" >&2
        exit 2
    fi
    RUN_DIR_EXISTS=1
fi

export PYTHONPATH=.

run_code_preflight() {
    pytest -q \
        tests/test_hust_strict_cache.py \
        tests/test_hust_source_protocol.py \
        tests/test_hust_runner_contracts.py \
        tests/test_hust_tuning.py
    python -c \
        'import pathlib, py_compile, sys, tempfile; directory=tempfile.TemporaryDirectory(); [py_compile.compile(path, cfile=str(pathlib.Path(directory.name) / f"{index}.pyc"), doraise=True) for index, path in enumerate(sys.argv[1:])]' \
        Lib/hust_strict_protocol.py \
        Lib/hust_physical_fault_evidence.py \
        Lib/hust_source_training.py \
        Dataset/HUSTStrict.py \
        main_src_dtcc_hust_strict.py \
        main_src_0711_hust_strict.py \
        main_tta_dtcc_hust_strict.py \
        main_tta_0711_hust_strict.py \
        tools/build_hust_strict_cache.py \
        tools/tune_hust_0711_strict.py \
        tools/summarize_hust_dtcc_0711.py
    bash -n "$SCRIPT_PATH"
}

validate_cache_root() {
    python -c \
        'import sys; from Lib.hust_strict_protocol import validate_cache; print(validate_cache(sys.argv[1])["manifest_sha256"])' \
        "$1"
}

audit_primary_checkpoints() {
    python -c \
        'import sys; from pathlib import Path; from tools.tune_hust_0711_strict import _source_output, load_config, plan_stage_jobs; config=load_config(Path(sys.argv[1])); jobs=plan_stage_jobs(config, "beginning", Path(sys.argv[2])); unique={str(job["source_checkpoint_path"]): job for job in jobs if job["kind"] == "target"}; audited=[_source_output(job)[0] for job in unique.values()]; assert len(unique) == 8, f"expected 8 strict source checkpoints, found {len(unique)}"; assert all(row.get("carrier_identity_check") is True and row.get("target_labels_consumed") is False and len(row.get("carrier_parameters", ())) == 3 for row in audited), "source checkpoint identity contract mismatch"; print("audited source checkpoints: 8")' \
        "$CONFIG" "$RUN_DIR"
}

discover_gpus() {
    nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
}

case "$STAGE" in
    report)
        if [[ "$RUN_DIR_EXISTS" != 1 || ! -d "$RUN_DIR/state" ]]; then
            printf 'report requires an owned RUN_DIR with state inputs: %s\n' "$RUN_DIR" >&2
            exit 2
        fi
        shopt -s nullglob
        REPORT_STATE_FILES=("$RUN_DIR"/state/*.json)
        shopt -u nullglob
        if (( ${#REPORT_STATE_FILES[@]} == 0 )); then
            printf 'report requires persisted state JSON inputs: %s\n' "$RUN_DIR" >&2
            exit 2
        fi
        ;;
    cache)
        run_code_preflight
        python "$CACHE_BUILDER" \
            --raw-root Dataset/HUST \
            --output Dataset/HUST_STRICT_CACHE_V2 \
            --seed 2025 \
            --dry-run
        ;;
    source|all)
        run_code_preflight
        validate_cache_root Dataset/HUST_STRICT_CACHE_V2
        discover_gpus
        ;;
    beginning|baseline|tune|final)
        run_code_preflight
        validate_cache_root Dataset/HUST_STRICT_CACHE_V2
        audit_primary_checkpoints
        discover_gpus
        ;;
    load-audit)
        run_code_preflight
        validate_cache_root Dataset/HUST_STRICT_LOAD_CACHE_V2
        discover_gpus
        ;;
esac

if [[ "$RUN_DIR_EXISTS" != 1 ]]; then
    if ! mkdir -- "$RUN_DIR"; then
        printf 'RUN_DIR collision; refusing to reuse: %s\n' "$RUN_DIR" >&2
        exit 2
    fi
    MARKER_TEMP=$RUN_DIR/$MARKER_NAME.tmp.$$
    (umask 077; printf '%s' "$EXPECTED_MARKER" > "$MARKER_TEMP")
    mv -- "$MARKER_TEMP" "$RUN_DIR/$MARKER_NAME"
fi

case "$STAGE" in
    cache)
        python "$CACHE_BUILDER" \
            --raw-root Dataset/HUST \
            --output Dataset/HUST_STRICT_CACHE_V2 \
            --seed 2025
        ;;
    report)
        orchestrator_command report
        "${ORCHESTRATOR_COMMAND[@]}"
        ;;
    all)
        orchestrator_command source
        "${ORCHESTRATOR_COMMAND[@]}"
        audit_primary_checkpoints
        orchestrator_command all
        "${ORCHESTRATOR_COMMAND[@]}"
        ;;
    *)
        orchestrator_command "$STAGE"
        "${ORCHESTRATOR_COMMAND[@]}"
        ;;
esac
