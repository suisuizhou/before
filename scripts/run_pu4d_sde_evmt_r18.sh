#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-../.conda/bin/python}
SOURCE=${SOURCE:-0}
TARGET=${TARGET:-1}
SEED=${SEED:-1}

for variant in R2 R3 R4 R5 R6; do
  "$PYTHON_BIN" main_tta_sde_evmt_r18.py \
    +variant="$variant" \
    seed="$SEED" \
    +only_task="[$SOURCE,$TARGET]"
done
