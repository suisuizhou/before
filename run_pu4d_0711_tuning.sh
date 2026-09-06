#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

python -m pytest tests/test_pu4d_0711_tuning.py -q
python -m py_compile \
  tools/tune_pu4d_0711_strict.py \
  tools/summarize_pu4d_0711_tuning.py \
  main_tta_0711_strict_online.py

exec python tools/tune_pu4d_0711_strict.py "$@"
