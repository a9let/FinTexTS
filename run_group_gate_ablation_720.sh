#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$CODE:$CODE/forecasting_task${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$CODE"
exec python -u "$CODE/run_group_gate_queue.py" --code "$CODE" "$@"
