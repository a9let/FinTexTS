#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$CODE:$CODE/forecasting_task${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd -- "$CODE"
exec python -u "$CODE/run_group_memory_queue.py" "$@"
