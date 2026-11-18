#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$CODE:$CODE/forecasting_task${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$CODE"
exec python -u "$CODE/run_huang_query_queue.py" --code "$CODE" "$@"
