#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$CODE:$CODE/forecasting_task${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd -- "$CODE"
exec python -u "$CODE/run_context_price_mean_lag_grid.py" "$@"
