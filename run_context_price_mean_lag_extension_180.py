"""Six NVDA checks and 180 new training runs for mean60 and mean80."""

import argparse
import itertools
from pathlib import Path

import mean_lag_grid_audit as audit
from run_context_price_mean_lag_grid import SCRIPTS, run_queue


RESULTS = audit.RUNS / 'context_price_mean_lag_grid_extension_10_v1'
WINDOWS = (60, 80)
CONFIGS = {
    audit.cell_name(window, lags): dict(audit.configuration(audit.MEAN_GRID, 32, window, 'price_only'),
                                       price_gate_lags=list(lags))
    for window, lags in itertools.product(WINDOWS, audit.LAG_SETS)
}
EXTENSION_SCRIPTS = (*SCRIPTS, 'run_context_price_mean_lag_extension_180.py',
                     'run_context_price_mean_lag_extension_180.sh')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=audit.shared.DATA)
    parser.add_argument('--results', type=Path, default=RESULTS)
    args = parser.parse_args()
    try:
        run_queue(Path(__file__).resolve().parent, args.data, args.results, None,
                  configurations=CONFIGS, scripts=EXTENSION_SCRIPTS)
    except audit.AUDIT_ERRORS as error:
        parser.exit(1, f'STOP: {error}\n')


if __name__ == '__main__':
    main()
