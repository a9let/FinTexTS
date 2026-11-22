"""Compare mean40, mean60 and mean80 on matched ticker/seed pairs per lag set."""

import argparse
from collections import Counter
from pathlib import Path
import re
from statistics import mean

import mean_lag_grid_audit as audit
from analyze_context_price_mean_lag_grid import fmt, improvement
from run_context_price_mean_lag_extension_180 import CONFIGS, RESULTS, WINDOWS


METRICS = audit.prior.METRICS
REFERENCE_WINDOW = 40


def read_commit(root):
    sha = (root/'code_commit.txt').read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', sha):
        raise ValueError(f'{root}: expected a full code SHA')
    return sha


def collect(results, control_root, data):
    cache = audit.prior.ReplayCache(data)
    rows = []
    commits = {}
    for root in (results, control_root):
        try:
            commits[root] = read_commit(root), ''
        except audit.AUDIT_ERRORS as error:
            commits[root] = None, str(error)
    for lags in audit.LAG_SETS:
        for ticker, seed in audit.PAIRS:
            control = None
            for window in (REFERENCE_WINDOW, *WINDOWS):
                cell = audit.cell_name(window, lags)
                is_control = window == REFERENCE_WINDOW
                root = control_root if is_control else results
                config = audit.CONFIGS[cell] if is_control else CONFIGS[cell]
                destination = audit.grid_destination(root, cell, ticker, seed)
                commit, commit_error = commits[root]
                row = dict(window=window, lags=','.join(map(str, lags)), ticker=ticker, seed=seed,
                           path=str(destination/'output'), code_commit=commit, checkpoint_sha256=None,
                           verification='unconfirmed', common_epochs=None, reason=commit_error,
                           **dict.fromkeys(METRICS))
                rows.append(row)
                if commit is None:
                    continue
                try:
                    run = audit.checked_run(destination, config, data, commit, ticker, seed, cache)
                except audit.AUDIT_ERRORS as error:
                    row.update(reason=str(error), verification='unconfirmed'
                               if isinstance(error, FileNotFoundError) else 'mismatch')
                    continue
                row.update({metric: run[0][metric] for metric in METRICS})
                row['checkpoint_sha256'] = run[2]['checkpoint_sha256']
                if is_control:
                    control = run
                if control is None:
                    row['reason'] = 'mean40 checkpoint is unconfirmed for this ticker, seed and lag set'
                    continue
                try:
                    row['common_epochs'] = audit.compare_grid_runs(run, control)
                    row['verification'] = 'full'
                    for metric in METRICS:
                        row[metric+'_difference'] = row[metric] - control[0][metric]
                        row[metric+'_improvement_pct'] = improvement(row[metric], control[0][metric])
                except audit.AUDIT_ERRORS as error:
                    row.update(verification='mismatch', reason=str(error))
    return rows


def summarize(rows):
    table = []
    for lag_set in audit.LAG_SETS:
        lags = ','.join(map(str, lag_set))
        cells = []
        for window in (REFERENCE_WINDOW, *WINDOWS):
            runs = [r for r in rows if r['lags'] == lags and r['window'] == window]
            verified = [r for r in runs if r['verification'] == 'full']
            count = len(audit.PAIRS)
            complete = len(verified) == count
            status = ('full' if complete else 'mismatch' if any(r['verification']=='mismatch' for r in runs)
                      else 'partial' if verified else 'unconfirmed')
            row = dict(window=window, lags=lags, verification=status, verified_pairs=len(verified),
                       validation_rank=None)
            for metric in METRICS:
                values = [r[metric] for r in runs if r[metric] is not None]
                row[metric] = mean(values) if len(values) == count else None
            cells.append(row)
        control = cells[0]
        for row in cells:
            for metric in METRICS:
                row[metric+'_improvement_pct'] = (improvement(row[metric], control[metric])
                                                   if row['verification'] == 'full' else None)
        if all(cell['verification'] == 'full' for cell in cells):
            cells.sort(key=lambda row: (row['val_mse'], row['window']))
            for rank, row in enumerate(cells, 1):
                row['validation_rank'] = rank
        table.extend(cells)
    return table


def analyze(results, control_root, data, output):
    if output.exists():
        raise FileExistsError(f'Report already exists: {output}; choose another --output')
    rows = collect(results, control_root, data)
    table = summarize(rows)
    output.mkdir(parents=True, exist_ok=False)
    audit.prior.write_table(output/'cells.csv', table)
    audit.prior.write_table(output/'paired_runs.csv', rows)
    print('Лаги | Место Val | Окно | Val MSE | Test MSE | Test MAE | Выигрыш к mean40 Val/Test MSE/Test MAE, % | Сверка')
    print('--- | --- | --- | --- | --- | --- | --- | ---')
    for row in table:
        deltas = ' / '.join(fmt(row[key+'_improvement_pct'], 2) for key in METRICS)
        print(' | '.join([row['lags'], str(row['validation_rank'] or '—'), f'mean{row["window"]}',
                          *(fmt(row[key]) for key in METRICS), deltas,
                          f'{row["verification"]} {row["verified_pairs"]}/{len(audit.PAIRS)}']))
    statuses = Counter(row['verification'] for row in rows)
    print('Сверка по прогонам: ' + ', '.join(f'{status}={count}' for status, count in sorted(statuses.items())))
    print(f'Причины и парные разницы: {output/"paired_runs.csv"}')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=audit.shared.DATA)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--control-root', type=Path, default=audit.RESULTS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        analyze(args.results, args.control_root, args.data,
                args.output or args.results/'diagnostics'/'mean_lag_extension')
    except audit.AUDIT_ERRORS as error:
        parser.exit(1, f'STOP: {error}\n')
