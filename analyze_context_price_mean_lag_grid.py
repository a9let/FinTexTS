"""Audit and report the nine-cell mean/price grid from saved artifacts on CPU."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

import mean_lag_grid_audit as audit
import run_han_head_ablation_queue as shared


BASELINE_MSE = 0.015328073783851212
BASELINE_ROOT = audit.RUNS / 'film_compare_50ep_v1'
MEAN5_ROOT = audit.RUNS / 'context_price_mean5_10_v1'
METRICS = audit.prior.METRICS


def improvement(candidate, reference):
    return 100 * (1 - candidate/reference) if candidate is not None and reference else None


def historical_references(cache, controls, baseline_root, mean5_root):
    rows = []
    for label, root, subdir in (('original_baseline', baseline_root, 'baseline'),
                                ('mean5', mean5_root, 'context_price_only')):
        for ticker, seed in audit.PAIRS:
            path = root / 'full' / subdir / ticker / f'seed{seed}' / 'output'
            control = controls.get((ticker, seed))
            reference = control[1] if control else None
            try:
                row, metadata, _ = audit.prior.inspect_run(
                    audit.prior.Run('original_baseline', ticker, seed, path), cache, reference)
            except audit.AUDIT_ERRORS as error:
                row = dict(ticker=ticker, seed=seed, output=str(path), window_verification='mismatch',
                           comparable=False, missing_fields='[]', mismatches=json.dumps([str(error)]),
                           **dict.fromkeys(METRICS))
                metadata = None
            row['variant'] = label
            if metadata:
                args = metadata['args']
                missing = json.loads(row['missing_fields'])
                mismatches = json.loads(row['mismatches'])
                # Historical runners use an aggregated joint linear text input or the mean5 pool.
                if label == 'original_baseline':
                    checkpoint = torch.load(path/'best_checkpoint.pt', map_location='cpu', weights_only=True)
                    state = checkpoint.get('text_model_state_dict')
                    if state is None:
                        missing.append('saved joint linear head state')
                    else:
                        weights = [tuple(value.shape) for name, value in state.items()
                                   if name.endswith('weight') and value.ndim == 2]
                        if weights != [(3, 384)]:
                            mismatches.append('historical baseline is not a single 384→3 text head')
                    if args.get('text_num_layers') != 1:
                        mismatches.append('historical baseline text_num_layers differs from 1')
                    if args.get('text_temporal_pool', 'last') != 'last' or args.get('group_gate'):
                        mismatches.append('historical baseline temporal pooling or price gating differs')
                elif args.get('text_temporal_pool') != 'mean5' or args.get('group_gate') != 'context_price_only':
                    mismatches.append('reference is not mean5 with price-only gating')
                if not control:
                    missing.append('verified mean20/[1,5] pair')
                row.update(missing_fields=json.dumps(missing), mismatches=json.dumps(mismatches),
                           window_verification='mismatch' if mismatches else 'partial' if missing else 'full',
                           comparable=not mismatches)
            rows.append(row)
    return rows


def gather(results, data, normal_root):
    commit = (results/'code_commit.txt').read_text(encoding='utf-8').strip()
    cache = audit.prior.ReplayCache(data)
    decision = audit.decisions(results)
    if decision is None:
        raise ValueError('No recorded queue decision for the mean20/[1,5] control')
    controls = {}
    normal_records = decision['records']
    if decision['source'] == 'historical':
        if Path(decision['normal_root']).resolve() != normal_root.resolve():
            raise ValueError('Historical control path differs from the recorded queue decision')
        candidate, normal_records = audit.inspect_normal_cohort(normal_root, data, cache)
        if len(candidate) == len(audit.PAIRS) and normal_records == decision['records']:
            controls = candidate
    rows, grid = [], {}
    # Confirm the control before comparing the other eight cells.
    for cell in (audit.CONTROL, *(name for name in audit.CONFIGS if name != audit.CONTROL)):
        config = audit.CONFIGS[cell]
        for ticker, seed in audit.PAIRS:
            key = ticker, seed
            historical = cell == audit.CONTROL and decision['source'] == 'historical'
            destination = (audit.normal_destination(normal_root, *key) if historical
                           else audit.grid_destination(results, cell, *key))
            row = dict(cell=cell, window=config['text_window'], lags=','.join(map(str, config['price_gate_lags'])),
                       group_weights='price_only', ticker=ticker, seed=seed, output=str(destination/'output'),
                       source='historical_xTime_normal' if historical else 'trained', verification='unconfirmed',
                       reason='', common_epochs=None, checkpoint_sha256=None, **dict.fromkeys(METRICS))
            try:
                if historical:
                    if key not in controls:
                        raise ValueError('Pinned xTime cohort failed re-audit; resume the queue with a matched control')
                    run = controls[key]
                else:
                    run = audit.checked_run(destination, config, data, commit, ticker, seed, cache)
                row.update({metric: run[0][metric] for metric in METRICS})
                row['checkpoint_sha256'] = run[2]['checkpoint_sha256']
                if cell == audit.CONTROL:
                    controls[key] = run
                if key not in controls:
                    raise ValueError('Matched control pair is unavailable or unconfirmed')
                row['common_epochs'] = audit.compare_grid_runs(run, controls[key])
                row['verification'] = 'full'
                grid[cell, ticker, seed] = row
            except audit.AUDIT_ERRORS as error:
                row.update(reason=str(error),
                           verification='unconfirmed' if isinstance(error, (OSError, KeyError)) else 'mismatch')
            rows.append(row)
    return rows, grid, cache, controls, decision, normal_records


def summarize(rows, baseline_status):
    cells = {}
    for cell, config in audit.CONFIGS.items():
        cohort = [row for row in rows if row['cell'] == cell]
        complete = len(cohort) == len(audit.PAIRS) and all(row['verification'] == 'full' for row in cohort)
        summary = dict(cell=cell, window=config['text_window'], lags=','.join(map(str, config['price_gate_lags'])),
                       group_weights='price_only', confirmed_runs=sum(row['verification']=='full' for row in cohort),
                       verification='full' if complete else 'mismatch' if any(row['verification']=='mismatch' for row in cohort) else 'unconfirmed',
                       baseline_verification=baseline_status)
        summary.update({metric: mean(row[metric] for row in cohort) if complete else None for metric in METRICS})
        summary['baseline_improvement_pct'] = improvement(summary['test_mse'], BASELINE_MSE)
        cells[cell] = summary
    for summary in cells.values():
        summary['control_improvement_pct'] = improvement(summary['test_mse'], cells[audit.CONTROL]['test_mse'])
    return cells


def fixed_slices(cells, fixed_lags):
    pairs = (
        ((audit.cell_name(before, lags), audit.cell_name(after, lags)) for lags, (before, after)
         in itertools.product(audit.LAG_SETS, zip(audit.WINDOWS, audit.WINDOWS[1:])))
        if fixed_lags else
        ((audit.cell_name(window, before), audit.cell_name(window, after)) for window, (before, after)
         in itertools.product(audit.WINDOWS, zip(audit.LAG_SETS, audit.LAG_SETS[1:])))
    )
    rows = []
    for old, new in pairs:
        a, b = cells[old], cells[new]
        row = dict(before=old, after=new, common_verified=a['verification']==b['verification']=='full')
        for metric in METRICS:
            row[metric+'_difference'] = b[metric]-a[metric] if row['common_verified'] else None
            row[metric+'_improvement_pct'] = improvement(b[metric], a[metric]) if row['common_verified'] else None
        rows.append(row)
    return rows


def ticker_results(cells, grid):
    rows = []
    for cell, ticker in itertools.product(cells, shared.TICKERS):
        pairs = [(grid.get((cell, ticker, seed)), grid.get((audit.CONTROL, ticker, seed))) for seed in shared.SEEDS]
        if not all(a and b for a, b in pairs):
            continue
        row = dict(cell=cell, ticker=ticker)
        for metric in METRICS:
            row[metric] = mean(a[metric] for a, _ in pairs)
            row['control_'+metric] = mean(b[metric] for _, b in pairs)
            row[metric+'_difference'] = row[metric] - row['control_'+metric]
        row['validation_win'] = row['val_mse_difference'] < 0
        rows.append(row)
    return rows


def baseline_verification(references):
    rows = [row for row in references if row['variant'] == 'original_baseline']
    observed = [row['test_mse'] for row in rows if row['test_mse'] is not None]
    aggregate = mean(observed) if len(observed) == len(audit.PAIRS) else None
    if any(row['window_verification'] == 'mismatch' for row in rows):
        return 'mismatch', aggregate
    if aggregate is not None and not np.isclose(aggregate, BASELINE_MSE, rtol=1e-4, atol=1e-6):
        return 'mismatch', aggregate
    return ('full' if aggregate is not None and all(row['window_verification']=='full' for row in rows)
            else 'partial'), aggregate


def reference_metrics(references, label):
    rows = [row for row in references if row['variant'] == label and row['window_verification'] in ('full', 'partial')]
    if len(rows) != len(audit.PAIRS):
        return None
    return {key: mean(row[key] for row in rows) if all(row[key] is not None for row in rows) else None
            for key in METRICS}


def fmt(value, precision=8):
    return '—' if value is None else f'{value:.{precision}f}'


def analyze(results, data, normal_root, baseline_root, mean5_root, output):
    rows, grid, cache, controls, decision, records = gather(results, data, normal_root)
    references = historical_references(cache, controls, baseline_root, mean5_root)
    baseline_status, observed = baseline_verification(references)
    baseline_reasons = Counter(reason for row in references if row['variant'] == 'original_baseline'
                               for field in ('missing_fields', 'mismatches') for reason in json.loads(row[field]))
    if observed is not None and not np.isclose(observed, BASELINE_MSE, rtol=1e-4, atol=1e-6):
        baseline_reasons['mean Test MSE differs from the supplied denominator'] = len(audit.PAIRS)
    mean5 = reference_metrics(references, 'mean5')
    cells = summarize(rows, baseline_status)
    by_ticker = ticker_results(cells, grid)
    fixed_lags, fixed_window = fixed_slices(cells, True), fixed_slices(cells, False)
    winner = min(cells.values(), key=lambda cell: cell['val_mse']) if all(
        cell['verification']=='full' for cell in cells.values()) else None
    wins = sum(row['validation_win'] for row in by_ticker if winner and row['cell']==winner['cell'])
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (('cells', list(cells.values())), ('per_run', rows), ('per_ticker', by_ticker),
                        ('fixed_lags', fixed_lags), ('fixed_window', fixed_window),
                        ('historical_references', references)):
        audit.prior.write_table(output/f'{name}.csv', table)
    details = dict(control=decision, normal_reaudit=records, original_baseline_status=baseline_status,
                   original_baseline_denominator=BASELINE_MSE, original_baseline_observed=observed,
                   original_baseline_reasons=dict(baseline_reasons), historical_mean5=mean5,
                   target_test_mse=BASELINE_MSE*.8, validation_winner=winner and winner['cell'],
                   validation_ticker_wins=wins if winner else None)
    (output/'verification.json').write_text(json.dumps(details, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('Окно | Лаги gate | Веса групп | Val MSE | Test MSE | Test MAE | К бейзлайну, % | К mean20+[1,5], % | Сверка')
    print('--- | --- | --- | --- | --- | --- | --- | --- | ---')
    for row in cells.values():
        print(' | '.join([str(row['window']), row['lags'], 'ценовой gate',
                          *(fmt(row[key]) for key in METRICS), fmt(row['baseline_improvement_pct'], 2),
                          fmt(row['control_improvement_pct'], 2), row['verification']]))
    print(f'Контроль: {decision["source"]}. {decision["reason"]}.')
    print(f'Бейзлайн: {baseline_status}, знаменатель {BASELINE_MSE}, среднее из артефактов '
          f'{fmt(observed, 12)}, цель −20%: {BASELINE_MSE*.8:.8f}.')
    if baseline_status != 'full':
        print('Процент к фиксированному знаменателю описательный. Причины сверки: historical_references.csv.')
        for reason, count in baseline_reasons.most_common(4):
            print(f'  {reason} ({count}/{len(audit.PAIRS)})')
    if mean5:
        print('Справка mean5, Val MSE / Test MSE / Test MAE: ' + ' / '.join(fmt(mean5[key]) for key in METRICS))
    for title, comparisons in (('Фиксированные лаги', fixed_lags), ('Фиксированное окно', fixed_window)):
        print(title)
        for row in comparisons:
            print(f'  {row["before"]} → {row["after"]}: Δ Val MSE {fmt(row["val_mse_difference"])}, '
                  f'выигрыш Test MSE {fmt(row["test_mse_improvement_pct"], 2)}%')
    if winner:
        print(f'Выбор по Validation MSE: {winner["cell"]}, побед по тикерам над mean20+[1,5]: {wins}/{len(shared.TICKERS)}.')
    else:
        print('Выбор по Validation MSE ожидает подтверждения всех девяти ячеек.')
    print(f'CSV и причины сверки: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=audit.RESULTS)
    parser.add_argument('--data', type=Path, default=shared.DATA)
    parser.add_argument('--normal-root', type=Path, default=audit.NORMAL_ROOT)
    parser.add_argument('--baseline-root', type=Path, default=BASELINE_ROOT)
    parser.add_argument('--mean5-root', type=Path, default=MEAN5_ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.results/'diagnostics'/'mean_lag_grid'
    try:
        if output.exists():
            raise FileExistsError(f'Report already exists: {output}; pass a new --output path')
        analyze(args.results, args.data, args.normal_root, args.baseline_root, args.mean5_root, output)
    except audit.AUDIT_ERRORS as error:
        parser.exit(1, f'STOP: {error}\n')
