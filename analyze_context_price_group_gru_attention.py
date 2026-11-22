"""Matched last/Group-GRU reports and artifact checks on CPU, without training."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_context_price_only import COMMON, PRICE_FIELDS, TICKERS, SEEDS
from analyze_persistence_baseline import Run, load_metadata, model_metrics, check_scalers
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.models.GroupText import TEXT_GROUPS
from forecasting_task.models.GroupGRUAttention import temporal_structure
from forecasting_task.window_evidence import SPLIT_POLICY, window_evidence, check_saved_test_windows, check_validation_dates


RUNS = Path('/auto/datasets/kondyurin/FinTexTS/runs')
DEFAULT_RESULTS = RUNS / 'context_price_group_gru_attention_10_v1'
DEFAULT_BASELINE = RUNS / 'film_compare_50ep_v1'
DEFAULT_DATA = Path('/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts')
VARIANTS = ('last', 'group_gru_attention', 'original_baseline')
METRICS = ('val_mse', 'test_mse', 'test_mae')


class ReplayCache:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.datasets = {}

    def get(self, ticker, args):
        extended = (args['text_window'] if args.get('text_family') == 'mean_price_grid'
                    and args['text_window'] > args['seq_len'] else None)
        key = (ticker, args['seq_len'], args['label_len'], args['pred_len'], tuple(args['used_col_prefixes']), extended)
        if key not in self.datasets:
            tickers = pd.read_parquet(self.data_root / (ticker + '.parquet'), columns=['ticker'])['ticker']
            if tickers.isna().any() or set(tickers) != {ticker}:
                raise ValueError(f'Data file contains a different ticker: {ticker}')
            self.datasets[key] = {flag: ContextNoveltyDataset(Dataset_Custom(
                args['seq_len'], args['label_len'], args['pred_len'], flag, args['used_col_prefixes'],
                root_path=str(self.data_root), data_path=ticker + '.parquet'),
                extended_text_window=extended) for flag in ('train', 'val', 'test')}
        return self.datasets[key]


def configuration_issues(run, summary, args, checkpoint, reference):
    if args.get('text_temporal_pool') == 'group_gate_experiment':
        from group_gate_audit import configuration_issues as experiment_issues
        return experiment_issues(run, summary, args, checkpoint)
    missing, mismatch = [], []
    expected = {**COMMON, 'model_type': 'FiLM', 'used_col_prefixes': list(TEXT_GROUPS), 'text_weight': .1}
    if reference is not None:
        expected.update({key: reference[key] for key in PRICE_FIELDS})
    for key, value in expected.items():
        if key not in args:
            missing.append('args.' + key)
        else:
            actual = str(args[key]).lower() if key == 'model_type' else args[key]
            expected_value = str(value).lower() if key == 'model_type' else value
            if actual != expected_value:
                mismatch.append(f'args.{key}={args[key]!r}; expected {value!r}')
    for key in PRICE_FIELDS:
        if key not in args and 'args.' + key not in missing:
            missing.append('args.' + key)
    if Path(str(args.get('data_path', '')).replace('\\', '/')).name != run.ticker + '.parquet':
        mismatch.append('data_path/ticker mismatch')
    if 'fnspid' in str(args.get('data_path', '')).lower():
        mismatch.append('forward-fill preprocessing differs')
    for key, value in (('category_pool', None), ('price_context_lags', [1, 5]),
                       ('gate_text_history_mode', 'none'), ('text_history_mode', 'last')):
        if args.get(key, value) != value:
            mismatch.append(f'unsupported {key}')
    if run.variant != 'original_baseline':
        parameters = {}
        if run.variant in ('huang_recurrent_query', 'huang_constant_query', 'huang_recurrent_replace'):
            from forecasting_task.models.GroupHuangQuery import huang_options
            parameters = huang_options(args)
            for source_name, source in (('args', args), ('summary', summary), ('checkpoint', checkpoint),
                                         ('text_model_config', args.get('text_model_config', {}))):
                if huang_options(source) != parameters:
                    mismatch.append(f'{source_name}.huang_parameters mismatch')
        for source_name, source in (('args', args), ('summary', summary), ('checkpoint', checkpoint)):
            for key, value in (('text_temporal_pool', run.variant), ('group_gate', 'context_price_only'),
                               ('fixed_alpha', .1), ('text_weight_mode', 'fixed'), ('split_policy', SPLIT_POLICY),
                               ('temporal_pool_structure', temporal_structure(run.variant, **parameters))):
                if key not in source:
                    missing.append(f'{source_name}.{key}')
                elif source[key] != value:
                    mismatch.append(f'{source_name}.{key} mismatch')
            if source.get('code_commit') != args.get('code_commit') or not source.get('code_commit'):
                mismatch.append(f'{source_name}.code_commit missing/inconsistent')
        config = args.get('text_model_config', {})
        for key, value in (('text_temporal_pool', run.variant), ('group_gate', 'context_price_only'),
                           ('pred_len', args['pred_len']), ('num_layers', 1), ('initial_text_weight', .1),
                           ('fusion_mode', 'blend'), ('used_col_prefixes', list(TEXT_GROUPS)),
                           ('temporal_pool_structure', temporal_structure(run.variant, **parameters))):
            if config.get(key) != value:
                mismatch.append(f'text_model_config.{key} mismatch')
        for key, value in (('text_num_layers', 1), ('fusion_mode', 'blend'),
                           ('group_weight_mode', 'context_price_only'), ('sampling_policy', 'independent_train_generator')):
            if args.get(key) != value:
                mismatch.append(f'args.{key} mismatch')
        expected_gate = {'type': 'Linear', 'in_features': 18, 'out_features': 5, 'parameters': 95, 'initialization': 'zeros'}
        if args.get('gate_structure') != expected_gate:
            mismatch.append('price gate structure mismatch')
    elif not any(key in source for source in (args, checkpoint) for key in ('split_policy', 'split_config', 'split_params')):
        missing.append('historical saved split declaration (current fixed policy replayed)')
    return missing, mismatch


def dataset_scaler(dataset):
    return {key: np.asarray(getattr(dataset.scaler, key + '_')) for key in ('mean', 'scale', 'var')}


def checked_predictions(output, expected):
    path = output / 'test_io.pt'
    check_saved_test_windows(path, expected)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    values = {}
    for key, shape in (('inputs', [expected['windows'], *expected['input_shape']]),
                       ('ground_truths', [expected['windows'], *expected['target_shape']]),
                       ('outputs', [expected['windows'], *expected['target_shape']])):
        if key not in payload or payload[key] is None:
            raise ValueError(f'Missing normalized {key}; actual keys={list(payload)}')
        value = torch.as_tensor(payload[key]).detach().cpu().double().numpy()
        if list(value.shape) != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid normalized {key} shape/values in {path}')
        values[key] = value
    return values


def inspect_run(run, cache, reference=None):
    row = {'variant': run.variant, 'ticker': run.ticker, 'seed': run.seed, 'output': str(run.output),
           'window_verification': 'not_checked', 'comparable': False,
           'missing_fields': '', 'mismatches': '', 'confirmed': '', **dict.fromkeys(METRICS)}
    metadata, payload, evidence = None, None, None
    missing, mismatch, confirmed = [], [], []
    try:
        summary = json.loads((run.output / 'summary.json').read_text(encoding='utf-8'))
        metrics, _ = model_metrics(summary)
        row.update({metric: metrics[metric] for metric in METRICS})
        summary, args, scaler, split = load_metadata(run)
        checkpoint = torch.load(run.output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
        missing, mismatch = configuration_issues(run, summary, args, checkpoint, reference['args'] if reference else None)
        missing.extend('summary.' + metric for metric in METRICS if row[metric] is None)
        if mismatch:
            raise ValueError('Saved configuration mismatch')
        if run.variant != 'original_baseline' and missing:
            raise ValueError('New matched run has incomplete saved configuration')
        datasets = cache.get(run.ticker, args)
        for dataset in datasets.values():
            check_scalers(dataset_scaler(dataset), scaler, str(run.output))
        confirmed.extend(['scaler matches train-only replay', 'current fixed split replay'])
        if reference:
            check_scalers(reference['scaler'], scaler, str(run.output))
            if reference['split'] != split:
                raise ValueError('Saved split mismatch')
            if run.variant != 'original_baseline' and args.get('code_commit') != reference['args'].get('code_commit'):
                raise ValueError('Matched modes have different code SHA')
        evidence = {flag: window_evidence(datasets[flag]) for flag in ('val', 'test')}
        if reference and evidence != reference['evidence']:
            raise ValueError('Reconstructed paired input windows/targets/dates mismatch')
        full_evidence = True
        for name, source in (('args', args), ('summary', summary), ('checkpoint', checkpoint)):
            if 'window_evidence' not in source:
                full_evidence = False
                missing.append(name + '.window_evidence (historical validation inputs/targets/dates unconfirmed)')
            elif source['window_evidence'] != evidence:
                raise ValueError(f'{name} validation/test window fingerprints mismatch')
        if full_evidence:
            confirmed.append('persisted validation/test input, target and date fingerprints')
        dates_path = run.output / 'validation_group_diagnostics.csv'
        if dates_path.is_file():
            check_validation_dates(dates_path, evidence['val'])
            confirmed.append('saved validation window dates')
        else:
            missing.append('validation_group_diagnostics.csv dates')
        if (run.output / 'test_io.pt').is_file():
            # Contradictory input/target hashes block comparisons even if predictions are missing.
            check_saved_test_windows(run.output / 'test_io.pt', evidence['test'])
            confirmed.append('saved normalized test inputs and targets')
            stored = torch.load(run.output / 'test_io.pt', map_location='cpu', weights_only=False)
            if stored.get('outputs') is None:
                missing.append(f'normalized test outputs; actual keys={list(stored)}')
            else:
                payload = checked_predictions(run.output, evidence['test'])
                errors = payload['outputs'] - payload['ground_truths']
                for key, actual in (('test_mse', np.square(errors).mean()), ('test_mae', np.abs(errors).mean())):
                    if row[key] is not None and not np.isclose(row[key], actual, rtol=1e-4, atol=1e-6):
                        raise ValueError(f'{key} saved prediction metric mismatch')
        else:
            missing.append('test_io.pt normalized windows/predictions')
        if run.variant != 'original_baseline' and (not full_evidence or payload is None):
            raise ValueError('Matched runs require saved window fingerprints and normalized predictions')
        if run.variant != 'original_baseline' and any(row[metric] is None for metric in METRICS):
            raise ValueError('Missing required matched model metrics')
        row['window_verification'] = 'partial' if missing else 'full'
        row['comparable'] = True
        metadata = {'args': args, 'scaler': scaler, 'split': split, 'evidence': evidence, 'datasets': datasets}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        if isinstance(error, (OSError, KeyError)):
            missing.append(str(error))
        else:
            mismatch.append(str(error))
            row['window_verification'] = 'mismatch'
    row.update(missing_fields=json.dumps(missing), mismatches=json.dumps(mismatch), confirmed=json.dumps(confirmed))
    return row, metadata, payload


def train_return_thresholds(train, percentile=90):
    """Original-scale Close returns from forecast windows wholly inside train."""
    returns = []
    for index in range(len(train)):
        x, y = train[index][:2]
        last = train.scaler.inverse_transform(x[-1:])[0, 3]
        future = train.scaler.inverse_transform(y[-train.pred_len:])[:, 3]
        if not np.isfinite(last) or last <= 0 or not np.isfinite(future).all():
            raise ValueError('Train return diagnostic requires finite positive last Close')
        returns.append(np.abs(future / last - 1))
    if not returns:
        raise ValueError('No complete train forecast windows for return thresholds')
    return np.percentile(np.asarray(returns), percentile, axis=0), len(returns)


def movement_records(ticker, seed, metadata, predictions, required_variants=('last', 'group_gru_attention')):
    if any(predictions.get(mode) is None for mode in required_variants):
        return [], [{'ticker': ticker, 'seed': seed, 'reason': 'matched normalized predictions unavailable'}]
    last = predictions['last']
    for mode, payload in predictions.items():
        if payload is not None and (not np.array_equal(payload['inputs'], last['inputs'])
                                    or not np.array_equal(payload['ground_truths'], last['ground_truths'])):
            raise ValueError(f'Strong-movement paired windows/targets mismatch for {mode}')
    thresholds, train_count = train_return_thresholds(metadata['datasets']['train'])
    scaler = metadata['scaler']
    close = last['inputs'][:, -1, 3] * scaler['scale'][3] + scaler['mean'][3]
    future = last['ground_truths'][:, :, 3] * scaler['scale'][3] + scaler['mean'][3]
    if np.any(close <= 0):
        raise ValueError('Test return diagnostic requires positive last Close')
    returns = np.abs(future / close[:, None] - 1)
    rows = []
    for horizon, threshold in enumerate(thresholds):
        selected = returns[:, horizon] >= threshold
        control_errors = last['outputs'][:, horizon] - last['ground_truths'][:, horizon]
        total_sse = np.square(control_errors).sum()
        selected_control_sse = np.square(control_errors[selected]).sum()
        for mode, payload in predictions.items():
            if payload is None:
                continue
            errors = payload['outputs'][:, horizon] - payload['ground_truths'][:, horizon]
            rows.append({'ticker': ticker, 'seed': seed, 'variant': mode, 'horizon': horizon + 1,
                         'train_percentile': 90, 'train_windows': train_count, 'threshold_abs_close_return': float(threshold),
                         'selected_windows': int(selected.sum()), 'total_windows': len(selected),
                         'elements': int(selected.sum() * 4), 'sse': float(np.square(errors[selected]).sum()),
                         'mse': float(np.square(errors[selected]).mean()) if selected.any() else None,
                         'last_selected_sse': float(selected_control_sse), 'last_total_sse': float(total_sse),
                         'share_of_last_sse': float(selected_control_sse / total_sse) if total_sse else None})
    return rows, []


def mean_available(values, count):
    return float(np.mean(values)) if len(values) == count and all(v is not None for v in values) else None


def aggregate(rows):
    by_seed, across, ticker_means, paired, wins, reductions = [], [], [], [], [], []
    for variant in VARIANTS:
        cohort = [row for row in rows if row['variant'] == variant]
        for ticker in TICKERS:
            group = [r for r in cohort if r['ticker'] == ticker]
            comparable = len(group) == 3 and all(r['comparable'] for r in group)
            ticker_means.append({'variant': variant, 'ticker': ticker, 'comparable': comparable,
                                 **{m: mean_available([r[m] for r in group], 3) if comparable else None for m in METRICS}})
        for seed in SEEDS:
            group = [r for r in cohort if r['seed'] == seed]
            comparable = len(group) == 10 and all(r['comparable'] for r in group)
            by_seed.append({'variant': variant, 'seed': seed, 'comparable_runs': sum(r['comparable'] for r in group),
                            **{m: mean_available([r[m] for r in group], 10) if comparable else None for m in METRICS}})
        for metric in METRICS:
            values = [r[metric] for r in by_seed if r['variant'] == variant]
            complete = None not in values
            across.append({'variant': variant, 'metric': metric, 'mean': mean_available(values, 3),
                           'sample_std': float(np.std(values, ddof=1)) if complete else None, 'std_ddof': 1})
    lookup = {(r['variant'], r['ticker']): r for r in ticker_means}
    run_lookup = {(r['variant'], r['ticker'], r['seed']): r for r in rows}
    for left, right in (('group_gru_attention', 'last'), ('group_gru_attention', 'original_baseline'), ('last', 'original_baseline')):
        for ticker in TICKERS:
            a, b = lookup[left, ticker], lookup[right, ticker]
            for metric in METRICS:
                comparable = a['comparable'] and b['comparable'] and a[metric] is not None and b[metric] is not None
                statuses = {run_lookup[v, ticker, seed]['window_verification'] for v in (left, right) for seed in SEEDS}
                paired.append({'left': left, 'right': right, 'ticker': ticker, 'metric': metric,
                               'left_mean': a[metric], 'right_mean': b[metric], 'comparable': comparable,
                               'window_verification': ('partial' if 'partial' in statuses else 'full') if comparable else 'mismatch',
                               'difference': a[metric] - b[metric] if comparable else None,
                               'improvement_pct': 100 * (b[metric] - a[metric]) / b[metric] if comparable and b[metric] else None,
                               'baseline_target_0_8': .8 * b[metric] if comparable and right == 'original_baseline' and metric == 'test_mse' else None})
        for metric in METRICS:
            group = [r for r in paired if r['left'] == left and r['right'] == right and r['metric'] == metric and r['comparable']]
            wins.append({'left': left, 'right': right, 'metric': metric, 'left_wins': sum(r['difference'] < 0 for r in group),
                         'ties': sum(r['difference'] == 0 for r in group), 'comparable_tickers': len(group)})
        eligible = [r for r in paired if r['left'] == left and r['right'] == right and r['metric'] == 'test_mse' and r['comparable']]
        complete = len(eligible) == len(TICKERS)
        a = float(np.mean([r['left_mean'] for r in eligible])) if complete else None
        b = float(np.mean([r['right_mean'] for r in eligible])) if complete else None
        reductions.append({'left': left, 'right': right, 'metric': 'test_mse', 'comparable_tickers': len(eligible),
                           'expected_tickers': len(TICKERS), 'cohort_complete': complete,
                           'left_mean': a, 'right_mean': b, 'improvement_pct': 100 * (b - a) / b if b else None,
                           'baseline_target_0_8': .8 * b if b is not None and right == 'original_baseline' else None,
                           'window_verification': ('partial' if any(r['window_verification'] == 'partial' for r in eligible) else 'full') if complete else 'not_checked',
                           'limitation': '' if complete else 'Complete matched ten-ticker cohort could not be confirmed; aggregate relative gain unavailable'})
    primary = [r for r in rows if r['variant'] != 'original_baseline']
    valid = len(primary) == 60 and all(r['comparable'] and r['window_verification'] == 'full' for r in primary)
    scores = {r['variant']: r['mean'] for r in across if r['metric'] == 'val_mse' and r['variant'] != 'original_baseline'}
    valid = valid and None not in scores.values()
    selection = {'metric': 'val_mse', 'scores': scores, 'status': 'full' if valid else 'blocked',
                 'selected': [v for v, score in scores.items() if score == min(scores.values())] if valid else [],
                 'test_used_for_selection': False, 'strong_movements_used_for_selection': False}
    return {'per_ticker': ticker_means, 'by_seed': by_seed, 'across_seeds': across,
            'paired_by_ticker': paired, 'wins': wins, 'test_mse_reductions': reductions}, selection


def inventory(results_root, baseline_root):
    roots = {mode: Path(results_root) / 'full' / mode for mode in VARIANTS[:2]}
    roots['original_baseline'] = Path(baseline_root) / 'full' / 'baseline'
    expected = set(itertools.product(TICKERS, SEEDS))
    identities = set()
    for mode in VARIANTS[:2]:
        paths = list(roots[mode].glob('*/seed*/output/best_checkpoint.pt'))
        keys = {(p.parents[2].name, p.parents[1].name) for p in paths}
        if len(paths) != 30 or keys != {(t, f'seed{s}') for t, s in expected}:
            raise ValueError(f'{roots[mode]}: expected exactly 30 unique ticker/seed checkpoints')
        for path in paths:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if not stat.st_size or identity in identities:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            identities.add(identity)
    return roots


def write_table(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record)) or ['ticker', 'seed', 'reason']
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def aggregate_movements(rows):
    result = []
    for variant, seed, horizon in itertools.product(VARIANTS, SEEDS, (1, 2, 3)):
        group = [r for r in rows if r['variant'] == variant and r['seed'] == seed and r['horizon'] == horizon]
        elements = sum(r['elements'] for r in group)
        denominator = sum(r['last_total_sse'] for r in group)
        result.append({'variant': variant, 'seed': seed, 'horizon': horizon, 'available_tickers': len(group),
                       'expected_tickers': len(TICKERS), 'selected_windows': sum(r['selected_windows'] for r in group),
                       'total_windows': sum(r['total_windows'] for r in group),
                       'mse': sum(r['sse'] for r in group) / elements if elements else None,
                       'share_of_last_sse': sum(r['last_selected_sse'] for r in group) / denominator if denominator else None,
                       'aggregation': 'sum SSE / all selected OHLC elements; horizon separately'})
    return result


def analyze(results_root=DEFAULT_RESULTS, baseline_root=DEFAULT_BASELINE, data_root=DEFAULT_DATA, output_dir=None):
    output = Path(output_dir) if output_dir else Path(results_root) / 'diagnostics' / 'comparison'
    if output.exists():
        raise FileExistsError(f'Report directory exists: {output}')
    roots = inventory(results_root, baseline_root)
    cache = ReplayCache(data_root)
    rows, movements, movement_issues = [], [], []
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        pair, reference, predictions = [], None, {}
        for mode in VARIANTS:
            run = Run(mode, ticker, seed, roots[mode] / ticker / f'seed{seed}' / 'output')
            row, metadata, payload = inspect_run(run, cache, reference)
            pair.append(row)
            if mode == 'last':
                reference = metadata
            if row['comparable']:
                predictions[mode] = payload
        primary = pair[:2]
        if not all(r['comparable'] and r['window_verification'] == 'full' for r in primary):
            for row in primary:
                row['comparable'] = False
        elif reference is not None:
            try:
                records, issues = movement_records(ticker, seed, reference, predictions)
                movements.extend(records)
                movement_issues.extend(issues)
            except (ValueError, KeyError, RuntimeError) as error:
                movement_issues.append({'ticker': ticker, 'seed': seed, 'reason': str(error)})
        rows.extend(pair)
    tables, selection = aggregate(rows)
    tables = {'per_run': rows, **tables, 'strong_movements': movements, 'strong_movements_by_seed': aggregate_movements(movements),
              'strong_movement_issues': movement_issues,
              'window_verification': [{k: r[k] for k in ('variant', 'ticker', 'seed', 'window_verification', 'confirmed', 'missing_fields', 'mismatches')} for r in rows]}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        write_table(output / f'{name}.csv', records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
    for name in ('window_verification', 'by_seed', 'across_seeds', 'wins', 'test_mse_reductions', 'strong_movements_by_seed'):
        print(name.upper())
        for record in tables[name]:
            print(json.dumps(record, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Reports: {output}')
    if selection['status'] == 'blocked':
        raise ValueError('Matched comparison blocked; see saved window_verification.csv')
    return tables, selection


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--baseline_root', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--data_root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    analyze(args.results_root, args.baseline_root, args.data_root, args.output_dir)


if __name__ == '__main__':
    cli()
