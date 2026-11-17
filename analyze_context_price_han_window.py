"""Compare saved HAN-5/10 with HAN-20 and last; no training or model inference."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import analyze_context_price_han_head_ablation as heads
from analyze_persistence_baseline import Run
from forecasting_task.models.HANHeadAblation import HAN_WINDOW_MODES
from run_han_head_ablation_queue import file_hash


DEFAULT_RESULTS = heads.han.RUNS / 'context_price_han_window_10_v1'
DEFAULT_REFERENCE = heads.DEFAULT_RESULTS
HAN20, LAST = 'han_h16_linear', 'last_matched'
WINDOWS = (*HAN_WINDOW_MODES, HAN20)
MODES = (*WINDOWS, LAST)
PAIRS = tuple(itertools.combinations(MODES, 2))
METRICS, TICKERS, SEEDS = heads.METRICS, heads.TICKERS, heads.SEEDS


class ReplayCache(heads.han.prior.ReplayCache):
    def __init__(self, data_root):
        super().__init__(data_root)
        self.file_hashes = {}

    def parquet_hash(self, ticker):
        if ticker not in self.file_hashes:
            self.file_hashes[ticker] = file_hash(self.data_root / (ticker + '.parquet'))
        return self.file_hashes[ticker]


def inspect(run, cache):
    result = heads.inspect(run, cache)
    row, metadata, _, _ = result
    if row['comparable']:
        try:
            args = metadata['args']
            if args.get('seed') != run.seed:
                raise ValueError('Saved seed differs from the requested ticker/seed pair')
            recorded = args['source_data_evidence'].get('parquet_sha256')
            if recorded is None:
                row.update(comparable=False, match_verification='partial', evidence_reason='Missing source Parquet SHA')
            elif recorded != cache.parquet_hash(run.ticker):
                raise ValueError('Original Parquet file fingerprint differs')
        except OSError as error:
            row.update(comparable=False, match_verification='partial', evidence_reason=str(error))
        except (KeyError, TypeError, ValueError) as error:
            row.update(comparable=False, match_verification='mismatch', evidence_reason=str(error))
    return result


def compare_pair(left, right):
    """Only text-window length may differ among the three h16 architectures."""
    epochs = heads.compare_common(left, right)
    a, b = left[1]['args'], right[1]['args']
    for key in ('d_model', 'dropout', 'novelty_policy', 'sampling_policy', 'gate_structure'):
        if key not in a or key not in b:
            raise KeyError(f'Missing shared configuration field: {key}')
        if a[key] != b[key]:
            raise ValueError(f'Shared configuration differs: {key}')
    if left[0]['variant'] in WINDOWS and right[0]['variant'] in WINDOWS:
        if a['head_ablation_initialization'] != b['head_ablation_initialization']:
            raise ValueError('Initial h16 BiGRU/projection/HAN attention parameters differ')
        if a['common_initialization']['heads_sha256'] != b['common_initialization']['heads_sha256']:
            raise ValueError('Initial direct Linear(32,3) heads differ')
    if left[0]['variant'] in HAN_WINDOW_MODES and right[0]['variant'] in HAN_WINDOW_MODES:
        if a['code_commit'] != b['code_commit']:
            raise ValueError('Code SHA differs between new window conditions')
    return epochs


def pair_record(left, right):
    a, b = left[0], right[0]
    record = {'left': a['variant'], 'right': b['variant'], 'ticker': a['ticker'], 'seed': a['seed'],
              'window_verification': 'not_checked', 'common_epochs': None, 'reason': '', 'confirmed': ''}
    if not a['comparable'] or not b['comparable']:
        statuses = [r['match_verification'] for r in (a, b)]
        record['window_verification'] = ('mismatch' if 'mismatch' in statuses else
                                         'partial' if 'partial' in statuses or 'full' in statuses else 'not_checked')
        record['reason'] = json.dumps([{key: r.get(key) for key in ('variant', 'match_verification', 'missing_fields',
                                                                  'mismatches', 'evidence_missing', 'evidence_reason')}
                                      for r in (a, b) if not r['comparable']])
    else:
        try:
            record['common_epochs'] = compare_pair(left, right)
            record['window_verification'] = 'full'
            record['confirmed'] = ('original Parquet and normalized source text/prices/dates; validation/test price windows, '
                                   'targets and dates; train-only scaler and split; FiLM/gate initial parameters; actual '
                                   'batch orders on common epochs' +
                                   ('; h16 encoder and direct heads initial parameters' if a['variant'] in WINDOWS and b['variant'] in WINDOWS else ''))
        except KeyError as error:
            record.update(window_verification='partial', reason=str(error))
        except (TypeError, ValueError) as error:
            record.update(window_verification='mismatch', reason=str(error))
    return record


def aggregate(rows, verification):
    by_seed, across, per_ticker, paired, wins = [], [], [], [], []
    for mode in MODES:
        for seed in SEEDS:
            group = [r for r in rows if r['variant'] == mode and r['seed'] == seed]
            by_seed.append({'variant': mode, 'seed': seed, 'verified_runs': sum(r['comparable'] for r in group),
                            **{m: heads.han.prior.mean_available([r[m] for r in group], 10) for m in METRICS}})
        for metric in METRICS:
            values = [r[metric] for r in by_seed if r['variant'] == mode]
            across.append({'variant': mode, 'metric': metric, 'mean': heads.han.prior.mean_available(values, 3),
                           'sample_std': float(np.std(values, ddof=1)) if None not in values else None, 'std_ddof': 1})
        for ticker in TICKERS:
            group = [r for r in rows if r['variant'] == mode and r['ticker'] == ticker]
            per_ticker.append({'variant': mode, 'ticker': ticker,
                               **{m: heads.han.prior.mean_available([r[m] for r in group], 3) for m in METRICS}})
    lookup = {(r['variant'], r['ticker']): r for r in per_ticker}
    for left, right in PAIRS:
        for metric in METRICS:
            differences = []
            for ticker in TICKERS:
                checks = [r for r in verification if (r['left'], r['right'], r['ticker']) == (left, right, ticker)]
                a, b = lookup[left, ticker][metric], lookup[right, ticker][metric]
                valid = len(checks) == 3 and all(r['window_verification'] == 'full' for r in checks) and a is not None and b is not None
                difference = a - b if valid else None
                paired.append({'left': left, 'right': right, 'ticker': ticker, 'metric': metric, 'comparable': valid,
                               'left_seed_mean': a, 'right_seed_mean': b, 'difference': difference,
                               'improvement_pct': 100 * (b - a) / b if valid and b else None})
                if valid:
                    differences.append(difference)
            wins.append({'left': left, 'right': right, 'metric': metric, 'eligible_tickers': len(differences),
                         'left_wins': sum(d < 0 for d in differences), 'right_wins': sum(d > 0 for d in differences),
                         'ties': sum(d == 0 for d in differences)})
    scores = {r['variant']: r['mean'] for r in across if r['variant'] in WINDOWS and r['metric'] == 'val_mse'}
    primary = [r for r in verification if (r['left'], r['right']) == HAN_WINDOW_MODES]
    primary_ok = len(primary) == 30 and all(r['window_verification'] == 'full' for r in primary)
    historical = [r for r in verification if r['right'] == HAN20]
    historical_ok = len(historical) == 60 and all(r['window_verification'] == 'full' for r in historical)
    eligible = list(WINDOWS if historical_ok else HAN_WINDOW_MODES) if primary_ok else []
    eligible = [mode for mode in eligible if scores[mode] is not None]
    selection = {'status': 'full' if primary_ok and historical_ok else 'partial' if primary_ok else 'blocked',
                 'metric': 'val_mse', 'scores': scores, 'eligible_windows': eligible,
                 'selected_among_confirmed_windows': [m for m in eligible if scores[m] == min(scores[e] for e in eligible)],
                 'test_used_for_selection': False, 'old_controls_retrained': False,
                 'limitation': '' if historical_ok else 'HAN-20 comparison unconfirmed; no strict choice across all three window lengths'}
    return {'by_seed': by_seed, 'across_seeds': across, 'per_ticker': per_ticker, 'paired_by_ticker': paired, 'wins': wins}, selection


def analyze(results_root=DEFAULT_RESULTS, reference_root=DEFAULT_REFERENCE,
            data_root=heads.han.prior.DEFAULT_DATA, output_dir=None):
    results_root, reference_root = Path(results_root), Path(reference_root)
    output = Path(output_dir) if output_dir else results_root / 'diagnostics' / 'window_comparison'
    if output.exists():
        raise FileExistsError(f'Report exists: {output}')
    roots = {mode: (results_root if mode in HAN_WINDOW_MODES else reference_root) / 'full' / mode for mode in MODES}
    expected = set(itertools.product(TICKERS, (f'seed{s}' for s in SEEDS)))
    physical, inventories = set(), []
    for mode, root in roots.items():
        paths = list(root.glob('*/seed*/output/best_checkpoint.pt'))
        actual = {(p.parents[2].name, p.parents[1].name) for p in paths}
        if len(actual) != len(paths) or not actual <= expected or (mode in HAN_WINDOW_MODES and actual != expected):
            raise ValueError(f'{root}: unexpected, missing or duplicate ticker/seed checkpoints')
        for path in paths:
            stat = path.stat()
            identity = stat.st_dev, stat.st_ino
            if not stat.st_size or identity in physical:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            physical.add(identity)
        inventories.append({'variant': mode, 'checkpoints': len(paths), 'expected': 30,
                            'status': 'full' if actual == expected else 'partial' if actual else 'not_checked'})
    cache, rows, verification = ReplayCache(data_root), [], []
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        runs = {mode: inspect(Run(mode, ticker, seed, root / ticker / f'seed{seed}' / 'output'), cache) for mode, root in roots.items()}
        verification.extend(pair_record(runs[a], runs[b]) for a, b in PAIRS)
        rows.extend(item[0] for item in runs.values())
    tables, selection = aggregate(rows, verification)
    tables = {'per_run': rows, 'inventory': inventories, 'window_verification': verification, **tables}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        heads.han.prior.write_table(output / (name + '.csv'), records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    for name in ('window_verification', 'by_seed', 'across_seeds', 'per_ticker', 'paired_by_ticker', 'wins'):
        print(name.upper())
        for record in tables[name]:
            print(json.dumps(record, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Reports: {output}')
    if selection['status'] == 'blocked':
        raise ValueError('HAN-5/10 comparison blocked; see saved evidence report')
    return tables, selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--reference_root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--data_root', type=Path, default=heads.han.prior.DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    analyze(args.results_root, args.reference_root, args.data_root, args.output_dir)
