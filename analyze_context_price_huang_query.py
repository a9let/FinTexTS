"""CPU artifact comparison: recurrent vs constant group queries and saved last."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import analyze_context_price_han_head_ablation as heads
from analyze_persistence_baseline import Run
from forecasting_task.models.GroupHuangQuery import HUANG_MODES
from forecasting_task.models.GroupText import TEXT_GROUPS
from run_han_head_ablation_queue import file_hash


DEFAULT_RESULTS = heads.han.RUNS / 'context_price_huang_query_10_v1'
DEFAULT_REFERENCE = heads.DEFAULT_RESULTS
LAST = 'last_matched'
MODES = (*HUANG_MODES, LAST)
PAIRS = tuple(itertools.combinations(MODES, 2))
TICKERS, SEEDS, METRICS = heads.TICKERS, heads.SEEDS, heads.METRICS


class ReplayCache(heads.han.prior.ReplayCache):
    def __init__(self, data_root):
        super().__init__(data_root)
        self.hashes = {}

    def parquet_hash(self, ticker):
        if ticker not in self.hashes:
            self.hashes[ticker] = file_hash(self.data_root / (ticker + '.parquet'))
        return self.hashes[ticker]


def inspect(run, cache):
    result = heads.inspect(run, cache)
    row, metadata, _, _ = result
    if not row['comparable']:
        return result
    try:
        args = metadata['args']
        if args.get('seed') != run.seed:
            raise ValueError('Saved seed differs')
        digest = args['source_data_evidence'].get('parquet_sha256')
        if digest is None:
            raise KeyError('Missing source Parquet SHA')
        if digest != cache.parquet_hash(run.ticker):
            raise ValueError('Source Parquet differs')
        summary = json.loads((run.output / 'summary.json').read_text())
        checkpoint = torch.load(run.output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
        for key in (('huang_initialization', 'initial_prediction_evidence') if run.variant in HUANG_MODES else ('initial_prediction_evidence',)):
            if any(key not in source for source in (args, summary, checkpoint)):
                raise KeyError(f'Missing {key}')
            if args[key] != summary[key] or args[key] != checkpoint[key]:
                raise ValueError(f'Inconsistent {key}')
        proof = args['initial_prediction_evidence']
        if (not proof or proof.get('matches_last_matched_formula') is not True or proof.get('max_abs_difference') != 0
                or proof.get('input_end_date') != metadata['datasets']['val'].window_end_date(0)):
            raise ValueError('Initial last-formula equality unconfirmed')
        digests = [proof.get('prediction_sha256', '')]
        if run.variant in HUANG_MODES:
            if set(args['huang_initialization']) != set(TEXT_GROUPS):
                raise ValueError('Missing per-group common news initialization')
            digests.extend(args['huang_initialization'].values())
        for digest in digests:
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid initialization/prediction fingerprint')
    except (OSError, KeyError) as error:
        row.update(comparable=False, match_verification='partial', evidence_reason=str(error))
    except (TypeError, ValueError, RuntimeError) as error:
        row.update(comparable=False, match_verification='mismatch', evidence_reason=str(error))
    return result


def compare_pair(left, right):
    epochs = heads.compare_common(left, right)
    a, b = left[1]['args'], right[1]['args']
    for key in ('d_model', 'dropout', 'novelty_policy', 'sampling_policy', 'gate_structure'):
        if key not in a or key not in b:
            raise KeyError(f'Missing common setting: {key}')
        if a[key] != b[key]:
            raise ValueError(f'Common setting differs: {key}')
    if a['common_initialization'] != b['common_initialization']:
        raise ValueError('Initial FiLM/current heads/price gate differ')
    if a['initial_prediction_evidence'] != b['initial_prediction_evidence']:
        raise ValueError('Actual initial predictions or dates differ')
    if left[0]['variant'] in HUANG_MODES and right[0]['variant'] in HUANG_MODES:
        if a['code_commit'] != b['code_commit']:
            raise ValueError('Code SHA differs between new query modes')
        if a['huang_initialization'] != b['huang_initialization']:
            raise ValueError('Initial shared U/V, LSTM cells or correction heads differ')
    return epochs


def verify_pair(left, right):
    a, b = left[0], right[0]
    row = {'left': a['variant'], 'right': b['variant'], 'ticker': a['ticker'], 'seed': a['seed'],
           'window_verification': 'not_checked', 'common_epochs': None, 'reason': ''}
    if not a['comparable'] or not b['comparable']:
        statuses = [r['match_verification'] for r in (a, b)]
        row['window_verification'] = 'mismatch' if 'mismatch' in statuses else 'partial' if any(s in ('full', 'partial') for s in statuses) else 'not_checked'
        row['reason'] = json.dumps([{k: r.get(k) for k in ('variant', 'match_verification', 'missing_fields', 'mismatches', 'evidence_missing', 'evidence_reason')}
                                    for r in (a, b) if not r['comparable']])
    else:
        try:
            row['common_epochs'] = compare_pair(left, right)
            row['window_verification'] = 'full'
        except KeyError as error:
            row.update(window_verification='partial', reason=str(error))
        except (ValueError, TypeError) as error:
            row.update(window_verification='mismatch', reason=str(error))
    return row


def aggregate(rows, verification):
    by_seed, across, by_ticker, pairs, per_run_pairs, wins = [], [], [], [], [], []
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
            by_ticker.append({'variant': mode, 'ticker': ticker,
                              **{m: heads.han.prior.mean_available([r[m] for r in group], 3) for m in METRICS}})
    lookup = {(r['variant'], r['ticker']): r for r in by_ticker}
    runs = {(r['variant'], r['ticker'], r['seed']): r for r in rows}
    for check in verification:
        for metric in METRICS:
            a = runs[check['left'], check['ticker'], check['seed']][metric]
            b = runs[check['right'], check['ticker'], check['seed']][metric]
            valid = check['window_verification'] == 'full' and a is not None and b is not None
            per_run_pairs.append({**check, 'metric': metric, 'left_error': a, 'right_error': b,
                                  'difference': a - b if valid else None})
    for left, right in PAIRS:
        for metric in METRICS:
            differences = []
            for ticker in TICKERS:
                checks = [r for r in verification if (r['left'], r['right'], r['ticker']) == (left, right, ticker)]
                a, b = lookup[left, ticker][metric], lookup[right, ticker][metric]
                valid = len(checks) == 3 and all(r['window_verification'] == 'full' for r in checks) and a is not None and b is not None
                difference = a - b if valid else None
                pairs.append({'left': left, 'right': right, 'ticker': ticker, 'metric': metric, 'comparable': valid,
                              'left_seed_mean': a, 'right_seed_mean': b, 'difference': difference,
                              'improvement_pct': 100 * (b - a) / b if valid and b else None})
                if valid:
                    differences.append(difference)
            wins.append({'left': left, 'right': right, 'metric': metric, 'eligible_tickers': len(differences),
                         'left_wins': sum(d < 0 for d in differences), 'right_wins': sum(d > 0 for d in differences), 'ties': sum(d == 0 for d in differences)})
    primary = [r for r in verification if (r['left'], r['right']) == HUANG_MODES]
    valid = len(primary) == 30 and all(r['window_verification'] == 'full' for r in primary)
    scores = {r['variant']: r['mean'] for r in across if r['metric'] == 'val_mse' and r['variant'] in HUANG_MODES}
    valid = valid and len(scores) == 2 and None not in scores.values()
    selection = {'status': 'full' if valid else 'blocked', 'metric': 'val_mse', 'scores': scores,
                 'selected': [m for m, v in scores.items() if v == min(scores.values())] if valid else [],
                 'test_used_for_selection': False, 'last_retrained': False,
                 'last_comparison': 'full' if all(r['window_verification'] == 'full' for r in verification if r['right'] == LAST) else 'partial'}
    return {'by_seed': by_seed, 'across_seeds': across, 'per_ticker': by_ticker, 'paired_by_run': per_run_pairs, 'paired_by_ticker': pairs, 'wins': wins}, selection


def analyze(results_root=DEFAULT_RESULTS, reference_root=DEFAULT_REFERENCE, data_root=heads.han.prior.DEFAULT_DATA, output_dir=None):
    results_root, reference_root = Path(results_root), Path(reference_root)
    output = Path(output_dir) if output_dir else results_root / 'diagnostics' / 'query_comparison'
    if output.exists():
        raise FileExistsError(f'Report exists: {output}')
    roots = {m: (reference_root if m == LAST else results_root) / 'full' / m for m in MODES}
    expected = set(itertools.product(TICKERS, (f'seed{s}' for s in SEEDS)))
    identities = set()
    for mode, root in roots.items():
        paths = list(root.glob('*/seed*/output/best_checkpoint.pt'))
        actual = {(p.parents[2].name, p.parents[1].name) for p in paths}
        if len(paths) != len(actual) or not actual <= expected or (mode in HUANG_MODES and actual != expected):
            raise ValueError(f'{root}: expected 30 unique ticker/seed runs (saved last may be incomplete)')
        for path in paths:
            stat = path.stat()
            if not stat.st_size or (stat.st_dev, stat.st_ino) in identities:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            identities.add((stat.st_dev, stat.st_ino))
    cache, rows, verification = ReplayCache(data_root), [], []
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        runs = {m: inspect(Run(m, ticker, seed, root / ticker / f'seed{seed}' / 'output'), cache) for m, root in roots.items()}
        verification.extend(verify_pair(runs[a], runs[b]) for a, b in PAIRS)
        rows.extend(r[0] for r in runs.values())
    tables, selection = aggregate(rows, verification)
    tables = {'per_run': rows, 'window_verification': verification, **tables}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        heads.han.prior.write_table(output / (name + '.csv'), records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    for name, records in tables.items():
        print(name.upper())
        for record in records:
            print(json.dumps(record, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Reports: {output}')
    if not selection['status'] == 'full':
        raise ValueError('Query comparison blocked; see saved verification report')
    return tables, selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--reference_root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--data_root', type=Path, default=heads.han.prior.DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    analyze(args.results_root, args.reference_root, args.data_root, args.output_dir)
