"""HAN temporal/uniform and strictly verified saved controls, using CPU artifacts."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import analyze_context_price_group_gru_attention as prior
from analyze_persistence_baseline import Run
from forecasting_task.experiment_evidence import source_data_evidence
from forecasting_task.models.GroupHANTemporal import HAN_MODES


RUNS = prior.RUNS
DEFAULT_RESULTS = RUNS / 'context_price_han_temporal_10_v1'
DEFAULT_REFERENCE = RUNS / 'context_price_group_gru_attention_10_v1'
VARIANTS = (*HAN_MODES, 'last', 'group_gru_attention')
METRICS = prior.METRICS
TICKERS, SEEDS = prior.TICKERS, prior.SEEDS


def provenance(output, metadata):
    """Evidence absent from historical runs remains explicitly unconfirmed."""
    summary = json.loads((output / 'summary.json').read_text())
    checkpoint = torch.load(output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
    args = metadata['args']
    missing = []
    for key in ('common_initialization', 'source_data_evidence'):
        values = [source.get(key) for source in (args, summary, checkpoint)]
        if any(value is None for value in values):
            missing.append(key)
        elif any(value != values[0] for value in values[1:]):
            raise ValueError(f'Inconsistent {key} between args/summary/checkpoint')
    if args.get('text_temporal_pool') in HAN_MODES:
        values = [source.get('temporal_initialization_sha256') for source in (args, summary, checkpoint)]
        if any(value is None for value in values):
            missing.append('initial HAN BiGRU/bridge/scorer parameters')
        elif (any(value != values[0] for value in values[1:]) or len(values[0]) != 64
              or any(c not in '0123456789abcdef' for c in values[0])):
            raise ValueError('Initial HAN temporal parameter fingerprint mismatch')
    if args.get('source_data_evidence') is not None:
        current = source_data_evidence(metadata['datasets'])
        for flag in ('train', 'val', 'test'):
            if args['source_data_evidence'].get(flag) != current[flag]:
                raise ValueError('Original group text/price/date source data mismatch')
    initial = args.get('common_initialization')
    if initial is not None:
        for key in ('film_sha256', 'heads_sha256', 'price_gate_sha256'):
            digest = initial.get(key, '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'Invalid common initialization fingerprint: {key}')
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines() if line.strip()]
    if not history or [r['epoch'] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError('Invalid training history epoch sequence')
    orders = [r.get('train_batch_order') for r in history]
    if any(order is None for order in orders) or 'train_batch_orders' not in summary or 'train_batch_orders' not in checkpoint:
        missing.append('actual train batch order by epoch')
    else:
        if summary['train_batch_orders'] != orders or checkpoint['train_batch_orders'] != orders[:checkpoint['epoch']]:
            raise ValueError('Saved batch order history/checkpoint mismatch')
        if 'epochs_completed' not in summary:
            missing.append('epochs_completed')
        elif len(history) != summary['epochs_completed']:
            raise ValueError('History length and completed epochs disagree')
        for order in orders:
            digest = order.get('sha256', '')
            if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                    or order.get('windows', 0) <= 0 or order.get('batches', 0) <= 0
                    or order.get('batch_size') != args['batch_size'] or order.get('drop_last') is not True):
                raise ValueError('Invalid actual train batch order evidence')
    source = args.get('source_data_evidence')
    semantic_source = {flag: source.get(flag) for flag in ('train', 'val', 'test')} if source else None
    return {'initialization': initial, 'source': semantic_source, 'orders': orders}, missing


def inspect(run, cache):
    # Compare across code SHAs through saved common-state and data evidence.
    row, metadata, payload = prior.inspect_run(run, cache, reference=None)
    row.update(match_verification=row['window_verification'], evidence_missing='', evidence_reason='', common_epochs=None)
    evidence = None
    if row['comparable']:
        try:
            evidence, missing = provenance(run.output, metadata)
            row['evidence_missing'] = json.dumps(missing)
            if missing or row['window_verification'] != 'full':
                row.update(match_verification='partial', comparable=False)
        except OSError as error:
            row.update(match_verification='partial', comparable=False, evidence_reason=str(error))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            row.update(match_verification='mismatch', comparable=False, evidence_reason=str(error))
    return row, metadata, payload, evidence


def compare_evidence(left, right):
    if left[1]['evidence'] != right[1]['evidence']:
        raise ValueError('Input/target/date window evidence differs')
    prior.check_scalers(left[1]['scaler'], right[1]['scaler'], 'paired conditions')
    if left[1]['split'] != right[1]['split']:
        raise ValueError('Split policy differs')
    for key in (*prior.COMMON, *prior.PRICE_FIELDS):
        if left[1]['args'].get(key) != right[1]['args'].get(key):
            raise ValueError(f'Common setting differs: {key}')
    left_args, right_args = left[1]['args'], right[1]['args']
    if left_args.get('text_temporal_pool') in HAN_MODES and right_args.get('text_temporal_pool') in HAN_MODES:
        if left_args.get('code_commit') != right_args.get('code_commit'):
            raise ValueError('HAN temporal/uniform code SHA differs')
        if left_args.get('temporal_initialization_sha256') != right_args.get('temporal_initialization_sha256'):
            raise ValueError('Initial HAN BiGRU/bridge/scorer parameters differ')
    a, b = left[3], right[3]
    for key in ('initialization', 'source'):
        if a[key] != b[key]:
            raise ValueError(f'Common {key} evidence differs')
    common_epochs = min(len(a['orders']), len(b['orders']))
    if a['orders'][:common_epochs] != b['orders'][:common_epochs]:
        raise ValueError('Actual training batch order differs')
    return common_epochs


def aggregate(rows):
    by_seed, across, by_ticker, pairs, wins = [], [], [], [], []
    for mode in VARIANTS:
        for seed in SEEDS:
            group = [r for r in rows if r['variant'] == mode and r['seed'] == seed]
            eligible = len(group) == 10 and all(r['comparable'] for r in group)
            by_seed.append({'variant': mode, 'seed': seed, 'eligible_runs': sum(r['comparable'] for r in group),
                            **{m: prior.mean_available([r[m] for r in group], 10) if eligible else None for m in METRICS}})
        for metric in METRICS:
            values = [r[metric] for r in by_seed if r['variant'] == mode]
            across.append({'variant': mode, 'metric': metric, 'mean': prior.mean_available(values, 3),
                           'sample_std': float(np.std(values, ddof=1)) if None not in values else None, 'std_ddof': 1})
        for ticker in TICKERS:
            group = [r for r in rows if r['variant'] == mode and r['ticker'] == ticker]
            eligible = len(group) == 3 and all(r['comparable'] for r in group)
            by_ticker.append({'variant': mode, 'ticker': ticker, 'comparable': eligible,
                              **{m: prior.mean_available([r[m] for r in group], 3) if eligible else None for m in METRICS}})
    lookup = {(r['variant'], r['ticker']): r for r in by_ticker}
    comparisons = [(HAN_MODES[0], HAN_MODES[1]), *((mode, control) for mode in HAN_MODES for control in ('last', 'group_gru_attention'))]
    for left, right in comparisons:
        for metric in METRICS:
            eligible = []
            for ticker in TICKERS:
                a, b = lookup[left, ticker], lookup[right, ticker]
                comparable = a['comparable'] and b['comparable'] and a[metric] is not None and b[metric] is not None
                difference = a[metric] - b[metric] if comparable else None
                pairs.append({'left': left, 'right': right, 'ticker': ticker, 'metric': metric, 'comparable': comparable,
                              'left_seed_mean': a[metric], 'right_seed_mean': b[metric], 'difference': difference,
                              'improvement_pct': 100 * (b[metric] - a[metric]) / b[metric] if comparable and b[metric] else None})
                if comparable:
                    eligible.append(difference)
            wins.append({'left': left, 'right': right, 'metric': metric, 'eligible_tickers': len(eligible),
                         'left_wins': sum(v < 0 for v in eligible), 'ties': sum(v == 0 for v in eligible)})
    primary = [r for r in rows if r['variant'] in HAN_MODES]
    valid = len(primary) == 60 and all(r['comparable'] for r in primary)
    scores = {r['variant']: r['mean'] for r in across if r['metric'] == 'val_mse' and r['variant'] in (*HAN_MODES, 'last') and r['mean'] is not None}
    selection = {'status': 'full' if valid else 'blocked', 'metric': 'val_mse', 'scores': scores,
                 'selected': [mode for mode, value in scores.items() if value == min(scores.values())] if valid else [],
                 'last_confirmed': 'last' in scores, 'test_used_for_selection': False,
                 'strong_movements_used_for_selection': False,
                 'gru_reference': 'additional comparison; date weights and pooled representations both differ'}
    return {'by_seed': by_seed, 'across_seeds': across, 'per_ticker': by_ticker, 'paired_by_ticker': pairs, 'wins': wins}, selection


def analyze(results_root=DEFAULT_RESULTS, reference_root=DEFAULT_REFERENCE, last_root=None,
            data_root=prior.DEFAULT_DATA, output_dir=None):
    results_root, reference_root = Path(results_root), Path(reference_root)
    output = Path(output_dir) if output_dir else results_root / 'diagnostics' / ('matched_last' if last_root else 'saved_last')
    if output.exists():
        raise FileExistsError(f'Report directory exists: {output}')
    roots = {mode: results_root / 'full' / mode for mode in HAN_MODES}
    roots['last'] = (Path(last_root) if last_root else reference_root) / 'full' / 'last'
    roots['group_gru_attention'] = reference_root / 'full' / 'group_gru_attention'
    expected = {(t, f'seed{s}') for t, s in itertools.product(TICKERS, SEEDS)}
    identities = set()
    for mode in HAN_MODES:
        paths = list(roots[mode].glob('*/seed*/output/best_checkpoint.pt'))
        if len(paths) != 30 or {(p.parents[2].name, p.parents[1].name) for p in paths} != expected:
            raise ValueError(f'{roots[mode]}: expected exactly 30 unique ticker/seed runs')
        for path in paths:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if not stat.st_size or identity in identities:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            identities.add(identity)
    cache = prior.ReplayCache(data_root)
    rows, movements, issues = [], [], []
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        runs = {mode: inspect(Run(mode, ticker, seed, root / ticker / f'seed{seed}' / 'output'), cache) for mode, root in roots.items()}
        anchor = runs[HAN_MODES[0]]
        for mode in VARIANTS[1:]:
            candidate = runs[mode]
            if anchor[0]['comparable'] and candidate[0]['comparable']:
                try:
                    candidate[0]['common_epochs'] = compare_evidence(anchor, candidate)
                except (ValueError, KeyError, TypeError) as error:
                    candidate[0].update(comparable=False, match_verification='mismatch', evidence_reason=str(error))
        if not all(runs[mode][0]['comparable'] for mode in HAN_MODES):
            for mode in HAN_MODES:
                runs[mode][0]['comparable'] = False
        if all(runs[mode][0]['comparable'] for mode in (*HAN_MODES, 'last')):
            try:
                predictions = {mode: item[2] for mode, item in runs.items() if item[0]['comparable']}
                records, errors = prior.movement_records(ticker, seed, runs['last'][1], predictions,
                                                        required_variants=('last', *HAN_MODES))
                movements.extend(records)
                issues.extend(errors)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                issues.append({'ticker': ticker, 'seed': seed, 'reason': str(error)})
        else:
            issues.append({'ticker': ticker, 'seed': seed, 'reason': 'Strong-movement comparison awaits a strictly matched last control'})
        rows.extend(item[0] for item in runs.values())
    tables, selection = aggregate(rows)
    tables = {'per_run': rows, **tables, 'strong_movements': movements, 'strong_movement_issues': issues,
              'window_verification': [{k: row[k] for k in ('variant', 'ticker', 'seed', 'window_verification', 'match_verification',
                                                        'common_epochs', 'missing_fields', 'mismatches', 'evidence_missing', 'evidence_reason')} for row in rows]}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        prior.write_table(output / (name + '.csv'), records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    for name in ('window_verification', 'by_seed', 'across_seeds', 'wins'):
        print(name.upper())
        for row in tables[name]:
            print(json.dumps(row, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Reports: {output}')
    if selection['status'] == 'blocked':
        raise ValueError('HAN temporal/uniform comparison blocked; see saved evidence report')
    return tables, selection


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--reference_root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--last_root', type=Path)
    parser.add_argument('--data_root', type=Path, default=prior.DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    analyze(args.results_root, args.reference_root, args.last_root, args.data_root, args.output_dir)


if __name__ == '__main__':
    cli()
