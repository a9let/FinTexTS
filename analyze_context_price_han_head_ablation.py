"""Five matched head ablations and an audited historical HAN bridge reference."""

import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import analyze_context_price_han_temporal as han
from analyze_blend_price_anchor import execution_device, film_module
from analyze_persistence_baseline import Run
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
from forecasting_task.models.GroupText import build_text_model
from forecasting_task.models.HANHeadAblation import ABLATION_MODES, head_ablation_structure


DEFAULT_RESULTS = han.RUNS / 'context_price_han_head_ablation_10_v1'
DEFAULT_REFERENCE = han.RUNS / 'context_price_han_temporal_10_v1'
BRIDGE = 'group_han_temporal'
MODES = (*ABLATION_MODES, BRIDGE)
METRICS, TICKERS, SEEDS = han.METRICS, han.TICKERS, han.SEEDS
PAIRS = (('han_h16_linear', 'han_h32_linear'), ('han_h32_mlp', 'han_h32_linear'),
         ('last_han_h32_residual', 'last_matched'), ('han_h32_linear', BRIDGE))


class BridgePartialError(ValueError):
    """The historical initialization could not be reproduced in this runtime."""


def inspect(run, cache):
    result = han.inspect(run, cache)
    row, metadata, payload, evidence = result
    if row['comparable'] and run.variant in ABLATION_MODES:
        try:
            args = metadata['args']
            summary = json.loads((run.output / 'summary.json').read_text())
            checkpoint = torch.load(run.output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
            structure = head_ablation_structure(run.variant, args['pred_len'])
            if args.get('text_mlp_structure') != structure['head_structure']:
                raise ValueError('Actual group head structure differs')
            for key in ('head_ablation_initialization', 'initial_prediction_evidence'):
                if any(key not in source for source in (args, summary, checkpoint)):
                    raise ValueError(f'Missing {key}')
                if args[key] != summary[key] or args[key] != checkpoint[key]:
                    raise ValueError(f'Inconsistent {key}')
            if run.variant != 'last_matched':
                for key in ('encoder_input_projection_sha256', 'encoder_sha256'):
                    value = args['head_ablation_initialization'].get(key, '')
                    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                        raise ValueError(f'Invalid {key}')
                if args['head_ablation_initialization']['encoder_sha256'] != args['temporal_initialization_sha256']:
                    raise ValueError('Encoder initialization records differ')
            if run.variant in ('last_matched', 'last_han_h32_residual'):
                proof = args['initial_prediction_evidence']
                if not proof or proof.get('matches_last_matched_formula') is not True or proof.get('max_abs_difference') != 0:
                    raise ValueError('Initial A/E prediction equality unconfirmed')
                digest = proof.get('prediction_sha256', '')
                if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('Missing initial A/E forecast fingerprint')
                if proof.get('input_end_date') != metadata['datasets']['val'].window_end_date(0):
                    raise ValueError('Initial A/E forecast window date differs')
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            row.update(comparable=False, match_verification='mismatch', evidence_reason=str(error))
    return row, metadata, payload, evidence


def compare_common(left, right):
    a, b = left[1], right[1]
    if a['evidence'] != b['evidence'] or a['split'] != b['split']:
        raise ValueError('Price input/target/date windows or split differ')
    han.prior.check_scalers(a['scaler'], b['scaler'], 'head ablation pair')
    for key in (*han.prior.COMMON, *han.prior.PRICE_FIELDS):
        if a['args'].get(key) != b['args'].get(key):
            raise ValueError(f'Common setting differs: {key}')
    for key in ('film_sha256', 'price_gate_sha256'):
        if left[3]['initialization'][key] != right[3]['initialization'][key]:
            raise ValueError(f'Common initial parameter differs: {key}')
    if left[3]['source'] != right[3]['source']:
        raise ValueError('Original group text/price/date sources differ')
    common_epochs = min(len(left[3]['orders']), len(right[3]['orders']))
    if left[3]['orders'][:common_epochs] != right[3]['orders'][:common_epochs]:
        raise ValueError('Actual training batch order differs')
    return common_epochs


def compare_ablations(anchor, candidate):
    epochs = compare_common(anchor, candidate)
    a, b = anchor[1]['args'], candidate[1]['args']
    if a['code_commit'] != b['code_commit']:
        raise ValueError('Code SHA differs across new ablations')
    if candidate[0]['variant'] == 'last_han_h32_residual':
        if a['common_initialization']['heads_sha256'] != b['common_initialization']['heads_sha256']:
            raise ValueError('Initial current-summary heads differ between A and E')
        if a['initial_prediction_evidence'] != b['initial_prediction_evidence']:
            raise ValueError('Initial E prediction differs from A')
    return epochs


def compare_encoders(runs):
    reference = runs['han_h32_linear'][1]['args']['head_ablation_initialization']
    for mode in ABLATION_MODES[2:]:
        args = runs[mode][1]['args']
        proof = args['head_ablation_initialization']
        if proof['encoder_input_projection_sha256'] != reference['encoder_input_projection_sha256']:
            raise ValueError(f'Shared input projection initialization differs: {mode}')
        if mode != 'han_h16_linear' and proof['encoder_sha256'] != reference['encoder_sha256']:
            raise ValueError(f'Shared h32 BiGRU/attention initialization differs: {mode}')


@lru_cache(maxsize=8)
def replay_bridge_initialization(serialized):
    """Initialize the preserved old architecture; never load trained weights."""
    config = json.loads(serialized)
    with torch.random.fork_rng(devices=[]), execution_device(torch.device('cpu')):
        torch.manual_seed(config['seed'])
        model = film_module.Model(SimpleNamespace(**config))
        text = build_text_model(config['text_model_config'], 'context_price_only')
        return {'common': common_initialization(model, text), 'full_pool': state_fingerprint(text.temporal_pool),
                'encoder': state_fingerprint(text.temporal_pool, exclude_prefixes=('output_projections.',))}


def verify_bridge(direct, bridge):
    compare_common(direct, bridge)
    args = bridge[1]['args']
    fields = (*han.prior.PRICE_FIELDS, 'seq_len', 'label_len', 'pred_len', 'seed', 'd_model', 'dropout', 'text_model_config')
    serialized = json.dumps({key: args[key] for key in fields}, sort_keys=True)
    replay = replay_bridge_initialization(serialized)
    if replay['common'] != args['common_initialization'] or replay['full_pool'] != args['temporal_initialization_sha256']:
        raise BridgePartialError('Old bridge initialization could only be checked partially in this runtime')
    if replay['encoder'] != direct[1]['args']['head_ablation_initialization']['encoder_sha256']:
        raise ValueError('Initial old/new HAN encoders differ')


def aggregate(rows):
    seed_means, across, ticker_means, paired, wins = [], [], [], [], []
    for mode in MODES:
        for seed in SEEDS:
            group = [r for r in rows if r['variant'] == mode and r['seed'] == seed]
            eligible = len(group) == 10 and all(r['comparable'] for r in group)
            seed_means.append({'variant': mode, 'seed': seed, 'eligible_tickers': sum(r['comparable'] for r in group),
                               **{m: han.prior.mean_available([r[m] for r in group], 10) if eligible else None for m in METRICS}})
        for metric in METRICS:
            values = [r[metric] for r in seed_means if r['variant'] == mode]
            across.append({'variant': mode, 'metric': metric, 'mean': han.prior.mean_available(values, 3),
                           'sample_std': float(np.std(values, ddof=1)) if None not in values else None, 'std_ddof': 1})
        for ticker in TICKERS:
            group = [r for r in rows if r['variant'] == mode and r['ticker'] == ticker]
            eligible = len(group) == 3 and all(r['comparable'] for r in group)
            ticker_means.append({'variant': mode, 'ticker': ticker, 'comparable': eligible,
                                 **{m: han.prior.mean_available([r[m] for r in group], 3) if eligible else None for m in METRICS}})
    lookup = {(r['variant'], r['ticker']): r for r in ticker_means}
    for left, right in PAIRS:
        for metric in METRICS:
            differences = []
            for ticker in TICKERS:
                a, b = lookup[left, ticker], lookup[right, ticker]
                valid = a['comparable'] and b['comparable'] and a[metric] is not None and b[metric] is not None
                difference = a[metric] - b[metric] if valid else None
                paired.append({'left': left, 'right': right, 'ticker': ticker, 'metric': metric,
                               'comparable': valid, 'left_seed_mean': a[metric], 'right_seed_mean': b[metric],
                               'difference': difference, 'improvement_pct': 100 * (b[metric] - a[metric]) / b[metric] if valid and b[metric] else None})
                if valid:
                    differences.append(difference)
            wins.append({'left': left, 'right': right, 'metric': metric, 'eligible_tickers': len(differences),
                         'left_wins': sum(d < 0 for d in differences), 'right_wins': sum(d > 0 for d in differences),
                         'ties': sum(d == 0 for d in differences)})
    new = [r for r in rows if r['variant'] in ABLATION_MODES]
    valid = len(new) == 150 and all(r['comparable'] for r in new)
    scores = {r['variant']: r['mean'] for r in across if r['variant'] in ABLATION_MODES and r['metric'] == 'val_mse'}
    valid = valid and None not in scores.values()
    selection = {'status': 'full' if valid else 'blocked', 'metric': 'val_mse', 'scores': scores,
                 'selected': [mode for mode, value in scores.items() if value == min(scores.values())] if valid else [],
                 'test_used_for_selection': False,
                 'bridge_comparison': 'full' if all(r['comparable'] for r in rows if r['variant'] in (BRIDGE, 'han_h32_linear')) else 'partial'}
    return {'by_seed': seed_means, 'across_seeds': across, 'per_ticker': ticker_means, 'paired_by_ticker': paired, 'wins': wins}, selection


def analyze(results_root=DEFAULT_RESULTS, reference_root=DEFAULT_REFERENCE, data_root=han.prior.DEFAULT_DATA, output_dir=None):
    results_root, reference_root = Path(results_root), Path(reference_root)
    output = Path(output_dir) if output_dir else results_root / 'diagnostics' / 'comparison'
    if output.exists():
        raise FileExistsError(f'Report exists: {output}')
    roots = {mode: results_root / 'full' / mode for mode in ABLATION_MODES}
    roots[BRIDGE] = reference_root / 'full' / BRIDGE
    expected = {(ticker, f'seed{seed}') for ticker, seed in itertools.product(TICKERS, SEEDS)}
    physical = set()
    for mode in ABLATION_MODES:
        paths = list(roots[mode].glob('*/seed*/output/best_checkpoint.pt'))
        if len(paths) != 30 or {(p.parents[2].name, p.parents[1].name) for p in paths} != expected:
            raise ValueError(f'{roots[mode]}: expected exactly 30 unique ticker/seed checkpoints')
        for path in paths:
            stat = path.stat()
            identity = stat.st_dev, stat.st_ino
            if not stat.st_size or identity in physical:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            physical.add(identity)
    cache = han.prior.ReplayCache(data_root)
    rows = []
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        runs = {mode: inspect(Run(mode, ticker, seed, root / ticker / f'seed{seed}' / 'output'), cache) for mode, root in roots.items()}
        if all(runs[mode][0]['comparable'] for mode in ABLATION_MODES):
            try:
                for mode in ABLATION_MODES[1:]:
                    runs[mode][0]['common_epochs'] = compare_ablations(runs['last_matched'], runs[mode])
                compare_encoders(runs)
            except (KeyError, TypeError, ValueError) as error:
                for mode in ABLATION_MODES:
                    runs[mode][0].update(comparable=False, match_verification='mismatch', evidence_reason=str(error))
        else:
            for mode in ABLATION_MODES:
                runs[mode][0]['comparable'] = False
        bridge = runs[BRIDGE]
        if bridge[0]['comparable'] and runs['han_h32_linear'][0]['comparable']:
            try:
                verify_bridge(runs['han_h32_linear'], bridge)
            except (BridgePartialError, KeyError, RuntimeError) as error:
                bridge[0].update(comparable=False, match_verification='partial', evidence_reason=str(error))
            except (TypeError, ValueError) as error:
                bridge[0].update(comparable=False, match_verification='mismatch', evidence_reason=str(error))
        rows.extend(item[0] for item in runs.values())
    tables, selection = aggregate(rows)
    tables = {'per_run': rows, **tables,
              'window_verification': [{k: row[k] for k in ('variant', 'ticker', 'seed', 'window_verification', 'match_verification',
                                                        'common_epochs', 'missing_fields', 'mismatches', 'evidence_missing', 'evidence_reason')} for row in rows]}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        han.prior.write_table(output / (name + '.csv'), records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    for name in ('window_verification', 'by_seed', 'across_seeds', 'per_ticker', 'paired_by_ticker', 'wins'):
        print(name.upper())
        for record in tables[name]:
            print(json.dumps(record, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Reports: {output}')
    if selection['status'] == 'blocked':
        raise ValueError('Head ablation comparison blocked; see saved evidence report')
    return tables, selection


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--reference_root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--data_root', type=Path, default=han.prior.DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    analyze(args.results_root, args.reference_root, args.data_root, args.output_dir)


if __name__ == '__main__':
    cli()
