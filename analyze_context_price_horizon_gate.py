"""CPU comparison of saved horizon gates on the original grouped-embedding path.

No training entry point is called. Validation is replayed from the best checkpoint
to obtain horizon metrics; test metrics use normalized saved test_io tensors.
Historical validation dates + reproduced MSE are only partial window evidence.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_blend_price_anchor import check_scaler, execution_device, film_module
from analyze_context_price_only import COMMON, PRICE_FIELDS, TICKERS, SEEDS
from analyze_persistence_baseline import Run, load_metadata, check_configuration, check_scalers
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.models.GroupText import build_text_model, TEXT_GROUPS
from forecasting_task.window_evidence import window_evidence, check_saved_test_windows, check_validation_dates


RUNS = Path('/auto/datasets/kondyurin/FinTexTS/runs')
DEFAULT_RESULTS = RUNS / 'context_price_horizon_gate_10_v1'
DEFAULT_REFERENCE = RUNS / 'context_price_only_10_v1'
DEFAULT_DATA = Path('/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts')
METRICS = ('val_mse', 'test_mse', 'test_mae') + tuple(
    f'{split}_{error}_h{h}' for split, error in (('val', 'mse'), ('test', 'mse'), ('test', 'mae'))
    for h in (1, 2, 3))


class ConfigurationError(ValueError):
    """Windows have not been checked because the configuration is unsupported."""


def validate_config(run, summary, args, reference=None):
    check_configuration(run, args, reference)
    for key, expected in COMMON.items():
        if args.get(key) != expected:
            raise ValueError(f'{key}: expected {expected}, got {args.get(key)}')
    for key in PRICE_FIELDS:
        if key not in args or reference is not None and args[key] != reference[key]:
            raise ValueError(f'FiLM configuration mismatch/unavailable: {key}')
    for key in ('novelty_policy', 'sampling_policy', 'text_mlp_structure'):
        if key not in args or reference is not None and args[key] != reference[key]:
            raise ValueError(f'Controlled configuration mismatch/unavailable: {key}')
    if 'fnspid' in str(args['data_path']).lower():
        raise ValueError('Expected original minilm grouped data without forward fill')
    for source in (summary, args):
        if (source.get('group_gate') != 'context_price_only' or source.get('fixed_alpha') != .1
                or source.get('text_weight_mode') != 'fixed'
                or source.get('price_gate_mode', 'shared') != run.variant):
            raise ValueError('Condition/alpha/gate mode mismatch')
    config = args['text_model_config']
    if (args.get('group_weight_mode') != 'context_price_only' or args.get('fusion_mode') != 'blend'
            or args.get('text_weight') != .1 or args.get('text_num_layers') != 1
            or tuple(args.get('used_col_prefixes', [])) != TEXT_GROUPS
            or config.get('group_gate') != 'context_price_only'
            or config.get('price_gate_mode', 'shared') != run.variant
            or config.get('pred_len') != 3 or config.get('num_layers') != 1
            or config.get('initial_text_weight') != .1 or config.get('fusion_mode') != 'blend'
            or tuple(config.get('used_col_prefixes', [])) != TEXT_GROUPS):
        raise ValueError('Text heads/fusion mismatch')
    # Reject accidental use of category/temporal/history/multiscale branches.
    for key, expected in (('text_temporal_pool', 'last'), ('text_history_mode', 'last'),
                          ('gate_text_history_mode', 'none'), ('category_pool', None),
                          ('price_context_lags', [1, 5])):
        for source in (args, config):
            if source.get(key, expected) != expected:
                raise ValueError(f'Unsupported {key}')
    expected_outputs = 15 if run.variant == 'per_horizon' else 5
    structure = args.get('gate_structure', {})
    if (structure.get('in_features') != 18 or structure.get('out_features') != expected_outputs
            or structure.get('parameters') != 19 * expected_outputs
            or structure.get('initialization') != 'zeros'):
        raise ValueError('Gate structure mismatch')


def error_metrics(prediction, target):
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 4:
        raise ValueError('Prediction/target shape mismatch')
    error = prediction.double() - target.double()
    if not error.numel() or not torch.isfinite(error).all():
        raise ValueError('Empty/nonfinite errors')
    return {'mse': error.square().mean().item(), 'mae': error.abs().mean().item(),
            **{f'{name}_h{h + 1}': operation(error[:, h]).mean().item()
               for name, operation in (('mse', torch.square), ('mae', torch.abs))
               for h in range(target.shape[1])}}


@torch.no_grad()
def evaluate_validation(film, text, dataset, batch_size):
    film.eval()
    text.eval()
    # Element sums, including the final incomplete batch, then divide once.
    squares = torch.zeros(dataset.pred_len, dtype=torch.float64)
    absolutes = torch.zeros_like(squares)
    count = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0):
        x, y, e, xm, ym = [value.float() for value in batch[:5]]
        target = y[:, -dataset.pred_len:]
        decoder = torch.cat((y[:, :dataset.label_len], torch.zeros_like(target)), 1)
        metadata = {key: value.float() for key, value in batch[6].items()}
        output = text(film(x, xm, decoder, ym), x[:, -1], e[:, -1], price_window=x, **metadata)
        error = output.double() - target.double()
        if error.shape != target.shape or not torch.isfinite(error).all():
            raise ValueError('Invalid validation output')
        squares += error.square().sum((0, 2))
        absolutes += error.abs().sum((0, 2))
        count += error.shape[0] * error.shape[2]
    if not count:
        raise ValueError('Empty validation split')
    return {'mse': squares.sum().item() / (count * dataset.pred_len),
            **{f'mse_h{h + 1}': squares[h].item() / count for h in range(dataset.pred_len)}}


def verification(output, evidence, args, summary, checkpoint):
    """Full requires persisted fingerprints for BOTH splits, never just a replay."""
    persisted = []
    for source in (args, summary, checkpoint):
        if 'window_evidence' in source:
            if source['window_evidence'] != evidence:
                raise ValueError('Saved validation/test window fingerprints mismatch')
            persisted.append(source['window_evidence'])
    check_saved_test_windows(output / 'test_io.pt', evidence['test'])
    dates = output / 'validation_group_diagnostics.csv'
    if dates.is_file():
        check_validation_dates(dates, evidence['val'])
    if persisted and dates.is_file():
        return 'full', 'Saved val/test input, target and date hashes; test tensors; validation dates; replayed val MSE'
    return 'partial', ('Replayed current-data validation MSE; exact saved test inputs/targets; '
                       + ('saved validation dates. ' if dates.is_file() else 'validation dates unavailable. ')
                       + 'Historical validation input/target fingerprints unavailable; use matched shared control.')


@torch.no_grad()
def evaluate_run(run, data_root, reference=None, reference_scaler=None, reference_split=None):
    summary, args, scaler, split = load_metadata(run)
    try:
        validate_config(run, summary, args, reference)
    except (ValueError, KeyError, TypeError) as error:
        raise ConfigurationError(str(error)) from error
    if reference_scaler is not None:
        check_scalers(reference_scaler, scaler, str(run.output))
        if split != reference_split:
            raise ValueError('Split policy mismatch')
    checkpoint_path = run.output / 'best_checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('price_gate_mode', args.get('price_gate_mode', 'shared')) != run.variant:
        raise ValueError('Checkpoint gate mode mismatch')
    datasets = {flag: ContextNoveltyDataset(Dataset_Custom(
        args['seq_len'], args['label_len'], args['pred_len'], flag, args['used_col_prefixes'],
        root_path=str(data_root), data_path=run.ticker + '.parquet')) for flag in ('val', 'test')}
    for dataset in datasets.values():
        check_scaler(dataset, checkpoint, checkpoint_path)
    evidence = {flag: window_evidence(dataset) for flag, dataset in datasets.items()}
    with execution_device(torch.device('cpu')):
        film = film_module.Model(SimpleNamespace(**args)).cpu()
        text = build_text_model(args['text_model_config'], args['group_weight_mode']).cpu()
        film.load_state_dict(checkpoint['model_state_dict'], strict=True)
        text.load_state_dict(checkpoint['text_model_state_dict'], strict=True)
        val = evaluate_validation(film, text, datasets['val'], args['batch_size'])
    saved = torch.load(run.output / 'test_io.pt', map_location='cpu', weights_only=False)
    if any(saved.get(key) is None for key in ('outputs', 'ground_truths')):
        raise ValueError(f'Missing normalized outputs/ground_truths; keys={list(saved)}')
    test = error_metrics(torch.as_tensor(saved['outputs']), torch.as_tensor(saved['ground_truths']))
    for actual, key in ((val['mse'], 'best_val_mse'), (test['mse'], 'best_test_mse'), (test['mae'], 'best_test_mae')):
        if not np.isclose(actual, summary[key], rtol=1e-4, atol=1e-6):
            raise ValueError(f'{checkpoint_path}: {key} replay={actual} saved={summary[key]}')
    status, reason = verification(run.output, evidence, args, summary, checkpoint)
    metrics = {**{'val_' + k: v for k, v in val.items()}, **{'test_' + k: v for k, v in test.items()}}
    return metrics, status, reason, evidence, args, scaler, split


def aggregate(rows):
    by_seed, across, paired, wins = [], [], [], []
    for variant in ('per_horizon', 'shared'):
        for seed in SEEDS:
            cohort = [r for r in rows if r['variant'] == variant and r['seed'] == seed]
            complete = len(cohort) == 10 and all(r['comparable'] for r in cohort)
            by_seed.append({'variant': variant, 'seed': seed, 'complete': complete,
                            **{m: float(np.mean([r[m] for r in cohort])) if complete else None for m in METRICS}})
        for metric in METRICS:
            values = [r[metric] for r in by_seed if r['variant'] == variant]
            complete = len(values) == 3 and None not in values
            across.append({'variant': variant, 'metric': metric,
                           'mean': float(np.mean(values)) if complete else None,
                           'sample_std': float(np.std(values, ddof=1)) if complete else None, 'std_ddof': 1})
    for ticker in TICKERS:
        cohort = [r for r in rows if r['ticker'] == ticker]
        if len(cohort) != 6 or not all(r['comparable'] for r in cohort):
            continue
        for metric in METRICS:
            a, b = (float(np.mean([r[metric] for r in cohort if r['variant'] == v]))
                    for v in ('per_horizon', 'shared'))
            paired.append({'ticker': ticker, 'metric': metric, 'per_horizon': a, 'shared': b,
                           'difference': a - b, 'change_pct': 100 * (a - b) / b if b else None})
    for metric in METRICS:
        cohort = [r for r in paired if r['metric'] == metric]
        wins.append({'metric': metric, 'per_horizon_wins': sum(r['difference'] < 0 for r in cohort),
                     'ties': sum(r['difference'] == 0 for r in cohort), 'comparable_tickers': len(cohort)})
    scores = {r['variant']: r['mean'] for r in across if r['metric'] == 'val_mse'}
    comparable = len(rows) == 60 and all(r['comparable'] for r in rows) and None not in scores.values()
    full = comparable and all(r['window_verification'] == 'full' for r in rows)
    selected = [v for v, value in scores.items() if value == min(scores.values())] if comparable else []
    selection = {'metric': 'val_mse', 'test_used_for_selection': False, 'scores': scores,
                 'status': 'full' if full else 'partial' if comparable else 'blocked',
                 'selected': selected if full else [], 'provisional_selected': selected if comparable and not full else [],
                 'limitation': '' if full else 'Run matched shared control for a strict comparison; no partial evidence is called full.'}
    return {'by_seed': by_seed, 'across_seeds': across, 'paired_by_ticker': paired, 'wins': wins}, selection


def analyze(results_root=DEFAULT_RESULTS, reference_root=DEFAULT_REFERENCE, data_root=DEFAULT_DATA,
            output_dir=None, matched_shared=False):
    output = Path(output_dir) if output_dir else Path(results_root) / 'diagnostics' / ('matched_shared' if matched_shared else 'historical_shared')
    if output.exists():
        raise FileExistsError(f'Report already exists: {output}')
    roots = {'per_horizon': Path(results_root) / 'full' / 'per_horizon',
             'shared': Path(reference_root) / 'full' / ('shared' if matched_shared else 'context_price_only')}
    expected = set(itertools.product(TICKERS, SEEDS))
    physical = set()
    for variant, root in roots.items():
        paths = list(root.glob('*/seed*/output/best_checkpoint.pt'))
        keys = {(p.parents[2].name, int(p.parents[1].name[4:])) for p in paths}
        if keys != expected or len(paths) != 30:
            raise ValueError(f'{root}: expected exactly 30 unique matching ticker/seed checkpoints')
        for path in paths:
            stat = path.stat()
            identity = stat.st_dev, stat.st_ino
            if not stat.st_size or identity in physical:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            physical.add(identity)
    rows = []
    common_args = None
    for ticker, seed in itertools.product(TICKERS, SEEDS):
        reference = None
        for variant in ('per_horizon', 'shared'):
            run = Run(variant, ticker, seed, roots[variant] / ticker / f'seed{seed}' / 'output')
            row = {'variant': variant, 'ticker': ticker, 'seed': seed, 'output': str(run.output),
                   'window_verification': 'not_checked', 'comparable': False, 'reason': '',
                   **dict.fromkeys(METRICS)}
            try:
                result = evaluate_run(run, data_root, reference[4] if reference else common_args,
                                      reference[5] if reference else None, reference[6] if reference else None)
                metrics, status, reason, evidence, args, _, _ = result
                if reference is not None and evidence != reference[3]:
                    raise ValueError('Current replay windows/targets differ between paired conditions')
                if (variant == 'per_horizon' or matched_shared) and status != 'full':
                    raise ValueError('New runs require full persisted window evidence')
                if matched_shared and reference is not None:
                    if not args.get('code_commit') or args['code_commit'] != reference[4].get('code_commit'):
                        raise ConfigurationError('Matched shared must use the same code SHA as per_horizon')
                row.update(metrics, window_verification=status, comparable=True, reason=reason)
                if reference is None:
                    reference = result
                    common_args = args
            except (OSError, KeyError, ValueError, TypeError, RuntimeError) as error:
                status = 'not_checked' if isinstance(error, (ConfigurationError, OSError, KeyError)) else 'mismatch'
                row.update(window_verification=status, reason=str(error))
            rows.append(row)
    # A failed member invalidates the entire pair; never aggregate mismatched windows.
    for ticker, seed in expected:
        pair = [r for r in rows if (r['ticker'], r['seed']) == (ticker, seed)]
        if not all(r['comparable'] for r in pair):
            for row in pair:
                row['comparable'] = False
    tables, selection = aggregate(rows)
    tables = {'per_run': rows, **tables,
              'window_verification': [{k: row[k] for k in ('variant', 'ticker', 'seed', 'window_verification', 'reason')} for row in rows]}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        with (output / f'{name}.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ['ticker', 'metric'])
            writer.writeheader()
            writer.writerows(records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
    for name in ('window_verification', 'by_seed', 'across_seeds', 'wins'):
        print(name.upper())
        for record in tables[name]:
            print(json.dumps(record, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Report: {output}')
    if selection['status'] == 'blocked':
        raise ValueError('Comparison blocked by mismatches; see saved window_verification.csv')
    return tables, selection


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--reference_root', type=Path)
    parser.add_argument('--data_root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--matched_shared', action='store_true')
    args = parser.parse_args()
    reference = args.reference_root or (Path(str(args.results_root) + '_matched_shared') if args.matched_shared else DEFAULT_REFERENCE)
    analyze(args.results_root, reference, args.data_root, args.output_dir, args.matched_shared)


if __name__ == '__main__':
    cli()
