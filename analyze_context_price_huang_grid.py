"""Matched 2x2 H/steps Huang grid, using the original query artifact analyzer."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import analyze_context_price_huang_query as query
from analyze_persistence_baseline import Run
from forecasting_task.models.GroupHuangQuery import huang_options
from run_huang_grid_queue import CONFIGURATIONS, RESULTS


OLD = 'h32_s20_l10'
LAST = query.LAST
NEW = tuple(CONFIGURATIONS)
CELLS = (*NEW, OLD)
MODES = (*CELLS, LAST)
EFFECTS = (('hidden_at_steps5', 'h16_s5_l10', 'h32_s5_l10'),
           ('hidden_at_steps20', 'h16_s20_l10', OLD),
           ('steps_at_hidden16', 'h16_s5_l10', 'h16_s20_l10'),
           ('steps_at_hidden32', 'h32_s5_l10', OLD))
PAIRS = tuple((a, b) for _, a, b in EFFECTS) + (('h16_s5_l10', OLD), ('h32_s5_l10', 'h16_s20_l10')) + tuple((m, LAST) for m in CELLS)
OLD_RESULTS = query.DEFAULT_RESULTS
LAST_RESULTS = query.DEFAULT_REFERENCE
PARAMETERS = {**CONFIGURATIONS, OLD: {'text_temporal_pool': 'huang_recurrent_query', 'huang_hidden_size': 32,
                                    'huang_recurrent_steps': 20, 'huang_attention_lookback': 10}}


def inspect(run, cache):
    mode = LAST if run.variant == LAST else 'huang_recurrent_query'
    result = query.inspect(Run(mode, run.ticker, run.seed, run.output), cache)
    row, metadata, _, _ = result
    row['variant'] = run.variant
    if run.variant in CELLS:
        row.update(huang_options(PARAMETERS[run.variant]))
        if row['comparable'] and huang_options(metadata['args']) != huang_options(PARAMETERS[run.variant]):
            row.update(comparable=False, match_verification='mismatch', evidence_reason='Saved H/steps/lookback differ from grid cell')
    return result


def compare_pair(left, right):
    a, b = left[0]['variant'], right[0]['variant']
    epochs = query.compare_pair(left, right, allow_architecture_difference=True, require_same_code=a in NEW and b in NEW)
    if a in CELLS and b in CELLS:
        args_a, args_b = left[1]['args'], right[1]['args']
        if huang_options(args_a)['huang_hidden_size'] == huang_options(args_b)['huang_hidden_size']:
            digests = [args.get('temporal_initialization_sha256') for args in (args_a, args_b)]
            if any(not digest for digest in digests):
                raise KeyError('Initial Huang state including query is unconfirmed')
            if digests[0] != digests[1]:
                raise ValueError('Initial Huang parameters including query differ at equal H')
    return epochs


def grid_tables(tables, verification, base_selection, *, cells=CELLS, new_cells=NEW, effects=EFFECTS, old=OLD):
    lookup = {(r['variant'], r['metric']): r for r in tables['across_seeds']}
    grid = [{'huang_hidden_size': hidden, 'huang_attention_lookback': 10, 'metric': metric,
             **{f'steps{steps}_mean': lookup[f'h{hidden}_s{steps}_l10', metric]['mean'] for steps in (5, 20)}}
            for metric in query.METRICS for hidden in (16, 32)]
    effect_rows, across_effects = [], []
    for name, left, right in effects:
        for metric in query.METRICS:
            for seed in query.SEEDS:
                pairs = [r for r in tables['paired_by_run'] if (r['left'], r['right'], r['metric'], r['seed']) == (left, right, metric, seed)]
                values = [r['difference'] for r in pairs]
                effect_rows.append({'effect': name, 'left': left, 'right': right, 'metric': metric, 'seed': seed,
                                'verified_tickers': sum(v is not None for v in values),
                                'mean_difference': query.heads.han.prior.mean_available(values, 10)})
            values = [r['mean_difference'] for r in effect_rows if r['effect'] == name and r['metric'] == metric]
            across_effects.append({'effect': name, 'left': left, 'right': right, 'metric': metric,
                                   'mean_difference': query.heads.han.prior.mean_available(values, 3),
                                   'sample_std': float(np.std(values, ddof=1)) if None not in values else None, 'std_ddof': 1})
    old_pairs = [r for r in verification if old in (r['left'], r['right']) and LAST not in (r['left'], r['right'])]
    old_confirmed = old is None or (len(old_pairs) == len(new_cells) * 30 and all(r['window_verification'] == 'full' for r in old_pairs))
    new_confirmed = base_selection['status'] == 'full'
    eligible = list(cells if old_confirmed else new_cells) if new_confirmed else []
    scores = {m: lookup[m, 'val_mse']['mean'] for m in cells}
    eligible = [m for m in eligible if scores[m] is not None]
    selection = {**base_selection, 'status': 'full' if new_confirmed and old_confirmed else 'partial' if new_confirmed else 'blocked',
                 'scores': scores, 'eligible_cells': eligible,
                 'selected': [m for m in eligible if scores[m] == min(scores[k] for k in eligible)],
                 'historical_h32_s20': 'full' if old_confirmed else 'partial',
                 'limitation': '' if old_confirmed else 'No strict choice across all four grid cells: saved 32/20/10 comparison unconfirmed'}
    if old is None:
        selection.pop('historical_h32_s20')
        selection.update(all_grid_cells_trained_from_scratch=True, historical_huang_used=False)
    return {'grid_2x2': grid, 'effects_by_seed': effect_rows, 'effects_across_seeds': across_effects}, selection


def analyze(results_root=RESULTS, old_root=OLD_RESULTS, last_root=LAST_RESULTS,
            data_root=query.heads.han.prior.DEFAULT_DATA, output_dir=None):
    results_root = Path(results_root)
    output = Path(output_dir) if output_dir else results_root / 'diagnostics' / 'grid_comparison'
    if output.exists():
        raise FileExistsError(f'Report exists: {output}')
    roots = {m: results_root / 'full' / m for m in NEW}
    roots[OLD] = Path(old_root) / 'full' / 'huang_recurrent_query'
    roots[LAST] = Path(last_root) / 'full' / LAST
    expected = set(itertools.product(query.TICKERS, (f'seed{s}' for s in query.SEEDS)))
    identities, inventory = set(), []
    for mode, root in roots.items():
        paths = list(root.glob('*/seed*/output/best_checkpoint.pt'))
        actual = {(p.parents[2].name, p.parents[1].name) for p in paths}
        if len(paths) != len(actual) or not actual <= expected or (mode in NEW and actual != expected):
            raise ValueError(f'{root}: unexpected, missing or duplicate ticker/seed checkpoints')
        for path in paths:
            stat = path.stat()
            if not stat.st_size or (stat.st_dev, stat.st_ino) in identities:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            identities.add((stat.st_dev, stat.st_ino))
        inventory.append({'variant': mode, 'checkpoints': len(paths), 'expected': 30,
                          'status': 'full' if actual == expected else 'partial' if actual else 'not_checked'})
    cache, rows, verification = query.ReplayCache(data_root), [], []
    for ticker, seed in itertools.product(query.TICKERS, query.SEEDS):
        runs = {m: inspect(Run(m, ticker, seed, root / ticker / f'seed{seed}/output'), cache) for m, root in roots.items()}
        verification.extend(query.verify_pair(runs[a], runs[b], comparator=compare_pair) for a, b in PAIRS)
        rows.extend(r[0] for r in runs.values())
    tables, base_selection = query.aggregate(rows, verification, modes=MODES, pairs=PAIRS, selection_modes=NEW)
    extra, selection = grid_tables(tables, verification, base_selection)
    tables = {'per_run': rows, 'inventory': inventory, 'window_verification': verification, **tables, **extra}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        query.heads.han.prior.write_table(output / (name + '.csv'), records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    for metric in query.METRICS:
        print(f'GRID {metric}: H / steps | 5 | 20')
        for r in tables['grid_2x2']:
            if r['metric'] == metric:
                print(f"{r['huang_hidden_size']} | {r['steps5_mean']} | {r['steps20_mean']}")
    for name in ('window_verification', 'by_seed', 'across_seeds', 'paired_by_run', 'paired_by_ticker', 'wins', 'effects_by_seed', 'effects_across_seeds'):
        print(name.upper())
        for record in tables[name]:
            print(json.dumps(record, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Reports: {output}')
    if selection['status'] == 'blocked':
        raise ValueError('New Huang grid comparison blocked; see verification report')
    return tables, selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=RESULTS)
    parser.add_argument('--old_root', type=Path, default=OLD_RESULTS)
    parser.add_argument('--last_root', type=Path, default=LAST_RESULTS)
    parser.add_argument('--data_root', type=Path, default=query.heads.han.prior.DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    analyze(args.results_root, args.old_root, args.last_root, args.data_root, args.output_dir)
