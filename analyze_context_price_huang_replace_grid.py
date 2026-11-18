"""Four freshly trained replace cells vs saved last; no additive Huang controls."""

import argparse
import itertools
import json
from pathlib import Path

import analyze_context_price_huang_query as query
import analyze_context_price_huang_grid as grid
from analyze_persistence_baseline import Run
from forecasting_task.models.GroupHuangQuery import HUANG_REPLACE, huang_options
from run_huang_replace_grid_queue import CONFIGURATIONS, RESULTS


CELLS = tuple(CONFIGURATIONS)
LAST = query.LAST
MODES = (*CELLS, LAST)
EFFECTS = grid.EFFECTS
PAIRS = grid.PAIRS
LAST_RESULTS = query.DEFAULT_REFERENCE


def inspect(run, cache):
    actual_mode = LAST if run.variant == LAST else HUANG_REPLACE
    result = query.inspect(Run(actual_mode, run.ticker, run.seed, run.output), cache)
    row, metadata, _, _ = result
    row['variant'] = run.variant
    if run.variant in CELLS:
        parameters = huang_options(CONFIGURATIONS[run.variant])
        row.update(parameters, text_branch='huang_replaces_last')
        if row['comparable'] and (metadata['args']['text_temporal_pool'] != HUANG_REPLACE or huang_options(metadata['args']) != parameters):
            row.update(comparable=False, match_verification='mismatch', evidence_reason='Expected replace mode and matching H/steps/lookback')
    return result


def compare_pair(left, right):
    a, b = left[0]['variant'], right[0]['variant']
    epochs = query.compare_pair(left, right, allow_architecture_difference=True, require_same_code=a in CELLS and b in CELLS,
                                compare_text_heads=False, compare_initial_prediction=False)
    if a in CELLS and b in CELLS:
        args_a, args_b = left[1]['args'], right[1]['args']
        if huang_options(args_a)['huang_hidden_size'] == huang_options(args_b)['huang_hidden_size']:
            values = [args.get('temporal_initialization_sha256') for args in (args_a, args_b)]
            if any(not v for v in values):
                raise KeyError('Full Huang initial parameters including query are unconfirmed')
            if values[0] != values[1]:
                raise ValueError('Initial Huang parameters differ at equal H')
    return epochs


def analyze(results_root=RESULTS, last_root=LAST_RESULTS, data_root=query.heads.han.prior.DEFAULT_DATA, output_dir=None):
    results_root = Path(results_root)
    output = Path(output_dir) if output_dir else results_root / 'diagnostics' / 'replace_grid_comparison'
    if output.exists():
        raise FileExistsError(f'Report exists: {output}')
    roots = {m: results_root / 'full' / m for m in CELLS}
    roots[LAST] = Path(last_root) / 'full' / LAST
    expected = set(itertools.product(query.TICKERS, (f'seed{s}' for s in query.SEEDS)))
    identities, inventory = set(), []
    for mode, root in roots.items():
        paths = list(root.glob('*/seed*/output/best_checkpoint.pt'))
        actual = {(p.parents[2].name, p.parents[1].name) for p in paths}
        if len(paths) != len(actual) or not actual <= expected or (mode in CELLS and actual != expected):
            raise ValueError(f'{root}: unexpected, missing or duplicate ticker/seed checkpoints')
        for path in paths:
            stat = path.stat()
            if not stat.st_size or (stat.st_dev, stat.st_ino) in identities:
                raise ValueError(f'Empty/duplicate checkpoint: {path}')
            identities.add((stat.st_dev, stat.st_ino))
        inventory.append({'variant': mode, 'checkpoints': len(paths), 'expected': 30,
                          'status': 'full' if actual == expected else 'partial' if actual else 'not_checked'})
    rows, verification, cache = [], [], query.ReplayCache(data_root)
    for ticker, seed in itertools.product(query.TICKERS, query.SEEDS):
        runs = {m: inspect(Run(m, ticker, seed, root / ticker / f'seed{seed}/output'), cache) for m, root in roots.items()}
        verification.extend(query.verify_pair(runs[a], runs[b], comparator=compare_pair) for a, b in PAIRS)
        rows.extend(r[0] for r in runs.values())
    tables, base = query.aggregate(rows, verification, modes=MODES, pairs=PAIRS, selection_modes=CELLS)
    extra, selection = grid.grid_tables(tables, verification, base, cells=CELLS, new_cells=CELLS, effects=EFFECTS, old=None)
    last_val = next(r['mean'] for r in tables['across_seeds'] if r['variant'] == LAST and r['metric'] == 'val_mse')
    all_scores = {**selection['scores'], LAST: last_val}
    confirmed = selection['status'] == 'full' and selection['last_comparison'] == 'full' and last_val is not None
    selection['selected_grid'] = selection['selected']
    selection['including_last'] = {'status': 'full' if confirmed else 'partial', 'metric': 'val_mse', 'scores': all_scores,
                                   'selected': [m for m, v in all_scores.items() if v == min(all_scores.values())] if confirmed else [],
                                   'test_used_for_selection': False}
    tables = {'per_run': rows, 'inventory': inventory, 'window_verification': verification, **tables, **extra}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in tables.items():
        query.heads.han.prior.write_table(output / (name + '.csv'), records)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    for name in ('grid_2x2', 'window_verification', 'by_seed', 'across_seeds', 'paired_by_run', 'paired_by_ticker', 'wins', 'effects_by_seed', 'effects_across_seeds'):
        print(name.upper())
        for record in tables[name]:
            print(json.dumps(record, allow_nan=False))
    print(json.dumps(selection, allow_nan=False))
    print(f'Reports: {output}')
    if selection['status'] == 'blocked':
        raise ValueError('Replace grid comparison blocked; see verification report')
    return tables, selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=RESULTS)
    parser.add_argument('--last_root', type=Path, default=LAST_RESULTS)
    parser.add_argument('--data_root', type=Path, default=query.heads.han.prior.DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    analyze(args.results_root, args.last_root, args.data_root, args.output_dir)
