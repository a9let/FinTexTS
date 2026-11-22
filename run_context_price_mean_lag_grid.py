"""Run the nine-cell mean/price grid with a verified all-or-none control."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
from statistics import mean
import subprocess

import torch

import mean_lag_grid_audit as audit
import run_han_head_ablation_queue as shared
from run_group_memory_queue import execute, queue_lock


SCRIPTS = ('run_context_price_mean_lag_grid.sh', 'run_context_price_mean_lag_grid.py',
           'mean_lag_grid_audit.py', 'run_group_memory_queue.py', 'run_han_head_ablation_queue.py')


def jobs(configurations=audit.CONFIGS):
    return shared.jobs(tuple(configurations))


def record_json(path, value):
    shared.record_file(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def select_control(results, normal_root, data, force):
    previous = audit.decisions(results)
    if previous and previous['source'] == 'matched':
        return previous
    records = []
    if force:
        source, reason = 'matched', 'Requested with --force-matched-control'
    else:
        runs, records = audit.inspect_normal_cohort(normal_root, data, audit.prior.ReplayCache(data))
        passed = len(runs)
        source = 'historical' if passed == len(audit.PAIRS) else 'matched'
        reason = f'xTime normal: {passed}/{len(audit.PAIRS)} passed the audit'
        if previous and previous['records'] != records:
            source = 'matched'
            reason += '; previously pinned control evidence has changed'
    decision = dict(source=source, reason=reason, normal_root=str(normal_root), records=records)
    name = 'reused_control.json' if source == 'historical' else 'matched_control.json'
    record_json(results/name, decision)
    return decision


def run_queue(code, data, results, normal_root, force=False, *, configurations=audit.CONFIGS, scripts=SCRIPTS):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    code, data, results = (Path(p).resolve() for p in (code, data, results))
    if normal_root is not None:
        normal_root = Path(normal_root).resolve()
    commit = os.environ.get('CODE_COMMIT') or subprocess.check_output(
        ['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if not re.fullmatch(r'[a-fA-F0-9]{40}|[a-fA-F0-9]{64}', commit):
        raise ValueError('CODE_COMMIT must be the full Git SHA')
    for ticker in (*shared.TICKERS, 'NVDA'):
        if not (data / f'{ticker}.parquet').is_file():
            raise FileNotFoundError(data / f'{ticker}.parquet')
    results.mkdir(parents=True, exist_ok=True)
    with queue_lock(results / '.queue.lock'):
        shared.record_file(results / 'code_commit.txt', (commit+'\n').encode())
        for script in scripts:
            shared.record_file(results / script, (code / script).read_bytes())
        inventory = jobs(configurations)
        record_json(results / 'manifest.json', dict(code_commit=commit, data=str(data),
                    normal_root=str(normal_root) if normal_root is not None else None,
                    settings=shared.SETTINGS, configurations=configurations, jobs=inventory))
        for area, cell, ticker, seed, epochs in inventory:
            if area != 'checks':
                break
            destination = audit.grid_destination(results, cell, ticker, seed, area)
            execute(code, data, commit, destination, configurations[cell], ticker, seed, epochs, [], None)
            print(f'OK check={cell}', flush=True)

        reused = 0
        if normal_root is not None:
            decision = select_control(results, normal_root, data, force)
            reused = len(audit.PAIRS) if decision['source'] == 'historical' else 0
            print(f'CONTROL {decision["source"]}: {decision["reason"]}', flush=True)
            for record in decision['records']:
                if record['status'] == 'rejected':
                    print(f'CONTROL_REJECT {record["ticker"]}/seed{record["seed"]}: {record["reason"]}', flush=True)
        planned = [(cell, ticker, seed, epochs) for area, cell, ticker, seed, epochs in inventory
                   if area == 'full' and (not reused or cell != audit.CONTROL)]
        remaining = len(planned)
        total = remaining + reused
        print(f'GRID trained=0 reused={reused} remaining={remaining} total={total}', flush=True)
        timing = defaultdict(list)
        for trained, (cell, ticker, seed, epochs) in enumerate(planned, 1):
            destination = audit.grid_destination(results, cell, ticker, seed)
            marker = execute(code, data, commit, destination, configurations[cell], ticker, seed, epochs, [], None)
            timing[cell].append(marker['wall_seconds'])
            remaining = len(planned) - trained
            overall = mean(seconds for values in timing.values() for seconds in values)
            eta = sum(mean(timing[label]) if label in timing else overall
                      for label, *_ in planned[trained:])
            print(f'OK {cell}/{ticker}/seed{seed} trained={trained} reused={reused} remaining={remaining} '
                  f'completed={trained+reused}/{total} cell_mean_seconds={mean(timing[cell]):.1f} '
                  f'eta_seconds={eta:.1f}', flush=True)
        print(f'ALL {total} GRID RESULTS COMPLETE trained={len(planned)} reused={reused}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=shared.DATA)
    parser.add_argument('--results', type=Path, default=audit.RESULTS)
    parser.add_argument('--normal-root', type=Path, default=audit.NORMAL_ROOT)
    parser.add_argument('--force-matched-control', action='store_true')
    args = parser.parse_args()
    try:
        run_queue(Path(__file__).resolve().parent, args.data, args.results, args.normal_root, args.force_matched_control)
    except audit.AUDIT_ERRORS as error:
        parser.exit(1, f'STOP: {error}\n')


if __name__ == '__main__':
    main()
