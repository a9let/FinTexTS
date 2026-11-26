"""Cache-only budget experiment. This entry point cannot train or infer experts."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from forecasting_task.mean_router import fit_router
from forecasting_task.mean_router_protocol import atomic_json
from run_group_memory_queue import queue_lock
import router_budget_source as source

BASE_SHA = source.SOURCE_SHA


def implementation_sha():
    value = os.environ.get('CODE_COMMIT') or subprocess.check_output(
        ['git','-C',str(Path(__file__).resolve().parent),'rev-parse','HEAD'],text=True).strip()
    if len(value)!=40 or any(c not in '0123456789abcdef' for c in value.lower()):
        raise ValueError('CODE_COMMIT must be a full implementation SHA')
    return value


def task_plan(args):
    return [dict(mode=mode,seed=seed,epochs=args.max_epochs,patience=args.patience,include_initial=True,
                 source_cache=str(args.source_root/'cache'/f'seed{seed}_validation.pt'),
                 output=str(args.output/'router'/mode/f'seed{seed}'))
            for mode,seed in itertools.product(source.MODES,source.SEEDS)]


def check_roots(args):
    args.source_root,args.output = args.source_root.resolve(),args.output.resolve()
    if (args.source_root==args.output or args.source_root in args.output.parents
            or args.output in args.source_root.parents):
        raise ValueError('Source and output must be disjoint directories')


def audited_source(args,sha):
    print(f'AUDIT source={args.source_root} source_code_sha={source.SOURCE_SHA} implementation_sha={sha}',flush=True)
    snapshot = source.source_snapshot(args.source_root)
    serialized = json.dumps(snapshot,sort_keys=True).encode()
    provenance = dict(source_code_sha=snapshot['source_code_sha'],implementation_sha=sha,
                      source_snapshot_sha256=hashlib.sha256(serialized).hexdigest())
    record = dict(provenance=provenance,snapshot=snapshot,test_previously_inspected=True)
    path = args.output/'source_audit.json'
    if path.exists():
        if source.read_json(path)!=record:
            raise ValueError('Source or implementation differs from the recorded audit; files preserved')
    else:
        atomic_json(path,record)
    print('AUDIT verified saved artifacts; raw-data and expert-forward replay were not performed',flush=True)
    return record


def train(args,audit):
    manifest,_ = source.verify_manifest(args.source_root)
    planned = task_plan(args)
    started = time.monotonic()
    for i,task in enumerate(planned,1):
        mode,seed = task['mode'],task['seed']
        # Only validation caches are accessible on this code path.
        cache = source.validation_cache(args.source_root,manifest,seed)
        print(f'START mode={mode} seed={seed} max_steps={args.max_epochs} patience={args.patience}',flush=True)
        fit_router(cache,Path(task['output']),mode,seed,epochs=args.max_epochs,patience=args.patience,
                   lr=.001,batch_size=128,regularization=.001,resume=args.resume,
                   include_initial=True,provenance=audit['provenance'])
        _,old_last = source.old_state(args.source_root,mode,seed,cache)
        current = torch.load(Path(task['output'])/'last_checkpoint.pt',map_location='cpu',weights_only=True)
        trajectory = source.compare_history(old_last['history'],current['history'])
        atomic_json(Path(task['output'])/'old_new_trajectory.json',trajectory)
        print(f'OK mode={mode} seed={seed} completed={i}/9 '
              f'ETA_seconds={(time.monotonic()-started)/i*(9-i):.1f}',flush=True)
    # Source fingerprints are pinned again before accepting completion.
    if source.source_snapshot(args.source_root)!=audit['snapshot']:
        raise ValueError('Read-only source changed during the fit queue')
    print('ALL 9 ROUTER BUDGET FITS COMPLETED',flush=True)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','audit','train','analyze','queue'))
    parser.add_argument('--source-root',type=Path,default=source.SOURCE_ROOT)
    parser.add_argument('--output',type=Path,default=source.OUTPUT)
    parser.add_argument('--max-epochs',type=int,default=1000)
    parser.add_argument('--patience',type=int,default=100)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-name',default='final')
    parser.add_argument('--tie-atol',type=float,default=1e-6)
    parser.add_argument('--tie-rtol',type=float,default=1e-4)
    args = parser.parse_args(argv)
    check_roots(args)
    if (args.max_epochs<1 or args.patience<1
            or any(not math.isfinite(v) or v<0 for v in (args.tie_atol,args.tie_rtol))):
        parser.error('Budgets must be positive and tie tolerances finite and nonnegative')
    if len(Path(args.report_name).parts)!=1 or args.report_name in ('.','..'):
        parser.error('report-name must be one directory name')
    if args.stage=='plan':
        print(json.dumps(dict(base_sha=BASE_SHA,tasks=task_plan(args),fits=9,
                              source_code_sha=source.SOURCE_SHA,read_only=True),indent=2))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with queue_lock(args.output/'.queue.lock'):
        sha = implementation_sha()
        audit = audited_source(args,sha)
        if args.stage in ('train','queue'):
            train(args,audit)
        if args.stage in ('analyze','queue'):
            from router_budget_report import analyze,verify_report
            if args.stage=='queue' and (args.output/'reports'/args.report_name).exists():
                verify_report(args,audit)
                print('SKIP verified completed report',flush=True)
            else:
                analyze(args,audit)


if __name__=='__main__':
    cli()
