"""Memory experiment dispatcher. Every failed attempt stays available for inspection."""
import argparse
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import time

import torch
import run_han_head_ablation_queue as shared
from run_group_gate_queue import configuration
from forecasting_task.memory_phases import MEMTO_EPOCHS, XTIME_SPECIALIST_EPOCHS, XTIME_ROUTER_EPOCHS


RESULTS = shared.DATA.parents[1]/'runs'/'group_memory_ablation_720_v1'
FAMILIES = ('memto_group','titans_linear_group','xtime_group')
class RejectedTrainingData(ValueError):
    """Keep the failed attempt and advance to the next configuration."""


CONFIGURATIONS = {}
for family, options in zip(FAMILIES, ((8,16),(16,32),(0.,.5))):
    for window, value, weights in itertools.product((5,20),options,('price_only','uniform')):
        label = f'{family}_w{window}_{("beta"+str(value) if family=="xtime_group" else str(value))}_{weights}'
        CONFIGURATIONS[label] = dict(configuration(family,32 if family=='xtime_group' else value,window,weights),
                                     kd_beta=value if family=='xtime_group' else 0.)


def jobs(stage='full', family=None, ticker='AAPL', seed=7):
    configs = {k:v for k,v in CONFIGURATIONS.items() if family is None or v['text_family']==family}
    if stage=='control':
        return [('controls','last_uniform',t,s) for t,s in itertools.product(shared.TICKERS,shared.SEEDS)]
    if stage=='smoke':
        return [('checks',k,'NVDA',7) for k in configs]
    pairs = [(ticker,seed)] if stage=='pilot' else list(itertools.product(shared.TICKERS,shared.SEEDS))
    return [('full',k,t,s) for k,(t,s) in itertools.product(configs,pairs)]


@contextmanager
def queue_lock(path):
    with path.open('a+b') as handle:
        if os.name=='nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def execute(code, data, commit, destination, config, ticker, seed, epochs, flags, progress):
    expected = shared.expected_config(config['text_family'],ticker,seed,epochs,commit,data,configuration=config)
    if destination.exists():
        marker = shared.completion(destination,expected)
        print(f'SKIP {destination}',flush=True)
        return marker
    destination.parent.mkdir(parents=True,exist_ok=True)
    attempt_index = 1
    while (attempt:=destination.with_name(f'{destination.name}.attempt-{attempt_index}')).exists():
        attempt_index += 1
    attempt.mkdir()
    cmd = shared.training_command(config['text_family'],ticker,seed,epochs,commit,data,attempt/'output',configuration=config)
    if 'kd_beta' in config:
        cmd += ['--kd_beta',str(config['kd_beta'])]
    if progress is not None:
        cmd += ['--memory_progress',str(progress)]
    start = time.monotonic()
    print(f'START {destination}',flush=True)
    status = shared.execute_training(cmd+flags,attempt/'train.log',code)
    if status:
        tail = (attempt/'train.log').read_text(encoding='utf-8',errors='replace').splitlines()[-24:]
        if status == 42:
            reasons = [line for line in tail if line.startswith('TRAIN_DATA_INSUFFICIENT:')]
            raise RejectedTrainingData(reasons[-1] if reasons else str(attempt))
        raise RuntimeError(f'{attempt} exited with {status}\n'+'\n'.join(tail))
    marker = shared.completion(attempt,expected,time.monotonic()-start)
    attempt.rename(destination)
    return marker


def run(code,data,results,stage,family=None,ticker='AAPL',seed=7):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    commit = os.environ.get('CODE_COMMIT') or subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    if not re.fullmatch(r'[a-fA-F0-9]{40}|[a-fA-F0-9]{64}',commit):
        raise ValueError('CODE_COMMIT must be the full Git SHA')
    data,results = Path(data).resolve(),Path(results).resolve()
    planned = jobs(stage,family,ticker,seed)
    for symbol in {j[2] for j in planned}:
        if not (data/f'{symbol}.parquet').is_file():
            raise FileNotFoundError(data/f'{symbol}.parquet')
    results.mkdir(parents=True,exist_ok=True)
    with queue_lock(results/'.queue.lock'):
        shared.record_file(results/'code_commit.txt',(commit+'\n').encode())
        for name in ('run_group_memory_ablation_720.sh','run_group_memory_queue.py','run_han_head_ablation_queue.py'):
            shared.record_file(results/name,(Path(code)/name).read_bytes())
        manifest = {'code_commit':commit,'data':str(data),'configurations':CONFIGURATIONS,
                    'budgets':{'supervised':50,'patience':10,'memto_epochs':MEMTO_EPOCHS,
                               'memto_nearest':'minimum_squared_euclidean','xtime_specialists_each':XTIME_SPECIALIST_EPOCHS,
                               'xtime_router':XTIME_ROUTER_EPOCHS,'smoke_per_phase':1},
                    'full_jobs':jobs(),'smoke_jobs':jobs('smoke'),'control_jobs':jobs('control')}
        shared.record_file(results/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
        timing = {}
        phase_seen = set()
        completed = 0
        rejected = 0
        for visited,(area,label,symbol,run_seed) in enumerate(planned,1):
            config = dict(CONFIGURATIONS[label]) if area!='controls' else configuration('last',32,20,'uniform')
            kind, window = config['text_family'], config['text_window']
            smoke = area=='checks'
            flags = ['--memory_smoke'] if smoke else []
            epochs = 1 if smoke else 50
            destination = results/area/label/symbol/f'seed{run_seed}'
            if area=='controls':
                destination = results/'controls'/'full'/label/symbol/f'seed{run_seed}'
            progress = results/'progress'/area/label/symbol/f'seed{run_seed}' if area!='controls' else None
            if kind=='memto_group':
                cache = results/'phases'/area/kind/f'w{window}_k{config["family_hidden_size"]}'/symbol/f'seed{run_seed}'
                flags += ['--memory_cache',str(cache)]
            if kind=='xtime_group':
                cache = results/'phases'/area/kind/f'w{window}_{config["group_weight_condition"]}'/symbol/f'seed{run_seed}'
                normal = dict(config,kd_beta=0.,memory_stage='normal',memory_smoke=smoke)
                try:
                    normal_marker = execute(code,data,commit,cache/'normal',normal,symbol,run_seed,epochs,
                                            flags+['--memory_stage','normal','--memory_cache',str(cache)],cache/'progress')
                except RejectedTrainingData as error:
                    rejected += 1
                    print(f'REJECT {label}/{symbol}/seed{run_seed} {error}',flush=True)
                    continue
                if cache not in phase_seen:
                    print(f'PHASE_DONE {kind}/normal wall_seconds={normal_marker["wall_seconds"]:.2f}',flush=True)
                    phase_seen.add(cache)
                flags += ['--memory_cache',str(cache)]
                epochs = 1
            if area!='controls':
                config.update(memory_stage='final',memory_smoke=smoke)
            try:
                marker = execute(code,data,commit,destination,config,symbol,run_seed,epochs,flags,progress)
            except RejectedTrainingData as error:
                rejected += 1
                print(f'REJECT {label}/{symbol}/seed{run_seed} {error}',flush=True)
                continue
            completed += 1
            times = timing.setdefault(kind,[])
            normal_cost = normal_marker['wall_seconds']/2 if kind=='xtime_group' else 0.
            times.append(marker['wall_seconds']+normal_cost)
            remaining = sum((CONFIGURATIONS[k]['text_family'] if k in CONFIGURATIONS else 'last')==kind
                            for _,k,_,_ in planned[visited:])
            print(f'OK {kind} completed={completed}/{len(planned)} family_completed={len(times)} '
                  f'family_eta_seconds={remaining*sum(times)/len(times):.2f}',flush=True)
        print(f'DONE completed={completed}/{len(planned)} insufficient_train_data={rejected}',flush=True)
        if rejected:
            raise SystemExit(2)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--data',type=Path,default=shared.DATA)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--stage',choices=('full','smoke','pilot','control'),default='full')
    parser.add_argument('--family',choices=FAMILIES)
    parser.add_argument('--ticker',choices=shared.TICKERS,default='AAPL')
    parser.add_argument('--seed',type=int,choices=shared.SEEDS,default=7)
    args = parser.parse_args()
    run(args.code,args.data,args.results,args.stage,args.family,args.ticker,args.seed)
