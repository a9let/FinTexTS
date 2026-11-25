"""FiLM Q1 experts and frozen-expert router experiment, explicit stages only."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

import mean_lag_grid_audit as grid
import run_han_head_ablation_queue as shared
from run_group_memory_queue import queue_lock
from forecasting_task import run
from forecasting_task.data_provider.context_novelty import get_context_novelty_dataloader, gate_batch_inputs
from forecasting_task.experiment_evidence import state_fingerprint
from forecasting_task.mean_router_protocol import (
    BASE_SHA, BASE_DATASET_SHA256, BASELINE_SOURCE, BACKBONES, PERIODS, PROTOCOL, MatchedBaseline,
    atomic_json, atomic_save, assignments, evidence, file_hash, period_evidence)
from forecasting_task.mean_router import (
    EPS, SCHEMA, Router, features, fit_router, subset, cache_hashes, validate_cache)
from forecasting_task.models.GroupText import build_text_model

RESULTS = grid.RUNS/'context_price_mean_router_q1_v1'
CONFIGS = {grid.cell_name(w,lags):dict(grid.configuration(grid.MEAN_GRID,32,w,'price_only'), price_gate_lags=list(lags))
           for w,lags in itertools.product((10,20,40,60,80), grid.LAG_SETS)}
MODES = ('constant','price','price_text')
FACTORIES = {name:getattr(run,name) for name in BACKBONES}


def inventory(backbone, root):
    tasks = []
    for kind, configs in (('experts',CONFIGS), ('controls',{'matched_baseline':CONFIGS['mean20_l1-5']})):
        for cell, config in configs.items():
            for ticker, seed in grid.PAIRS:
                tasks.append(dict(kind=kind, cell=cell, ticker=ticker, seed=seed, config=config,
                                  backbone=backbone, output=str(root/kind/cell/ticker/f'seed{seed}'/'output')))
    return tasks


def datasets(args, data):
    return {flag:get_context_novelty_dataloader(batch_size=args.batch_size, seq_len=args.seq_len,
        label_len=args.label_len, pred_len=args.pred_len, flag=flag, used_col_prefixes=args.used_col_prefixes,
        root_path=str(data), data_path=args.data_path, include_calendar_days=True,
        matched_baseline=getattr(args,'matched_original_baseline',False),
        extended_text_window=args.text_window if args.text_window>args.seq_len else None)[0]
        for flag in ('train','val','test')}


def verify_completed(task, sha, data, epochs=50):
    output = Path(task['output'])
    marker = json.loads((output/'completed.json').read_text(encoding='utf-8'))
    for name, digest in marker['artifacts'].items():
        if file_hash(output/name) != digest:
            raise ValueError(f'{output}: modified artifact {name}')
    if set(marker['artifacts']) != {'best_checkpoint.pt','last_checkpoint.pt','summary.json','history.jsonl','args.yaml'}:
        raise ValueError('Incomplete completion marker')
    saved = torch.load(output/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    args = saved['args']
    expected = {**shared.SETTINGS, **task['config'], 'model_type':task['backbone'],
                'seed':task['seed'], 'selection_protocol':PROTOCOL}
    for key in ('text_temporal_pool',):
        expected.pop(key, None)
    expected.update(code_commit=sha, matched_original_baseline=task['kind']=='controls', num_epoch=epochs)
    if any(args.get(key) != value for key,value in expected.items()):
        raise ValueError(f'{output}: task configuration mismatch')
    if args['data_path'] != task['ticker']+'.parquet':
        raise ValueError('Task ticker mismatch')
    replay = evidence(datasets(SimpleNamespace(**args), data))
    if replay != saved['evidence'] or marker['signature'] != saved['signature']:
        raise ValueError('Completed expert dataset/signature mismatch')
    last = torch.load(output/'last_checkpoint.pt', map_location='cpu', weights_only=False)
    history = last['history']
    best = min(history, key=lambda row:row['expert_selection_mse'])
    if (saved['epoch'] != best['epoch'] or saved['best_mse'] != best['expert_selection_mse']
            or (last['epoch'] != epochs and last['patience_counter'] < args['patience'])):
        raise ValueError('Expert Q1 selection or completion budget mismatch')
    return dict(**task, checkpoint=str(output/'best_checkpoint.pt'), checkpoint_sha256=file_hash(output/'best_checkpoint.pt'),
                source_sha=sha, evidence=replay, initial=args['common_initialization'],
                q1_mse=saved['best_mse'],
                train_batch_orders=[row['train_batch_order'] for row in history])


def code_sha():
    sha = os.environ.get('CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha.lower()):
        raise ValueError('CODE_COMMIT must be a full SHA')
    return sha


def manifest_path(args):
    return args.expert_manifest or args.output/'expert_manifest.json'


def write_manifest(args, sha):
    confirmed = []
    for task in inventory(args.backbone, args.output):
        confirmed.append(verify_completed(task, sha, args.data))
    for ticker,seed in grid.PAIRS:
        paired = [task for task in confirmed if task['ticker']==ticker and task['seed']==seed]
        reference = paired[0]
        for task in paired[1:]:
            epochs = min(len(reference['train_batch_orders']),len(task['train_batch_orders']))
            if (task['initial']['film_sha256']!=reference['initial']['film_sha256']
                    or task['train_batch_orders'][:epochs]!=reference['train_batch_orders'][:epochs]):
                raise ValueError('Paired common initialization or TRAIN batch order differs')
    value = dict(protocol=PROTOCOL, backbone=args.backbone,
                 base_sha=BASE_SHA, source_sha=sha, tasks=confirmed, test_previously_inspected=True)
    path = manifest_path(args)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Existing expert manifest differs; preserve it and use a separate output')
    else:
        atomic_json(path,value)


def audit_data(args):
    dataset_source = Path(__file__).resolve().parent/'forecasting_task/data_provider/dataset.py'
    source_hash = hashlib.sha256(dataset_source.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if source_hash != BASE_DATASET_SHA256:
        raise ValueError('Dataset_Custom source differs from the pinned base TRAIN/test/scaler implementation')
    records = []
    # Compare the exact old loader policy before any optimization or protocol masks.
    for ticker in shared.TICKERS:
        common = SimpleNamespace(seq_len=64,label_len=16,pred_len=3,batch_size=32,
                                 used_col_prefixes=list(grid.TEXT_GROUPS),data_path=ticker+'.parquet',text_window=40)
        reference = evidence(datasets(common,args.data))
        for window in (10,20,40,60,80):
            current = copy.copy(common)
            current.text_window = window
            actual = evidence(datasets(current,args.data))
            if actual != reference:
                raise ValueError(f'{ticker}/W{window}: original train/test policy changed')
        historical = []
        for root in (grid.RUNS/'context_price_mean_lag_grid_10_v1',
                     grid.RUNS/'context_price_mean_lag_grid_extension_10_v1'):
            for path in root.glob(f'full/*/{ticker}/seed7/output/best_checkpoint.pt'):
                saved = torch.load(path,map_location='cpu',weights_only=False)
                old_args = saved['args']
                if old_args.get('text_family') != grid.MEAN_GRID:
                    continue
                old_datasets = datasets(SimpleNamespace(**old_args),args.data)
                replay = evidence(old_datasets)
                if any(saved.get('scaler_'+key) != reference['scaler'][key] for key in ('mean','scale','var')):
                    raise ValueError(f'{path}: historical scaler differs')
                if saved.get('source_data_evidence') != reference['source'] or replay['windows'] != reference['windows']:
                    raise ValueError(f'{path}: historical TRAIN/windows differ')
                if saved.get('window_evidence') != {key:reference['windows'][key] for key in ('val','test')}:
                    raise ValueError(f'{path}: historical saved dates/targets differ')
                historical.append(str(path))
        records.append(dict(ticker=ticker, evidence=reference, current_policy_verification='full',
                            historical_artifact_verification='full' if historical else 'not_checked',
                            historical_paths=historical))
        print(f'AUDIT {ticker} excluded_cross_boundary_windows={reference["excluded_validation_windows"]}',flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output/'data_audit.json', dict(base_sha=BASE_SHA, records=records,
        base_dataset_sha256=BASE_DATASET_SHA256, actual_dataset_sha256=source_hash,
        baseline_source=BASELINE_SOURCE))
    print(f'AUDIT {len(records)} tickers; exact TRAIN/test replay; historical counts={[len(r["historical_paths"]) for r in records]}')


def train_tasks(args, kind, sha, smoke=False):
    if args.backbone != 'FiLM':
        raise ValueError('This queue trains FiLM only; other registered backbones use plan')
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.empty(1,device='cuda')
    tasks = inventory(args.backbone,args.output)
    tasks = [task for task in tasks if task['kind']==kind]
    if smoke:
        tasks = [dict(task,ticker='NVDA',seed=7,
                      output=str(args.output/'checks'/kind/task['cell']/'output'))
                 for task in tasks if task['ticker']==shared.TICKERS[0] and task['seed']==7]
    elapsed = 0.
    for i, task in enumerate(tasks,1):
        output = Path(task['output'])
        if (output/'completed.json').exists():
            verify_completed(task,sha,args.data,1 if smoke else 50)
            continue
        if output.exists() and not args.resume:
            raise ValueError(f'Incomplete run requires --resume: {output}')
        output.parent.mkdir(parents=True,exist_ok=True)
        command = shared.training_command(grid.MODE,task['ticker'],task['seed'],1 if smoke else 50,
                    sha,args.data,output,dict(task['config']))
        command[command.index('--model_type')+1] = args.backbone
        command += ['--selection_protocol',PROTOCOL]
        if kind == 'controls':
            command += ['--matched_original_baseline']
        if args.resume:
            command += ['--protocol_resume']
        if args.save_epochs and not smoke:
            command += ['--save_epochs', *map(str,args.save_epochs)]
        started = time.monotonic()
        print(f'START {kind} {task["cell"]}/{task["ticker"]}/seed{task["seed"]}',flush=True)
        with (output.parent/'train.log').open('a',encoding='utf-8') as log:
            status = subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
        if status:
            tail = (output.parent/'train.log').read_text(errors='replace').splitlines()[-30:]
            raise ValueError('\n'.join(tail))
        verify_completed(task,sha,args.data,1 if smoke else 50)
        elapsed += time.monotonic()-started
        print(f'OK completed={i}/{len(tasks)} ETA_seconds={elapsed/i*(len(tasks)-i):.1f}',flush=True)


def load_expert(task, data):
    output = Path(task['output'])
    if file_hash(task['checkpoint']) != task['checkpoint_sha256']:
        raise ValueError('Expert checkpoint changed after manifest selection')
    checkpoint = torch.load(task['checkpoint'], map_location='cpu', weights_only=False)
    args = SimpleNamespace(**checkpoint['args'])
    with torch.random.fork_rng(devices=[]):
        model = FACTORIES[task['backbone']](args)
        text = MatchedBaseline(3) if task['kind']=='controls' else build_text_model(args.text_model_config,args.group_weight_mode)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    text.load_state_dict(checkpoint['text_model_state_dict'],strict=True)
    for module in (model,text):
        module.eval().requires_grad_(False)
    ds = datasets(args,data)
    if evidence(ds) != task['evidence']:
        raise ValueError('Manifest source data differs')
    return model,text,args,ds


def cache_path(args,seed,test=False):
    return args.output/'cache'/f'seed{seed}_{"test" if test else "validation"}.pt'


@torch.no_grad()
def create_cache(args,seed,*,test=False):
    manifest = json.loads(manifest_path(args).read_text(encoding='utf-8'))
    if manifest['backbone'] != args.backbone or manifest['protocol'] != PROTOCOL:
        raise ValueError('Manifest protocol/backbone mismatch')
    records = [r for r in manifest['tasks'] if r['seed']==seed]
    expected = {(kind,cell,ticker) for kind, configs in (('experts',CONFIGS),('controls',{'matched_baseline':None}))
                for cell in configs for ticker in shared.TICKERS}
    if {(r['kind'],r['cell'],r['ticker']) for r in records} != expected or len(records)!=160:
        raise ValueError('Expected 150 exact experts and ten controls per seed, without duplicates')
    source = dict(manifest_sha256=file_hash(manifest_path(args)), checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in records},
                  split='test' if test else 'val', backbone=args.backbone, seed=seed)
    path = cache_path(args,seed,test)
    if path.exists():
        cache = validate_cache(torch.load(path,weights_only=True))
        if cache['source'] != source:
            raise ValueError('Cache provenance mismatch')
        for checkpoint,digest in source['checkpoints'].items():
            if file_hash(checkpoint)!=digest:
                raise ValueError('Cached expert changed')
        return cache
    parts = {key:[] for key in ('predictions','targets','features','ticker','components','baseline')}
    dates, target_dates, group_weights, masks = [], [], [], []
    for ticker_id,ticker in enumerate(shared.TICKERS):
        predictions, components, baseline = [], [], None
        reference = None
        for cell in (*CONFIGS,'matched_baseline'):
            task = next(r for r in records if r['ticker']==ticker and r['cell']==cell)
            model,text,config,ds = load_expert(task,args.data)
            split = ds['test' if test else 'val']
            before = (state_fingerprint(model),state_fingerprint(text))
            outputs, decomposition, weights, presence, inputs, targets = [], [], [], [], [], []
            batch_features = []
            for batch in DataLoader(split,batch_size=args.inference_batch_size,shuffle=False):
                x,e,xm,ym,decoder,y = run._prepare_batch(batch,torch.device('cpu'),3,16)
                financial = model(x,xm,decoder,ym)
                metadata = gate_batch_inputs(batch[6],e,torch.device('cpu'),use_window=True)
                if cell=='matched_baseline':
                    out = text(financial,x[:,-1],e[:,-1],**metadata)
                else:
                    out,detail = text(financial,x[:,-1],e[:,-1],price_window=x,return_details=True,**metadata)
                    component = torch.cat((.9*financial[:,None], (.1*x[:,-1:, :].expand(-1,3,-1))[:,None],
                        detail['contributions'][...,None].expand(-1,-1,-1,4)),1)
                    torch.testing.assert_close(component.sum(1),out,rtol=1e-5,atol=1e-6)
                    decomposition.append(component)
                    weights.append(detail['weights'])
                    presence.append(detail['present'])
                batch_features.append(features(x,e))
                outputs.append(out)
                inputs.append(x)
                targets.append(y)
            if before != (state_fingerprint(model),state_fingerprint(text)):
                raise ValueError('Frozen expert parameters/buffers changed')
            if not test:
                selection,_ = assignments(split)
                indices = selection['expert_selection']
                replay_mse = (torch.cat(outputs)[indices]-torch.cat(targets)[indices]).square().mean().item()
                if not np.isclose(replay_mse,task['q1_mse'],rtol=1e-4,atol=1e-6):
                    raise ValueError(f'{task["checkpoint"]}: Q1 MSE differs: {replay_mse} vs {task["q1_mse"]}')
            alignment = (torch.cat(inputs),torch.cat(targets))
            if reference is not None:
                for a,b in zip(alignment,reference):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
            reference = alignment
            if cell=='matched_baseline':
                baseline = torch.cat(outputs)
            else:
                predictions.append(torch.cat(outputs))
                components.append(torch.cat(decomposition))
                group_weights.append(torch.cat(weights))
                masks.append(torch.cat(presence))
        parts['predictions'].append(torch.stack(predictions,1))
        parts['components'].append(torch.stack(components,1))
        parts['targets'].append(reference[1])
        parts['features'].append(torch.cat(batch_features))
        parts['baseline'].append(baseline)
        parts['ticker'].append(torch.full((len(split),),ticker_id,dtype=torch.long))
        dates.extend(split.window_end_date(i) for i in range(len(split)))
        target_dates.extend([[str(date) for date in split.row_dates[i+64:i+67]] for i in range(len(split))])
    cache = {key:torch.cat(value) for key,value in parts.items()}
    cache.update(dates=dates,target_dates=target_dates,source=source,schema=SCHEMA,eps=EPS,
                 group_weights=torch.cat([torch.stack(group_weights[i:i+15],1) for i in range(0,len(group_weights),15)]),
                 masks=torch.cat([torch.stack(masks[i:i+15],1) for i in range(0,len(masks),15)]))
    cache['hashes'] = cache_hashes(cache)
    validate_cache(cache)
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(path,cache)
    return cache


def write_csv(path,rows):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def router_models(args, seed, cache):
    models = {}
    for mode in MODES:
        root = args.output/'router'/mode/f'seed{seed}'
        marker = json.loads((root/'completed.json').read_text())
        for name,digest in marker['artifacts'].items():
            if file_hash(root/name) != digest:
                raise ValueError('Router completed artifact hash mismatch')
        if marker['best_sha256'] != file_hash(root/'best_checkpoint.pt'):
            raise ValueError('Router best checkpoint hash mismatch')
        saved = torch.load(root/'best_checkpoint.pt',weights_only=True)
        if (saved['signature'] != marker['signature'] or saved['signature']['cache']!=cache['hashes']
                or saved['signature']['source']!=cache['source']):
            raise ValueError('Router selected on another cache')
        model = Router(mode).eval().requires_grad_(False)
        model.load_state_dict(saved['model'])
        last = torch.load(root/'last_checkpoint.pt',weights_only=True)
        if last.get('optimization_mode') != 'full_batch' or last.get('optimizer_steps') != last['epoch']:
            raise ValueError('Router optimizer step count differs from its epoch count')
        saved['fit_optimizer_steps'] = last['optimizer_steps']
        models[mode] = model,saved
    return models


def report(args):
    destination = args.output/'reports'/args.report_name
    if destination.exists():
        raise ValueError('Report exists; choose a new --report-name')
    # Freeze all selections from Q4 before permitting any test cache.
    choices = {}
    validation_caches = {}
    for seed in shared.SEEDS:
        cache = create_cache(args,seed)
        validation_caches[seed] = cache
        models = router_models(args,seed,cache)
        q4 = subset(cache,*PERIODS['router_validation'])
        errors = (q4['predictions']-q4['targets'][:,None]).square().mean((2,3))
        by_ticker = torch.stack([errors[q4['ticker']==i].mean(0) for i in range(10)])
        choices[seed] = dict(global_expert=int(by_ticker.mean(0).argmin()),
                             per_ticker=by_ticker.argmin(1), models=models)
    destination.mkdir(parents=True,exist_ok=False)
    rows, diagnostic_rows, daily = [], [], []
    for seed in shared.SEEDS:
        val = validation_caches[seed]
        for split, cache in (('validation',val), ('test',create_cache(args,seed,test=True))):
            p, y, ids = (cache[k] for k in ('predictions','targets','ticker'))
            n,k,h,c = p.shape
            names = list(CONFIGS)
            weights = {name:torch.nn.functional.one_hot(torch.full((n,),i),k).float() for i,name in enumerate(names)}
            weights['equal_mixture'] = torch.full((n,k),1/k)
            weights['best_global'] = torch.nn.functional.one_hot(torch.full((n,),choices[seed]['global_expert']),k).float()
            weights['best_per_ticker'] = torch.nn.functional.one_hot(choices[seed]['per_ticker'][ids],k).float()
            for mode,(model,saved) in choices[seed]['models'].items():
                weights[mode] = model.weights((cache['features']-saved['scaler_mean'])/saved['scaler_scale'])
            outputs = {name:(w[:,:,None,None]*p).sum(1) for name,w in weights.items()}
            outputs['matched_baseline'] = cache['baseline']
            baseline_errors = (cache['baseline']-y).square().mean((1,2))
            for name,prediction in outputs.items():
                errors = prediction-y
                mse,mae = errors.square().mean((1,2)),errors.abs().mean((1,2))
                for ticker_id,ticker in enumerate(shared.TICKERS):
                    for period,bounds in ((('test',None),) if split=='test' else
                                         (('full_validation_2022',None),('expert_selection',PERIODS['expert_selection']),
                                          ('router_validation',PERIODS['router_validation']))):
                        selected = ids == ticker_id
                        if bounds:
                            selected &= torch.tensor([all(bounds[0]<=date<bounds[1] for date in target)
                                                      for target in cache['target_dates']])
                        if not selected.any():
                            raise ValueError('Empty reporting period')
                        rows.append(dict(method=name,ticker=ticker,seed=seed,period=period,
                                         mse=mse[selected].mean().item(),mae=mae[selected].mean().item()))
                if split=='test':
                    for i in range(n):
                        daily.append(dict(method=name,ticker=shared.TICKERS[ids[i]],seed=seed,date=cache['dates'][i],
                                          mse_difference=(mse[i]-baseline_errors[i]).item()))
            for mode in MODES:
                w = weights[mode]
                blended = (cache['components']*w[:,:,None,None,None]).sum(1)
                torch.testing.assert_close(blended.sum(1),outputs[mode],rtol=1e-5,atol=1e-6)
                # Persist arrays once, preserving signed horizon/channel components and every mask.
                atomic_save(destination/f'{split}_{mode}_seed{seed}_diagnostics.pt',
                            dict(dates=cache['dates'],ticker=ids,weights=w,group_weights=cache['group_weights'],
                                 masks=cache['masks'],alpha=.1,components=blended,
                                 entropy=-(w*w.clamp_min(1e-12).log()).sum(1),
                                 expert_components=cache['components'],source=cache['source']))
    summary = []
    for method in sorted({r['method'] for r in rows}):
        item = dict(method=method,test_previously_inspected=True)
        if method in MODES:
            counts = {str(seed):choices[seed]['models'][method][1]['fit_optimizer_steps'] for seed in shared.SEEDS}
            item.update(optimization_mode='full_batch', optimizer_steps_total=sum(counts.values()),
                        optimizer_steps_by_seed=json.dumps(counts,sort_keys=True))
        for period in ('expert_selection','router_validation','full_validation_2022','test'):
            selected = [r for r in rows if r['method']==method and r['period']==period]
            for metric in ('mse','mae'):
                item[period+'_'+metric] = sum(r[metric] for r in selected)/len(selected)
        summary.append(item)
    lookup = {r['method']:r for r in summary}
    for row in summary:
        for control in ('matched_baseline','mean80_l1-2-5-10-20','best_global','best_per_ticker'):
            row['test_improvement_pct_vs_'+control] = 100*(1-row['test_mse']/lookup[control]['test_mse'])
    summary.sort(key=lambda r:r['router_validation_mse'])
    write_csv(destination/'per_run.csv',rows)
    write_csv(destination/'summary.csv',summary)
    write_csv(destination/'daily_paired_errors.csv',daily)
    # Average seeds before resampling contiguous blocks of dates, jointly across tickers.
    uncertainty = []
    for method in lookup:
        values = {}
        for row in daily:
            if row['method']==method:
                values.setdefault(row['date'],[]).append(row['mse_difference'])
        dates = sorted(values)
        series = np.asarray([np.mean(values[date]) for date in dates])
        rng = np.random.default_rng(7)
        draws = []
        block = min(20,len(series))
        for _ in range(500):
            indices = np.concatenate([(np.arange(block)+start)%len(series)
                                      for start in rng.integers(len(series),size=int(np.ceil(len(series)/block)))])[:len(series)]
            draws.append(series[indices].mean())
        uncertainty.append(dict(method=method,block_rows=block,draws=500,mean_mse_difference=float(series.mean()),
                                ci025=float(np.quantile(draws,.025)),ci975=float(np.quantile(draws,.975))))
    write_csv(destination/'paired_block_uncertainty.csv',uncertainty)
    atomic_json(destination/'selection.json',dict(test_previously_inspected=True,selection_period=PERIODS['router_validation'],
        selected_router=min(MODES,key=lambda m:lookup[m]['router_validation_mse']),
        fixed_experts={str(seed):dict(global_expert=choice['global_expert'],per_ticker=choice['per_ticker'].tolist())
                       for seed,choice in choices.items()},
        historical=dict(mean80_test_mse=.01381580,baseline_test_mse=.01532807,verification='partial'),
        interpretation='Expert weights can select financial forecasts; they are not causal effects or group importance.'))
    for row in summary:
        print(f'{row["method"]}: Q1={row["expert_selection_mse"]:.8f} Q4={row["router_validation_mse"]:.8f} '
              f'Val2022={row["full_validation_2022_mse"]:.8f} Test={row["test_mse"]:.8f} MAE={row["test_mae"]:.8f} '
              f'improvement={row["test_improvement_pct_vs_matched_baseline"]:.3f}%')


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','audit','smoke','train-experts','train-controls','cache','train-router','analyze','queue'])
    parser.add_argument('--backbone',choices=BACKBONES,default='FiLM')
    parser.add_argument('--protocol',choices=[PROTOCOL],default=PROTOCOL)
    parser.add_argument('--router-mode',choices=[*MODES,'all'],default='all')
    parser.add_argument('--expert-manifest',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data',type=Path,default=shared.DATA)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--save-epochs',type=int,nargs='*',default=[])
    parser.add_argument('--router-lr',type=float,default=.001)
    parser.add_argument('--router-epochs',type=int,default=200)
    parser.add_argument('--router-patience',type=int,default=20)
    parser.add_argument('--router-batch-size',type=int,default=128)
    parser.add_argument('--router-regularization',type=float,default=.001)
    parser.add_argument('--inference-batch-size',type=int,default=32)
    parser.add_argument('--report-name',default='final')
    args = parser.parse_args()
    args.output = (args.output or RESULTS/args.backbone).resolve()
    from forecasting_task.models import FiLM
    FiLM.device = torch.device('cpu') if args.stage in ('cache','analyze') else FiLM.device
    if args.stage=='plan':
        tasks = inventory(args.backbone,args.output)
        print(json.dumps(dict(base_sha=BASE_SHA,backbones=BACKBONES,tasks=tasks,
                              router_tasks=[dict(backbone=args.backbone,mode=mode,seed=seed,
                                  output=str(args.output/'router'/mode/f'seed{seed}'))
                                  for seed,mode in itertools.product(shared.SEEDS,MODES)],
                              counts=dict(experts=450,controls=30,router=9,total=489),
                              all_backbones=dict(experts=5400,controls=360,router=108),
                              missing=[t for t in tasks if not (Path(t['output'])/'completed.json').exists()]),indent=2))
        return
    if args.router_epochs<1 or args.router_patience<1 or args.router_batch_size<1 or args.router_lr<=0:
        parser.error('Router settings must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    sha = code_sha()
    with queue_lock(args.output/'.queue.lock'):
        stages = ['audit','smoke','train-experts','train-controls','cache','train-router','analyze'] if args.stage=='queue' else [args.stage]
        for stage in stages:
            print(f'STAGE {stage}',flush=True)
            if stage=='audit':
                audit_data(args)
            elif stage=='smoke':
                train_tasks(args,'experts',sha,smoke=True)
                train_tasks(args,'controls',sha,smoke=True)
            elif stage in ('train-experts','train-controls'):
                train_tasks(args,'experts' if stage=='train-experts' else 'controls',sha)
            elif stage=='cache':
                write_manifest(args,sha)
                FiLM.device = torch.device('cpu')
                for seed in shared.SEEDS:
                    create_cache(args,seed)
            elif stage=='train-router':
                tasks = list(itertools.product(shared.SEEDS,MODES if args.router_mode=='all' else [args.router_mode]))
                started = time.monotonic()
                for index,(seed,mode) in enumerate(tasks,1):
                    cache = validate_cache(torch.load(cache_path(args,seed),weights_only=True))
                    if cache['source']['manifest_sha256'] != file_hash(manifest_path(args)):
                        raise ValueError('Router cache expert manifest changed')
                    for path,digest in cache['source']['checkpoints'].items():
                        if file_hash(path)!=digest:
                            raise ValueError('Router cache expert checkpoint changed')
                    fit_router(cache,args.output/'router'/mode/f'seed{seed}',mode,seed,lr=args.router_lr,
                               epochs=args.router_epochs,patience=args.router_patience,batch_size=args.router_batch_size,
                               regularization=args.router_regularization,resume=args.resume)
                    print(f'ROUTER OK {mode}/seed{seed} completed={index}/{len(tasks)} '
                          f'ETA_seconds={(time.monotonic()-started)/index*(len(tasks)-index):.1f}',flush=True)
            elif stage=='analyze':
                FiLM.device = torch.device('cpu')
                report(args)
        if args.stage=='queue':
            print('ALL 489 FITS COMPLETED: experts=450 controls=30 router=9',flush=True)


if __name__=='__main__':
    cli()
