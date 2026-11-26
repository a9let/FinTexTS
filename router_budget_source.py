"""Read-only artifact verification for the router budget experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from forecasting_task.mean_router import SCHEMA, EPS, validate_cache, fit_scaler, subset
from forecasting_task.mean_router_protocol import BASE_SHA, BASE_DATASET_SHA256, PERIODS, PROTOCOL, file_hash
from forecasting_task.models.GroupText import TEXT_GROUPS
from mean_router_experiment import CONFIGS, MODES
from run_han_head_ablation_queue import TICKERS, SEEDS, SETTINGS

SOURCE_SHA = 'df0fdd20a9722935fa73489cb807db9f877c7d86'
RUNS = Path('/auto/datasets/kondyurin/FinTexTS/runs')
SOURCE_ROOT = RUNS/'context_price_mean_router_q1_v1'/'FiLM'
OUTPUT = RUNS/'context_price_mean_router_budget_v1'/'FiLM'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_artifacts(root, names):
    marker = read_json(root/'completed.json')
    if set(marker.get('artifacts',{})) != set(names):
        raise ValueError(f'{root}: completion artifact list is incomplete')
    for name,digest in marker['artifacts'].items():
        if file_hash(root/name) != digest:
            raise ValueError(f'{root/name}: saved digest mismatch')
    return marker


def verify_manifest(source):
    manifest = read_json(source/'expert_manifest.json')
    audit = read_json(source/'data_audit.json')
    if (manifest.get('protocol')!=PROTOCOL or manifest.get('backbone')!='FiLM'
            or manifest.get('source_sha')!=SOURCE_SHA or manifest.get('base_sha')!=BASE_SHA):
        raise ValueError('Source manifest protocol/backbone/code SHA differs')
    if audit.get('base_sha')!=BASE_SHA or audit.get('actual_dataset_sha256')!=BASE_DATASET_SHA256:
        raise ValueError('Missing or incompatible original data audit')
    records = audit.get('records',[])
    if len(records)!=10 or {row['ticker'] for row in records}!=set(TICKERS):
        raise ValueError('Original data audit must identify all ten tickers')
    expected = {(kind,cell,ticker,seed) for kind,cells in
                (('experts',CONFIGS),('controls',{'matched_baseline':None}))
                for cell in cells for ticker in TICKERS for seed in SEEDS}
    tasks = manifest.get('tasks',[])
    keys = [(t['kind'],t['cell'],t['ticker'],t['seed']) for t in tasks]
    if len(keys)!=480 or len(set(keys))!=480 or set(keys)!=expected:
        raise ValueError('Manifest must bind exactly 450 experts and 30 baselines')
    for row in records:
        if row.get('current_policy_verification')!='full':
            raise ValueError(f'{row["ticker"]}: original current-data policy is unverified')
    return manifest,audit


def verify_experts(source,manifest,audit):
    """Check saved states/evidence, without constructing or forwarding any expert."""
    audit_by_ticker = {row['ticker']:row for row in audit['records']}
    for task in manifest['tasks']:
        output = Path(task['output'])
        expected = source/task['kind']/task['cell']/task['ticker']/f'seed{task["seed"]}'/'output'
        if output.resolve()!=expected.resolve() or Path(task['checkpoint']).resolve()!=(output/'best_checkpoint.pt').resolve():
            raise ValueError('Manifest output is not the canonical source output')
        marker = verify_artifacts(output,('best_checkpoint.pt','last_checkpoint.pt','summary.json','history.jsonl','args.yaml'))
        if task['checkpoint_sha256']!=file_hash(output/'best_checkpoint.pt') or task['source_sha']!=SOURCE_SHA:
            raise ValueError('Manifest expert hash/source SHA differs')
        saved = torch.load(output/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        args,proof = saved['args'],saved['evidence']
        if args.get('used_col_prefixes')!=list(TEXT_GROUPS):
            raise ValueError('Saved expert group order differs')
        text_state = saved['text_model_state_dict']
        if task['kind']=='experts':
            shapes = {f'core.group_mlps.{group}.0.{suffix}':shape
                      for group in TEXT_GROUPS for suffix,shape in (('weight',(3,384)),('bias',(3,)))}
            shapes.update({'core.gate.weight':(5,4*len(CONFIGS[task['cell']]['price_gate_lags'])+10),
                           'core.gate.bias':(5,), 'core.fixed_alpha':()})
            if not np.isclose(float(text_state['core.fixed_alpha']),.1,atol=1e-7):
                raise ValueError('Expert fixed alpha differs')
        else:
            shapes = {'layers.0.weight':(3,384),'layers.0.bias':(3,)}
        if set(text_state)!=set(shapes) or any(tuple(text_state[key].shape)!=shape for key,shape in shapes.items()):
            raise ValueError('Source text checkpoint architecture differs')
        if any(tuple(saved['model_state_dict'].get(key,torch.empty(0)).shape)!=shape
               for key,shape in (('affine_weight',(1,1,4)),('affine_bias',(1,1,4)),('mlp.weight',(1,3)))):
            raise ValueError('Source financial checkpoint is not the expected full FiLM')
        config = CONFIGS[task['cell']] if task['kind']=='experts' else CONFIGS['mean20_l1-5']
        for key,value in {**SETTINGS,**config,'model_type':'FiLM','seed':task['seed'],
                          'code_commit':SOURCE_SHA,'selection_protocol':PROTOCOL,'num_epoch':50,
                          'matched_original_baseline':task['kind']=='controls'}.items():
            if args.get(key)!=value:
                raise ValueError(f'{output}: expert configuration differs at {key}')
        reference = audit_by_ticker[task['ticker']]['evidence']
        if proof!=task['evidence'] or proof!=reference or saved['scaler']!=proof['scaler']:
            raise ValueError('Expert split/scaler evidence differs from manifest/data audit')
        if marker['signature']!=saved['signature'] or saved['best_mse']!=task['q1_mse']:
            raise ValueError('Expert selection evidence differs')
        last = torch.load(output/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        if (last['signature']!=saved['signature'] or last['evidence']!=proof or last['scaler']!=proof['scaler']
                or [row['epoch'] for row in last['history']]!=list(range(1,last['epoch']+1))
                or (last['epoch']!=50 and last['patience_counter']!=10)
                or last['epoch']>50):
            raise ValueError('Source expert last checkpoint budget/resume evidence differs')
        orders = [row['train_batch_order'] for row in last['history']]
        if orders!=task['train_batch_orders'] or args.get('common_initialization')!=task['initial']:
            raise ValueError('Source expert initialization or TRAIN order evidence differs')
        best = min(last['history'],key=lambda row:row['expert_selection_mse'])
        if (best['epoch']!=saved['epoch'] or best['expert_selection_mse']!=saved['best_mse']
                or saved.get('scheduler_state_dict','missing') is not None):
            raise ValueError('Expert best checkpoint is not Q1-selected')
    return {
        'status':'verified_saved_artifacts',
        'expert_inference_replay':'not_performed',
        'raw_data_replay':'not_performed',
        'historical_evidence':{row['ticker']:row.get('historical_artifact_verification','unverified')
                               for row in audit['records']},
        'expert_order_evidence':'pinned source code df0fdd2 and digest-bound manifest; no standalone order field in old cache',
    }


def check_cache(source,manifest,cache,seed,test=False):
    validate_cache(cache)
    tasks = [task for task in manifest['tasks'] if task['seed']==seed]
    expected_source = dict(manifest_sha256=file_hash(source/'expert_manifest.json'),
        checkpoints={t['checkpoint']:t['checkpoint_sha256'] for t in tasks},
        split='test' if test else 'val',backbone='FiLM',seed=seed)
    if cache['source']!=expected_source or list(cache['source']['checkpoints'])!=list(expected_source['checkpoints']):
        raise ValueError('Cache seed/backbone/source/checkpoint order differs')
    ordered = list(dict.fromkeys(t['cell'] for t in tasks if t['kind']=='experts'))
    if ordered!=list(CONFIGS) or cache['schema']!=SCHEMA or cache['eps']!=EPS:
        raise ValueError('Expert or feature order is unverified/incompatible')
    if set(cache['ticker'].tolist())!=set(range(10)):
        raise ValueError('Cache must contain exactly the ten source tickers')
    offset = 0
    for ticker_id,ticker in enumerate(TICKERS):
        indices = torch.where(cache['ticker']==ticker_id)[0]
        if not torch.equal(indices,torch.arange(offset,offset+len(indices))):
            raise ValueError('Cache ticker/window order differs from source construction')
        offset += len(indices)
        task = next(t for t in tasks if t['kind']=='experts' and t['ticker']==ticker)
        proof = task['evidence']['windows']['test' if test else 'val']
        dates = [cache['dates'][i] for i in indices.tolist()]
        targets = [cache['target_dates'][i] for i in indices.tolist()]
        if any(len(d)!=3 or not (date<d[0]<d[1]<d[2]) for date,d in zip(dates,targets)):
            raise ValueError('Cache target horizon/dates are invalid')
        if dates!=sorted(set(dates)):
            raise ValueError('Cache input dates are duplicated or unordered')
        dates_hash = hashlib.sha256(('\n'.join(dates)+'\n').encode()).hexdigest()
        target_dates_hash = hashlib.sha256(''.join('|'.join(d)+'\n' for d in targets).encode()).hexdigest()
        target_hash = hashlib.sha256(cache['targets'][indices].numpy().astype('<f4').tobytes()).hexdigest()
        if (proof['windows']!=len(indices) or proof['dates_sha256']!=dates_hash
                or proof['target_dates_sha256']!=target_dates_hash or proof['targets_sha256']!=target_hash):
            raise ValueError('Cache windows/dates/normalized targets differ from saved expert evidence')
        if test:
            if any(d[0]<'2023-01-01' for d in targets):
                raise ValueError('Test targets overlap fit/selection periods')
        else:
            q1 = torch.tensor([i for i in indices.tolist()
                               if all('2022-01-01'<=d<'2022-04-01' for d in cache['target_dates'][i])])
            if not len(q1):
                raise ValueError('Cache has no Q1 windows')
            for k,cell in enumerate(CONFIGS):
                expert = next(t for t in tasks if t['cell']==cell and t['ticker']==ticker)
                metric = (cache['predictions'][q1,k]-cache['targets'][q1]).square().mean().item()
                if not np.isclose(metric,expert['q1_mse'],rtol=1e-4,atol=1e-6):
                    raise ValueError('Cache expert axis/Q1 errors differ from manifest')
            baseline = next(t for t in tasks if t['kind']=='controls' and t['ticker']==ticker)
            metric = (cache['baseline'][q1]-cache['targets'][q1]).square().mean().item()
            if not np.isclose(metric,baseline['q1_mse'],rtol=1e-4,atol=1e-6):
                raise ValueError('Cached matched baseline Q1 MSE differs')
    return cache


def validation_cache(source,manifest,seed):
    path = source/'cache'/f'seed{seed}_validation.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Required full server cache is missing: {path}')
    return check_cache(source,manifest,torch.load(path,map_location='cpu',weights_only=True),seed)


def old_state(source,mode,seed,cache):
    root = source/'router'/mode/f'seed{seed}'
    marker = verify_artifacts(root,('best_checkpoint.pt','last_checkpoint.pt','history.json','summary.json'))
    best = torch.load(root/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    last = torch.load(root/'last_checkpoint.pt',map_location='cpu',weights_only=True)
    signature = dict(mode=mode,seed=seed,lr=.001,epochs=200,patience=20,batch_size=128,
                     regularization=.001,cache=cache['hashes'],source=cache['source'],schema=SCHEMA)
    if marker['signature']!=signature or best['signature']!=signature or last['signature']!=signature:
        raise ValueError('Original router configuration/cache differs from 200/20 protocol')
    if marker['best_sha256']!=file_hash(root/'best_checkpoint.pt'):
        raise ValueError('Original router best digest differs')
    selected = min(last['history'],key=lambda row:row['router_validation_mse'])
    if best['epoch']!=selected['epoch'] or best['best']!=selected['router_validation_mse']:
        raise ValueError('Original best is not the earliest Q4 minimum')
    if last['epoch']!=200 and last['wait']<20:
        raise ValueError('Original router fit is incomplete')
    mu,sigma = fit_scaler(subset(cache,*PERIODS['router_train'])['features'])
    for state in (best,last):
        torch.testing.assert_close(state['scaler_mean'],mu,rtol=0,atol=0)
        torch.testing.assert_close(state['scaler_scale'],sigma,rtol=0,atol=0)
        if (state.get('scheduler_state_dict','missing') is not None
                or state['optimizer_steps']!=state['epoch']):
            raise ValueError('Original scheduler/optimizer steps differ')
    return best,last


def compare_history(old,new):
    previous = {row['epoch']:row for row in old}
    current = {row['epoch']:row for row in new if row['epoch']>0}
    shared = sorted(previous.keys() & current.keys())
    if not shared or shared!=list(range(1,len(shared)+1)):
        raise ValueError('Common optimizer-step trajectory is incomplete')
    maximum = {}
    for key in ('router_train_mse','router_validation_mse'):
        a = np.asarray([previous[i][key] for i in shared])
        b = np.asarray([current[i][key] for i in shared])
        if not np.allclose(a,b,rtol=1e-5,atol=1e-7):
            raise ValueError(f'Old/new shared optimizer-step trajectory differs: {key}')
        maximum[key+'_max_abs_difference'] = float(np.abs(a-b).max())
    return dict(common_steps=len(shared),old_steps=len(previous),
                covers_old_stop=len(shared)==len(previous),verification='common_metric_trajectory_verified',
                parameter_gradient_replay='synthetic_regression_only',**maximum)


def source_snapshot(source):
    manifest,audit = verify_manifest(source)
    status = verify_experts(source,manifest,audit)
    hashes = {str(source/name):file_hash(source/name) for name in
              ('expert_manifest.json','data_audit.json','reports/final/selection.json','reports/final/summary.csv')}
    for task in manifest['tasks']:
        root = Path(task['output'])
        for name in ('completed.json','best_checkpoint.pt','last_checkpoint.pt','summary.json','history.jsonl','args.yaml'):
            hashes[str(root/name)] = file_hash(root/name)
    for seed in SEEDS:
        cache = validation_cache(source,manifest,seed)
        hashes[str(source/'cache'/f'seed{seed}_validation.pt')] = file_hash(source/'cache'/f'seed{seed}_validation.pt')
        for mode in MODES:
            old_state(source,mode,seed,cache)
            root = source/'router'/mode/f'seed{seed}'
            for name in ('completed.json','best_checkpoint.pt','last_checkpoint.pt','history.json','summary.json'):
                hashes[str(root/name)] = file_hash(root/name)
    return dict(source_root=str(source),source_code_sha=manifest['source_sha'],files=hashes,verification=status)
