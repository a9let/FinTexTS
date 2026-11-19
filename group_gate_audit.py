"""Artifact checks shared by the 720 report and the existing window replay inspector."""
import csv
import json
import numpy as np
import torch
from forecasting_task.models.GroupGateExperiment import MODE, experiment_structure
from forecasting_task.models.GroupText import TEXT_GROUPS
from forecasting_task.experiment_evidence import source_data_evidence
from forecasting_task.window_evidence import SPLIT_POLICY
import run_han_head_ablation_queue as queue
EPOCH_LIMIT = 50


def configuration_issues(run, summary, args, checkpoint):
    missing, mismatch = [], []
    expected = {**queue.SETTINGS, 'num_epoch': EPOCH_LIMIT, 'used_col_prefixes': list(TEXT_GROUPS), 'group_weight_mode': 'context_price_only'}
    for key, value in expected.items():
        if key not in args:
            missing.append(key)
        elif args[key] != value:
            mismatch.append(key)
    if args.get('data_path') != run.ticker + '.parquet' or args.get('seed') != run.seed:
        mismatch.append('ticker/seed')
    keys = ('text_family','family_hidden_size','text_window','group_weight_condition')
    try:
        structure = experiment_structure(*(args[k] for k in keys), args['pred_len'])
    except (KeyError,ValueError) as error:
        return missing, mismatch+[str(error)]
    for name, source in (('args',args),('summary',summary),('checkpoint',checkpoint),('model_config',args.get('text_model_config',{}))):
        for key, value in {'text_temporal_pool':MODE,'temporal_pool_structure':structure, **{k:args[k] for k in keys}}.items():
            if key not in source:
                missing.append(name+'.'+key)
            elif source[key] != value:
                mismatch.append(name+'.'+key)
    if args.get('fixed_alpha') != .1 or args.get('text_weight_mode') != 'fixed' or args.get('split_policy') != SPLIT_POLICY:
        mismatch.append('fixed fusion or split')
    if args.get('sampling_policy') != 'independent_train_generator':
        mismatch.append('sampling policy')
    for name,source in (('args',args),('summary',summary),('checkpoint',checkpoint)):
        if source.get('fixed_alpha') != .1 or source.get('group_gate') != 'context_price_only':
            mismatch.append(name+'.fixed_fusion')
        if not args.get('code_commit') or source.get('code_commit') != args['code_commit']:
            mismatch.append(name+'.code_commit')
    if args['group_weight_condition'] == 'uniform' and args.get('gate_structure') is not None:
        mismatch.append('uniform has a gate')
    return missing,mismatch


def provenance(output, metadata):
    args = metadata['args']
    summary = json.loads((output/'summary.json').read_text())
    checkpoint = torch.load(output/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    sources = (args,summary,checkpoint)
    missing = []
    for key in ('common_initialization','source_data_evidence'):
        if any(key not in source for source in sources):
            missing.append(key)
        elif any(source[key] != args[key] for source in sources):
            raise ValueError('Inconsistent '+key)
    if args.get('text_temporal_pool') == MODE:
        from forecasting_task.models.GroupText import build_text_model
        with torch.random.fork_rng(devices=[]):
            model = build_text_model(args['text_model_config'],args['group_weight_mode'])
            model.load_state_dict(checkpoint['text_model_state_dict'],strict=True)
        if any(g.get('lr') != args['lr'] or g.get('weight_decay',0) != 0 for g in checkpoint['optimizer_state_dict']['param_groups']):
            raise ValueError('Optimizer differs from the saved Adam protocol')
        if any(source.get('experiment_initialization') != args.get('experiment_initialization') for source in sources):
            raise ValueError('Inconsistent text initialization')
        digest = args.get('experiment_initialization','')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            missing.append('experiment_initialization')
    initial = args.get('common_initialization',{})
    if len(initial.get('film_sha256','')) != 64:
        missing.append('FiLM initialization')
    source = args.get('source_data_evidence')
    if source:
        replay = source_data_evidence(metadata['datasets'])
        if source != replay:
            raise ValueError('Source text/price/date/Parquet fingerprints differ')
    history = [json.loads(line) for line in (output/'history.jsonl').read_text().splitlines() if line.strip()]
    orders = [r.get('train_batch_order') for r in history]
    if not history or any(order is None for order in orders):
        missing.append('actual batch order')
    elif summary.get('train_batch_orders') != orders or checkpoint.get('train_batch_orders') != orders[:checkpoint['epoch']]:
        raise ValueError('Batch order records differ')
    else:
        for order in orders:
            if order.get('batch_size') != args['batch_size'] or order.get('drop_last') is not True or len(order.get('sha256','')) != 64:
                raise ValueError('Invalid batch order fingerprint')
    best = min(history,key=lambda r:r['val_mse']) if history else None
    if history and ([r['epoch'] for r in history] != list(range(1,len(history)+1)) or summary.get('epochs_completed') != len(history)):
        raise ValueError('Epoch history incomplete')
    if best and (summary.get('best_epoch') != best['epoch'] or checkpoint['epoch'] != best['epoch'] or checkpoint.get('metrics') != best):
        raise ValueError('Validation checkpoint selection differs')
    if best and any(summary.get('best_'+key) != best[key] for key in ('val_mse','test_mse','test_mae')):
        raise ValueError('Summary best-checkpoint errors differ')
    return {'initialization':initial,'source':source,'orders':orders,'encoder':args.get('experiment_initialization')}, missing


def compare(left, right, gate_pair=False):
    import analyze_context_price_group_gru_attention as prior
    a,b = left[1],right[1]
    if a['evidence'] != b['evidence'] or a['split'] != b['split']:
        raise ValueError('Price windows, targets, dates or splits differ')
    prior.check_scalers(a['scaler'],b['scaler'],'paired experiment')
    for key in (*prior.COMMON,*prior.PRICE_FIELDS,'d_model','dropout'):
        if a['args'].get(key) != b['args'].get(key):
            raise ValueError('Common setting differs: '+key)
    if left[3]['initialization']['film_sha256'] != right[3]['initialization']['film_sha256'] or left[3]['source'] != right[3]['source']:
        raise ValueError('Common FiLM initialization or source data differ')
    conditions=[m['args'].get('group_weight_condition','uniform' if m['args'].get('group_gate')=='static' else 'price_only') for m in (a,b)]
    if conditions == ['price_only','price_only']:
        gates=[r[3]['initialization'].get('price_gate_sha256') for r in (left,right)]
        if any(g is None for g in gates):
            raise KeyError('Initial price gate unconfirmed')
        if gates[0] != gates[1]:
            raise ValueError('Initial price gates differ')
    epochs = min(len(left[3]['orders']),len(right[3]['orders']))
    if left[3]['orders'][:epochs] != right[3]['orders'][:epochs]:
        raise ValueError('Actual train batch order differs')
    if gate_pair and left[3]['encoder'] != right[3]['encoder']:
        raise ValueError('Paired common text initialization differs')
    if a['args'].get('text_temporal_pool') == MODE and b['args'].get('text_temporal_pool') == MODE and a['args'].get('code_commit') != b['args'].get('code_commit'):
        raise ValueError('New experiment code commits differ')
    return epochs


def contribution_rows(output, args):
    """Check signed group algebra and every OHLC output component before exporting."""
    rows = list(csv.DictReader((output/'validation_group_diagnostics.csv').open(encoding='utf-8',newline='')))
    sums, counts = {}, {}
    for row in rows:
        i = int(row['window_index'])
        weight, present = float(row['weight']),int(row['present'])
        if not 0 <= weight <= 1 or present not in (0,1):
            raise ValueError('Invalid weights/masks')
        if args.get('group_weight_condition') == 'uniform' and not np.isclose(weight,.2,atol=1e-7):
            raise ValueError('Nonuniform fixed coefficient')
        sums.setdefault(i,np.zeros(3))
        counts.setdefault(i,[]).append(weight)
        for h in range(1,4):
            c = float(row[f'contribution_h{h}'])
            if not np.isclose(c,.1*weight*present*float(row[f'head_h{h}']),rtol=1e-5,atol=1e-6):
                raise ValueError('Group contribution formula differs')
            sums[i][h-1] += c
    if any(len(v)!=5 or not np.isclose(sum(v),1.,atol=1e-6) for v in counts.values()):
        raise ValueError('Weights do not sum to one over all five groups')
    components = list(csv.DictReader((output/'validation_prediction_components.csv').open(encoding='utf-8',newline='')))
    if len(components) != len(sums)*12:
        raise ValueError('Incomplete OHLC reconstruction components')
    for r in components:
        c = sums[int(r['window_index'])][int(r['horizon'])-1]
        if not np.isclose(float(r['text_part']),c,rtol=1e-5,atol=1e-6) or not np.isclose(float(r['prediction']),float(r['film_part'])+float(r['anchor_part'])+c,rtol=1e-5,atol=1e-6):
            raise ValueError('Forecast reconstruction differs')
    return rows
