"""Opt-in Q1 checkpoint selection using the unchanged runner epoch implementation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from forecasting_task.window_evidence import window_evidence, SPLIT_POLICY
from forecasting_task.experiment_evidence import source_data_evidence
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset

PROTOCOL = 'temporally_separated'
PERIODS = {'expert_selection': ('2022-01-01', '2022-04-01'),
           'router_train': ('2022-04-01', '2022-10-01'),
           'router_validation': ('2022-10-01', '2023-01-01')}
BASE_SHA = 'da3cbe771cd8329943d4321a492bfa07c8ffa316'
BASE_DATASET_SHA256 = 'cb25a4d444fbe158fa4338215ef0eefe338448b0ef8284b2d7c0e78151a4758f'
BACKBONES = ('Autoformer', 'Crossformer', 'DLinear', 'FiLM', 'Informer', 'iTransformer',
             'Nonstationary_Transformer', 'PatchTST', 'Reformer', 'TiDE', 'Transformer', 'TSMixer')
BASELINE_SOURCE = 'db35a2d:forecasting_task/run.py TextModel; dataset.py groupby(stan).mean; run_baselines.sh'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignments(dataset):
    result = {name: [] for name in PERIODS}
    excluded = []
    for index in range(len(dataset)):
        dates = dataset.row_dates[index+dataset.seq_len:index+dataset.seq_len+dataset.pred_len]
        matches = [name for name, (start, end) in PERIODS.items()
                   if len(dates) == dataset.pred_len and all(start <= date < end for date in dates)]
        if len(matches) > 1:
            raise ValueError('Overlapping period assignments')
        (result[matches[0]] if matches else excluded).append(index)
    return result, excluded


def period_evidence(dataset, indices):
    h = hashlib.sha256()
    dates = []
    for index in indices:
        batch = dataset[index]
        h.update(np.asarray(batch[0], dtype='<f4').tobytes())
        h.update(np.asarray(batch[1][-dataset.pred_len:], dtype='<f4').tobytes())
        target_dates = dataset.row_dates[index+dataset.seq_len:index+dataset.seq_len+dataset.pred_len]
        h.update(('|'.join(target_dates)+'\n').encode())
        dates.extend(target_dates)
    return dict(windows=len(indices), sha256=h.hexdigest(),
                first_target=min(dates) if dates else None, last_target=max(dates) if dates else None)


def evidence(datasets):
    groups, excluded = assignments(datasets['val'])
    if any(not value for value in groups.values()):
        raise ValueError('Every named validation period must contain complete target windows')
    result = dict(protocol=PROTOCOL, base_sha=BASE_SHA, split_policy=SPLIT_POLICY,
                  windows={name: window_evidence(ds) for name, ds in datasets.items()},
                  source=source_data_evidence(datasets),
                  scaler={name: getattr(datasets['train'].scaler, name+'_').tolist() for name in ('mean','scale','var')},
                  assignments={name: period_evidence(datasets['val'], indices) for name, indices in groups.items()},
                  excluded_validation_windows=len(excluded))
    test = result['windows']['test']
    dates = datasets['test'].row_dates[datasets['test'].seq_len:]
    if any(date < '2023-01-01' for date in dates):
        raise ValueError('Original test targets overlap expert/router fit or selection periods')
    return result


class BaselineDataset(ContextNoveltyDataset):
    """Historical per-dimension mean skips NaNs before filling remaining missing values."""
    def __init__(self, base, include_calendar_days=False, extended_text_window=None):
        super().__init__(base, include_calendar_days, extended_text_window)
        frame = pd.read_parquet(Path(base.root_path)/base.data_path)
        frame = frame[('2019-01-01'<=frame['date']) & (frame['date']<'2023-12-19')]
        values = np.stack([frame[[f'{g}_emb{i}' for i in range(384)]].to_numpy(dtype=np.float64)
                           for g in base.used_col_prefixes],1)
        count = (~np.isnan(values)).sum(1)
        pooled = np.nansum(values,axis=1)/count.clip(1)
        dates = frame['date'].astype(str).to_numpy()
        indices = {date:i for i,date in enumerate(dates)}
        self.baseline_text = pooled[[indices[d] for d in self.row_dates]].astype(np.float32)

    def __getitem__(self, index):
        *fields, metadata = super().__getitem__(index)
        metadata['baseline_embedding'] = self.baseline_text[index+self.seq_len-1]
        return (*fields, metadata)


class MatchedBaseline(nn.Module):
    """Historical single linear head, fixed anchored blend and original mean-group mask."""
    def __init__(self, pred_len):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(384, pred_len)])

    def get_alpha(self):
        return self.layers[0].weight.new_tensor(.1)

    def fusion_state(self):
        return dict(text_alpha=.1, baseline_source=BASELINE_SOURCE)

    def forward(self, price, last, embeddings, **kwargs):
        pooled = kwargs.get('baseline_embedding', embeddings.mean(1))
        present = pooled.abs().sum(-1) >= 1e-6
        forecast = (last[:, None] + self.layers[0](pooled)[..., None]) * present[:, None, None]
        return .9*price + .1*forecast


def capture_rng(generator):
    numpy_state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                generator=generator.get_state())


def restore_rng(state, generator):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    generator.set_state(state['generator'])


def train_selected(runner, args, model, text, optimizer, datasets, loaders, generator, device):
    """Reuse _run_epoch and the exact original train loader; evaluate Q1 alone."""
    root = Path(args.logdir)
    proof = evidence(datasets)
    groups, _ = assignments(datasets['val'])
    selection = DataLoader(Subset(datasets['val'], groups['expert_selection']),
                           batch_size=args.batch_size, shuffle=False)
    if not len(loaders['train']):
        raise ValueError('Original TRAIN has no complete batch; preserve its policy and inspect the data')
    signature = dict(args=vars(args).copy(), evidence=proof)
    for key in ('protocol_resume', 'logdir'):
        signature['args'].pop(key, None)
    last_path = root/'last_checkpoint.pt'
    history, best, patience, first = [], float('inf'), 0, 1
    saved_best_epoch = None
    if last_path.exists():
        if not args.protocol_resume:
            raise ValueError('Existing last checkpoint requires --protocol_resume')
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Resume configuration/data evidence mismatch')
        if saved.get('scheduler_state_dict', 'missing') is not None:
            raise ValueError('Expert protocol has no scheduler; incompatible saved scheduler state')
        if saved['scaler'] != proof['scaler']:
            raise ValueError('Saved price scaler differs from the original train-only scaler')
        model.load_state_dict(saved['model_state_dict'])
        text.load_state_dict(saved['text_model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        restore_rng(saved['rng'], generator)
        history, best, patience = saved['history'], saved['best_mse'], saved['patience_counter']
        saved_best_epoch = saved['best_epoch']
        first = saved['epoch']+1
    started = time.monotonic()
    for epoch in range(first, args.num_epoch+1):
        if patience >= args.patience:
            break
        before = time.monotonic()
        train_mse, train_mae = runner._run_epoch(loaders['train'], datasets['train'], model, text, args, device, optimizer)
        with torch.no_grad():
            val_mse, val_mae = runner._run_epoch(selection, datasets['val'], model, text, args, device)
        history.append(dict(epoch=epoch, train_mse=train_mse, train_mae=train_mae,
                            expert_selection_mse=val_mse, expert_selection_mae=val_mae,
                            train_batch_order=loaders['train'].fintexts_batch_order,
                            seconds=time.monotonic()-before))
        improved = val_mse < best
        if improved:
            best, patience = val_mse, 0
        else:
            patience += 1
        state = dict(signature=signature, args=vars(args).copy(), epoch=epoch,
                     model_state_dict=copy.deepcopy(model.state_dict()),
                     text_model_state_dict=copy.deepcopy(text.state_dict()),
                     optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=None,
                     scaler=proof['scaler'], evidence=proof, best_mse=best,
                     best_epoch=epoch if improved else saved_best_epoch,
                     patience_counter=patience, rng=capture_rng(generator), history=history,
                     test_previously_inspected=True)
        saved_best_epoch = state['best_epoch']
        if improved:
            atomic_save(root/'best_checkpoint.pt', state)
        atomic_save(last_path, state)
        if epoch in args.save_epochs:
            atomic_save(root/f'epoch{epoch}.pt', state)
        (root/'history.jsonl.tmp').write_text(''.join(json.dumps(row)+'\n' for row in history), encoding='utf-8')
        os.replace(root/'history.jsonl.tmp', root/'history.jsonl')
        print(f'Q1 epoch={epoch} train={train_mse:.8f} selection={val_mse:.8f} patience={patience}', flush=True)
    best_state = torch.load(root/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    summary = dict(protocol=PROTOCOL, selection_policy=PERIODS['expert_selection'], scheduler=None,
                   args=vars(args), evidence=proof, best_epoch=best_state['epoch'], epochs_completed=len(history),
                   expert_selection_mse=best_state['best_mse'], test_previously_inspected=True,
                   seconds=sum(row['seconds'] for row in history),
                   estimated_checkpoint_bytes=(root/'last_checkpoint.pt').stat().st_size,
                   optional_epoch_disk_bytes=(root/'last_checkpoint.pt').stat().st_size*len(args.save_epochs))
    atomic_json(root/'summary.json', summary)
    atomic_json(root/'completed.json', dict(signature=signature,
        artifacts={name:file_hash(root/name) for name in ('best_checkpoint.pt','last_checkpoint.pt','summary.json','history.jsonl','args.yaml')}))
    return summary
