"""Backbone-independent mixtures, causal features and equal-ticker router fitting."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from forecasting_task.models.GroupText import TEXT_GROUPS, price_context
from forecasting_task.mean_router_protocol import atomic_save, atomic_json, file_hash, capture_rng, restore_rng

EPS = 1e-6
PRICE_LAGS = (1,2,5,10,20)
SCHEMA = [f'price_diff_l{lag}_{channel}' for lag in PRICE_LAGS for channel in ('open','high','low','close')]
SCHEMA += [f'{g}_{feature}' for g in TEXT_GROUPS for feature in ('current_present','fraction20','cosine_distance','cosine_available')]


def features(price, text):
    if text.shape[1] < 21:
        raise ValueError('Router text features require current and 20 previous rows')
    current = text[:, -1].float()
    history = text[:, -21:-1].float()
    present = history.abs().sum(-1) >= EPS
    mean = (history*present[...,None]).sum(1) / present.sum(1).clamp_min(1)[...,None]
    current_present = current.abs().sum(-1) >= EPS
    norm = current.norm(dim=-1)*mean.norm(dim=-1)
    available = current_present & present.any(1) & (norm >= EPS)
    cosine = (current*mean).sum(-1) / norm.clamp_min(EPS)
    distance = torch.where(available, 1-cosine.clamp(-1,1), torch.zeros_like(norm))
    fraction = (text[:, -20:].float().abs().sum(-1) >= EPS).float().mean(1)
    text_features = torch.stack((current_present.float(), fraction, distance, available.float()), -1).flatten(1)
    return torch.cat((price_context(price, PRICE_LAGS), text_features), -1)


class Router(nn.Module):
    def __init__(self, mode, experts=15):
        super().__init__()
        if mode not in ('constant','price','price_text'):
            raise ValueError('Unknown router mode')
        self.mode = mode
        if mode == 'constant':
            self.logits = nn.Parameter(torch.zeros(experts))
        else:
            with torch.random.fork_rng(devices=[]):
                self.linear = nn.Linear(20 if mode == 'price' else 40, experts)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def weights(self, features):
        logits = self.logits.expand(len(features),-1) if self.mode == 'constant' else self.linear(features[:, :self.linear.in_features])
        return logits.softmax(-1)

    def forward(self, predictions, features):
        if predictions.ndim != 4:
            raise ValueError('Expected expert predictions [N,K,H,C]')
        return (self.weights(features)[:,:,None,None]*predictions).sum(1)

    def penalty(self):
        return torch.cat([parameter.flatten() for parameter in self.parameters()]).square().mean()


def fit_scaler(train_features):
    mean = train_features.mean(0)
    scale = train_features.std(0, unbiased=False)
    return mean, torch.where(scale > EPS, scale, torch.ones_like(scale))


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def cache_hashes(cache):
    result = {key:tensor_hash(cache[key]) for key in
              ('predictions','targets','features','ticker','components','baseline','group_weights','masks')}
    result['metadata'] = hashlib.sha256(json.dumps(
        {key:cache[key] for key in ('dates','target_dates','source','schema','eps')},
        sort_keys=True).encode()).hexdigest()
    return result


def validate_cache(cache):
    required = ('predictions','targets','features','ticker','dates','target_dates','components','source')
    if any(key not in cache for key in required):
        raise ValueError('Incomplete cache')
    p, y, x, ids = (cache[key] for key in ('predictions','targets','features','ticker'))
    if p.ndim != 4 or p.shape[1:] != (15,3,4) or y.shape != (len(p),3,4) or x.shape != (len(p),40) or ids.shape != (len(p),):
        raise ValueError('Cache shapes differ')
    if any(not torch.isfinite(value).all() for value in (p,y,x,cache['components'])):
        raise ValueError('Nonfinite cache')
    if cache['components'].shape != (len(p),15,7,3,4):
        raise ValueError('Component cache shape differs')
    torch.testing.assert_close(cache['components'].sum(2), p, rtol=1e-5, atol=1e-6)
    if len(cache['dates']) != len(p) or len(cache['target_dates']) != len(p):
        raise ValueError('Cache dates differ')
    if any(len(dates) != 3 for dates in cache['target_dates']):
        raise ValueError('Cache must contain all three target dates for each window')
    if cache['baseline'].shape != y.shape or cache['group_weights'].shape != (len(p),15,5) or cache['masks'].shape != (len(p),15,5):
        raise ValueError('Baseline/group diagnostics cache shape mismatch')
    expected = cache_hashes(cache)
    if cache.get('hashes') != expected:
        raise ValueError('Cache fingerprint mismatch')
    if cache.get('schema') != SCHEMA or cache.get('eps') != EPS:
        raise ValueError('Cache feature schema mismatch')
    return cache


def subset(cache, start, end):
    indices = torch.tensor([i for i, dates in enumerate(cache['target_dates']) if all(start <= d < end for d in dates)])
    if not len(indices):
        raise ValueError(f'Empty cache period {start}/{end}')
    return {key:cache[key][indices] for key in ('predictions','targets','features','ticker')}


def equal_ticker_mse(prediction, target, ticker):
    errors = (prediction-target).square().mean((1,2))
    return torch.stack([errors[ticker==value].mean() for value in ticker.unique()]).mean()


def full_batch_step(model, optimizer, train, mean, scale, generator, batch_size, regularization):
    ids = train['ticker'].unique()
    weights = train['predictions'].new_empty(len(train['ticker']))
    for value in ids:
        present = train['ticker'] == value
        weights[present] = 1.0/(len(ids)*int(present.sum()))
    order = torch.randperm(len(weights), generator=generator)
    optimizer.zero_grad()
    for indices in order.split(batch_size):
        predicted = model(train['predictions'][indices], (train['features'][indices]-mean)/scale)
        loss = ((predicted-train['targets'][indices]).square().mean((1,2))*weights[indices]).sum()
        loss.backward()
    (regularization*model.penalty()).backward()
    optimizer.step()


def fit_router(cache, root, mode, seed, *, lr=.001, epochs=200, patience=20, batch_size=128,
               regularization=.001, resume=False, include_initial=False, provenance=None):
    validate_cache(cache)
    if cache['source'].get('split','val') != 'val':
        raise ValueError('Router fitting requires a validation cache, never test')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = subset(cache, '2022-04-01', '2022-10-01')
    val = subset(cache, '2022-10-01', '2023-01-01')
    mu, sigma = fit_scaler(train['features'])
    model = Router(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    signature = dict(mode=mode, seed=seed, lr=lr, epochs=epochs, patience=patience, batch_size=batch_size,
                     regularization=regularization, cache=cache['hashes'], source=cache['source'], schema=SCHEMA)
    if include_initial:
        signature['include_initial'] = True
    if provenance is not None:
        signature['provenance'] = provenance
    artifacts = {'best_checkpoint.pt','last_checkpoint.pt','history.json','summary.json'}
    if include_initial:
        artifacts.add('initial_checkpoint.pt')
    first, best, wait, history, optimizer_steps = 1, float('inf'), 0, [], 0
    if (root/'completed.json').exists():
        completed = json.loads((root/'completed.json').read_text())
        if completed['signature'] != signature:
            raise ValueError('Completed router configuration/cache differs')
        if set(completed.get('artifacts',{})) != artifacts:
            raise ValueError('Incomplete router completion marker')
        for name,digest in completed['artifacts'].items():
            if file_hash(root/name)!=digest:
                raise ValueError('Completed router artifact changed')
        saved = torch.load(root/'best_checkpoint.pt',weights_only=True)
        model.load_state_dict(saved['model'])
        return model.eval()
    if (root/'last_checkpoint.pt').exists():
        if not resume:
            raise ValueError('Existing router requires --resume')
        saved = torch.load(root/'last_checkpoint.pt', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Router resume cache/config mismatch')
        if saved.get('scheduler_state_dict', 'missing') is not None:
            raise ValueError('Router has no scheduler; incompatible saved scheduler state')
        if saved.get('optimization_mode') != 'full_batch' or saved.get('optimizer_steps') != saved['epoch']:
            raise ValueError('Router full-batch optimizer step evidence is inconsistent')
        if ((saved['optimizer_steps'] == 0 and (not include_initial or saved['optimizer']['state']))
                or (saved['optimizer_steps'] > 0 and (not saved['optimizer']['state'] or any(
                    int(state['step']) != saved['optimizer_steps'] for state in saved['optimizer']['state'].values())))):
            raise ValueError('AdamW state step counter differs from the recorded optimizer steps')
        if not torch.equal(saved['generator'], saved['rng']['generator']):
            raise ValueError('Router generator states disagree')
        torch.testing.assert_close(saved['scaler_mean'], mu, rtol=0, atol=0)
        torch.testing.assert_close(saved['scaler_scale'], sigma, rtol=0, atol=0)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], generator)
        optimizer_steps = saved['optimizer_steps']
        first, best, wait, history = saved['epoch']+1, saved['best'], saved['wait'], saved['history']
    elif include_initial:
        if any(root.iterdir()):
            raise ValueError('Incomplete new fit has no last checkpoint; preserve and inspect its files')
        with torch.no_grad():
            initial_prediction = model(val['predictions'], (val['features']-mu)/sigma)
            torch.testing.assert_close(initial_prediction, val['predictions'].mean(1), rtol=1e-5, atol=1e-6)
            best = equal_ticker_mse(initial_prediction, val['targets'], val['ticker']).item()
            train_mse = equal_ticker_mse(model(train['predictions'], (train['features']-mu)/sigma),
                                         train['targets'], train['ticker']).item()
        history = [dict(epoch=0, optimizer_steps=0, initial=True,
                        router_train_mse=train_mse, router_validation_mse=best)]
        state = dict(signature=signature, epoch=0, best=best, wait=0, history=history,
                     model=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
                     generator=generator.get_state(), rng=capture_rng(generator),
                     scheduler_state_dict=None, optimization_mode='full_batch', optimizer_steps=0,
                     scaler_mean=mu, scaler_scale=sigma)
        atomic_save(root/'initial_checkpoint.pt', state)
        atomic_save(root/'best_checkpoint.pt', state)
        atomic_save(root/'last_checkpoint.pt', state)
        atomic_json(root/'history.json', history)
        print(f'ROUTER mode={mode} seed={seed} steps=0/{epochs} train_mse={train_mse:.8f} '
              f'Q4_mse={best:.8f} best_step=0 patience=0/{patience}',flush=True)
    for epoch in range(first, epochs+1):
        if wait >= patience:
            break
        full_batch_step(model, optimizer, train, mu, sigma, generator, batch_size, regularization)
        optimizer_steps += 1
        with torch.no_grad():
            mse = equal_ticker_mse(model(val['predictions'], (val['features']-mu)/sigma), val['targets'], val['ticker']).item()
            train_mse = equal_ticker_mse(model(train['predictions'], (train['features']-mu)/sigma),
                                         train['targets'], train['ticker']).item()
        improved = mse < best
        best, wait = (mse, 0) if improved else (best, wait+1)
        history.append(dict(epoch=epoch, optimizer_steps=optimizer_steps,
                            router_train_mse=train_mse, router_validation_mse=mse))
        state = dict(signature=signature, epoch=epoch, best=best, wait=wait, history=history,
                     model=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
                     generator=generator.get_state(), rng=capture_rng(generator),
                     scheduler_state_dict=None, optimization_mode='full_batch', optimizer_steps=optimizer_steps,
                     scaler_mean=mu, scaler_scale=sigma)
        if improved:
            atomic_save(root/'best_checkpoint.pt', state)
        atomic_save(root/'last_checkpoint.pt', state)
        atomic_json(root/'history.json', history)
        best_step = min(history,key=lambda row:row['router_validation_mse'])['epoch']
        print(f'ROUTER mode={mode} seed={seed} steps={epoch}/{epochs} train_mse={train_mse:.8f} '
              f'Q4_mse={mse:.8f} best_step={best_step} patience={wait}/{patience}',flush=True)
    selected = torch.load(root/'best_checkpoint.pt', weights_only=True)
    atomic_json(root/'summary.json', dict(optimization_mode='full_batch', optimizer_steps=optimizer_steps,
        epochs_completed=optimizer_steps, best_epoch=selected['epoch'],
        best_optimizer_steps=selected['optimizer_steps'], scheduler=None))
    atomic_json(root/'completed.json', dict(signature=signature,best_sha256=file_hash(root/'best_checkpoint.pt'),
        artifacts={name:file_hash(root/name) for name in sorted(artifacts)}))
    if include_initial:
        model.load_state_dict(selected['model'])
        return model.eval()
    return model
