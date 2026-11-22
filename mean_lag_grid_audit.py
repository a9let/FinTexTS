"""Shared experiment definitions and read-only checks for saved control runs."""

import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

import analyze_context_price_group_gru_attention as prior
from analyze_blend_price_anchor import execution_device, film_module
from group_gate_audit import provenance
import run_han_head_ablation_queue as shared
from run_group_gate_queue import configuration
from forecasting_task.experiment_evidence import common_initialization, state_fingerprint
from forecasting_task.models.GroupGateExperiment import MEAN_GRID, MODE, experiment_structure
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model


RUNS = shared.DATA.parents[1] / 'runs'
RESULTS = RUNS / 'context_price_mean_lag_grid_10_v1'
NORMAL_ROOT = RUNS / 'group_memory_ablation_720_v1/phases/full/xtime_group/w20_price_only'
BASE_COMMIT = 'cb03e54192529912b6a41b24ce074fd91275b818'
EPOCH_LIMIT = 50
WINDOWS = (10, 20, 40)
LAG_SETS = ((1, 5), (1, 5, 20), (1, 2, 5, 10, 20))
PAIRS = tuple(itertools.product(shared.TICKERS, shared.SEEDS))
AUDIT_ERRORS = (OSError, KeyError, ValueError, TypeError, RuntimeError, AssertionError,
                EOFError, pickle.UnpicklingError, yaml.YAMLError)


def cell_name(window, lags):
    return f'mean{window}_l' + '-'.join(map(str, lags))


CONTROL = cell_name(20, LAG_SETS[0])
CONFIGS = {cell_name(window, lags): dict(configuration(MEAN_GRID, 32, window, 'price_only'),
                                       price_gate_lags=list(lags))
           for window, lags in itertools.product(WINDOWS, LAG_SETS)}
NORMAL = dict(configuration('xtime_group', 32, 20, 'price_only'),
              kd_beta=0., memory_stage='normal', memory_smoke=False)


def normal_destination(root, ticker, seed):
    return root / ticker / f'seed{seed}' / 'normal'


def grid_destination(root, cell, ticker, seed, area='full'):
    return root / area / cell / ticker / f'seed{seed}'


def replay_orders(size, batch_size, seed, epochs, best_epoch):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(range(size), batch_size=batch_size, shuffle=True,
                        drop_last=True, generator=generator)
    orders = []
    for epoch in range(1, epochs + 1):
        digest = hashlib.sha256()
        windows = batches = 0
        for batch in loader:
            indices = np.asarray(batch, dtype='<i8')
            digest.update(np.asarray([len(indices)], dtype='<i8').tobytes())
            digest.update(indices.tobytes())
            windows += len(indices)
            batches += 1
        orders.append(dict(sha256=digest.hexdigest(), windows=windows, batches=batches,
                           batch_size=batch_size, drop_last=True))
        if epoch == best_epoch:
            best_state = generator.get_state()
    return orders, best_state


def grid_config(text_config):
    config = {key: value for key, value in text_config.items() if key != 'kd_beta'}
    config.update(text_family=MEAN_GRID, price_gate_lags=[1, 5])
    config['temporal_pool_structure'] = experiment_structure(MEAN_GRID, 32, 20, 'price_only',
                                                            config['pred_len'])
    return config


@torch.no_grad()
def prove_normal_forward(old, new, dataset):
    """Compare saved weights on real validation inputs and explicit empty groups."""
    new.gate.load_state_dict(old.gate.state_dict(), strict=True)
    for group in TEXT_GROUPS:
        new.group_mlps[group][0].load_state_dict(old.temporal_pool.experts[0][group].state_dict(), strict=True)
    old.eval()
    new.eval()
    indices = sorted({0, len(dataset)//2, len(dataset)-1})
    samples = [dataset[i] for i in indices]
    price = torch.as_tensor(np.stack([sample[0] for sample in samples]), dtype=torch.float32)
    text = torch.as_tensor(np.stack([sample[2] for sample in samples]), dtype=torch.float32)
    text = torch.cat((text, text[:1].clone(), torch.zeros_like(text[:1])))
    text[-2, :, 0] = 0
    text[-2, -1, 1] = 0
    text[-2, -2, 1, 0] = 1  # Text exists in the mean window but is absent today.
    price = torch.cat((price, price[:1].expand(2, -1, -1)))
    last = price[:, -1]
    film_prediction = price[:, :3]
    kwargs = dict(text_window=text, price_window=price, novelty=torch.zeros(len(price), 5),
                  pair_available=torch.zeros(len(price), 5), return_details=True)
    old_y, old_detail = old(film_prediction, last, text[:, -1], **kwargs)
    new_y, new_detail = new(film_prediction, last, text[:, -1], **kwargs)
    torch.testing.assert_close(new_y, old_y, rtol=0, atol=0)
    for key in ('present', 'weights', 'contributions'):
        torch.testing.assert_close(new_detail[key], old_detail[key], rtol=0, atol=0)


def check_initialization(args, checkpoint, historical, dataset):
    text_config = args['text_model_config']
    with torch.random.fork_rng(devices=[]), execution_device(torch.device('cpu')):
        torch.random.default_generator.manual_seed(args['seed'])
        film = film_module.Model(SimpleNamespace(**args))
        after_film = torch.get_rng_state()
        text = build_text_model(text_config, args['group_weight_mode'])
        initial = common_initialization(film, text)
        if initial != args['common_initialization']:
            raise ValueError('Saved FiLM, gate or head initialization differs from seeded replay')
        if args['experiment_initialization'] != state_fingerprint(text.core, exclude_prefixes=('gate.',)):
            raise ValueError('Saved text initialization differs from seeded replay')
        if not all(torch.count_nonzero(param) == 0 for param in text.gate.parameters()):
            raise ValueError('Initial gate weights and bias must be zero')
        heads_sha = initial['heads_sha256']
        if historical:
            if args['temporal_initialization_sha256'] != state_fingerprint(text.temporal_pool):
                raise ValueError('Saved xTime normal initialization differs from seeded replay')
            torch.set_rng_state(after_film)
            new = build_text_model(grid_config(text_config), args['group_weight_mode'])
            for group in TEXT_GROUPS:
                if state_fingerprint(text.temporal_pool.experts[0][group]) != state_fingerprint(new.group_mlps[group][0]):
                    raise ValueError('xTime and grid heads have different initial parameters')
            heads_sha = state_fingerprint(new.group_mlps)
            text.load_state_dict(checkpoint['text_model_state_dict'], strict=True)
            if not text.temporal_pool.normal_only.item():
                raise ValueError('Checkpoint does not enable xTime normal-only')
            prove_normal_forward(text, new, dataset)
    return {**initial, 'heads_sha256': heads_sha}


def checked_run(destination, config, data, commit, ticker, seed, cache, *, historical=False):
    output = destination / 'output'
    expected = shared.expected_config(MODE, ticker, seed, EPOCH_LIMIT, commit, data, config)
    summary = shared.validate_completed(destination, expected)
    marker = json.loads((destination / 'completed.json').read_text(encoding='utf-8'))
    if marker.get('config') != expected or not math.isfinite(marker['wall_seconds']) or marker['wall_seconds'] < 0:
        raise ValueError('Completion marker settings or runtime differ')
    if marker.get('artifact_sha256') != {name: shared.file_hash(output/name) for name in shared.ARTIFACTS}:
        raise ValueError('Completion marker artifact hashes differ')
    if (summary['epochs_completed'] != EPOCH_LIMIT
            and summary['epochs_completed'] - summary['best_epoch'] != expected['patience']):
        raise ValueError('Training ended before the epoch limit or the configured early stopping point')
    row, metadata, _ = prior.inspect_run(prior.Run(MODE, ticker, seed, output), cache)
    if row['window_verification'] != 'full':
        raise ValueError(f'Window verification {row["window_verification"]}: {row["mismatches"]} {row["missing_fields"]}')
    proof, missing = provenance(output, metadata)
    if missing:
        raise ValueError('Missing protocol evidence: ' + ', '.join(missing))
    args = metadata['args']
    checkpoint = torch.load(output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
    if args.get('extended_text_evidence'):
        replay = {split: dataset.extended_text_evidence() for split, dataset in metadata['datasets'].items()}
        if args['extended_text_evidence'] != replay:
            raise ValueError('Extended text rows, dates or window bounds differ')
    orders, best_state = replay_orders(len(metadata['datasets']['train']), args['batch_size'], seed,
                                      summary['epochs_completed'], summary['best_epoch'])
    if orders != proof['orders'] or not torch.equal(best_state, checkpoint['train_generator_state']):
        raise ValueError('Saved training order or generator state differs from seeded replay')
    price_width = 4 * len(config.get('price_gate_lags', (1, 5))) + 10
    expected_gate = dict(type='Linear', in_features=price_width, out_features=5,
                         parameters=5*(price_width+1), initialization='zeros')
    if args.get('gate_structure') != expected_gate:
        raise ValueError('Price gate structure differs')
    initial = check_initialization(args, checkpoint, historical, metadata['datasets']['val'])
    return row, metadata, {**proof, 'initialization': initial,
                           'checkpoint_sha256': marker['artifact_sha256']['best_checkpoint.pt']}


def compare_grid_runs(run, control):
    _, left, a = run
    _, right, b = control
    if left['split'] != right['split'] or left['evidence'] != right['evidence'] or a['source'] != b['source']:
        raise ValueError('Paired source data, splits, windows, targets or dates differ')
    prior.check_scalers(left['scaler'], right['scaler'], 'paired mean grid')
    for key in (*shared.SETTINGS, *prior.PRICE_FIELDS, 'd_model', 'dropout', 'used_col_prefixes', 'seed'):
        if left['args'].get(key) != right['args'].get(key):
            raise ValueError(f'Paired setting differs: {key}')
    for key in ('film_sha256', 'heads_sha256'):
        if a['initialization'][key] != b['initialization'][key]:
            raise ValueError(f'Paired initialization differs: {key}')
    if (left['args'].get('price_gate_lags', [1, 5]) == right['args'].get('price_gate_lags', [1, 5])
            and a['initialization']['price_gate_sha256'] != b['initialization']['price_gate_sha256']):
        raise ValueError('Paired gate initialization differs at the same lags')
    common_epochs = min(len(a['orders']), len(b['orders']))
    if not common_epochs or a['orders'][:common_epochs] != b['orders'][:common_epochs]:
        raise ValueError('Paired training orders differ')
    return common_epochs


def inspect_normal_cohort(root, data, cache):
    runs, records = {}, []
    for ticker, seed in PAIRS:
        path = normal_destination(root, ticker, seed)
        record = dict(ticker=ticker, seed=seed, path=str(path))
        try:
            run = checked_run(path, NORMAL, data, BASE_COMMIT, ticker, seed, cache, historical=True)
            runs[ticker, seed] = run
            record.update(status='full', checkpoint_sha256=run[2]['checkpoint_sha256'], reason='')
        except AUDIT_ERRORS as error:
            record.update(status='rejected', reason=str(error))
        records.append(record)
    return runs, records


def decisions(root):
    """A matched cohort takes precedence once scheduled, including forced reruns."""
    for name in ('matched_control.json', 'reused_control.json'):
        path = root / name
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
    return None
