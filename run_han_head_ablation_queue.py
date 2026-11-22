"""Resumable 5-smoke + 150-run queue with validated completion manifests."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import torch
import yaml

from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.models.HANHeadAblation import ABLATION_MODES
from forecasting_task.models.GroupHuangQuery import HUANG_MODES, HUANG_ALL_MODES, HUANG_REPLACE, huang_options
from forecasting_task.models.GroupGRUAttention import temporal_structure
from forecasting_task.window_evidence import SPLIT_POLICY
from forecasting_task.models.GroupGateExperiment import experiment_structure


DATA = Path('/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts')
RESULTS = Path('/auto/datasets/kondyurin/FinTexTS/runs/context_price_han_head_ablation_10_v1')
TICKERS = ('AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'AMAT', 'AMD', 'AMGN', 'AMZN')
SEEDS = (7, 17, 27)
SETTINGS = {'model_type': 'FiLM', 'seq_len': 64, 'label_len': 16, 'pred_len': 3, 'batch_size': 32,
            'text_num_layers': 1, 'text_hidden_size': 64, 'text_weight': .1, 'lr': .001,
            'patience': 10, 'e_layers': 2, 'fusion_mode': 'blend', 'group_gate': 'context_price_only'}
ARTIFACTS = ('args.yaml', 'history.jsonl', 'summary.json', 'best_checkpoint.pt', 'test_io.pt',
             'validation_group_diagnostics.csv', 'validation_text_dates.csv', 'validation_prediction_components.csv')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jobs(modes=ABLATION_MODES):
    checks = [('checks', mode, 'NVDA', 7, 1) for mode in modes]
    full = [('full', mode, ticker, seed, 50) for mode in modes for ticker in TICKERS for seed in SEEDS]
    return checks + full


def expected_config(mode, ticker, seed, epochs, commit, data, configuration=None):
    settings = dict(SETTINGS, **(configuration or {}))
    actual_mode = settings.pop('text_temporal_pool', mode)
    parameters = huang_options(settings) if actual_mode in HUANG_ALL_MODES else {}
    if actual_mode == 'group_gate_experiment':
        structure = experiment_structure(settings['text_family'], settings['family_hidden_size'],
                                         settings['text_window'], settings['group_weight_condition'],
                                         settings['pred_len'], settings.get('kd_beta', 0.),
                                         settings.get('price_gate_lags', (1, 5)))
    else:
        structure = temporal_structure(actual_mode, settings['pred_len'], **parameters)
    return {**settings, 'text_temporal_pool': actual_mode, 'ticker': ticker, 'seed': seed, 'num_epoch': epochs,
            'code_commit': commit, 'root_path': str(Path(data).resolve()),
            'used_col_prefixes': list(TEXT_GROUPS), 'structure': structure}


def validate_completed(destination, expected):
    """Validate finished outputs before they can be skipped or marked complete."""
    destination = Path(destination)
    output = destination / 'output'
    for name in ARTIFACTS:
        path = output / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Incomplete run {destination}: missing/empty {name}; preserve and inspect this directory')
    args = yaml.safe_load((output / 'args.yaml').read_text(encoding='utf-8'))
    summary = json.loads((output / 'summary.json').read_text())
    checkpoint = torch.load(output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
    if args != checkpoint['args']:
        raise ValueError(f'{destination}: args.yaml/checkpoint disagree')
    for key in SETTINGS:
        if args.get(key) != expected[key]:
            raise ValueError(f'{destination}: setting {key} differs')
    for key in ('seed', 'num_epoch', 'code_commit', 'used_col_prefixes', 'text_temporal_pool'):
        if args.get(key) != expected[key]:
            raise ValueError(f'{destination}: setting {key} differs')
    if (Path(args['root_path']).resolve() != Path(expected['root_path']) or args['data_path'] != expected['ticker'] + '.parquet'
            or args.get('group_weight_mode') != 'context_price_only' or args.get('text_mlp_structure') != expected['structure']['head_structure']):
        raise ValueError(f'{destination}: data path, heads or group gate differ')
    for key in ('code_commit', 'text_temporal_pool', 'seed'):
        if summary.get(key) != expected[key]:
            raise ValueError(f'{destination}: summary {key} differs')
    initialization_key = 'experiment_initialization' if expected['text_temporal_pool'] == 'group_gate_experiment' else 'huang_initialization' if expected['text_temporal_pool'] in HUANG_ALL_MODES else 'head_ablation_initialization'
    for key in ('window_evidence', 'source_data_evidence', 'common_initialization', initialization_key,
                'initial_prediction_evidence', 'temporal_initialization_sha256', 'split_policy', 'temporal_pool_structure'):
        if key not in args or key not in summary or key not in checkpoint or args[key] != summary[key] or args[key] != checkpoint[key]:
            raise ValueError(f'{destination}: inconsistent saved {key}')
    if args['split_policy'] != SPLIT_POLICY or args['temporal_pool_structure'] != expected['structure']:
        raise ValueError(f'{destination}: split/architecture mismatch')
    if expected['text_temporal_pool'] in HUANG_ALL_MODES:
        for source in (args, summary, checkpoint, args['text_model_config']):
            if huang_options(source) != huang_options(expected):
                raise ValueError(f'{destination}: Huang parameters differ')
    if expected['text_temporal_pool'] == 'group_gate_experiment':
        family_keys = ('text_family', 'family_hidden_size', 'text_window', 'group_weight_condition')
        if 'price_gate_lags' in expected:
            family_keys += ('price_gate_lags',)
        for key in family_keys:
            if any(source.get(key) != expected[key] for source in (args, summary, checkpoint, args['text_model_config'])):
                raise ValueError(f'{destination}: experiment configuration differs: {key}')
        for key in ('kd_beta', 'memory_stage', 'memory_smoke'):
            if key in expected and args.get(key) != expected[key]:
                raise ValueError(f'{destination}: phase configuration differs: {key}')
        if expected['text_family'] == 'mean_price_grid' and expected['text_window'] > expected['seq_len']:
            evidence = args.get('extended_text_evidence')
            if (not evidence or set(evidence) != {'train', 'val', 'test'}
                    or summary.get('extended_text_evidence') != evidence
                    or checkpoint.get('extended_text_evidence') != evidence
                    or any(split['requested_rows'] != expected['text_window'] for split in evidence.values())):
                raise ValueError(f'{destination}: extended text window evidence differs')
    data_file = Path(expected['root_path']) / (expected['ticker'] + '.parquet')
    if args['source_data_evidence'].get('parquet_sha256') != file_hash(data_file):
        raise ValueError(f'{destination}: source Parquet changed')
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines() if line.strip()]
    if (not history or [r['epoch'] for r in history] != list(range(1, len(history) + 1))
            or len(history) != summary['epochs_completed'] or len(history) > expected['num_epoch']):
        raise ValueError(f'{destination}: incomplete/inconsistent epoch history')
    best = min(history, key=lambda r: r['val_mse'])
    if ('price_gate_lags' in expected and len(history) != expected['num_epoch']
            and len(history) - best['epoch'] != expected['patience']):
        raise ValueError(f'{destination}: epoch limit/early stopping completion unconfirmed')
    if (summary['best_epoch'] != best['epoch'] or checkpoint['epoch'] != best['epoch']
            or checkpoint['metrics'] != best or summary['best_val_mse'] != best['val_mse']
            or summary['best_test_mse'] != best['test_mse'] or summary['best_test_mae'] != best['test_mae']):
        raise ValueError(f'{destination}: selected checkpoint/metrics disagree')
    for record in history:
        for key in ('train_mse', 'val_mse', 'test_mse', 'test_mae'):
            if not math.isfinite(record[key]) or record[key] < 0:
                raise ValueError(f'{destination}: invalid {key}')
    orders = [record['train_batch_order'] for record in history]
    if summary.get('train_batch_orders') != orders or checkpoint.get('train_batch_orders') != orders[:best['epoch']]:
        raise ValueError(f'{destination}: actual training order evidence differs')
    if not checkpoint.get('model_state_dict') or not checkpoint.get('optimizer_state_dict'):
        raise ValueError(f'{destination}: missing training state')
    groups = checkpoint['optimizer_state_dict'].get('param_groups', [])
    if not groups or any(group.get('lr') != expected['lr'] or group.get('weight_decay', 0) != 0 for group in groups):
        raise ValueError(f'{destination}: optimizer settings differ')
    with torch.random.fork_rng(devices=[]):
        text = build_text_model(args['text_model_config'], args['group_weight_mode'])
        text.load_state_dict(checkpoint['text_model_state_dict'], strict=True)
    if expected['text_temporal_pool'] == HUANG_REPLACE and any(name.startswith('group_mlps.') for name in checkpoint['text_model_state_dict']):
        raise ValueError(f'{destination}: replace checkpoint contains last heads')
    if expected['text_temporal_pool'] in ('last_matched', 'last_han_h32_residual', *HUANG_MODES):
        proof = args['initial_prediction_evidence']
        if not proof or proof.get('matches_last_matched_formula') is not True or proof.get('max_abs_difference') != 0:
            raise ValueError(f'{destination}: initial last-formula equality unconfirmed')
    return summary


def completion(destination, expected, elapsed=None):
    try:
        summary = validate_completed(destination, expected)
    except (OSError, KeyError, ValueError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError, yaml.YAMLError) as error:
        raise ValueError(f'{destination}: completion could not be confirmed: {error}') from error
    marker_path = Path(destination) / 'completed.json'
    hashes = {name: file_hash(Path(destination) / 'output' / name) for name in ARTIFACTS}
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get('config') != expected or marker.get('artifact_sha256') != hashes:
            raise ValueError(f'{destination}: completion manifest disagrees with artifacts/settings')
        if not math.isfinite(marker.get('wall_seconds', -1)) or marker['wall_seconds'] < 0:
            raise ValueError(f'{destination}: invalid completed runtime')
        return marker
    seconds = elapsed if elapsed is not None else float(summary['total_seconds'])
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f'{destination}: invalid recorded runtime')
    marker = {'config': expected, 'artifact_sha256': hashes, 'wall_seconds': seconds,
              'timing_source': 'queue_wall_clock' if elapsed is not None else 'summary_total_seconds_recovered'}
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination, prefix='.completed-', suffix='.tmp', delete=False) as handle:
        json.dump(marker, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, marker_path)
    finally:
        temporary.unlink(missing_ok=True)
    return marker


def record_file(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Existing queue metadata differs: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(content)


def training_command(mode, ticker, seed, epochs, commit, data, output, configuration=None):
    command = [sys.executable, '-u', '-m', 'forecasting_task.run']
    for key, value in SETTINGS.items():
        command.extend(['--' + key, str(value)])
    configuration = configuration or {}
    if configuration:
        values = ({k: configuration[k] for k in ('text_family', 'family_hidden_size', 'text_window', 'group_weight_condition')} if 'text_family' in configuration else huang_options(configuration))
        for key, value in values.items():
            command.extend(['--' + key, str(value)])
        if 'price_gate_lags' in configuration:
            command.extend(['--price_gate_lags', ','.join(map(str, configuration['price_gate_lags']))])
    command.extend(['--root_path', str(data), '--data_path', ticker + '.parquet', '--group_weight_mode', 'fixed_uniform',
                    '--text_temporal_pool', configuration.get('text_temporal_pool', mode), '--used_col_prefixes', ','.join(TEXT_GROUPS),
                    '--num_epoch', str(epochs), '--seed', str(seed), '--code_commit', commit, '--logdir', str(output)])
    return command


def execute_training(command, log_path, code):
    with log_path.open('x', encoding='utf-8') as handle:
        return subprocess.run(command, cwd=code, stdout=handle, stderr=subprocess.STDOUT).returncode


def print_timing(markers, modes=ABLATION_MODES, configurations=None):
    times = defaultdict(list)
    for job, marker in markers.items():
        if job[0] == 'full':
            times[job[1]].append(marker['wall_seconds'])
    observed = [value for values in times.values() for value in values]
    if not observed:
        return
    overall = sum(observed) / len(observed)
    remaining, fallback = 0., []
    for mode in modes:
        values = times[mode]
        mean = sum(values) / len(values) if values else overall
        if not values:
            fallback.append(mode)
        remaining += (30 - len(values)) * mean
        print(f'TIMING mode={mode} completed={len(values)}/30 mean_seconds={sum(values)/len(values) if values else "pending"}', flush=True)
    if configurations and all('text_family' in c for c in configurations.values()):
        for family in sorted({c['text_family'] for c in configurations.values()}):
            labels = [label for label,c in configurations.items() if c['text_family'] == family]
            observed_f = [v for label in labels for v in times[label]]
            remaining_f = sum((30-len(times[label]))*(sum(times[label])/len(times[label]) if times[label] else overall) for label in labels)
            print(f'FAMILY_TIMING family={family} completed={len(observed_f)}/{30*len(labels)} mean_seconds={sum(observed_f)/len(observed_f) if observed_f else "pending"} ETA_seconds={remaining_f:.1f}', flush=True)
    print(f'ETA remaining_runs={30*len(modes)-len(observed)} estimated_seconds={remaining:.1f} estimated_hours={remaining/3600:.2f} '
          f'overall_mean_fallback_modes={",".join(fallback)}', flush=True)


def run_queue(code, data=DATA, results=RESULTS, commit=None, *, modes=ABLATION_MODES, scripts=None, configurations=None):
    total = len(modes) * len(TICKERS) * len(SEEDS)
    code, data, results = Path(code), Path(data).resolve(), Path(results)
    for ticker in (*TICKERS, 'NVDA'):
        if not (data / (ticker + '.parquet')).is_file():
            raise FileNotFoundError(f'Missing data: {data}/{ticker}.parquet')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    print(f'CUDA ready: {torch.cuda.get_device_name(0)}', flush=True)
    commit = commit or os.environ.get('CODE_COMMIT')
    if not commit:
        try:
            commit = subprocess.check_output(['git', '-C', str(code), 'rev-parse', '--verify', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError('Set CODE_COMMIT to the full code SHA when Git is unavailable') from error
    if len(commit) not in (40, 64) or any(c not in '0123456789abcdefABCDEF' for c in commit):
        raise ValueError('CODE_COMMIT must be a full hexadecimal code SHA')
    results.mkdir(parents=True, exist_ok=True)
    record_file(results / 'code_commit.txt', (commit + '\n').encode())
    record_file(results / 'selected_tickers.txt', ('\n'.join(TICKERS) + '\n').encode())
    for name in (scripts or ('run_context_price_han_head_ablation_150.sh', Path(__file__).name)):
        record_file(results / name, (code / name).read_bytes())
    planned, markers = jobs(modes), {}
    def options(mode):
        return {'configuration': configurations[mode]} if configurations is not None else {}
    for job in planned:
        stage, mode, ticker, seed, epochs = job
        destination = results / stage / mode / ticker / f'seed{seed}'
        if destination.exists():
            markers[job] = completion(destination, expected_config(mode, ticker, seed, epochs, commit, data, **options(mode)))
    completed = sum(job[0] == 'full' for job in markers)
    print_timing(markers, modes, configurations)
    for job in planned:
        stage, mode, ticker, seed, epochs = job
        destination = results / stage / mode / ticker / f'seed{seed}'
        if job in markers:
            print(f'SKIP stage={stage} mode={mode} ticker={ticker} seed={seed} completed={completed}/{total}', flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        print(f'START stage={stage} mode={mode} ticker={ticker} seed={seed} completed={completed}/{total}', flush=True)
        started = time.monotonic()
        status = execute_training(training_command(mode, ticker, seed, epochs, commit, data, destination / 'output', **options(mode)), destination / 'train.log', code)
        if status:
            tail = (destination / 'train.log').read_text(encoding='utf-8', errors='replace').splitlines()[-30:]
            print('\n'.join(tail), file=sys.stderr, flush=True)
            raise subprocess.CalledProcessError(status, str(destination))
        markers[job] = completion(destination, expected_config(mode, ticker, seed, epochs, commit, data, **options(mode)), time.monotonic() - started)
        if stage == 'full':
            completed += 1
        print(f'OK stage={stage} mode={mode} ticker={ticker} seed={seed} completed={completed}/{total}', flush=True)
        print_timing(markers, modes, configurations)
    if completed != total:
        raise ValueError(f'Unexpected completed count: {completed}/{total}')
    print(f'ALL {total} RUNS COMPLETED', flush=True)
    return markers


def cli(*, results=RESULTS, modes=ABLATION_MODES, scripts=None, configurations=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--results', type=Path, default=results)
    args = parser.parse_args()
    try:
        run_queue(args.code, args.data, args.results, modes=modes, scripts=scripts, configurations=configurations)
    except subprocess.CalledProcessError as error:
        print(f'FAILED exit={error.returncode}: {error.cmd}; incomplete files preserved', file=sys.stderr)
        raise SystemExit(error.returncode)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'STOP: {error}. Inspect incomplete directories and explicitly archive them before resuming.', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
