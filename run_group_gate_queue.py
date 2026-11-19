"""720 paired main runs; separate explicit last-control reruns, never automatic."""
import argparse
import itertools
from pathlib import Path
import sys
import run_han_head_ablation_queue as shared

RESULTS = shared.DATA.parents[1] / 'runs' / 'group_gate_ablation_720_v1'


def configuration(family, hidden, window, weights):
    return dict(text_temporal_pool='group_gate_experiment', text_family=family,
                family_hidden_size=hidden, text_window=window, group_weight_condition=weights)


CONFIGURATIONS = {f'{f}_h{h}_w{w}_{g}': configuration(f,h,w,g)
                  for f,h,w,g in itertools.product(('han','huang','tlstm'), (16,32), (5,20), ('price_only','uniform'))}
CONTROLS = {f'last_{g}': configuration('last',32,20,g) for g in ('price_only','uniform')}
SCRIPTS = ('run_group_gate_ablation_720.sh','run_group_gate_queue.py','run_han_head_ablation_queue.py')


def jobs(controls=None):
    configs = CONFIGURATIONS if controls is None else {k:v for k,v in CONTROLS.items() if controls == 'all' or k == 'last_'+controls}
    return shared.jobs(tuple(configs))


def run_queue(code, data=shared.DATA, results=RESULTS, commit=None, controls=None):
    configs = CONFIGURATIONS if controls is None else {k:v for k,v in CONTROLS.items() if controls == 'all' or k == 'last_'+controls}
    return shared.run_queue(code, data, Path(results) / 'controls' if controls else results, commit,
                            modes=tuple(configs), scripts=SCRIPTS, configurations=configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--data', type=Path, default=shared.DATA)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--controls', choices=['price_only','uniform','all'])
    args = parser.parse_args()
    try:
        run_queue(args.code,args.data,args.results,controls=args.controls)
    except (OSError,ValueError,RuntimeError,shared.subprocess.CalledProcessError) as error:
        print(f'STOP: {error}; existing files preserved',file=sys.stderr)
        raise SystemExit(1)
