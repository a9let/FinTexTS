"""Three new Huang grid cells: 3 NVDA checks + 90 main jobs, no control retraining."""

from run_han_head_ablation_queue import DATA, cli, jobs as shared_jobs, run_queue as shared_queue


RESULTS = DATA.parents[1] / 'runs' / 'context_price_huang_grid_10_v1'
CONFIGURATIONS = {
    f'h{hidden}_s{steps}_l10': {'text_temporal_pool': 'huang_recurrent_query', 'huang_hidden_size': hidden,
                              'huang_recurrent_steps': steps, 'huang_attention_lookback': 10}
    for hidden, steps in ((32, 5), (16, 20), (16, 5))}
SCRIPTS = ('run_context_price_huang_grid_90.sh', 'run_huang_grid_queue.py', 'run_han_head_ablation_queue.py')


def jobs():
    return shared_jobs(tuple(CONFIGURATIONS))


def run_queue(code, data=DATA, results=RESULTS, commit=None):
    return shared_queue(code, data, results, commit, modes=tuple(CONFIGURATIONS), scripts=SCRIPTS, configurations=CONFIGURATIONS)


if __name__ == '__main__':
    cli(results=RESULTS, modes=tuple(CONFIGURATIONS), scripts=SCRIPTS, configurations=CONFIGURATIONS)
