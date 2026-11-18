"""Four replace-only Huang cells: 4 NVDA checks + 120 main jobs."""

from run_han_head_ablation_queue import DATA, cli, jobs as shared_jobs, run_queue as shared_queue
from forecasting_task.models.GroupHuangQuery import HUANG_REPLACE


RESULTS = DATA.parents[1] / 'runs' / 'context_price_huang_replace_grid_10_v1'
CONFIGURATIONS = {
    f'h{hidden}_s{steps}_l10': {'text_temporal_pool': HUANG_REPLACE, 'huang_hidden_size': hidden,
                              'huang_recurrent_steps': steps, 'huang_attention_lookback': 10}
    for hidden, steps in ((16, 5), (16, 20), (32, 5), (32, 20))}
SCRIPTS = ('run_context_price_huang_replace_grid_120.sh', 'run_huang_replace_grid_queue.py', 'run_han_head_ablation_queue.py')


def jobs():
    return shared_jobs(tuple(CONFIGURATIONS))


def run_queue(code, data=DATA, results=RESULTS, commit=None):
    return shared_queue(code, data, results, commit, modes=tuple(CONFIGURATIONS), scripts=SCRIPTS, configurations=CONFIGURATIONS)


if __name__ == '__main__':
    cli(results=RESULTS, modes=tuple(CONFIGURATIONS), scripts=SCRIPTS, configurations=CONFIGURATIONS)
