"""Two NVDA checks plus 60 grouped Huang runs; saved last is never retrained."""

from run_han_head_ablation_queue import DATA, cli, jobs as shared_jobs, run_queue as shared_queue
from forecasting_task.models.GroupHuangQuery import HUANG_MODES


RESULTS = DATA.parents[1] / 'runs' / 'context_price_huang_query_10_v1'
SCRIPTS = ('run_context_price_huang_query_60.sh', 'run_huang_query_queue.py', 'run_han_head_ablation_queue.py')


def jobs():
    return shared_jobs(HUANG_MODES)


def run_queue(code, data=DATA, results=RESULTS, commit=None):
    return shared_queue(code, data, results, commit, modes=HUANG_MODES, scripts=SCRIPTS)


if __name__ == '__main__':
    cli(results=RESULTS, modes=HUANG_MODES, scripts=SCRIPTS)
