"""Two NVDA smoke checks and 60 HAN text-window runs, trained from scratch."""

from run_han_head_ablation_queue import cli, jobs as head_jobs, run_queue as head_queue, DATA
from forecasting_task.models.HANHeadAblation import HAN_WINDOW_MODES


RESULTS = DATA.parents[1] / 'runs' / 'context_price_han_window_10_v1'
SCRIPTS = ('run_context_price_han_window_60.sh', 'run_han_window_queue.py', 'run_han_head_ablation_queue.py')


def jobs():
    return head_jobs(HAN_WINDOW_MODES)


def run_queue(code, data=DATA, results=RESULTS, commit=None):
    return head_queue(code, data, results, commit, modes=HAN_WINDOW_MODES, scripts=SCRIPTS)


if __name__ == '__main__':
    cli(results=RESULTS, modes=HAN_WINDOW_MODES, scripts=SCRIPTS)
