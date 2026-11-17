"""Run all 2 + 60 jobs through a stub and reuse the old completion safeguards."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_han_window_queue as window
import run_han_head_ablation_queue as shared
import test_han_head_queue as core


ROOT = Path(__file__).resolve().parents[1]


class HANWindowQueueTests(unittest.TestCase):
    def setup_root(self, root):
        code, data, results = root / 'code', root / 'data', root / 'results'
        code.mkdir()
        data.mkdir()
        for name in window.SCRIPTS:
            (code / name).write_bytes((ROOT / name).read_bytes())
        for ticker in (*shared.TICKERS, 'NVDA'):
            (data / (ticker + '.parquet')).touch()
        return code, data, results

    @staticmethod
    def completion(destination, expected, elapsed=None):
        path = Path(destination) / 'output' / 'stub.json'
        if not path.is_file():
            raise ValueError(f'Incomplete run: {destination}')
        if json.loads(path.read_text()) != expected:
            raise ValueError(f'Settings differ: {destination}')
        return {'config': expected, 'wall_seconds': 2.}

    def test_all_jobs_resume_and_no_overwrites(self):
        planned = window.jobs()
        self.assertEqual(len(planned), 62)
        self.assertEqual(len(set(planned)), 62)
        self.assertEqual([j[2:] for j in planned[:2]], [('NVDA', 7, 1)] * 2)
        self.assertTrue(all(j[0] == 'full' and j[-1] == 50 for j in planned[2:]))
        self.assertEqual(len(shared.jobs()), 155)
        calls = []
        def trainer(command, log, code):
            flags = core.HeadQueueTests.flags(command)
            mode = flags['--text_temporal_pool']
            ticker = flags['--data_path'].split('.')[0]
            expected = shared.expected_config(mode, ticker, int(flags['--seed']), int(flags['--num_epoch']), flags['--code_commit'], flags['--root_path'])
            self.assertEqual(flags['--seq_len'], '64')
            self.assertEqual(flags['--group_gate'], 'context_price_only')
            self.assertEqual(expected['structure']['hidden_per_direction'], 16)
            calls.append(command)
            output = Path(flags['--logdir'])
            output.mkdir()
            (output / 'stub.json').write_text(json.dumps(expected))
            log.write_text('stub only')
            return 0
        with tempfile.TemporaryDirectory() as directory:
            code, data, results = self.setup_root(Path(directory))
            with patch.object(shared.torch.cuda, 'is_available', return_value=True), patch.object(shared.torch.cuda, 'get_device_name', return_value='mock'), \
                 patch.object(shared, 'execute_training', side_effect=trainer), patch.object(shared, 'completion', side_effect=self.completion), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                markers = window.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(markers), 62)
                self.assertEqual(len(calls), 62)
                self.assertEqual(log.getvalue().count('START '), 62)
                self.assertTrue(log.getvalue().rstrip().endswith('ALL 60 RUNS COMPLETED'))
                self.assertIn('completed=60/60', log.getvalue())
                self.assertIn('remaining_runs=0', log.getvalue())
                window.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(calls), 62)
                altered = results / 'full' / window.HAN_WINDOW_MODES[-1] / 'AMZN/seed27/output/stub.json'
                altered.write_text('incomplete preserved')
                old = altered.read_bytes()
                with self.assertRaises(ValueError):
                    window.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(altered.read_bytes(), old)
                self.assertEqual(len(calls), 62)

    def test_cuda_failure_and_training_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            code, data, results = self.setup_root(Path(directory))
            with patch.object(shared.torch.cuda, 'is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                window.run_queue(code, data, results, 'a' * 40)
            self.assertFalse(results.exists())
            def failed(command, log, code):
                log.write_text('synthetic failure preserved')
                return 9
            with patch.object(shared.torch.cuda, 'is_available', return_value=True), patch.object(shared.torch.cuda, 'get_device_name', return_value='mock'), \
                 patch.object(shared, 'execute_training', side_effect=failed), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as log:
                with self.assertRaises(subprocess.CalledProcessError):
                    window.run_queue(code, data, results, 'a' * 40)
                self.assertIn('synthetic failure preserved', log.getvalue())
                self.assertEqual(len(list(results.glob('checks/*/NVDA/seed7/train.log'))), 1)

    def test_bash_syntax(self):
        bash = r'C:\Program Files\Git\bin\bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash:
            self.skipTest('Bash unavailable; run bash -n on server')
        result = subprocess.run([bash, '-n', str(ROOT / 'run_context_price_han_window_60.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
