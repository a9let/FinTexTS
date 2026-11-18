"""Replace-only queue: 4 smoke checks, 120 main jobs, safe resume and failures."""

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

import run_huang_replace_grid_queue as queue
import run_han_head_ablation_queue as shared
import test_huang_queue as core


ROOT = Path(__file__).resolve().parents[1]


class ReplaceQueueTests(unittest.TestCase):
    def setup_root(self, root):
        code, data, results = root / 'code', root / 'data', root / 'results'
        code.mkdir()
        data.mkdir()
        for name in queue.SCRIPTS:
            (code / name).write_bytes((ROOT / name).read_bytes())
        for ticker in (*shared.TICKERS, 'NVDA'):
            (data / (ticker + '.parquet')).touch()
        return code, data, results

    def test_124_jobs_all_replace_flags_and_resume(self):
        planned = queue.jobs()
        self.assertEqual(len(planned), 124)
        self.assertEqual(len(set(planned)), 124)
        self.assertEqual([j[2:] for j in planned[:4]], [('NVDA', 7, 1)] * 4)
        self.assertTrue(all(j[0] == 'full' and j[-1] == 50 for j in planned[4:]))
        calls = []
        def trainer(command, log, code):
            flags = core.core.HeadQueueTests.flags(command)
            self.assertEqual(flags['--text_temporal_pool'], 'huang_recurrent_replace')
            self.assertEqual(flags['--seq_len'], '64')
            label = f"h{flags['--huang_hidden_size']}_s{flags['--huang_recurrent_steps']}_l{flags['--huang_attention_lookback']}"
            expected = shared.expected_config(label, flags['--data_path'].split('.')[0], int(flags['--seed']), int(flags['--num_epoch']),
                                              flags['--code_commit'], flags['--root_path'], configuration=queue.CONFIGURATIONS[label])
            self.assertEqual(expected['structure']['text_branch'], 'huang_replaces_last')
            self.assertIsNone(expected['structure']['residual_head'])
            calls.append(command)
            output = Path(flags['--logdir'])
            output.mkdir()
            (output / 'stub.json').write_text(json.dumps(expected))
            log.write_text('stub only')
            return 0
        with tempfile.TemporaryDirectory() as directory:
            code, data, results = self.setup_root(Path(directory))
            with patch.object(shared.torch.cuda, 'is_available', return_value=True), patch.object(shared.torch.cuda, 'get_device_name', return_value='mock'), \
                 patch.object(shared, 'execute_training', side_effect=trainer), patch.object(shared, 'completion', side_effect=core.HuangQueueTests.completion), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                queue.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(calls), 124)
                self.assertIn('completed=120/120', log.getvalue())
                self.assertIn('remaining_runs=0', log.getvalue())
                self.assertTrue(log.getvalue().rstrip().endswith('ALL 120 RUNS COMPLETED'))
                queue.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(calls), 124)
                altered = results / 'full/h32_s20_l10/AMZN/seed27/output/stub.json'
                payload = json.loads(altered.read_text())
                payload['text_temporal_pool'] = 'huang_recurrent_query'
                altered.write_text(json.dumps(payload))
                preserved = altered.read_bytes()
                with self.assertRaisesRegex(ValueError, 'Incomplete/different'):
                    queue.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(altered.read_bytes(), preserved)
                self.assertEqual(len(calls), 124)

    def test_cuda_failure_logs_and_bash(self):
        with tempfile.TemporaryDirectory() as directory:
            code, data, results = self.setup_root(Path(directory))
            with patch.object(shared.torch.cuda, 'is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                queue.run_queue(code, data, results, 'a' * 40)
            self.assertFalse(results.exists())
            def failure(command, log, code):
                log.write_text('synthetic error preserved')
                return 9
            with patch.object(shared.torch.cuda, 'is_available', return_value=True), patch.object(shared.torch.cuda, 'get_device_name', return_value='mock'), \
                 patch.object(shared, 'execute_training', side_effect=failure), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as log:
                with self.assertRaises(subprocess.CalledProcessError):
                    queue.run_queue(code, data, results, 'a' * 40)
                self.assertIn('synthetic error preserved', log.getvalue())
                self.assertEqual(len(list(results.glob('checks/*/NVDA/seed7/train.log'))), 1)
        bash = r'C:\Program Files\Git\bin\bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash:
            self.skipTest('Bash unavailable; check on server')
        checked = subprocess.run([bash, '-n', str(ROOT / 'run_context_price_huang_replace_grid_120.sh')], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == '__main__':
    unittest.main()
