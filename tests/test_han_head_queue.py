"""Exercise all 155 jobs and resume/error behavior using a stub trainer."""

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

import run_han_head_ablation_queue as queue


ROOT = Path(__file__).resolve().parents[1]


class HeadQueueTests(unittest.TestCase):
    def setup_root(self, root):
        code, data, results = root / 'code', root / 'data', root / 'results'
        code.mkdir()
        data.mkdir()
        for name in ('run_context_price_han_head_ablation_150.sh', 'run_han_head_ablation_queue.py'):
            (code / name).write_bytes((ROOT / name).read_bytes())
        for ticker in (*queue.TICKERS, 'NVDA'):
            (data / (ticker + '.parquet')).touch()
        return code, data, results

    @staticmethod
    def flags(command):
        return dict(zip(command[4::2], command[5::2]))

    def completion_stub(self, destination, expected, elapsed=None):
        path = Path(destination) / 'output' / 'stub.json'
        if not path.is_file():
            raise ValueError(f'Incomplete run: {destination}')
        flags = json.loads(path.read_text())
        for key in queue.SETTINGS:
            if flags['--' + key] != str(expected[key]):
                raise ValueError(f'Setting differs: {key}')
        for flag, value in (('text_temporal_pool', expected['text_temporal_pool']), ('data_path', expected['ticker'] + '.parquet'),
                            ('seed', str(expected['seed'])), ('num_epoch', str(expected['num_epoch'])), ('code_commit', expected['code_commit'])):
            if flags['--' + flag] != value:
                raise ValueError(f'Setting differs: {flag}')
        return {'config': expected, 'wall_seconds': 10. * (1 + queue.ABLATION_MODES.index(expected['text_temporal_pool']))}

    def test_all_jobs_resume_and_settings_are_checked(self):
        calls = []
        def trainer(command, log_path, code):
            calls.append(command)
            flags = self.flags(command)
            self.assertEqual(flags['--group_weight_mode'], 'fixed_uniform')
            self.assertEqual(flags['--used_col_prefixes'], ','.join(queue.TEXT_GROUPS))
            output = Path(flags['--logdir'])
            output.mkdir()
            (output / 'stub.json').write_text(json.dumps(flags))
            log_path.write_text('synthetic training only')
            return 0
        with tempfile.TemporaryDirectory() as directory:
            code, data, results = self.setup_root(Path(directory))
            with patch.object(queue.torch.cuda, 'is_available', return_value=True), patch.object(queue.torch.cuda, 'get_device_name', return_value='mock'), \
                 patch.object(queue, 'execute_training', side_effect=trainer), patch.object(queue, 'completion', side_effect=self.completion_stub):
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    markers = queue.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(calls), 155)
                self.assertEqual(len(markers), 155)
                full = [job for job in markers if job[0] == 'full']
                checks = [job for job in markers if job[0] == 'checks']
                self.assertEqual(len(set(full)), 150)
                self.assertEqual(len(checks), 5)
                self.assertTrue(all(job[2:] == ('NVDA', 7, 1) for job in checks))
                self.assertTrue(all(job[-1] == 50 for job in full))
                self.assertEqual(log.getvalue().count('START '), 155)
                self.assertIn('TIMING mode=han_h32_mlp', log.getvalue())
                self.assertIn('overall_mean_fallback_modes=', log.getvalue())
                self.assertIn('remaining_runs=0', log.getvalue())
                self.assertTrue(log.getvalue().rstrip().endswith('ALL 150 RUNS COMPLETED'))
                resumed = io.StringIO()
                with contextlib.redirect_stdout(resumed):
                    queue.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(calls), 155)
                self.assertEqual(resumed.getvalue().count('SKIP '), 155)
                altered = results / 'full/han_h16_linear/AMZN/seed27/output/stub.json'
                flags = json.loads(altered.read_text())
                flags['--lr'] = '0.9'
                altered.write_text(json.dumps(flags))
                preserved = altered.read_bytes()
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Setting differs'):
                    queue.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(altered.read_bytes(), preserved)
                self.assertEqual(len(calls), 155)

    def test_incomplete_run_and_failure_preserve_logs(self):
        for failure in ('incomplete', 'training'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                code, data, results = self.setup_root(Path(directory))
                old = results / 'full/last_han_h32_residual/AMZN/seed27'
                if failure == 'incomplete':
                    old.mkdir(parents=True)
                    (old / 'train.log').write_text('preserve interrupted run')
                def trainer(command, log_path, code):
                    log_path.write_text('synthetic failed training')
                    return 9
                with patch.object(queue.torch.cuda, 'is_available', return_value=True), patch.object(queue.torch.cuda, 'get_device_name', return_value='mock'), \
                     patch.object(queue, 'execute_training', side_effect=trainer) as execute, patch.object(queue, 'completion', side_effect=self.completion_stub), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
                    if failure == 'incomplete':
                        with self.assertRaisesRegex(ValueError, 'Incomplete run'):
                            queue.run_queue(code, data, results, 'a' * 40)
                        self.assertFalse(execute.called)
                        self.assertEqual((old / 'train.log').read_text(), 'preserve interrupted run')
                    else:
                        with self.assertRaises(subprocess.CalledProcessError) as caught:
                            queue.run_queue(code, data, results, 'a' * 40)
                        self.assertEqual(caught.exception.returncode, 9)
                        self.assertEqual(execute.call_count, 1)
                        self.assertIn('synthetic failed training', error.getvalue())
                        self.assertEqual((results / 'checks/last_matched/NVDA/seed7/train.log').read_text(), 'synthetic failed training')

    def test_cuda_and_bash_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            code, data, results = self.setup_root(Path(directory))
            with patch.object(queue.torch.cuda, 'is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                queue.run_queue(code, data, results, 'a' * 40)
            self.assertFalse(results.exists())
        bash = r'C:\Program Files\Git\bin\bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash:
            self.skipTest('Bash unavailable; run bash -n on server')
        checked = subprocess.run([bash, '-n', str(ROOT / 'run_context_price_han_head_ablation_150.sh')], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == '__main__':
    unittest.main()
