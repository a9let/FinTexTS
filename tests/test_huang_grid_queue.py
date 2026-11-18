"""All 3 + 90 grid jobs with explicit CLI parameters and validated resume."""

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

import run_huang_grid_queue as grid
import run_han_head_ablation_queue as shared
import test_huang_queue as core


ROOT = Path(__file__).resolve().parents[1]


class HuangGridQueueTests(unittest.TestCase):
    def setup_root(self, root):
        code, data, results = root / 'code', root / 'data', root / 'results'
        code.mkdir()
        data.mkdir()
        for name in grid.SCRIPTS:
            (code / name).write_bytes((ROOT / name).read_bytes())
        for ticker in (*shared.TICKERS, 'NVDA'):
            (data / (ticker + '.parquet')).touch()
        return code, data, results

    def test_93_unique_jobs_flags_resume_and_mismatch(self):
        planned = grid.jobs()
        self.assertEqual(len(planned), 93)
        self.assertEqual(len(set(planned)), 93)
        self.assertEqual([j[2:] for j in planned[:3]], [('NVDA', 7, 1)] * 3)
        self.assertTrue(all(j[0] == 'full' and j[-1] == 50 for j in planned[3:]))
        calls = []
        def trainer(command, log, code):
            flags = core.core.HeadQueueTests.flags(command)
            self.assertEqual(flags['--text_temporal_pool'], 'huang_recurrent_query')
            self.assertEqual(flags['--seq_len'], '64')
            label = f"h{flags['--huang_hidden_size']}_s{flags['--huang_recurrent_steps']}_l{flags['--huang_attention_lookback']}"
            expected = shared.expected_config(label, flags['--data_path'].split('.')[0], int(flags['--seed']),
                                              int(flags['--num_epoch']), flags['--code_commit'], flags['--root_path'], configuration=grid.CONFIGURATIONS[label])
            self.assertNotEqual(label, 'h32_s20_l10')
            calls.append((label, command))
            output = Path(flags['--logdir'])
            self.assertEqual(output.parents[2].name, label)
            output.mkdir()
            (output / 'stub.json').write_text(json.dumps(expected))
            log.write_text('stub only')
            return 0
        with tempfile.TemporaryDirectory() as directory:
            code, data, results = self.setup_root(Path(directory))
            with patch.object(shared.torch.cuda, 'is_available', return_value=True), patch.object(shared.torch.cuda, 'get_device_name', return_value='mock'), \
                 patch.object(shared, 'execute_training', side_effect=trainer), patch.object(shared, 'completion', side_effect=core.HuangQueueTests.completion), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                markers = grid.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(markers), 93)
                self.assertEqual(len(calls), 93)
                self.assertEqual(log.getvalue().count('START '), 93)
                self.assertIn('completed=90/90', log.getvalue())
                self.assertIn('remaining_runs=0', log.getvalue())
                self.assertTrue(log.getvalue().rstrip().endswith('ALL 90 RUNS COMPLETED'))
                grid.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(len(calls), 93)
                altered = results / 'full/h16_s5_l10/AMZN/seed27/output/stub.json'
                content = json.loads(altered.read_text())
                content['huang_hidden_size'] = 32
                altered.write_text(json.dumps(content))
                preserved = altered.read_bytes()
                with self.assertRaisesRegex(ValueError, 'Incomplete/different'):
                    grid.run_queue(code, data, results, 'a' * 40)
                self.assertEqual(altered.read_bytes(), preserved)
                self.assertEqual(len(calls), 93)

    def test_bash_syntax(self):
        bash = r'C:\Program Files\Git\bin\bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash:
            self.skipTest('Bash unavailable; check on server')
        checked = subprocess.run([bash, '-n', str(ROOT / 'run_context_price_huang_grid_90.sh')], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == '__main__':
    unittest.main()
