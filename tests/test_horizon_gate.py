"""CPU-only horizon gating, original initialization, checkpoint and diagnostics."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting_task'))
from forecasting_task import run
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS, build_text_model
import test_context_novelty as fixtures
make_dataset, make_frame = fixtures.make_dataset, fixtures.make_frame
import analyze_context_price_horizon_gate as analysis
from analyze_persistence_baseline import Run


class HorizonGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, mode='shared'):
        return ContextNoveltyTextModel(64, 3, 1, group_gate='context_price_only', price_gate_mode=mode)

    def test_shared_legacy_forward_and_per_horizon_equivalence(self):
        for batch in (1, 4):
            torch.manual_seed(7)
            shared = self.create()
            separate = self.create('per_horizon')
            separate.group_mlps.load_state_dict(shared.group_mlps.state_dict())
            with torch.no_grad():
                shared.gate.weight.normal_()
                shared.gate.bias.normal_()
                separate.gate.weight.copy_(shared.gate.weight.repeat(3, 1))
                separate.gate.bias.copy_(shared.gate.bias.repeat(3))
            x, e, price = torch.randn(batch, 8, 4), torch.randn(batch, 5, 384), torch.randn(batch, 3, 4)
            for missing in ('partial', 'all'):
                e[:, 2] = 0
                if missing == 'all':
                    e.zero_()
                kwargs = dict(price_window=x, novelty=torch.rand(batch, 5), pair_available=torch.ones(batch, 5))
                output, detail = shared(price, x[:, -1], e, return_details=True, **kwargs)
                context = torch.cat((x[:, -1] - x[:, -2], x[:, -1] - x[:, -6], torch.zeros(batch, 10)), -1)
                weights = shared.gate(context).softmax(-1)
                heads = torch.stack([shared.group_mlps[g](e[:, j]) for j, g in enumerate(TEXT_GROUPS)], 1)
                delta = (weights[..., None] * heads * (e.abs().sum(-1) >= 1e-6)[..., None]).sum(1)
                legacy = (1 - shared.fixed_alpha) * price + shared.fixed_alpha * (x[:, -1:, :] + delta[..., None])
                torch.testing.assert_close(output, legacy, rtol=0, atol=0)
                other, details = separate(price, x[:, -1], e, return_details=True, **kwargs)
                torch.testing.assert_close(other, output)
                torch.testing.assert_close(details['weights'].sum(1), torch.ones(batch, 3))
                self.assertTrue((details['contributions'][:, 2] == 0).all())
                if missing == 'all':
                    torch.testing.assert_close(other, .9 * price + .1 * x[:, -1:])
                before = details['weights'].detach().clone()
                with torch.no_grad():
                    separate.gate.bias[5] += 3
                _, changed = separate(price, x[:, -1], e, return_details=True, **kwargs)
                torch.testing.assert_close(changed['weights'][:, :, [0, 2]], before[:, :, [0, 2]], rtol=0, atol=0)
                self.assertFalse(torch.equal(changed['weights'][:, :, 1], before[:, :, 1]))
                with torch.no_grad():
                    separate.gate.bias[5] -= 3

    def test_initial_film_heads_predictions_rng_and_batches(self):
        for seed in (7, 17, 27):
            reference = None
            for mode in ('shared', 'per_horizon'):
                run.set_seed(seed)
                film = fixtures.ContextNoveltyTests.make_film()
                text = self.create(mode)
                rng = torch.get_rng_state().clone()
                self.assertEqual(text.gate.out_features, 5 if mode == 'shared' else 15)
                self.assertTrue(all((p == 0).all() for p in text.gate.parameters()))
                x, e = torch.randn(4, 6, 4), torch.randn(4, 5, 384)
                out = text(film(x, None, None, None), x[:, -1], e, price_window=x,
                           novelty=torch.ones(4, 5), pair_available=torch.ones(4, 5))
                loader = DataLoader(TensorDataset(torch.arange(33)), batch_size=4, shuffle=True, drop_last=True)
                run._configure_gate_sampling(loader, seed)
                orders = [torch.cat([batch[0] for batch in loader]) for _ in range(3)]
                current = film.state_dict(), text.group_mlps.state_dict(), rng, out.detach(), orders
                if reference is None:
                    reference = current
                else:
                    for actual, expected in zip(current[:2], reference[:2]):
                        for name in actual:
                            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                    for actual, expected in zip(current[2:4], reference[2:4]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for actual, expected in zip(orders, reference[4]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_independent_gradients_unique_parameters_and_restore(self):
        model = self.create('per_horizon')
        x, e = torch.randn(4, 7, 4), torch.randn(4, 5, 384)
        kw = dict(price_window=x, novelty=torch.ones(4, 5), pair_available=torch.ones(4, 5))
        price = torch.randn(4, 3, 4)
        out = model(price, x[:, -1], e, **kw)
        out[:, 1].square().mean().backward()
        self.assertGreater(model.gate.weight.grad[5:10, :8].abs().sum().item(), 0)
        self.assertTrue((model.gate.weight.grad[:5] == 0).all())
        self.assertTrue((model.gate.weight.grad[10:] == 0).all())
        self.assertTrue((model.gate.weight.grad[:, 8:] == 0).all())
        self.assertFalse(hasattr(model, 'G'))
        self.assertFalse(model.fixed_alpha.requires_grad)
        params = list(model.parameters())
        self.assertEqual(len(params), len({id(p) for p in params}))
        torch.optim.Adam(params, lr=.001).step()
        restored = self.create('per_horizon')
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        torch.testing.assert_close(restored(price, x[:, -1], e, **kw), model(price, x[:, -1], e, **kw), rtol=0, atol=0)
        # A legacy configuration/state has no price_gate_mode: shared is implicit.
        config = dict(hidden_size=64, pred_len=3, num_layers=1, group_gate='context_price_only')
        legacy = build_text_model(config, 'context_price_only')
        self.assertEqual(legacy.price_gate_mode, 'shared')
        legacy.load_state_dict(self.create().state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, 'requires context_price_only'):
            ContextNoveltyTextModel(64, 3, 1, group_gate='context_novelty', price_gate_mode='per_horizon')

    def test_runner_best_checkpoint_horizon_diagnostics_and_analysis(self):
        frame = make_frame()
        for mode in ('shared', 'per_horizon'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                datasets = {flag: make_dataset(frame, flag) for flag in ('train', 'val', 'test')}
                loaders = {flag: DataLoader(dataset, batch_size=4, shuffle=flag == 'train', drop_last=flag == 'train')
                           for flag, dataset in datasets.items()}
                def factory(**kwargs):
                    return datasets[kwargs['flag']], loaders[kwargs['flag']]
                argv = ['--model_type', 'FiLM', '--group_gate', 'context_price_only', '--price_gate_mode', mode,
                        '--code_commit', 'a' * 40, '--seq_len', '6', '--label_len', '2', '--e_layers', '1',
                        '--pred_len', '3', '--batch_size', '4', '--num_epoch', '1', '--patience', '10', '--logdir', directory,
                        '--data_path', 'SYN.parquet']
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader', side_effect=factory), \
                     patch('torch.cuda.is_available', return_value=False), \
                     patch('forecasting_task.models.FiLM.device', torch.device('cpu')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                root = Path(directory)
                summary = json.loads((root / 'summary.json').read_text())
                checkpoint = torch.load(root / 'best_checkpoint.pt', weights_only=True)
                self.assertEqual(summary['price_gate_mode'], mode)
                self.assertEqual(checkpoint['price_gate_mode'], mode)
                self.assertEqual(checkpoint['args']['text_model_config']['price_gate_mode'], mode)
                self.assertEqual(checkpoint['window_evidence'], summary['window_evidence'])
                with (root / 'validation_group_diagnostics.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    for h in (1, 2, 3):
                        self.assertIn(f'weight_h{h}', row)
                        if not int(row['present']):
                            self.assertEqual(float(row[f'contribution_h{h}']), 0)
                controlled = dict(analysis.COMMON, seq_len=6, label_len=2, batch_size=4, num_epoch=1)
                with patch('pandas.read_parquet', return_value=frame), patch.object(analysis, 'COMMON', controlled):
                    result = analysis.evaluate_run(Run(mode, 'SYN', 7, root), root)
                metrics, status = result[:2]
                self.assertEqual(status, 'full')
                self.assertAlmostEqual(metrics['val_mse'], summary['best_val_mse'], places=5)
                for split in ('val', 'test'):
                    self.assertAlmostEqual(metrics[f'{split}_mse'], sum(metrics[f'{split}_mse_h{h}'] for h in (1, 2, 3)) / 3)


if __name__ == '__main__':
    unittest.main()
