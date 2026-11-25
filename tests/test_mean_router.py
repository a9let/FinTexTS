import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from forecasting_task import run
from forecasting_task.mean_router import *
from forecasting_task.mean_router_protocol import *
from forecasting_task.data_provider.context_novelty import ContextNoveltyDataset
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.experiment_evidence import state_fingerprint
from forecasting_task.models.GroupText import build_text_model
from test_mean_lag_extension import toy_data
import mean_router_experiment as experiment


def synthetic_cache():
    generator = torch.Generator().manual_seed(7)
    dates = [f'2022-{month:02d}-{day:02d}' for month in (2,5,11) for day in range(1,7)]
    n = len(dates)*2
    predictions = torch.randn(n,15,3,4,generator=generator)
    components = torch.zeros(n,15,7,3,4)
    components[:,:,0] = predictions
    cache = dict(predictions=predictions,targets=predictions[:,0]+.1,
                 features=torch.randn(n,40,generator=generator),ticker=torch.arange(2).repeat_interleave(len(dates)),
                 dates=dates*2,target_dates=[[d]*3 for d in dates*2],components=components,
                 baseline=predictions[:,1],source={'sha':'test'},schema=SCHEMA,eps=EPS)
    cache.update(group_weights=torch.full((n,15,5),.2),masks=torch.ones(n,15,5,dtype=torch.bool))
    cache['hashes'] = cache_hashes(cache)
    return cache


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_logits_one_hot_two_backbones_and_components(self):
        for backbone in (nn.Linear(4,12),nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,12))):
            prices = torch.randn(3,4)
            p = backbone(prices).reshape(3,1,3,4).expand(-1,15,-1,-1).clone()
            p += torch.arange(15)[None,:,None,None]
            x = torch.randn(3,40)
            for mode in ('constant','price','price_text'):
                model = Router(mode)
                torch.testing.assert_close(model(p,x),p.mean(1))
            model = Router('constant')
            with torch.no_grad():
                model.logits.fill_(-1000)
                model.logits[4] = 0
            torch.testing.assert_close(model(p,x),p[:,4],rtol=0,atol=0)
        c = synthetic_cache()
        validate_cache(c)
        c['components'][0,0,1] = 1
        with self.assertRaises(AssertionError):
            validate_cache(c)

    def test_full_batch_gradient_equal_tickers_l2_once_and_zero_grad(self):
        generator = torch.Generator().manual_seed(41)
        # Unequal ticker counts and an incomplete chunk expose window-weighting errors.
        train = dict(predictions=torch.randn(13,15,3,4,generator=generator,dtype=torch.float64),
                     targets=torch.randn(13,3,4,generator=generator,dtype=torch.float64),
                     features=torch.randn(13,40,generator=generator,dtype=torch.float64),
                     ticker=torch.tensor([0]*2+[1]*4+[2]*7))
        mean,scale = fit_scaler(train['features'])
        reference = Router('price_text').double()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.copy_(torch.randn(parameter.shape,generator=generator,dtype=torch.float64)*.2)
        expected = equal_ticker_mse(reference(train['predictions'],(train['features']-mean)/scale),
                                    train['targets'],train['ticker']) + .001*reference.penalty()
        expected.backward()
        gradients = [p.grad.clone() for p in reference.parameters()]
        initial = copy.deepcopy(reference.state_dict())
        for batch_size in (1,4,13,128):
            model = Router('price_text').double()
            model.load_state_dict(initial)
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter,999)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
            original_step = optimizer.step
            captured = []
            def checked_step():
                captured.extend(p.grad.clone() for p in model.parameters())
                return original_step()
            with patch.object(optimizer,'step',side_effect=checked_step) as steps:
                full_batch_step(model,optimizer,train,mean,scale,torch.Generator().manual_seed(7),batch_size,.001)
            self.assertEqual(steps.call_count,1)
            for actual,wanted in zip(captured,gradients):
                torch.testing.assert_close(actual,wanted,rtol=1e-11,atol=1e-12)
            self.assertTrue(all(int(state['step'])==1 for state in optimizer.state.values()))

    def test_original_base_code_train_test_scaler_match(self):
        source = subprocess.check_output(['git','show',f'{BASE_SHA}:forecasting_task/data_provider/dataset.py'],text=True)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),BASE_DATASET_SHA256)
        namespace = {'__name__':'fintexts_base_dataset'}
        exec(compile(source,'base_dataset.py','exec'),namespace)
        frame = toy_data()
        with patch('pandas.read_parquet',return_value=frame):
            for flag in ('train','val','test'):
                arguments = (64,16,3,flag,list(TEXT_GROUPS))
                original = ContextNoveltyDataset(namespace['Dataset_Custom'](*arguments))
                current = ContextNoveltyDataset(Dataset_Custom(*arguments),extended_text_window=80)
                self.assertEqual(window_evidence(current),window_evidence(original))
                for field in ('mean_','var_','scale_','n_samples_seen_'):
                    np.testing.assert_array_equal(getattr(current.scaler,field),getattr(original.scaler,field))

    def test_full_rng_round_trip(self):
        generator = torch.Generator().manual_seed(17)
        state = capture_rng(generator)
        expected = (torch.rand(3),random.random(),np.random.rand(3),torch.rand(3,generator=generator))
        restore_rng(state,generator)
        actual = (torch.rand(3),random.random(),np.random.rand(3),torch.rand(3,generator=generator))
        torch.testing.assert_close(expected[0],actual[0],rtol=0,atol=0)
        self.assertEqual(expected[1],actual[1])
        np.testing.assert_array_equal(expected[2],actual[2])
        torch.testing.assert_close(expected[3],actual[3],rtol=0,atol=0)

    def test_contract_actual_film_and_dlinear_frozen_forecasts(self):
        from analyze_blend_price_anchor import execution_device
        args = SimpleNamespace(task_name='short_term_forecast',seq_len=64,label_len=16,
                               pred_len=3,output_attention=False,e_layers=2,enc_in=4,moving_avg=25)
        with execution_device(torch.device('cpu')), torch.no_grad():
            for factory in (run.FiLM,run.DLinear):
                backbone = factory(args).eval().requires_grad_(False)
                before = state_fingerprint(backbone)
                x = torch.randn(2,64,4)
                output = backbone(x,torch.zeros(2,64,3),torch.zeros(2,19,4),torch.zeros(2,19,3))
                self.assertEqual(output.shape,(2,3,4))
                predictions = output[:,None].expand(-1,15,-1,-1).clone()
                router = Router('price')
                torch.testing.assert_close(router(predictions,torch.randn(2,40)),output)
                self.assertEqual(state_fingerprint(backbone),before)
                self.assertFalse(any(p.requires_grad for p in backbone.parameters()))
                self.assertTrue(set(map(id,backbone.parameters())).isdisjoint(set(map(id,router.parameters()))))

    def test_features_missing_history_causal_and_scaler_train_only(self):
        x = torch.arange(64*4.).reshape(1,64,4)
        e = torch.zeros(1,64,5,384)
        e[:,-1,0,0] = 2
        e[:,-2,0,0] = 1
        f = features(x,e)
        self.assertEqual(f.shape,(1,40))
        torch.testing.assert_close(f[0,:4],torch.full((4,),4.))
        torch.testing.assert_close(f[0,20:24],torch.tensor([1.,.1,0.,1.]))
        self.assertEqual(torch.count_nonzero(f[0,24:]),0)
        e[:,:-21] = 300
        torch.testing.assert_close(features(x,e),f)
        c = synthetic_cache()
        t = subset(c,'2022-04-01','2022-10-01')
        expected = fit_scaler(t['features'])
        c['features'][-6:] = 1000
        actual = fit_scaler(subset(c,'2022-04-01','2022-10-01')['features'])
        for a,b in zip(expected,actual):
            torch.testing.assert_close(a,b)

    def test_router_resume_freeze_batching_and_signed_improvement(self):
        c = synthetic_cache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit_router(c,root/'continuous','price_text',7,epochs=4)
            original = atomic_save
            def interrupt(path,state):
                original(path,state)
                if Path(path).name=='last_checkpoint.pt' and state['epoch']==2:
                    raise RuntimeError('interruption')
            with patch('forecasting_task.mean_router.atomic_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    fit_router(c,root/'resumed','price_text',7,epochs=4)
            last_path = root/'resumed'/'last_checkpoint.pt'
            saved = torch.load(last_path,weights_only=True)
            invalid = copy.deepcopy(saved)
            invalid['scheduler_state_dict'] = {'last_epoch':2}
            atomic_save(last_path,invalid)
            with self.assertRaisesRegex(ValueError,'scheduler'):
                fit_router(c,root/'resumed','price_text',7,epochs=4,resume=True)
            atomic_save(last_path,saved)
            fit_router(c,root/'resumed','price_text',7,epochs=4,resume=True)
            a = torch.load(root/'continuous'/'last_checkpoint.pt',weights_only=True)
            b = torch.load(root/'resumed'/'last_checkpoint.pt',weights_only=True)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(a['optimizer_steps'],4)
            self.assertEqual(a['optimizer_steps'],b['optimizer_steps'])
            self.assertEqual(a['wait'],b['wait'])
            self.assertIsNone(a['scheduler_state_dict'])
            self.assertIsNone(b['scheduler_state_dict'])
            for key,state in a['optimizer']['state'].items():
                for name,value in state.items():
                    torch.testing.assert_close(value,b['optimizer']['state'][key][name],rtol=0,atol=0)
            summary = json.loads((root/'resumed'/'summary.json').read_text())
            self.assertEqual(summary['optimizer_steps'],4)
            model = Router('price_text')
            model.load_state_dict(a['model'])
            full = model(c['predictions'],c['features'])
            chunks = torch.cat([model(p,x) for p,x in zip(c['predictions'].split(5),c['features'].split(5))])
            torch.testing.assert_close(full,chunks)
            c['features'][0,0] += 1
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                validate_cache(c)
        self.assertLess(100*(1-.03/.02),0)

    def test_periods_train_test_and_expert_resume(self):
        frame = toy_data()
        # Cover every assignment and its crossing horizons.
        dates = [*pd.date_range('2021-01-01',periods=130),
                 *pd.date_range('2022-01-03','2022-12-30',freq='B'),
                 *pd.date_range('2023-01-03',periods=10,freq='B')]
        frame = frame.iloc[np.arange(len(dates))%len(frame)].reset_index(drop=True)
        frame['date'] = [str(d.date()) for d in dates]
        with patch('pandas.read_parquet',return_value=frame):
            ds = {flag:ContextNoveltyDataset(Dataset_Custom(64,16,3,flag,list(TEXT_GROUPS))) for flag in ('train','val','test')}
        proof = evidence(ds)
        self.assertEqual(proof['excluded_validation_windows'],4)
        groups,_ = assignments(ds['val'])
        for name,indices in groups.items():
            start,end = PERIODS[name]
            for i in indices:
                self.assertTrue(all(start<=d<end for d in ds['val'].row_dates[i+64:i+67]))
        self.assertEqual(proof['windows']['train'],window_evidence(ds['train']))
        loaders = {flag:DataLoader(value,batch_size=32,shuffle=flag=='train',drop_last=flag=='train')
                   for flag,value in ds.items()}
        self.assertEqual(proof['windows']['test'],window_evidence(ds['test']))
        class TinyPrice(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4,4)
            def forward(self,x,*args):
                return self.linear(x[:,-1:]).expand(-1,3,-1)
        with tempfile.TemporaryDirectory() as directory:
            def training(name,resume=False,force_ties=False):
                root = Path(directory)/name
                root.mkdir(exist_ok=True)
                (root/'args.yaml').write_text('synthetic: true')
                torch.manual_seed(7)
                price,text = TinyPrice(),MatchedBaseline(3)
                generator = run._configure_gate_sampling(loaders['train'],7)
                args = SimpleNamespace(logdir=str(root),protocol_resume=resume,save_epochs=[],
                                       batch_size=32,num_epoch=3,patience=2 if force_ties else 10,group_gate='context_price_only',
                                       pred_len=3,label_len=16,text_temporal_pool='group_gate_experiment')
                optimizer = torch.optim.Adam([*price.parameters(),*text.parameters()],lr=.001)
                def epoch(*inputs,**kwargs):
                    if len(inputs)<7 or inputs[6] is None:
                        self.assertEqual(inputs[0].dataset.indices,groups['expert_selection'])
                        return 1.,1.
                    return run._run_epoch(*inputs,**kwargs)
                runner = SimpleNamespace(_run_epoch=epoch) if force_ties else run
                train_selected(runner,args,price,text,optimizer,ds,loaders,generator,torch.device('cpu'))
                return torch.load(root/'last_checkpoint.pt',weights_only=False)
            continuous = training('continuous')
            original = atomic_save
            def interrupted(path,state):
                original(path,state)
                if Path(path).name=='last_checkpoint.pt' and state['epoch']==2:
                    raise RuntimeError('interruption')
            with patch('forecasting_task.mean_router_protocol.atomic_save',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    training('resumed')
            resumed = training('resumed',True)
            for key in continuous['model_state_dict']:
                torch.testing.assert_close(continuous['model_state_dict'][key],resumed['model_state_dict'][key],rtol=0,atol=0)
            best = torch.load(Path(directory)/'resumed'/'best_checkpoint.pt',weights_only=False)
            self.assertEqual(best['epoch'],min(resumed['history'],key=lambda r:r['expert_selection_mse'])['epoch'])
            self.assertEqual(continuous['rng']['generator'].tolist(),resumed['rng']['generator'].tolist())
            original_targets = ds['val'].base.data_price_y
            ds['val'].base.data_price_y = original_targets.copy()
            later = ds['val'].row_dates >= '2022-04-01'
            ds['val'].base.data_price_y[later] += 10000
            changed = training('later_labels')
            self.assertEqual([r['expert_selection_mse'] for r in continuous['history']],
                             [r['expert_selection_mse'] for r in changed['history']])
            for key in continuous['model_state_dict']:
                torch.testing.assert_close(continuous['model_state_dict'][key],changed['model_state_dict'][key],rtol=0,atol=0)
            ds['val'].base.data_price_y = original_targets
            last_path = Path(directory)/'resumed'/'last_checkpoint.pt'
            saved = torch.load(last_path,weights_only=False)
            invalid = copy.deepcopy(saved)
            invalid['scheduler_state_dict'] = {'last_epoch':3}
            atomic_save(last_path,invalid)
            with self.assertRaisesRegex(ValueError,'scheduler'):
                training('resumed',True)
            atomic_save(last_path,saved)
            tied = training('ties',force_ties=True)
            with patch('forecasting_task.mean_router_protocol.atomic_save',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    training('ties_resume',force_ties=True)
            tied_resume = training('ties_resume',True,True)
            self.assertEqual(tied_resume['best_epoch'],1)
            self.assertEqual(tied_resume['patience_counter'],2)
            self.assertEqual(tied['patience_counter'],tied_resume['patience_counter'])
            for key,value in tied['optimizer_state_dict']['state'].items():
                for name,tensor in value.items():
                    torch.testing.assert_close(tensor,tied_resume['optimizer_state_dict']['state'][key][name],rtol=0,atol=0)

    def test_runner_protocol_entry_and_baseline_nan_aggregation(self):
        from forecasting_task.mean_router_protocol import BaselineDataset
        frame = toy_data()
        dates = [*pd.date_range('2021-01-01',periods=130),
                 *pd.date_range('2022-01-03','2022-12-30',freq='B'),
                 *pd.date_range('2023-01-03',periods=10,freq='B')]
        frame = frame.iloc[np.arange(len(dates))%len(frame)].reset_index(drop=True)
        frame['date'] = [str(d.date()) for d in dates]
        frame.loc[63,'macro_emb0'] = np.nan
        with patch('pandas.read_parquet',return_value=frame):
            base = Dataset_Custom(64,16,3,'train',list(TEXT_GROUPS))
            ds = BaselineDataset(base)
        expected = np.nanmean([frame.loc[63,f'{g}_emb0'] for g in TEXT_GROUPS])
        self.assertAlmostEqual(ds[0][6]['baseline_embedding'][0],expected,places=5)
        class TinyPrice(nn.Module):
            def __init__(self,config):
                super().__init__()
                self.linear = nn.Linear(4,4)
            def forward(self,x,*args):
                return self.linear(x[:,-1:]).expand(-1,3,-1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_parquet(root/'SYN.parquet',index=False)
            for baseline in (False,True):
                output = root/str(baseline)
                argv = ['--text_family','mean_price_grid','--text_window','80','--model_type','FiLM',
                        '--root_path',str(root),'--data_path','SYN.parquet','--logdir',str(output),
                        '--code_commit','a'*40,'--selection_protocol',PROTOCOL,'--num_epoch','1']
                if baseline:
                    argv += ['--matched_original_baseline']
                from forecasting_task.data_provider.context_novelty import get_context_novelty_dataloader
                def local_loader(**kwargs):
                    dataset, _ = get_context_novelty_dataloader(**kwargs)
                    training = kwargs['flag']=='train'
                    return dataset,DataLoader(dataset,batch_size=kwargs['batch_size'],
                                              shuffle=training,drop_last=training)
                with (patch.object(run,'FiLM',TinyPrice),patch('torch.cuda.is_available',return_value=False),
                      patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader',
                            side_effect=local_loader),
                      contextlib.redirect_stdout(io.StringIO())):
                    run.main(argv)
                saved = torch.load(output/'best_checkpoint.pt',weights_only=False)
                self.assertEqual(saved['args']['selection_protocol'],PROTOCOL)
                self.assertTrue(saved['test_previously_inspected'])
                self.assertGreater(saved['history'][0]['train_batch_order']['batches'],0)
                self.assertEqual(saved['evidence']['assignments']['expert_selection']['last_target'],'2022-03-31')
                self.assertFalse((output/'test_io.pt').exists())

    def test_exact_plan_489_and_12_model_registry(self):
        tasks = experiment.inventory('FiLM',Path('/results'))
        self.assertEqual(len(tasks),480)
        self.assertEqual(sum(t['kind']=='experts' for t in tasks),450)
        self.assertEqual(len({t['output'] for t in tasks}),480)
        self.assertEqual(len(BACKBONES),12)
        self.assertEqual(len(experiment.CONFIGS),15)
        self.assertEqual(len(experiment.MODES)*3,9)

    def test_matched_baseline_mask_after_bias(self):
        model = MatchedBaseline(3)
        with torch.no_grad():
            model.layers[0].bias.fill_(10)
        p,last,e = torch.randn(2,3,4),torch.randn(2,4),torch.zeros(2,5,384)
        torch.testing.assert_close(model(p,last,e),.9*p)
        e[0,:,0] = 1
        z = e.mean(1)
        expected = .9*p+.1*(last[:,None]+model.layers[0](z)[...,None])*(z.abs().sum(-1)>=1e-6)[:,None,None]
        torch.testing.assert_close(model(p,last,e),expected)

    def test_report_all_methods_periods_and_decompositions(self):
        cache = synthetic_cache()
        for key in ('predictions','targets','features','components','baseline','group_weights','masks'):
            cache[key] = cache[key].repeat((5,)+ (1,)*(cache[key].ndim-1))
        cache['ticker'] = torch.arange(10).repeat_interleave(18)
        cache['dates'] *= 5
        cache['target_dates'] *= 5
        cache['hashes'] = cache_hashes(cache)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(output=Path(directory),report_name='final')
            for seed in (7,17,27):
                for mode in ('constant','price','price_text'):
                    fit_router(cache,args.output/'router'/mode/f'seed{seed}',mode,seed,epochs=2)
            with patch.object(experiment,'create_cache',return_value=cache):
                experiment.report(args)
            import csv
            with (args.output/'reports'/'final'/'per_run.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            # 15 experts + equal + two fixed choices + 3 routers + baseline.
            self.assertEqual(len(rows),22*10*3*4)
            self.assertEqual({r['period'] for r in rows},
                             {'expert_selection','router_validation','full_validation_2022','test'})
            diagnostic = torch.load(args.output/'reports'/'final'/'test_price_seed7_diagnostics.pt',weights_only=True)
            self.assertEqual(diagnostic['components'].shape,(180,7,3,4))
            self.assertEqual(diagnostic['weights'].shape,(180,15))
            with (args.output/'reports'/'final'/'summary.csv').open() as handle:
                summary = list(csv.DictReader(handle))
            for row in summary:
                if row['method'] in ('constant','price','price_text'):
                    self.assertEqual(int(row['optimizer_steps_total']),6)
                    self.assertEqual(json.loads(row['optimizer_steps_by_seed']),{'7':2,'17':2,'27':2})


if __name__=='__main__':
    unittest.main()
