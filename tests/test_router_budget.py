import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from forecasting_task.mean_router import fit_router, Router, cache_hashes, fit_scaler, subset
from forecasting_task.mean_router_protocol import atomic_save, atomic_json, file_hash
from test_mean_router import synthetic_cache
import router_budget as budget
import router_budget_source as source
import router_budget_report as report


def zero_cache():
    cache = synthetic_cache()
    cache['predictions'].zero_()
    cache['components'].zero_()
    cache['targets'].fill_(1)
    cache['hashes'] = cache_hashes(cache)
    return cache


def disk_cache(seed,test=False):
    dates = (['2023-01-03','2023-03-29','2023-05-03','2023-07-03','2023-10-03','2023-12-01'] if test else
             ['2022-01-03','2022-01-04','2022-04-01','2022-04-04','2022-10-03','2022-10-04'])
    targets = [[str(d.date()) for d in pd.bdate_range(pd.Timestamp(date)+pd.Timedelta(days=1),periods=3)] for date in dates]
    n = 10*len(dates)
    g = torch.Generator().manual_seed(seed)
    return dict(predictions=torch.zeros(n,15,3,4),components=torch.zeros(n,15,7,3,4),
        targets=torch.full((n,3,4),.05),baseline=torch.full((n,3,4),.2),
        features=torch.randn(n,40,generator=g),ticker=torch.arange(10).repeat_interleave(len(dates)),
        dates=dates*10,target_dates=targets*10,group_weights=torch.full((n,15,5),.2),
        masks=torch.ones(n,15,5,dtype=torch.bool),schema=source.SCHEMA,eps=source.EPS)


def proof(cache,ticker):
    indices = torch.where(cache['ticker']==ticker)[0]
    dates = [cache['dates'][i] for i in indices]
    targets = [cache['target_dates'][i] for i in indices]
    return dict(windows=len(indices),
                dates_sha256=hashlib.sha256(('\n'.join(dates)+'\n').encode()).hexdigest(),
                target_dates_sha256=hashlib.sha256(''.join('|'.join(d)+'\n' for d in targets).encode()).hexdigest(),
                targets_sha256=hashlib.sha256(cache['targets'][indices].numpy().astype('<f4').tobytes()).hexdigest())


def saved_source(root):
    root.mkdir()
    (root/'cache').mkdir()
    values = {seed:disk_cache(seed) for seed in source.SEEDS}
    tests = {seed:disk_cache(seed,True) for seed in source.SEEDS}
    tasks = []
    for kind,cells in (('experts',source.CONFIGS),('controls',{'matched_baseline':None})):
        for cell in cells:
            for ticker_id,ticker in enumerate(source.TICKERS):
                for seed in source.SEEDS:
                    output = root/kind/cell/ticker/f'seed{seed}'/'output'
                    cache = values[seed]
                    target = cache['targets'][ticker_id*6:ticker_id*6+2]
                    p = cache['baseline'][ticker_id*6:ticker_id*6+2] if kind=='controls' else cache['predictions'][ticker_id*6:ticker_id*6+2,0]
                    tasks.append(dict(kind=kind,cell=cell,ticker=ticker,seed=seed,output=str(output),
                        checkpoint=str(output/'best_checkpoint.pt'),checkpoint_sha256='a'*64,
                        source_sha=source.SOURCE_SHA,q1_mse=(p-target).square().mean().item(),
                        evidence={'windows':{'val':proof(cache,ticker_id),'test':proof(tests[seed],ticker_id)}}))
    manifest = dict(protocol=source.PROTOCOL,backbone='FiLM',source_sha=source.SOURCE_SHA,base_sha=source.BASE_SHA,tasks=tasks)
    atomic_json(root/'expert_manifest.json',manifest)
    atomic_json(root/'data_audit.json',dict(base_sha=source.BASE_SHA,actual_dataset_sha256=source.BASE_DATASET_SHA256,
        records=[dict(ticker=t,current_policy_verification='full',historical_artifact_verification='partial') for t in source.TICKERS]))
    for seed in source.SEEDS:
        records = [r for r in tasks if r['seed']==seed]
        for test,cache in ((False,values[seed]),(True,tests[seed])):
            cache['source'] = dict(manifest_sha256=file_hash(root/'expert_manifest.json'),
                checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in records},split='test' if test else 'val',
                backbone='FiLM',seed=seed)
            cache['hashes'] = cache_hashes(cache)
            atomic_save(root/'cache'/f'seed{seed}_{"test" if test else "validation"}.pt',cache)
        for mode in source.MODES:
            fit_router(values[seed],root/'router'/mode/f'seed{seed}',mode,seed)
    report_root = root/'reports'/'final'
    report_root.mkdir(parents=True)
    atomic_json(report_root/'selection.json',dict(fixed_experts={str(seed):dict(global_expert=0,per_ticker=[0]*10) for seed in source.SEEDS}))
    baseline = (values[7]['baseline']-values[7]['targets']).square().mean().item()
    (report_root/'summary.csv').write_text(f'method,test_mse\nmatched_baseline,{baseline}\n')
    return manifest,values


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_best_zero_ties_counts_and_returned_best(self):
        cache = zero_cache()
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            model = fit_router(cache,root,'price',7,epochs=5,patience=2,include_initial=True)
            initial = torch.load(root/'initial_checkpoint.pt',weights_only=True)
            best = torch.load(root/'best_checkpoint.pt',weights_only=True)
            last = torch.load(root/'last_checkpoint.pt',weights_only=True)
            self.assertEqual(initial['epoch'],0)
            self.assertEqual(initial['optimizer']['state'],{})
            self.assertEqual(best['epoch'],0)
            self.assertEqual(last['epoch'],2)
            self.assertEqual(len(last['history']),3)
            self.assertEqual(json.loads((root/'summary.json').read_text())['epochs_completed'],2)
            val = subset(cache,*source.PERIODS['router_validation'])
            torch.testing.assert_close(model(val['predictions'],val['features']),val['predictions'].mean(1))

    def test_nonzero_best_is_returned_after_early_stopping(self):
        values = iter((10.,10.,1.,9.,2.,8.,3.,7.))
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            with patch('forecasting_task.mean_router.equal_ticker_mse',side_effect=lambda *args:torch.tensor(next(values))):
                model = fit_router(synthetic_cache(),root,'price',7,epochs=10,patience=2,include_initial=True)
            best = torch.load(root/'best_checkpoint.pt',weights_only=True)
            last = torch.load(root/'last_checkpoint.pt',weights_only=True)
            self.assertEqual(best['epoch'],1)
            self.assertEqual(last['epoch'],3)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,best['model'][key],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(best['model'][key],last['model'][key]) for key in best['model']))

    def test_resume_from_zero_and_nonzero_wait(self):
        for stop in (0,2):
            with self.subTest(stop=stop),tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                cache = zero_cache()
                fit_router(cache,root/'full','price_text',17,epochs=5,patience=4,include_initial=True)
                def interrupted(path,state):
                    atomic_save(path,state)
                    if Path(path).name=='last_checkpoint.pt' and state['epoch']==stop:
                        raise RuntimeError('interrupted')
                with patch('forecasting_task.mean_router.atomic_save',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):
                        fit_router(cache,root/'resume','price_text',17,epochs=5,patience=4,include_initial=True)
                fit_router(cache,root/'resume','price_text',17,epochs=5,patience=4,include_initial=True,resume=True)
                a = torch.load(root/'full'/'last_checkpoint.pt',weights_only=True)
                b = torch.load(root/'resume'/'last_checkpoint.pt',weights_only=True)
                self.assertEqual(a['history'],b['history'])
                self.assertEqual(a['wait'],b['wait'])
                for key in a['model']:
                    torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
                self.assertTrue(torch.equal(a['generator'],b['generator']))
                with self.assertRaisesRegex(ValueError,'configuration'):
                    fit_router(cache,root/'resume','price_text',17,epochs=6,patience=4,include_initial=True,resume=True)

    def test_exact_common_trajectory_against_pinned_original_source(self):
        original = subprocess.check_output(['git','show',f'{source.SOURCE_SHA}:forecasting_task/mean_router.py'],text=True)
        namespace = {'__name__':'original_mean_router'}
        exec(compile(original,'original_mean_router.py','exec'),namespace)
        for mode in source.MODES:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                cache = synthetic_cache()
                gradients = {'old':[],'new':[]}
                actual_step = torch.optim.AdamW.step
                label = ['old']
                def recorded(optimizer,*args,**kwargs):
                    gradients[label[0]].append([p.grad.clone() for group in optimizer.param_groups for p in group['params']])
                    return actual_step(optimizer,*args,**kwargs)
                with patch.object(torch.optim.AdamW,'step',recorded):
                    namespace['fit_router'](cache,root/'old',mode,7,epochs=5,patience=20)
                    label[0] = 'new'
                    fit_router(cache,root/'new',mode,7,epochs=5,patience=100,include_initial=True)
                old = torch.load(root/'old'/'last_checkpoint.pt',weights_only=True)
                new = torch.load(root/'new'/'last_checkpoint.pt',weights_only=True)
                self.assertEqual(old['history'],new['history'][1:])
                for key in old['model']:
                    torch.testing.assert_close(old['model'][key],new['model'][key],rtol=0,atol=0)
                for old_step,new_step in zip(gradients['old'],gradients['new']):
                    for a,b in zip(old_step,new_step):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertEqual(source.compare_history(old['history'],new['history'])['common_steps'],5)

    def test_plan_is_read_only_and_exactly_nine(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()) as out:
            root = Path(directory)
            budget.cli(['plan','--source-root',str(root/'missing'),'--output',str(root/'new')])
            plan = json.loads(out.getvalue())
            self.assertEqual(plan['fits'],9)
            self.assertEqual(len({(t['mode'],t['seed']) for t in plan['tasks']}),9)
            self.assertFalse((root/'new').exists())
            self.assertTrue(all(t['epochs']==1000 and t['patience']==100 and t['include_initial'] for t in plan['tasks']))

    def test_cache_only_queue_report_and_read_only_source(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source_root = root/'source'
            manifest,values = saved_source(source_root)
            before = {str(p):file_hash(p) for p in source_root.rglob('*') if p.is_file()}
            verification = dict(raw_data_replay='not_performed',expert_inference_replay='not_performed')
            snapshot = dict(verification=verification,synthetic=True)
            audit = dict(snapshot=snapshot,provenance={'source_code_sha':source.SOURCE_SHA,
                         'implementation_sha':'b'*40,'source_snapshot_sha256':'c'*64})
            args = SimpleNamespace(source_root=source_root,output=root/'new',max_epochs=3,patience=2,
                                   resume=False,report_name='final',tie_atol=1e-6,tie_rtol=1e-4)
            original_load = torch.load
            def no_test(path,*a,**kw):
                if str(path).endswith('_test.pt'):
                    raise AssertionError('Train opened test cache')
                return original_load(path,*a,**kw)
            with (patch.object(source,'source_snapshot',return_value=snapshot),patch.object(torch,'load',side_effect=no_test),
                  patch('mean_router_experiment.create_cache',side_effect=AssertionError('cache generation')),
                  patch('mean_router_experiment.train_tasks',side_effect=AssertionError('expert training')),
                  patch('mean_router_experiment.load_expert',side_effect=AssertionError('expert inference'))):
                budget.train(args,audit)
            self.assertEqual(len(list((args.output/'router').glob('*/seed*/completed.json'))),9)
            report.analyze(args,audit)
            report.verify_report(args,audit)
            report_root = args.output/'reports'/'final'
            self.assertTrue((report_root/'REPORT.md').is_file())
            selection = json.loads((report_root/'selection.json').read_text())
            self.assertEqual(len(selection['test_cache_sha256']),3)
            self.assertEqual(before,{str(p):file_hash(p) for p in source_root.rglob('*') if p.is_file()})
            altered = copy.deepcopy(values[7])
            altered['targets'][0,0,0] += 1
            altered['hashes'] = cache_hashes(altered)
            with self.assertRaisesRegex(ValueError,'targets'):
                source.check_cache(source_root,manifest,altered,7)
            with self.assertRaises(FileNotFoundError):
                source.validation_cache(root/'absent',manifest,7)
            self.assertTrue(all(torch.load(path,weights_only=True)['epoch']==0
                                for path in (args.output/'router').glob('*/seed*/best_checkpoint.pt')))

    def test_negative_delta_and_practical_ties(self):
        self.assertAlmostEqual(report.percent(.018,.02),-10)
        result = report.practical_counts(np.array([-.001,.001,1e-8]),np.array([.02]*3),1e-6,1e-4)
        self.assertEqual(result,dict(wins=1,losses=1,ties=1))

    def test_saved_expert_schema_and_digest_are_checked_without_inference(self):
        from forecasting_task.models.GroupText import build_text_model
        from run_han_head_ablation_queue import SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = next(iter(source.CONFIGS))
            output = root/'experts'/cell/source.TICKERS[0]/'seed7'/'output'
            output.mkdir(parents=True)
            args = dict(SETTINGS,**source.CONFIGS[cell],seed=7,code_commit=source.SOURCE_SHA,
                        selection_protocol=source.PROTOCOL,num_epoch=50,matched_original_baseline=False,
                        used_col_prefixes=list(source.TEXT_GROUPS),common_initialization={'film_sha256':'a'*64})
            config = dict(source.CONFIGS[cell],hidden_size=64,pred_len=3,num_layers=1)
            text = build_text_model(config)
            evidence = dict(scaler={'mean':[0.]*4,'scale':[1.]*4,'var':[1.]*4})
            signature = dict(args=args,evidence=evidence)
            state = dict(args=args,evidence=evidence,signature=signature,scaler=evidence['scaler'],
                epoch=1,best_mse=.1,scheduler_state_dict=None,text_model_state_dict=text.state_dict(),
                model_state_dict={'affine_weight':torch.ones(1,1,4),'affine_bias':torch.zeros(1,1,4),
                                  'mlp.weight':torch.ones(1,3)})
            atomic_save(output/'best_checkpoint.pt',state)
            history = [dict(epoch=i,expert_selection_mse=.1 if i==1 else .2,train_batch_order={'epoch':i}) for i in range(1,12)]
            atomic_save(output/'last_checkpoint.pt',dict(state,epoch=11,history=history,patience_counter=10))
            for name in ('summary.json','history.jsonl','args.yaml'):
                (output/name).write_text('synthetic')
            artifacts = ('best_checkpoint.pt','last_checkpoint.pt','summary.json','history.jsonl','args.yaml')
            atomic_json(output/'completed.json',dict(signature=signature,
                        artifacts={name:file_hash(output/name) for name in artifacts}))
            task = dict(kind='experts',cell=cell,ticker=source.TICKERS[0],seed=7,output=str(output),
                checkpoint=str(output/'best_checkpoint.pt'),checkpoint_sha256=file_hash(output/'best_checkpoint.pt'),
                source_sha=source.SOURCE_SHA,evidence=evidence,q1_mse=.1,initial=args['common_initialization'],
                train_batch_orders=[r['train_batch_order'] for r in history])
            manifest = dict(tasks=[task])
            audit = dict(records=[dict(ticker=source.TICKERS[0],evidence=evidence)])
            verified = source.verify_experts(root,manifest,audit)
            self.assertEqual(verified['expert_inference_replay'],'not_performed')
            state['text_model_state_dict']['core.gate.weight'] = torch.zeros(5,99)
            atomic_save(output/'best_checkpoint.pt',state)
            with self.assertRaisesRegex(ValueError,'digest'):
                source.verify_experts(root,manifest,audit)


if __name__=='__main__':
    unittest.main()
