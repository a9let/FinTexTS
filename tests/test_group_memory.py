import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import test_huang_query as fixture
from forecasting_task.models.GroupGateExperiment import GroupGateExperiment
from forecasting_task.models.GroupMemory import LinearMemory, PrototypeMemory
from forecasting_task.models.GroupText import TEXT_GROUPS, build_text_model
from forecasting_task.memory_phases import epoch_phase, initialize_prototypes, student_loss, train_classes, returns_for_windows
import run_group_memory_queue as queue
import run_han_head_ablation_queue as shared
import analyze_group_memory_ablation as analysis


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_memto_masked_write_read_gradients_and_snapshot(self):
        memory = PrototypeMemory(8)
        x = torch.randn(3,5,384)
        mask = torch.tensor([[1,0,1,0,1],[0,0,0,0,0],[0,1,0,1,0]],dtype=torch.bool)
        q = memory.encode(x,mask)
        changed = x.clone()
        changed[:,2:] = 500
        torch.testing.assert_close(memory.encode(changed,mask)[:,:2],q[:,:2])
        candidates, write = memory.write(q,mask)
        _, read = memory.read(q,candidates)
        torch.testing.assert_close(write.sum(-1),mask.any(1)[:,None].expand(-1,8).float())
        torch.testing.assert_close(read.sum(-1),torch.ones(3,5))
        original = memory.bank.clone()
        loss, _ = memory.reconstruction(x,mask)
        loss.backward()
        self.assertGreater(memory.old_gate.weight.grad.abs().sum(),0)
        self.assertGreater(memory.new_gate.weight.grad.abs().sum(),0)
        torch.testing.assert_close(candidates[1],original)
        torch.testing.assert_close(memory.bank,original)
        memory.commit(candidates.detach(),mask)
        torch.testing.assert_close(memory.bank,candidates[[0,2]].mean(0))
        memory.eval()
        frozen = memory.bank.clone()
        with torch.no_grad():
            memory.features(x,mask)
            memory.commit(candidates.detach(),torch.zeros_like(mask))
        torch.testing.assert_close(memory.bank,frozen)

    def test_titans_analytic_gradient_and_window_reset(self):
        module = LinearMemory(16)
        matrix = torch.randn(2,16,16,requires_grad=True)
        key,value = torch.randn(2,16),torch.randn(2,16)
        loss = .5*((matrix@key[...,None]).squeeze(-1)-value).square().sum()
        torch.testing.assert_close(module.gradient(matrix,key,value),torch.autograd.grad(loss,matrix)[0])
        text = torch.randn(2,5,384)
        mask = torch.tensor([[1,0,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
        expected, diagnostics = module(text,mask)
        changed = text.clone()
        changed[0,1::2] = 200
        changed[0,-1] = 400
        module(torch.randn_like(text),torch.ones_like(mask))
        torch.testing.assert_close(module(changed,mask)[0],expected)
        self.assertTrue((diagnostics[~mask]==0).all())
        expected.square().sum().backward()
        self.assertGreater(module.initial.grad.abs().sum(),0)

    def test_grid_start_checkpoint_group_masks_and_xtime_fusion(self):
        x,text,film,kwargs = fixture.inputs(2)
        text[0,:,0] = 0
        outcomes = {}
        for label,c in queue.CONFIGURATIONS.items():
            torch.manual_seed(7)
            parameters = {k:v for k,v in c.items() if k!='text_temporal_pool'}
            config = dict(hidden_size=64,pred_len=3,num_layers=1,group_gate='context_price_only',**parameters)
            model = GroupGateExperiment(**config).eval()
            if c['text_family']=='xtime_group':
                model.temporal_pool.normal_only.fill_(False)
            output,detail = model(film,x[:,-1],text[:,-1],text_window=text,return_details=True,**kwargs)
            self.assertTrue((detail['contributions'][0,0]==0).all())
            clone = GroupGateExperiment(**config).eval()
            clone.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(clone(film,x[:,-1],text[:,-1],text_window=text,**kwargs),output,rtol=0,atol=0)
            pair = label.rsplit('_',1)[0] if label.endswith('_uniform') else label.removesuffix('_price_only')
            if pair in outcomes:
                torch.testing.assert_close(outcomes[pair],output,rtol=0,atol=0)
            outcomes[pair] = output
            outside = text.clone()
            outside[:,:-c['text_window']] = 500
            torch.testing.assert_close(model(film,x[:,-1],outside[:,-1],text_window=outside,**kwargs),output,rtol=0,atol=0)
            if 'expert_weights' in detail:
                weights = detail['expert_weights']
                self.assertTrue(((weights>0).sum(-1)==2).all())
                torch.testing.assert_close(weights.sum(-1),torch.ones(2))
                torch.testing.assert_close((weights[...,None,None]*detail['expert_forecasts']).sum(1),output)

    def test_xtime_tail_sign_and_beta_only_changes_distillation(self):
        predicted = torch.tensor([[.05,-.05,.001]])
        truth = torch.tensor([[.10,-.10,.001]])
        student = torch.zeros(1,3,4)
        target,teacher = student.clone(),student+2
        base = student_loss(student,teacher,target,truth,predicted,.02,.1,1,0.)
        flipped = student_loss(student,teacher,target,-truth,-predicted,.02,.1,1,0.)
        self.assertAlmostEqual(base.item(),flipped.item())
        over = student_loss(student,teacher,target,truth,2*truth-predicted,.02,.1,1,0.)
        self.assertGreater(base,over)
        kd = student_loss(student,teacher,target,truth,predicted,.02,.1,1,.5)
        self.assertAlmostEqual((kd-base).item(),.5*(2/3)**2)

    def test_queue_inventories_and_pilot_reuse(self):
        full = set(queue.jobs())
        self.assertEqual(len(full),720)
        self.assertEqual(len(queue.jobs('full','memto_group')),240)
        self.assertEqual(len(queue.jobs('smoke')),24)
        self.assertEqual(len(queue.jobs('control')),30)
        self.assertTrue(set(queue.jobs('pilot','xtime_group')) <= full)

    def test_phase_resume_after_completed_epoch(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)/'phase.pt'
            model = torch.nn.Linear(1,1)
            optimizer = torch.optim.Adam(model.parameters(),lr=.1)

            def interrupted(epoch):
                if epoch:
                    raise RuntimeError('interrupted')
                optimizer.zero_grad()
                model(torch.ones(1,1)).sum().backward()
                optimizer.step()
                return 'first'

            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                epoch_phase(path,{'budget':2},model,optimizer,2,interrupted,torch.device('cpu'))
            expected = model.weight.detach().clone()
            renewed = torch.nn.Linear(1,1)
            new_optimizer = torch.optim.Adam(renewed.parameters(),lr=.1)

            def resumed(epoch):
                self.assertEqual(epoch,1)
                torch.testing.assert_close(renewed.weight,expected)
                return 'second'

            result = epoch_phase(path,{'budget':2},renewed,new_optimizer,2,resumed,torch.device('cpu'))
            self.assertEqual(result['orders'],['first','second'])

    def test_dispatcher_reuses_exact_normal_path_and_keeps_pilot(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data,results = root/'data',root/'results'
            data.mkdir()
            (data/'AAPL.parquet').touch()
            calls = []

            def train(command, log, code):
                output = command[command.index('--logdir')+1]
                Path(output).mkdir()
                log.write_text('synthetic')
                calls.append(command)
                return 0

            def completion(destination, expected, elapsed=None):
                return {'wall_seconds':1.}

            code = Path(__file__).resolve().parents[1]
            with patch.dict('os.environ',{'CODE_COMMIT':'a'*40}), patch.object(queue.torch.cuda,'is_available',return_value=True), \
                 patch.object(shared,'execute_training',side_effect=train),patch.object(shared,'completion',side_effect=completion):
                queue.run(code,data,results,'pilot','xtime_group')
                queue.run(code,data,results,'pilot','xtime_group')
            self.assertEqual(len(calls),12)  # Four shared normal runs and eight final runs.
            normals = [command for command in calls if 'normal' in command]
            self.assertEqual(len(normals),4)
            self.assertTrue(all(command[command.index('--num_epoch')+1]=='50' for command in normals))
            self.assertEqual(len(list((results/'full').glob('*/AAPL/seed7/output'))),8)

    def test_report_partial_controls_do_not_block_gate_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def inspect(label,ticker,seed,path,cache,historical=False):
                control = label.startswith('last_')
                value = 1. if label.endswith('price_only') else 2.
                return ({'variant':label,'ticker':ticker,'seed':seed,'output':str(path),'completed':not control,
                         'comparable':not control,**{metric:value for metric in analysis.METRICS}},
                        {},None,{},[])

            def compare(left,right,gate_pair):
                return ('full','') if gate_pair else ('partial','historical evidence unavailable')

            output = root/'report'
            with patch.object(analysis,'inspect',side_effect=inspect),patch.object(analysis,'checked_pair',side_effect=compare), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                analysis.analyze(root/'runs',root/'data',output,root/'old')
            self.assertEqual(len((output/'metrics.csv').read_text().splitlines()),25)
            self.assertEqual(len((output/'gate_comparison.csv').read_text().splitlines()),13)
            self.assertIn('720/720',log.getvalue())
            self.assertIn('подтверждено 4/4',log.getvalue())
            self.assertIn('last_uniform — partial',log.getvalue())
            self.assertNotIn('AAPL',log.getvalue())

    def test_runner_phase_artifacts_and_completed_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data'
            data.mkdir()
            frame = fixture.core.core.frame()
            # The legacy synthetic filing group is empty. Populate this fixture
            # so MEMTO can legitimately fit all eight prototypes per group.
            rng = np.random.default_rng(7)
            for group in TEXT_GROUPS:
                frame[[f'{group}_emb{i}' for i in range(384)]] = rng.normal(size=(len(frame),384)).astype(np.float32)
            frame.to_parquet(data/'SYN.parquet',index=False)
            datasets = {flag:fixture.core.ContextNoveltyDataset(fixture.core.Dataset_Custom(
                20,2,3,flag,list(TEXT_GROUPS),root_path=str(data),data_path='SYN.parquet'),include_calendar_days=True)
                for flag in ('train','val','test')}
            with self.assertRaisesRegex(ValueError,'train split'):
                train_classes(datasets['val'])
            classes, thresholds = train_classes(datasets['train'])
            self.assertEqual(sum(thresholds['counts']),len(classes))
            self.assertEqual(len(returns_for_windows(datasets['train'])),len(classes))
            with self.assertRaisesRegex(ValueError,'train split'):
                initialize_prototypes(None,datasets['test'],None,None)

            def factory(**kwargs):
                flag = kwargs['flag']
                dataset = datasets[flag]
                return dataset,DataLoader(dataset,batch_size=4,shuffle=flag=='train',drop_last=flag=='train')

            cache = root/'preparation'
            common = ['--model_type','FiLM','--root_path',str(data),'--data_path','SYN.parquet','--code_commit','a'*40,
                      '--seq_len','20','--label_len','2','--batch_size','4','--e_layers','1','--patience','10',
                      '--num_epoch','1','--memory_smoke','--text_window','5']
            stages = [('memto_group','final',8,'price_only',root/'memto'),
                      ('memto_group','final',8,'uniform',root/'uniform'),
                      ('xtime_group','normal',32,'price_only',cache/'normal'),
                      ('xtime_group','final',32,'price_only',root/'xtime')]
            for family,stage,size,condition,destination in stages:
                phase_root = cache if family=='xtime_group' else root/'prototypes'
                argv = common + ['--text_family',family,'--family_hidden_size',str(size),'--group_weight_condition',condition,
                                 '--memory_stage',stage,'--memory_cache',str(phase_root),'--memory_progress',str(destination/'progress'),
                                 '--logdir',str(destination/'output')]
                with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader',side_effect=factory), \
                     patch('torch.cuda.is_available',return_value=False), \
                     patch('forecasting_task.models.FiLM.device',torch.device('cpu')),contextlib.redirect_stdout(io.StringIO()):
                    fixture.core.run.main(argv)
                output = destination/'output'
                cp = torch.load(output/'best_checkpoint.pt',weights_only=True)
                text = build_text_model(cp['args']['text_model_config'],'context_price_only')
                text.load_state_dict(cp['text_model_state_dict'],strict=True)
                settings = dict(shared.SETTINGS,seq_len=20,label_len=2,batch_size=4,e_layers=1)
                config = dict(queue.configuration(family,size,5,condition),kd_beta=0.,memory_stage=stage,memory_smoke=True)
                with patch.object(shared,'SETTINGS',settings):
                    expected = shared.expected_config(family,'SYN',7,1,'a'*40,data,configuration=config)
                    shared.completion(destination,expected,elapsed=1)
                summary = json.loads((output/'summary.json').read_text())
                error = analysis.validation_predictions(output,datasets['val'])
                self.assertAlmostEqual(np.square(error).mean(),summary['best_val_mse'],places=5)
                if family=='xtime_group' and stage=='final':
                    self.assertFalse(text.temporal_pool.normal_only)
                    self.assertEqual(summary['memory_phases']['moderate']['epochs'],1)
                    self.assertEqual(summary['memory_phases']['router']['epochs'],1)
                if family=='memto_group':
                    self.assertTrue(summary['memory_phases']['prepared_sha256'])
            self.assertTrue((root/'prototypes'/'prepared.pt').exists())


if __name__=='__main__':
    unittest.main()
