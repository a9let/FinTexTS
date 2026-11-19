"""Paired initializations, preserved encoders and exact Baytas calendar mechanics."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import test_huang_query as fixture
from forecasting_task.models.GroupText import build_text_model,TEXT_GROUPS
from forecasting_task.models.GroupGateExperiment import MODE,experiment_structure
from forecasting_task.models.GroupTLSTM import GroupTLSTM
from forecasting_task.experiment_evidence import state_fingerprint,common_initialization
from forecasting_task.gate_diagnostics import write_gate_validation
import run_group_gate_queue as queue
import run_han_head_ablation_queue as shared
import analyze_group_gate_ablation as analysis


def config(family='tlstm',hidden=16,window=5,weights='price_only'):
    return dict(hidden_size=64,pred_len=3,num_layers=1,group_gate='context_price_only',text_temporal_pool=MODE,
                text_family=family,family_hidden_size=hidden,text_window=window,group_weight_condition=weights,
                temporal_pool_structure=experiment_structure(family,hidden,window,weights))


def inputs():
    x,text,price,kwargs=fixture.inputs(3)
    kwargs['calendar_days']=torch.arange(19000,19064).expand(3,-1).float()
    text[0,:,0]=0
    return x,text,price,kwargs


class GroupGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_all_24_pairs_start_identical_masks_and_checkpoint(self):
        for family in ('han','huang','tlstm'):
            for hidden in (16,32):
                for window in (5,20):
                    values=inputs()
                    models=[]
                    for weights in ('price_only','uniform'):
                        torch.manual_seed(17)
                        model=build_text_model(config(family,hidden,window,weights),'context_price_only').eval()
                        models.append(model)
                    a,b=models
                    self.assertIsNone(b.gate)
                    self.assertFalse(any('gate.' in n for n,_ in b.named_parameters()))
                    self.assertEqual(state_fingerprint(a.core,('gate.',)),state_fingerprint(b.core,('gate.',)))
                    x,text,price,kwargs=values
                    out,detail=a(price,x[:,-1],text[:,-1],text_window=text,return_details=True,**kwargs)
                    other=b(price,x[:,-1],text[:,-1],text_window=text,**kwargs)
                    torch.testing.assert_close(out,other,rtol=0,atol=0)
                    self.assertTrue((detail['contributions'][0,0]==0).all())
                    for model in models:
                        opt=torch.optim.Adam(model.parameters())
                        self.assertEqual(len([p for g in opt.param_groups for p in g['params']]),len({id(p) for p in model.parameters()}))
                        clone=build_text_model(config(family,hidden,window,model.weights_mode),'context_price_only').eval()
                        clone.load_state_dict(model.state_dict(),strict=True)
                        torch.testing.assert_close(clone(price,x[:,-1],text[:,-1],text_window=text,**kwargs),out,rtol=0,atol=0)

    def test_source_modules_transferred_verbatim_and_han_equivalence(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('GroupHANTemporal.py','HANHeadAblation.py'):
            expected=subprocess.check_output(['git','show',f'6a62808:forecasting_task/models/{name}'],cwd=root)
            self.assertEqual((root/'forecasting_task/models'/name).read_bytes().replace(b'\r\n',b'\n'),expected.replace(b'\r\n',b'\n'))
        from forecasting_task.models.HANHeadAblation import HANHeadAblationTextModel
        for hidden in (16,32):
            torch.manual_seed(7)
            model=build_text_model(config('han',hidden,5),'context_price_only').eval()
            with torch.no_grad():
                for attention in model.core.temporal_pool.attentions.values():
                    attention.position.copy_(torch.arange(20)*.07)
            reference=HANHeadAblationTextModel(text_temporal_pool=f'han_h{hidden}_linear',hidden_size=64,pred_len=3,num_layers=1,group_gate='context_price_only').eval()
            reference.load_state_dict(model.core.state_dict())
            x,text,price,kwargs=inputs();kwargs.pop('calendar_days')
            masked=text.clone();masked[:,:-5]=0
            actual=model(price,x[:,-1],text[:,-1],text_window=text,**kwargs)
            expected=reference(price,x[:,-1],masked[:,-1],text_window=masked,**kwargs)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)

    def test_baytas_formula_calendar_weekend_gaps_freeze_and_empty(self):
        pool=GroupTLSTM(16,5,3)
        text=torch.zeros(1,5,5,384);text[:,0,0]=1;text[:,3,0]=2
        # Friday 5 Jan -> Monday 8 Jan: 3 calendar days, despite skipped rows.
        days=torch.tensor([[19727.,19728.,19729.,19730.,19731.]])
        group=pool.cells['macro']
        with torch.no_grad(): group.decomposition.bias.fill_(2)
        calls=[]
        handle=group.cell.register_forward_hook(lambda m,a,o:calls.append(a))
        output,present,detail=pool(text,days)
        handle.remove()
        self.assertEqual(len(calls),2)
        self.assertEqual(detail['tlstm_delta'][0,3,0].item(),3)
        self.assertEqual(detail['tlstm_delta'][0,0,0].item(),0)
        torch.testing.assert_close(calls[0][1][1],torch.zeros(1,16),rtol=0,atol=0)
        previous_h=detail['tlstm_states'][:,0,0];previous_c=detail['tlstm_memories'][:,0,0]
        short=torch.tanh(group.decomposition(previous_c))
        adjusted=previous_c-short+short/np.log(np.e+3)
        torch.testing.assert_close(calls[1][1][0],previous_h,rtol=0,atol=0)
        torch.testing.assert_close(calls[1][1][1],adjusted)
        torch.testing.assert_close(detail['tlstm_states'][:,4,0],detail['tlstm_states'][:,3,0],rtol=0,atol=0)
        self.assertTrue((output[:,1:]==0).all())
        self.assertTrue((detail['tlstm_states'][:,:,1:]==0).all())
        output.square().sum().backward()
        self.assertGreater(group.decomposition.weight.grad.abs().sum(),0)
        # A group with one vector has no decay regardless of earlier dates/text outside W.
        text[:,0,0]=0
        _,_,one=pool(text,days)
        self.assertEqual(one['tlstm_delta'][0,3,0].item(),0)

    def test_independence_causality_and_paired_common_film_batches(self):
        for family in ('han','huang','tlstm'):
            records=[]
            for weights in ('price_only','uniform'):
                fixture.core.run.set_seed(7)
                film=fixture.core.fixtures.ContextNoveltyTests.make_film()
                model=build_text_model(config(family,weights=weights),'context_price_only').eval()
                loader=DataLoader(TensorDataset(torch.arange(31)),batch_size=4,shuffle=True,drop_last=True)
                fixture.core.run._configure_gate_sampling(loader,7)
                records.append((common_initialization(film,model)['film_sha256'],[torch.cat([b[0] for b in loader]) for _ in range(3)]))
            self.assertEqual(records[0][0],records[1][0])
            for a,b in zip(records[0][1],records[1][1]): torch.testing.assert_close(a,b,rtol=0,atol=0)
            x,text,price,kwargs=inputs()
            _,detail=model(price,x[:,-1],text[:,-1],text_window=text,return_details=True,**kwargs)
            changed=text.clone();changed[:,-4,0]+=100
            _,other=model(price,x[:,-1],changed[:,-1],text_window=changed,return_details=True,**kwargs)
            torch.testing.assert_close(detail['contributions'][:,1:],other['contributions'][:,1:],rtol=0,atol=0)
            if family=='tlstm':
                changed[:,-2:]=100
                _,future=model(price,x[:,-1],changed[:,-1],text_window=changed,return_details=True,**kwargs)
                torch.testing.assert_close(other['tlstm_states'][:,:3],future['tlstm_states'][:,:3],rtol=0,atol=0)

    def test_synthetic_runner_audit_horizons_and_diagnostics_all_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir()
            fixture.core.core.frame().to_parquet(data/'SYN.parquet',index=False)
            for family in ('han','huang','tlstm','last'):
                for weights in ('price_only','uniform'):
                    label=f'{family}_{weights}';output=root/label/'output'
                    datasets={f:fixture.core.ContextNoveltyDataset(fixture.core.Dataset_Custom(30,2,3,f,list(TEXT_GROUPS),root_path=str(data),data_path='SYN.parquet'),include_calendar_days=True) for f in ('train','val','test')}
                    loaders={f:DataLoader(d,batch_size=4,shuffle=f=='train',drop_last=f=='train') for f,d in datasets.items()}
                    def factory(**kwargs):return datasets[kwargs['flag']],loaders[kwargs['flag']]
                    hidden,window=(32,20) if family=='last' else (16,5)
                    argv=['--text_family',family,'--family_hidden_size',str(hidden),'--text_window',str(window),'--group_weight_condition',weights,
                          '--model_type','FiLM','--root_path',str(data),'--data_path','SYN.parquet','--code_commit','a'*40,
                          '--seq_len','30','--label_len','2','--batch_size','4','--e_layers','1','--num_epoch','1','--patience','10','--logdir',str(output)]
                    with patch('forecasting_task.data_provider.context_novelty.get_context_novelty_dataloader',side_effect=factory),patch('torch.cuda.is_available',return_value=False),patch('forecasting_task.models.FiLM.device',torch.device('cpu')),contextlib.redirect_stdout(io.StringIO()):
                        fixture.core.run.main(argv)
                    saved=torch.load(output/'best_checkpoint.pt',weights_only=True)
                    with patch('forecasting_task.models.FiLM.device',torch.device('cpu')):film=fixture.core.run.FiLM(SimpleNamespace(**saved['args'])).eval()
                    film.load_state_dict(saved['model_state_dict'])
                    textmodel=build_text_model(saved['args']['text_model_config'],'context_price_only').eval();textmodel.load_state_dict(saved['text_model_state_dict'])
                    with tempfile.TemporaryDirectory() as repeated:
                        args=SimpleNamespace(**{**saved['args'],'logdir':repeated,'diagnostic_best_val_mse':saved['metrics']['val_mse']})
                        write_gate_validation(loaders['val'],datasets['val'],film,textmodel,args,torch.device('cpu'),fixture.core.run._prepare_batch)
                        self.assertEqual((Path(repeated)/'validation_group_diagnostics.csv').read_bytes(),(output/'validation_group_diagnostics.csv').read_bytes())
                    summary=json.loads((output/'summary.json').read_text())
                    np.testing.assert_allclose(np.mean([summary['validation_diagnostics']['horizon_metrics'][f'val_mse_h{h}'] for h in (1,2,3)]),summary['best_val_mse'],rtol=1e-5,atol=1e-7)
                    self.assertTrue(analysis.audit.contribution_rows(output,saved['args']))
                    actual=queue.configuration(family,hidden,window,weights)
                    settings=dict(shared.SETTINGS,seq_len=30,label_len=2,batch_size=4,e_layers=1)
                    common=dict(analysis.prior.COMMON,seq_len=30,label_len=2,batch_size=4,num_epoch=1)
                    with patch.object(shared,'SETTINGS',settings),patch.object(analysis.prior,'COMMON',common):
                        expected=shared.expected_config(MODE,'SYN',7,1,'a'*40,data,configuration=actual)
                        shared.completion(output.parent,expected,elapsed=1.)
                        # Audit uses the production budget; exercise the replay with a test budget.
                        with patch.object(analysis.audit,'EPOCH_LIMIT',1):
                            variant=f'{family}_h{hidden}_w{window}_{weights}' if family!='last' else 'last_'+weights
                            row=analysis.inspect(variant,'SYN',7,output,analysis.prior.ReplayCache(data))
                        self.assertTrue(row[0]['comparable'],row[0])
                        self.assertEqual(row[0]['val_mse'],summary['best_val_mse'])


if __name__=='__main__':unittest.main()
