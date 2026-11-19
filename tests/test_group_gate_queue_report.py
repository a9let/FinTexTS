"""720 unique jobs, family ETA, quiet CSV reporting and partial historical controls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_group_gate_queue as queue
import run_han_head_ablation_queue as shared
import test_huang_queue as core
import analyze_group_gate_ablation as analysis


ROOT=Path(__file__).resolve().parents[1]


class QueueReportTests(unittest.TestCase):
    def test_720_jobs_smoke_checks_resume_and_control_separation(self):
        jobs=queue.jobs()
        self.assertEqual(len(jobs),744)
        self.assertEqual(sum(j[0]=='full' for j in jobs),720)
        self.assertEqual(len(set(jobs)),744)
        self.assertEqual(len(queue.jobs('all')),62)
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';code=root/'code';results=root/'results'
            data.mkdir();code.mkdir()
            for ticker in (*shared.TICKERS,'NVDA'):(data/(ticker+'.parquet')).touch()
            for name in queue.SCRIPTS:(code/name).write_bytes((ROOT/name).read_bytes())
            def train(command,log,code):
                flags=core.core.HeadQueueTests.flags(command)
                self.assertEqual(flags['--text_temporal_pool'],'group_gate_experiment')
                label=f"{flags['--text_family']}_h{flags['--family_hidden_size']}_w{flags['--text_window']}_{flags['--group_weight_condition']}"
                configuration=queue.CONFIGURATIONS[label]
                expected=shared.expected_config(label,flags['--data_path'].split('.')[0],int(flags['--seed']),int(flags['--num_epoch']),flags['--code_commit'],data,configuration=configuration)
                output=Path(flags['--logdir']);output.mkdir()
                (output/'stub.json').write_text(json.dumps(expected));log.write_text('stub only')
                calls.append(command)
                return 0
            with patch.object(shared.torch.cuda,'is_available',return_value=True),patch.object(shared.torch.cuda,'get_device_name',return_value='mock'),patch.object(shared,'execute_training',side_effect=train),patch.object(shared,'completion',side_effect=core.HuangQueueTests.completion),contextlib.redirect_stdout(io.StringIO()) as log:
                queue.run_queue(code,data,results,'a'*40)
                self.assertEqual(len(calls),744)
                self.assertIn('FAMILY_TIMING family=tlstm',log.getvalue())
                self.assertTrue(log.getvalue().rstrip().endswith('ALL 720 RUNS COMPLETED'))
                queue.run_queue(code,data,results,'a'*40)
                self.assertEqual(len(calls),744)
                altered=next(results.glob('full/*/AAPL/seed7/output/stub.json'))
                altered.write_text('{}');preserved=altered.read_bytes()
                with self.assertRaises(ValueError):queue.run_queue(code,data,results,'a'*40)
                self.assertEqual(altered.read_bytes(),preserved)

    @staticmethod
    def inspect(label,ticker,seed,output,cache,historical=False):
        configurations={**queue.CONFIGURATIONS,**queue.CONTROLS}
        c=configurations[label]
        value=(.2 if c['group_weight_condition']=='price_only' else .25)+(.01 if c['text_family']=='tlstm' else 0)
        if label.startswith('last_'):value=.3
        row={'variant':label,'ticker':ticker,'seed':seed,'completed':True,'comparable':True,'verification':'full','reason':'',
             **{m:value for m in analysis.METRICS}}
        metadata={'args':dict(analysis.prior.COMMON,**{k:'same' for k in analysis.prior.PRICE_FIELDS},d_model=64,dropout=.1),
                  'evidence':{'same':True},'split':{},'scaler':{k:[1,2,3,4] for k in ('mean','scale','var')}}
        proof={'initialization':{'film_sha256':'a'*64,'price_gate_sha256':'b'*64},'source':'same','encoder':label.replace('_price_only','').replace('_uniform',''),'orders':['one']}
        return row,metadata,{},proof,[]

    def test_report_exact_rows_quiet_summary_and_partial_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            results=Path(directory)/'results'
            with patch.object(analysis,'inspect',side_effect=self.inspect),contextlib.redirect_stdout(io.StringIO()) as log:
                tables=analysis.analyze(results)
            self.assertEqual(len(tables['metrics']),26)
            self.assertEqual(len(tables['gate_comparison']),12)
            self.assertEqual(len(tables['architecture_comparison']),24)
            self.assertIn('Готово: 720/720.',log.getvalue())
            self.assertNotIn('AAPL',log.getvalue())
            self.assertNotIn('{',log.getvalue())
            self.assertTrue(all(r['val_mse_improvement_pct']>0 for r in tables['gate_comparison']))
            self.assertTrue(all(r['val_mse_winner']==r['variant'] for r in tables['architecture_comparison']))
            with self.assertRaises(FileExistsError):analysis.analyze(results)
            def partial(*args,**kwargs):
                result=self.inspect(*args,**kwargs)
                if result[0]['variant']=='last_uniform':result[0].update(comparable=False,verification='partial')
                return result
            with patch.object(analysis,'inspect',side_effect=partial),contextlib.redirect_stdout(io.StringIO()) as log:
                tables=analysis.analyze(results,output=Path(directory)/'partial')
            self.assertTrue(all(r['val_mse_improvement_pct'] is None for r in tables['architecture_comparison'] if r['control']=='last_uniform'))
            self.assertIn('last_uniform — partial',log.getvalue())
            self.assertIn('--controls uniform',(Path(directory)/'partial/control_rerun_commands.txt').read_text())

    def test_validation_test_disagreement_and_no_last_wins_message(self):
        rows=[]
        for label in {**queue.CONFIGURATIONS,**queue.CONTROLS}:
            for ticker in shared.TICKERS:
                for seed in shared.SEEDS:
                    r=self.inspect(label,ticker,seed,None,None)[0]
                    if label.startswith('last_'):r.update(val_mse=.01,test_mse=.01)
                    elif label.endswith('price_only'):r['test_mse']=.4
                    rows.append(r)
        metrics,_=analysis.aggregate(rows)
        gates=[];architectures=[]
        for label,c in queue.CONFIGURATIONS.items():
            targets=[('architecture','last_'+c['group_weight_condition'])]
            if label.endswith('price_only'):targets.append(('gate',label.replace('price_only','uniform')))
            for kind,other in targets:
                checks=[dict(ticker=t,seed=s,status='full') for t,s in __import__('itertools').product(shared.TICKERS,shared.SEEDS)]
                record=analysis.summarize_pair(label,other,rows,checks)
                (gates if kind=='gate' else architectures).append(record)
        text=analysis.render(metrics,gates,architectures,720,dict(price_only='full',uniform='full'),Path('reports'))
        self.assertIn('ни один вариант не обошёл last',text)
        self.assertTrue(all(r['validation_test_disagree'] for r in gates))


if __name__=='__main__':unittest.main()
