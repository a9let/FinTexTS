"""Quiet Russian report for 720 runs; detailed verified results stay in CSV."""
import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import analyze_context_price_group_gru_attention as prior
import group_gate_audit as audit
from analyze_persistence_baseline import Run
from forecasting_task.models.GroupGateExperiment import MODE
from run_group_gate_queue import CONFIGURATIONS, CONTROLS, RESULTS

RUNS = RESULTS.parent
DEFAULT_LAST = RUNS/'context_price_han_head_ablation_10_v1'/'full'/'last_matched'
DEFAULT_STATIC = RUNS/'context_novelty_gate_v1'/'full'/'static'
BASE_METRICS = ('val_mse','test_mse','test_mae')
METRICS = (*BASE_METRICS, *(f'{m}_h{h}' for m in BASE_METRICS for h in (1,2,3)))


def inspect(label, ticker, seed, output, cache, historical=False, configurations=CONFIGURATIONS):
    variant = 'original_baseline' if historical else MODE
    try:
        result = prior.inspect_run(Run(variant,ticker,seed,output), cache)
    except (EOFError,pickle.UnpicklingError) as error:
        return ({'variant':label,'ticker':ticker,'seed':seed,'completed':False,'comparable':False,
                 'verification':'mismatch','reason':str(error), **dict.fromkeys(METRICS)},None,None,None,[])
    row, metadata, payload = result
    row.update(variant=label, completed=(output/'summary.json').is_file() and (output/'best_checkpoint.pt').is_file(),
               verification=row['window_verification'], reason='', **{k:None for k in METRICS if k not in row})
    proof, contributions = None, []
    if metadata:
        args = metadata['args']
        try:
            expected = (CONTROLS if label.startswith('last_') else configurations)[label]
            if historical:
                gate='static' if expected['group_weight_condition']=='uniform' else 'context_price_only'
                if args.get('group_gate') != gate or args.get('fixed_alpha') != .1 or args.get('fusion_mode') != 'blend' or args.get('text_num_layers') != 1:
                    raise ValueError('Historical control has different weighting or anchored blend')
                if gate != 'static' and args.get('text_temporal_pool') != 'last_matched':
                    raise ValueError('Expected saved last_matched')
                state=torch.load(output/'best_checkpoint.pt',map_location='cpu',weights_only=True)['text_model_state_dict']
                if not np.isclose(float(state['fixed_alpha']),.1) or any(k in state for k in ('G','alpha_logit')):
                    raise ValueError('Historical control has trainable text/group weights')
                for group in ('macro','sector','targetCompany','relatedCompany','filing'):
                    if tuple(state[f'group_mlps.{group}.0.weight'].shape)!=(3,384):
                        raise ValueError('Historical control is not five direct last heads')
                if gate=='static' and any(k.startswith('gate.') for k in state):
                    raise ValueError('Static control contains a trainable gate')
            elif any(args.get(k) != v for k,v in expected.items()):
                raise ValueError('Experiment cell differs')
            proof, missing = audit.provenance(output,metadata)
            contributions = audit.contribution_rows(output,dict(args,group_weight_condition=expected['group_weight_condition']))
            if missing or row['window_verification'] != 'full':
                row.update(comparable=False,verification='partial',reason='; '.join(missing))
            else:
                row.update(comparable=True,verification='full')
            summary = json.loads((output/'summary.json').read_text())
            row.update(summary.get('validation_diagnostics',{}).get('horizon_metrics',{}))
        except (OSError,KeyError) as error:
            row.update(comparable=False,verification='partial',reason=str(error))
        except (ValueError,TypeError,RuntimeError) as error:
            row.update(comparable=False,verification='mismatch',reason=str(error))
        if payload:
            error = payload['outputs'] - payload['ground_truths']
            for h in range(3):
                row[f'test_mse_h{h+1}'] = float(np.square(error[:,h]).mean())
                row[f'test_mae_h{h+1}'] = float(np.abs(error[:,h]).mean())
    else:
        row.update(comparable=False,verification='not_checked',reason=row['missing_fields'] or row['mismatches'])
    return row,metadata,payload,proof,contributions


def pair(left,right,gate_pair=False,*,allow_cross_commit=False):
    status,reason='partial','incomplete evidence'
    if left[0]['comparable'] and right[0]['comparable']:
        try:
            audit.compare(left,right,gate_pair,allow_cross_commit=allow_cross_commit)
            status,reason='full',''
        except KeyError as error:
            reason=str(error)
        except (ValueError,TypeError) as error:
            status,reason='mismatch',str(error)
    elif 'mismatch' in (left[0]['verification'],right[0]['verification']):
        status='mismatch'
    return status,reason


def mean(values,count):
    return prior.mean_available(values,count)


def aggregate(rows):
    metrics, seeds = [], []
    for label,config in {**CONFIGURATIONS,**CONTROLS}.items():
        group=[r for r in rows if r['variant']==label]
        for seed in prior.SEEDS:
            sample=[r for r in group if r['seed']==seed]
            seeds.append({'variant':label,'seed':seed, **{m:mean([r[m] for r in sample],10) for m in METRICS}})
        records=[r for r in seeds if r['variant']==label]
        out={'variant':label,'family':config['text_family'],'hidden':None if config['text_family']=='last' else config['family_hidden_size'],
             'window':1 if config['text_family']=='last' else config['text_window'],'weights':config['group_weight_condition'],
             'completed_runs':sum(r['completed'] for r in group),'verified_runs':sum(r['comparable'] for r in group)}
        for m in METRICS:
            values=[r[m] for r in records]
            out[m]=mean(values,3)
            out[m+'_seed_std']=float(np.std(values,ddof=1)) if None not in values else None
        metrics.append(out)
    return metrics,seeds


def summarize_pair(left,right,rows,checks):
    lookup={(r['variant'],r['ticker'],r['seed']):r for r in rows}
    verified={(c['ticker'],c['seed']) for c in checks if c['status']=='full'}
    full=len(verified)==30
    mismatched=sum(c['status']=='mismatch' for c in checks)
    record={'variant':left,'control':right,'confirmed_pairs':len(verified),'expected_pairs':30,
            'mismatched_pairs':mismatched,'status':'full' if full else 'mismatch' if mismatched else 'partial',
            'eligible_tickers':0,'test_mse_wins':0}
    for metric in BASE_METRICS:
        a=mean([lookup[left,t,s][metric] for t,s in itertools.product(prior.TICKERS,prior.SEEDS)],30)
        b=mean([lookup[right,t,s][metric] for t,s in itertools.product(prior.TICKERS,prior.SEEDS)],30)
        record[metric+'_difference']=a-b if full and a is not None and b is not None else None
        record[metric+'_improvement_pct']=100*(b-a)/b if full and a is not None and b else None
    for t in prior.TICKERS:
        if all((t,s) in verified for s in prior.SEEDS):
            record['eligible_tickers']+=1
            a=np.mean([lookup[left,t,s]['test_mse'] for s in prior.SEEDS])
            b=np.mean([lookup[right,t,s]['test_mse'] for s in prior.SEEDS])
            record['test_mse_wins']+=int(a<b)
    for metric in ('val_mse','test_mse'):
        d=record[metric+'_difference']
        gate_pair=left.endswith('_price_only') and right.endswith('_uniform') and not right.startswith('last_')
        winner=(left if d<0 else right if d>0 else 'tie') if full and d is not None else 'unconfirmed'
        if gate_pair and winner not in ('tie','unconfirmed'):
            winner='price_only' if winner==left else 'uniform'
        record[metric+'_winner']=winner
    record['validation_test_disagree']= full and record['val_mse_winner'] != record['test_mse_winner']
    return record


def render(metrics,gates,architectures,completed,controls_status,output):
    def fmt(v): return 'н/д' if v is None else f'{v:.6g}'
    lines=[f"Готово: {completed}/720. Контроли: last_price_only — {controls_status['price_only']}, last_uniform — {controls_status['uniform']}"]
    winners=[]
    for family,title in (('han','HAN'),('huang','Huang'),('tlstm','T-LSTM')):
        comparisons=[g for g in gates if g['variant'].startswith(family+'_')]
        gains=sum(g['status']=='full' and (g['val_mse_improvement_pct'] or 0)>0 for g in comparisons)
        confirmed=sum(g['status']=='full' for g in comparisons)
        lines.append(f'{title}: ценовой gate улучшил validation в {gains}/4 конфигураций (подтверждено {confirmed}/4).')
        eligible=[r for r in metrics if r['family']==family and r['verified_runs']==30 and r['val_mse'] is not None]
        if not eligible:
            lines.append('     Выбор по validation пока не подтверждён.')
            continue
        best=min(eligible,key=lambda r:r['val_mse'])
        winners.append(best)
        relative=next(r for r in architectures if r['variant']==best['variant'])
        lines += [f"     Лучший среди подтверждённых по validation: H={best['hidden']}, окно={best['window']}, веса={best['weights']};",
                  f"     Val MSE {fmt(best['val_mse'])}, Test MSE {fmt(best['test_mse'])}, Test MAE {fmt(best['test_mae'])};",
                  f"     Test MSE относительно соответствующего last: {fmt(relative['test_mse_improvement_pct'])}% ({relative['status']})."]
        disagree=sum(g['status']=='full' and g['validation_test_disagree'] for g in comparisons)
        if disagree:
            lines.append(f'     В {disagree} парах победители validation и test различаются.')
    if winners:
        best=min(winners,key=lambda r:r['val_mse'])
        lines.append(f"Итог: лучший подтверждённый validation — {best['family']}, H={best['hidden']}, окно={best['window']}, {best['weights']}.")
    wins=[r['variant'] for r in architectures if r['status']=='full' and r['val_mse_improvement_pct'] is not None and r['val_mse_improvement_pct']>0]
    if wins:
        lines.append(f'Соответствующий last обошли по validation: {len(wins)}/24 вариантов.')
    else:
        lines.append('Среди подтверждённых сравнений ни один вариант не обошёл last по validation.')
    unknown=sum(r['status']!='full' for r in gates+architectures)
    lines.append(f'Неподтверждённых сравнений: {unknown}; partial не является строгим выводом.')
    if unknown:
        kinds=set()
        for r in gates:
            if r['status']!='full':kinds.add(r['variant'].split('_')[0]+'/price_only vs uniform')
        for r in architectures:
            if r['status']!='full':kinds.add(r['variant'].split('_')[0]+'/'+r['control'])
        lines.append('Неподтверждены: '+'; '.join(sorted(kinds))+'.')
    lines.append(f'Отчёты: {output}')
    return '\n'.join(lines)


def analyze(results=RESULTS,data=prior.DEFAULT_DATA,last_price=DEFAULT_LAST,last_uniform=DEFAULT_STATIC,output=None):
    results=Path(results)
    output=Path(output) if output else results/'diagnostics'/'gate_comparison'
    if output.exists(): raise FileExistsError(f'Report exists: {output}')
    cache=prior.ReplayCache(data)
    rows,checks=[],[]
    expected={(t,f'seed{s}') for t,s in itertools.product(prior.TICKERS,prior.SEEDS)}
    for label in CONFIGURATIONS:
        paths=list((results/'full'/label).glob('*/seed*/output/best_checkpoint.pt'))
        actual={(p.parents[2].name,p.parents[1].name) for p in paths}
        if len(actual)!=len(paths) or not actual<=expected:
            raise ValueError('Unexpected/duplicate main runs: '+label)
    output.mkdir(parents=True,exist_ok=False)
    contribution_file=(output/'contributions.csv').open('x',encoding='utf-8',newline='')
    fields=['variant','ticker','seed','window_index','input_end_date','group','present','weight',
            *[f'head_h{h}' for h in (1,2,3)],*[f'contribution_h{h}' for h in (1,2,3)]]
    writer=csv.DictWriter(contribution_file,fieldnames=fields,extrasaction='ignore');writer.writeheader()
    component_file=(output/'prediction_components.csv').open('x',encoding='utf-8',newline='')
    component_fields=['variant','ticker','seed','window_index','input_end_date','horizon','channel',
                      'film_part','anchor_part','text_part','prediction','reconstructed']
    component_writer=csv.DictWriter(component_file,fieldnames=component_fields,extrasaction='ignore');component_writer.writeheader()
    try:
        for ticker,seed in itertools.product(prior.TICKERS,prior.SEEDS):
            runs={label:inspect(label,ticker,seed,results/'full'/label/ticker/f'seed{seed}/output',cache) for label in CONFIGURATIONS}
            for condition,legacy in (('price_only',last_price),('uniform',last_uniform)):
                label='last_'+condition
                matched=results/'controls'/'full'/label/ticker/f'seed{seed}/output'
                historical=not (matched/'best_checkpoint.pt').exists()
                path=Path(legacy)/ticker/f'seed{seed}/output' if historical else matched
                runs[label]=inspect(label,ticker,seed,path,cache,historical=historical)
            anchor=next((r for label,r in runs.items() if label in CONFIGURATIONS and r[0]['comparable']),None)
            if anchor:
                for label,r in runs.items():
                    if label in CONFIGURATIONS and r[0]['comparable']:
                        try: audit.compare(anchor,r)
                        except (ValueError,KeyError,TypeError) as error:
                            r[0].update(comparable=False,verification='mismatch',reason=str(error))
            for label,configuration in CONFIGURATIONS.items():
                if configuration['group_weight_condition']=='price_only':
                    other=label.replace('_price_only','_uniform')
                    status,reason=pair(runs[label],runs[other],True)
                    checks.append(dict(kind='gate',variant=label,control=other,ticker=ticker,seed=seed,status=status,reason=reason))
                other='last_'+configuration['group_weight_condition']
                status,reason=pair(runs[label],runs[other])
                checks.append(dict(kind='architecture',variant=label,control=other,ticker=ticker,seed=seed,status=status,reason=reason))
            for label,item in runs.items():
                rows.append(item[0])
                if item[0]['comparable']:
                    for r in item[4]: writer.writerow(dict(r,variant=label,ticker=ticker,seed=seed))
                    if item[4]:
                        with (Path(item[0]['output'])/'validation_prediction_components.csv').open(encoding='utf-8',newline='') as handle:
                            for r in csv.DictReader(handle):component_writer.writerow(dict(r,variant=label,ticker=ticker,seed=seed))
    finally:
        contribution_file.close()
        component_file.close()
    metrics,seeds=aggregate(rows)
    lookup={(r['variant'],r['ticker'],r['seed']):r for r in rows}
    for c in checks:
        a,b=lookup[c['variant'],c['ticker'],c['seed']],lookup[c['control'],c['ticker'],c['seed']]
        for metric in BASE_METRICS:
            c[metric+'_difference']=a[metric]-b[metric] if c['status']=='full' and a[metric] is not None and b[metric] is not None else None
    by_ticker=[]
    for kind,left,right in sorted({(r['kind'],r['variant'],r['control']) for r in checks}):
        for ticker in prior.TICKERS:
            sample=[c for c in checks if (c['kind'],c['variant'],c['ticker'])==(kind,left,ticker)]
            valid=len(sample)==3 and all(c['status']=='full' for c in sample)
            record=dict(kind=kind,variant=left,control=right,ticker=ticker,status='full' if valid else 'partial')
            for metric in BASE_METRICS:
                a=mean([lookup[left,ticker,s][metric] for s in prior.SEEDS],3)
                b=mean([lookup[right,ticker,s][metric] for s in prior.SEEDS],3)
                record[metric+'_difference']=a-b if valid and a is not None and b is not None else None
                record[metric+'_improvement_pct']=100*(b-a)/b if valid and a is not None and b else None
            by_ticker.append(record)
    gates,architectures=[],[]
    for kind,target in (('gate',gates),('architecture',architectures)):
        pairs=sorted({(r['variant'],r['control']) for r in checks if r['kind']==kind})
        for left,right in pairs:
            target.append(summarize_pair(left,right,rows,[c for c in checks if c['kind']==kind and c['variant']==left]))
    for name,records in (('metrics',metrics),('by_seed',seeds),('per_run',rows),('verification',checks),('paired_by_ticker',by_ticker),('gate_comparison',gates),('architecture_comparison',architectures)):
        prior.write_table(output/(name+'.csv'),records)
    controls_status={}
    for condition in ('price_only','uniform'):
        control='last_'+condition
        group=[r for r in rows if r['variant']==control]
        available=[c for c in checks if c['kind']=='architecture' and c['control']==control and lookup[c['variant'],c['ticker'],c['seed']]['comparable']]
        controls_status[condition]='full' if len(group)==30 and all(r['comparable'] for r in group) and all(c['status']=='full' for c in available) else 'partial'
    missing=[c for c,status in controls_status.items() if status!='full']
    (output/'control_rerun_commands.txt').write_text('\n'.join(f'bash run_group_gate_ablation_720.sh --results {results} --controls {c}' for c in missing),encoding='utf-8')
    text=render(metrics,gates,architectures,sum(r['completed'] for r in rows if r['variant'] in CONFIGURATIONS),controls_status,output)
    (output/'summary_ru.txt').write_text(text+'\n',encoding='utf-8')
    print(text)
    return {'metrics':metrics,'gate_comparison':gates,'architecture_comparison':architectures,'verification':checks,'per_run':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--data',type=Path,default=prior.DEFAULT_DATA)
    parser.add_argument('--last_price',type=Path,default=DEFAULT_LAST)
    parser.add_argument('--last_uniform',type=Path,default=DEFAULT_STATIC)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    analyze(args.results,args.data,args.last_price,args.last_uniform,args.output)
