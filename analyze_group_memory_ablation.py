"""Check matching artifacts, report ordinary and rare errors, print a concise summary."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

import analyze_group_gate_ablation as previous
from forecasting_task.memory_phases import returns_for_windows, train_classes, XTIME_SPECIALIST_EPOCHS, XTIME_ROUTER_EPOCHS
from run_group_memory_queue import CONFIGURATIONS, FAMILIES, RESULTS


prior = previous.prior
CONTROLS = ('last_price_only','last_uniform')
OLD = RESULTS.parent/'group_gate_ablation_720_v1'
TAIL_METRICS = tuple(f'{split}_{category}_mse' for split,category in
                     itertools.product(('val','test'),('normal','moderate','extreme','rare')))
METRICS = (*previous.METRICS,*TAIL_METRICS)


def validation_predictions(output, dataset):
    size = (len(dataset),dataset.pred_len,4)
    forecast = np.full(size,np.nan)
    channels = ('open','high','low','close')
    with (output/'validation_prediction_components.csv').open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            i,h,c = int(row['window_index']),int(row['horizon'])-1,channels.index(row['channel'])
            if not (0<=i<size[0] and 0<=h<size[1]) or row['input_end_date'] != dataset.window_end_date(i) or not np.isnan(forecast[i,h,c]):
                raise ValueError('Validation prediction windows are duplicated or misaligned')
            forecast[i,h,c] = float(row['prediction'])
    if not np.isfinite(forecast).all():
        raise ValueError('Validation prediction windows are incomplete')
    targets = np.stack([dataset[i][1][-dataset.pred_len:] for i in range(len(dataset))]).astype(np.float32)
    return forecast-targets


def stratified(errors, dataset, thresholds, split):
    severity = np.abs(returns_for_windows(dataset)).max(1)
    losses = np.square(errors).mean((1,2))
    low,high = thresholds['p90'],thresholds['p97']
    categories = {'normal':severity<=low, 'moderate':(severity>low)&(severity<=high),
                  'extreme':severity>high,'rare':severity>low}
    return {f'{split}_{group}_{stat}': (int(mask.sum()) if stat=='count' else
            float(losses[mask].mean()) if mask.any() else None)
            for group,mask in categories.items() for stat in ('count','mse')}


def inspect(label,ticker,seed,output,cache,historical=False):
    item = previous.inspect(label,ticker,seed,output,cache,historical,CONFIGURATIONS)
    row,metadata,test,_,_ = item
    row.update({k:None for k in TAIL_METRICS})
    if not row['comparable']:
        return item
    try:
        datasets = metadata['datasets']
        family = metadata['args'].get('text_family')
        _, thresholds = train_classes(datasets['train'], require_all_classes=family=='xtime_group')
        validation = validation_predictions(output,datasets['val'])
        testing = test['outputs']-test['ground_truths']
        if not np.isclose(np.square(validation).mean(),row['val_mse'],rtol=1e-5,atol=1e-7):
            raise ValueError('Validation trace does not reproduce saved MSE')
        row.update(stratified(validation,datasets['val'],thresholds,'val'),
                   **stratified(testing,datasets['test'],thresholds,'test'))
        for h in range(validation.shape[1]):
            row[f'val_mse_h{h+1}'] = float(np.square(validation[:,h]).mean())
        phases = metadata['args'].get('memory_phases',{})
        if metadata['args'].get('memory_smoke') or family=='xtime_group' and metadata['args'].get('memory_stage')!='final':
            raise ValueError('Expected a full final run')
        if family=='memto_group' and not phases.get('prepared_sha256'):
            raise KeyError('Missing MEMTO preparation checksum')
        if family=='xtime_group':
            for phase,epochs in (('moderate',XTIME_SPECIALIST_EPOCHS),('extreme',XTIME_SPECIALIST_EPOCHS),('router',XTIME_ROUTER_EPOCHS)):
                if len(phases.get(phase,{}).get('orders',())) != epochs:
                    raise KeyError(f'Missing or incomplete xTime {phase} history')
            if not phases.get('normal_checkpoint_sha256') or not phases.get('normal_orders'):
                raise KeyError('Missing xTime normal checkpoint proof')
        if 'thresholds' in phases and phases['thresholds'] != thresholds:
            raise ValueError('xTime rarity thresholds disagree with train replay')
        for phase,values in phases.items():
            if isinstance(values,dict) and 'seconds' in values:
                row[phase+'_seconds'] = values['seconds']
        for key in ('normal_seconds','kmeans_seconds','standardization_seconds'):
            if key in phases:
                row[key] = phases[key]
        summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        row['total_seconds'] = summary['total_seconds']
        history = [json.loads(line) for line in (output/'history.jsonl').read_text(encoding='utf-8').splitlines() if line]
        if family != 'xtime_group':
            row['supervised_seconds'] = sum(record['train_seconds'] for record in history)
        else:
            row['final_scoring_seconds'] = sum(record['val_seconds']+record['test_seconds'] for record in history)
        for key,value in phases.get('normal_metrics',{}).items():
            row['normal_only_'+key] = value
    except (KeyError,OSError) as error:
        row.update(comparable=False,verification='partial',reason=str(error))
    except (ValueError,TypeError,RuntimeError) as error:
        row.update(comparable=False,verification='mismatch',reason=str(error))
    return item


def checked_pair(left,right,gate_pair):
    if gate_pair and left[0]['comparable'] and right[0]['comparable']:
        a,b = (item[1]['args'] for item in (left,right))
        if a['code_commit'] != b['code_commit']:
            return 'mismatch','Different code commits'
        pa,pb = (args.get('memory_phases',{}) for args in (a,b))
        if a.get('text_family') == 'memto_group':
            if pa.get('prepared_sha256') != pb.get('prepared_sha256'):
                return 'mismatch','Different MEMTO preparation checkpoints'
        if a.get('text_family') == 'xtime_group':
            for phase in ('moderate','extreme','router'):
                if pa.get(phase,{}).get('orders') != pb.get(phase,{}).get('orders'):
                    return 'mismatch',f'Different {phase} batch order'
            aa,bb = pa.get('normal_orders'),pb.get('normal_orders')
            if not aa or not bb:
                return 'partial','Missing normal-stage batch evidence'
            overlap = min(len(aa),len(bb))
            if aa[:overlap] != bb[:overlap]:
                return 'mismatch','Different normal-stage batch order'
            # The final evaluation record carries the normal best epoch's
            # batch order, which may be from different epochs in the two runs.
            left = (*left[:3],{**left[3],'orders':aa},left[4])
            right = (*right[:3],{**right[3],'orders':bb},right[4])
    if left[0]['comparable'] and right[0]['comparable'] and not gate_pair:
        if left[1]['args'].get('text_family') == 'xtime_group':
            orders = left[1]['args']['memory_phases']['normal_orders']
            left = (*left[:3],{**left[3],'orders':orders},left[4])
    return previous.pair(left,right,gate_pair,allow_cross_commit=not gate_pair)


def summarize_pair(label,control,rows,checks):
    out = previous.summarize_pair(label,control,rows,checks)
    left_rows = {(r['ticker'],r['seed']):r for r in rows if r['variant']==label}
    right_rows = {(r['ticker'],r['seed']):r for r in rows if r['variant']==control}
    for metric in TAIL_METRICS:
        eligible = [(r[metric], right_rows[k][metric]) for k,r in left_rows.items()
                    if k in right_rows and r[metric] is not None and right_rows[k][metric] is not None]
        out[metric+'_eligible_pairs'] = len(eligible) if out['status']=='full' else 0
        difference = improvement = None
        if out['status']=='full' and eligible:
            a,b = np.mean(eligible, axis=0).tolist()
            difference = a-b
            improvement = 100*(b-a)/b if b else None
        out[metric+'_difference'] = difference
        out[metric+'_improvement_pct'] = improvement
    return out


def aggregate(rows):
    result = []
    for label in (*CONFIGURATIONS,*CONTROLS):
        entries = [r for r in rows if r['variant']==label]
        verified = sum(r['comparable'] for r in entries)
        if label in CONTROLS and verified != 30:
            continue
        record = {'variant':label,'verified_runs':verified,'completed_runs':sum(r['completed'] for r in entries)}
        columns = (*METRICS, 'normal_only_val_mse','normal_only_test_mse','normal_only_test_mae')
        for metric in columns:
            means = []
            for seed in prior.SEEDS:
                sample = [r.get(metric) for r in entries if r['seed']==seed]
                if metric in TAIL_METRICS and verified==30:
                    sample = [value for value in sample if value is not None]
                means.append(prior.mean_available(sample,len(sample) if metric in TAIL_METRICS and verified==30 else 10) if sample else None)
            record[metric] = prior.mean_available(means,3)
            record[metric+'_seed_std'] = float(np.std(means,ddof=1)) if None not in means else None
        for metric in TAIL_METRICS:
            counts = [r.get(metric.replace('_mse','_count')) for r in entries]
            record[metric.replace('_mse','_window_seed_count')] = sum(counts) if len(counts)==30 and None not in counts else None
        result.append(record)
    return result


def diagnostics(output, label, ticker, seed, item):
    if not item[0]['comparable']:
        return
    source = Path(item[0]['output'])
    destination = output/'diagnostics'/label/ticker/f'seed{seed}'
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('validation_group_diagnostics.csv','validation_prediction_components.csv','validation_text_dates.csv'):
        path = source/name
        if path.exists():
            (destination/name).write_bytes(path.read_bytes())


def render(rows, metrics, gates, vs_last, statuses, output):
    def number(value):
        return 'н/д' if value is None else f'{value:.6g}'

    completed = sum(r['completed'] for r in rows if r['variant'] in CONFIGURATIONS)
    lines = [f'Готово {completed}/720. last_price_only — {statuses[0]}, last_uniform — {statuses[1]}.']
    comparisons = {r['variant']:r for r in vs_last}
    for family,title in zip(FAMILIES,('MEMTO','Titans','xTime')):
        local = [r for r in gates if CONFIGURATIONS[r['variant']]['text_family']==family]
        wins = sum((r['val_mse_improvement_pct'] or 0)>0 for r in local)
        verified = sum(r['status']=='full' for r in local)
        eligible = [r for r in metrics if r['variant'].startswith(family+'_') and r['verified_runs']==30 and r['val_mse'] is not None]
        line = f'{title}. Price-only выиграл validation в {wins}/4 пар, подтверждено {verified}/4.'
        if eligible:
            winner = min(eligible,key=lambda r:r['val_mse'])
            baseline = comparisons[winner['variant']]
            changed = sum(r['validation_test_disagree'] for r in local if r['status']=='full')
            line += (f' Лучший по validation среди готовых — {winner["variant"]}. Val MSE {number(winner["val_mse"])}, '
                     f'Test MSE {number(winner["test_mse"])}, MAE {number(winner["test_mae"])}. Выигрыш к last по Test MSE '
                     f'{number(baseline["test_mse_improvement_pct"])}% на всех, '
                     f'{number(baseline["test_rare_mse_improvement_pct"])}% на редких. Расхождений validation/test {changed}.')
        lines.append(line)
    winners = [r['variant'] for r in vs_last if (r['val_mse_improvement_pct'] or 0)>0]
    unconfirmed = sorted({r['control'] for r in vs_last if r['status']!='full'})
    lines.append('Last обошли по validation — '+(', '.join(winners) if winners else 'ни один вариант в подтверждённых парах')+'.')
    lines.append(f'Сравнения с last без полной сверки — {", ".join(unconfirmed) or "нет"}. '
                 'Редкие окна оцениваются по train-порогам, их число указано в отчёте.')
    lines.append(f'Отчёты в {output}')
    return '\n'.join(lines)


def analyze(results,data,output,old):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cache = prior.ReplayCache(data)
    rows, gate_pairs, architecture_pairs, checks = [], {}, {}, []
    for ticker,seed in itertools.product(prior.TICKERS,prior.SEEDS):
        runs = {label:inspect(label,ticker,seed,results/'full'/label/ticker/f'seed{seed}'/'output',cache)
                for label in CONFIGURATIONS}
        price_paths = [old/'controls'/'full'/'last_price_only', previous.DEFAULT_LAST]
        price = price_paths[0]/ticker/f'seed{seed}'/'output'
        historical = not (price/'best_checkpoint.pt').exists()
        if historical:
            price = price_paths[1]/ticker/f'seed{seed}'/'output'
        runs['last_price_only'] = inspect('last_price_only',ticker,seed,price,cache,historical)
        uniform = results/'controls'/'full'/'last_uniform'/ticker/f'seed{seed}'/'output'
        runs['last_uniform'] = inspect('last_uniform',ticker,seed,uniform,cache)
        for label,config in CONFIGURATIONS.items():
            condition = config['group_weight_condition']
            pairs = [(architecture_pairs,'last_'+condition,False)]
            if condition=='price_only':
                pairs.append((gate_pairs,label.removesuffix('price_only')+'uniform',True))
            for collection,control,gate in pairs:
                status,reason = checked_pair(runs[label],runs[control],gate)
                check = dict(ticker=ticker,seed=seed,status=status,reason=reason)
                collection.setdefault((label,control),[]).append(check)
                checks.append(dict(variant=label,control=control,**check))
        for label,item in runs.items():
            rows.append(item[0])
            diagnostics(output,label,ticker,seed,item)
    metrics = aggregate(rows)
    gates = [summarize_pair(*key,rows,value) for key,value in gate_pairs.items()]
    vs_last = [summarize_pair(*key,rows,value) for key,value in architecture_pairs.items()]
    lookup = {(r['variant'],r['ticker'],r['seed']):r for r in rows}
    statuses = tuple('full' if sum(r['comparable'] for r in rows if r['variant']==c)==30
                     and all(r['status']=='full' for r in checks if r['control']==c
                             and lookup[r['variant'],r['ticker'],r['seed']]['comparable'])
                     else 'partial' for c in CONTROLS)
    metrics = [r for r in metrics if r['variant'] not in CONTROLS or statuses[CONTROLS.index(r['variant'])]=='full']
    for name,records in (('metrics',metrics),('gate_comparison',gates),('vs_last',vs_last),('per_run',rows),('verification',checks)):
        prior.write_table(output/(name+'.csv'),records)
    historical_report = old/'diagnostics'/'gate_comparison'/'metrics.csv'
    if historical_report.is_file():
        (output/'historical_metrics.csv').write_bytes(historical_report.read_bytes())
    protocol = Path(__file__).with_name('MEMORY_ABLATION.md').read_text(encoding='utf-8')
    totals = []
    for family in FAMILIES:
        group = [r for r in rows if r['variant'].startswith(family+'_')]
        columns = sorted({k for r in group for k in r if k.endswith('_seconds')})
        totals.append(f'### {family}')
        for key in columns:
            values = [r[key] for r in group if key in r]
            totals.append(f'- {key} — mean {np.mean(values):.2f} s across {len(values)} recorded runs.')
    totals = '\n'.join(totals)
    (output/'protocol.md').write_text(protocol+'\n\n## Recorded phase time\n\n'+totals+'\n',encoding='utf-8')
    print(render(rows,metrics,gates,vs_last,statuses,output))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--data',type=Path,default=prior.DEFAULT_DATA)
    parser.add_argument('--old',type=Path,default=OLD)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    analyze(args.results,args.data,args.output or args.results/'reports'/'comparison',args.old)
