"""Q4-only selection followed by read-only cached test diagnostics."""
import csv

import numpy as np
import torch

from forecasting_task.mean_router import Router, equal_ticker_mse, fit_scaler, subset
from forecasting_task.mean_router_protocol import PERIODS, atomic_json, file_hash
from mean_router_experiment import write_csv
import router_budget_source as source

SERIES = ('A_200_20','B_200_20_with_initial','C_1000_100_with_initial')


def load_model(state,mode):
    model = Router(mode).eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    return model


def state_weights(state,mode,cache):
    return load_model(state,mode).weights((cache['features']-state['scaler_mean'])/state['scaler_scale'])


def zero_state(mode,cache):
    mu,sigma = fit_scaler(subset(cache,*PERIODS['router_train'])['features'])
    return dict(model=Router(mode).state_dict(),epoch=0,optimizer_steps=0,scaler_mean=mu,scaler_scale=sigma)


def q4_mse(state,mode,cache):
    val = subset(cache,*PERIODS['router_validation'])
    prediction = load_model(state,mode)(val['predictions'],(val['features']-state['scaler_mean'])/state['scaler_scale'])
    return equal_ticker_mse(prediction,val['targets'],val['ticker']).item()


def new_state(args,audit,mode,seed,cache):
    root = args.output/'router'/mode/f'seed{seed}'
    marker = source.verify_artifacts(root,('initial_checkpoint.pt','best_checkpoint.pt','last_checkpoint.pt','history.json','summary.json'))
    best = torch.load(root/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    last = torch.load(root/'last_checkpoint.pt',map_location='cpu',weights_only=True)
    initial = torch.load(root/'initial_checkpoint.pt',map_location='cpu',weights_only=True)
    signature = dict(mode=mode,seed=seed,lr=.001,epochs=args.max_epochs,patience=args.patience,batch_size=128,
                     regularization=.001,cache=cache['hashes'],source=cache['source'],schema=source.SCHEMA,
                     include_initial=True,provenance=audit['provenance'])
    if any(state['signature']!=signature for state in (best,last,initial)) or marker['signature']!=signature:
        raise ValueError('New router provenance/config/cache signature differs')
    if marker['best_sha256']!=file_hash(root/'best_checkpoint.pt'):
        raise ValueError('New router best digest differs')
    history = last['history']
    if [row['epoch'] for row in history]!=list(range(last['epoch']+1)) or initial['epoch']!=0:
        raise ValueError('Initial or real-step history incomplete')
    if initial['optimizer_steps']!=0 or initial['optimizer']['state'] or initial['wait']!=0:
        raise ValueError('Invalid initial optimizer/patience state')
    for state in (initial,best,last):
        if state['optimizer_steps']!=state['epoch'] or state.get('scheduler_state_dict','missing') is not None:
            raise ValueError('New step/scheduler evidence differs')
    if last['epoch']!=args.max_epochs and last['wait']<args.patience:
        raise ValueError('New fit is incomplete')
    selected = min(history,key=lambda row:row['router_validation_mse'])
    if best['epoch']!=selected['epoch'] or best['best']!=selected['router_validation_mse']:
        raise ValueError('New selected checkpoint is not earliest Q4 minimum including zero')
    for parameter in load_model(initial,mode).parameters():
        if torch.count_nonzero(parameter):
            raise ValueError('Initial logits are not zero')
    mu,sigma = fit_scaler(subset(cache,*PERIODS['router_train'])['features'])
    for state in (initial,best,last):
        torch.testing.assert_close(state['scaler_mean'],mu,rtol=0,atol=0)
        torch.testing.assert_close(state['scaler_scale'],sigma,rtol=0,atol=0)
    if not np.isclose(q4_mse(best,mode,cache),best['best'],rtol=1e-5,atol=1e-6):
        raise ValueError('New best Q4 MSE replay mismatch')
    return best,last,history


@torch.no_grad()
def choose(args,audit,manifest):
    """Complete and persist every choice before the first test-cache load."""
    choices,caches,metadata,curves = {},{},[],{}
    old_selection = source.read_json(args.source_root/'reports/final/selection.json')
    for seed in source.SEEDS:
        cache = source.validation_cache(args.source_root,manifest,seed)
        caches[seed] = cache
        val = subset(cache,*PERIODS['router_validation'])
        errors = (val['predictions']-val['targets'][:,None]).square().mean((2,3))
        per_ticker = torch.stack([errors[val['ticker']==i].mean(0) for i in range(10)])
        fixed = dict(global_expert=int(per_ticker.mean(0).argmin()),per_ticker=per_ticker.argmin(1).tolist())
        if old_selection['fixed_experts'][str(seed)]!=fixed:
            raise ValueError('Replayed per-seed fixed choices differ from old selection.json')
        states = {}
        for mode in source.MODES:
            old,last_old = source.old_state(args.source_root,mode,seed,cache)
            initial = zero_state(mode,cache)
            if not np.isclose(q4_mse(old,mode,cache),old['best'],rtol=1e-5,atol=1e-6):
                raise ValueError('Old best Q4 MSE replay mismatch')
            b = initial if q4_mse(initial,mode,cache)<=q4_mse(old,mode,cache) else old
            current,last,history = new_state(args,audit,mode,seed,cache)
            curves[mode,seed] = last_old['history'],history
            source.compare_history(last_old['history'],history)
            for series,state in zip(SERIES,(old,b,current)):
                label = series+'_'+mode
                states[label] = state
                is_new = series==SERIES[2]
                metadata.append(dict(method=label,series=series,mode=mode,seed=seed,best_step=state['epoch'],
                    optimizer_steps=last['epoch'] if is_new else last_old['epoch'],
                    max_steps=args.max_epochs if is_new else 200,
                    patience=args.patience if is_new else 20,
                    initial_admitted=series!=SERIES[0],
                    selected_from='equal_step0' if state['epoch']==0 else 'new_best' if is_new else 'old_best',
                    reached_step_limit=last['epoch']==args.max_epochs if is_new else last_old['epoch']==200,
                    q4_mse=q4_mse(state,mode,cache)))
        choices[seed] = dict(fixed=fixed,states=states)
    return choices,caches,metadata,curves


def control_weights(cache,fixed):
    n,k = cache['predictions'].shape[:2]
    return dict(
        equal_mixture=torch.full((n,k),1/k),
        best_global=torch.nn.functional.one_hot(torch.full((n,),fixed['global_expert']),k).float(),
        best_per_ticker=torch.nn.functional.one_hot(torch.tensor(fixed['per_ticker'])[cache['ticker']],k).float(),
        mean10_l1_5=torch.nn.functional.one_hot(torch.full((n,),list(source.CONFIGS).index('mean10_l1-5')),k).float(),
        mean80_l1_2_5_10_20=torch.nn.functional.one_hot(
            torch.full((n,),list(source.CONFIGS).index('mean80_l1-2-5-10-20')),k).float())


def percent(model,control):
    return 100*(model/control-1) if control else None


def practical_counts(differences,controls,atol,rtol):
    ties = np.abs(differences)<=atol+rtol*np.abs(controls)
    return dict(wins=int(((differences<0)&~ties).sum()),
                losses=int(((differences>0)&~ties).sum()),ties=int(ties.sum()))


def bootstrap(paired,block,resamples=5000,seed=7):
    # Average seeds in each ticker/date, then resample common date blocks jointly.
    dates = sorted({key[1] for key in paired})
    matrix = np.full((10,len(dates)),np.nan)
    for i,ticker in enumerate(source.TICKERS):
        for j,date in enumerate(dates):
            values = paired.get((ticker,date))
            if values:
                matrix[i,j] = np.mean(values)
    actual = min(block,len(dates))
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0,resamples,100):
        size = min(100,resamples-start)
        starts = rng.integers(len(dates),size=(size,int(np.ceil(len(dates)/actual))))
        indices = ((starts[:,:,None]+np.arange(actual))%len(dates)).reshape(size,-1)[:,:len(dates)]
        samples = np.nanmean(np.nanmean(matrix[:,indices],axis=2),axis=0)
        if not np.isfinite(samples).all():
            raise ValueError('Bootstrap draw has an unrepresented ticker; interval is unverified')
        estimates.extend(samples.tolist())
    return dict(block_rows=block,effective_block_rows=actual,resamples=resamples,bootstrap_seed=seed,
                mean_mse_difference=float(np.nanmean(np.nanmean(matrix,axis=1))),
                ci025=float(np.quantile(estimates,.025)),ci975=float(np.quantile(estimates,.975)))


def plot_curves(root,curves):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for (mode,seed),(old,new) in curves.items():
        fig = Figure(figsize=(10,4),tight_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1,2)
        for ax,key,title in zip(axes,('router_train_mse','router_validation_mse'),('Train MSE','Q4 MSE')):
            for values,label in ((old,'A 200/20'),(new,'C 1000/100 + initial')):
                ax.plot([r['optimizer_steps'] for r in values],[r[key] for r in values],label=label)
                selected = min(values,key=lambda row:row['router_validation_mse'])
                ax.axvline(selected['optimizer_steps'],linestyle=':',alpha=.5)
            initial = new[0][key]
            ax.axhline(initial,color='gray',linestyle='--',label='equal mixture')
            ax.set(xlabel='Optimizer steps',ylabel=title,title=f'{mode}, seed {seed}')
            ax.legend(fontsize=7)
        fig.savefig(root/f'curves_{mode}_seed{seed}.png',dpi=130)


@torch.no_grad()
def analyze(args,audit):
    root = args.output/'reports'/args.report_name
    if root.exists():
        raise FileExistsError('Report exists; choose a new --report-name')
    manifest,_ = source.verify_manifest(args.source_root)
    choices,caches,metadata,curves = choose(args,audit,manifest)
    root.mkdir(parents=True,exist_ok=False)
    selected = dict(test_previously_inspected=True,provenance=audit['provenance'],
                    selection_period=PERIODS['router_validation'],tie_atol=args.tie_atol,tie_rtol=args.tie_rtol,
                    choices={str(seed):dict(fixed=value['fixed'],
                        routers={name:state['epoch'] for name,state in value['states'].items()})
                        for seed,value in choices.items()})
    atomic_json(root/'selection.json',selected)
    # Test is first read after the preceding persisted Q4 choices.
    rows,weights_rows,excluded,daily = [],[],[],{}
    for seed in source.SEEDS:
        test_path = args.source_root/'cache'/f'seed{seed}_test.pt'
        test = source.check_cache(args.source_root,manifest,
                                 torch.load(test_path,map_location='cpu',weights_only=True),seed,test=True)
        selected.setdefault('test_cache_sha256',{})[str(seed)] = file_hash(test_path)
        for split,cache in (('validation',caches[seed]),('test',test)):
            weights = control_weights(cache,choices[seed]['fixed'])
            for method,state in choices[seed]['states'].items():
                mode = method.rsplit('_',1)[-1] if not method.endswith('price_text') else 'price_text'
                weights[method] = state_weights(state,mode,cache)
            predictions = {name:(value[:,:,None,None]*cache['predictions']).sum(1) for name,value in weights.items()}
            predictions['matched_baseline'] = cache['baseline']
            q4 = torch.tensor([all('2022-10-01'<=d<'2023-01-01' for d in target) for target in cache['target_dates']])
            periods = {'Q4':q4,'validation_2022_descriptive':torch.ones(len(q4),dtype=torch.bool)}
            if split=='test':
                periods = {'test':torch.ones(len(q4),dtype=torch.bool)}
                for q,(start,end) in enumerate((('2023-01-01','2023-04-01'),('2023-04-01','2023-07-01'),
                                               ('2023-07-01','2023-10-01'),('2023-10-01','2024-01-01')),1):
                    periods[f'2023Q{q}'] = torch.tensor([all(start<=d<end for d in target) for target in cache['target_dates']])
                quarterly = torch.stack([periods[f'2023Q{q}'] for q in range(1,5)]).any(0)
                for i,ticker in enumerate(source.TICKERS):
                    excluded.append(dict(seed=seed,ticker=ticker,excluded_quarter_windows=int(((cache['ticker']==i)&~quarterly).sum()),
                                         overall_test_windows=int((cache['ticker']==i).sum())))
            for method,prediction in predictions.items():
                errors = prediction-cache['targets']
                mse,mae = errors.square().mean((1,2)),errors.abs().mean((1,2))
                for i,ticker in enumerate(source.TICKERS):
                    for period,mask in periods.items():
                        indices = mask&(cache['ticker']==i)
                        n = int(indices.sum())
                        rows.append(dict(method=method,seed=seed,ticker=ticker,period=period,windows=n,
                                         mse=mse[indices].mean().item() if n else None,
                                         mae=mae[indices].mean().item() if n else None))
                if split=='test':
                    daily[method,seed] = dict(mse=mse,dates=cache['dates'],ticker=cache['ticker'])
            for method,value in weights.items():
                if not method.startswith(('A_','B_','C_')):
                    continue
                entropy = -(value*value.clamp_min(1e-12).log()).sum(1)
                previous = {}
                for index,ticker_id in enumerate(cache['ticker'].tolist()):
                    prior = previous.get(ticker_id)
                    row = dict(method=method,seed=seed,split=split,ticker=source.TICKERS[ticker_id],date=cache['dates'][index],
                               entropy=float(entropy[index]),effective_experts=float(entropy[index].exp()),
                               max_weight=float(value[index].max()),
                               weight_l1_change=float((value[index]-prior).abs().sum()) if prior is not None else None)
                    row.update({f'pi_{name}':float(value[index,k]) for k,name in enumerate(source.CONFIGS)})
                    weights_rows.append(row)
                    previous[ticker_id] = value[index]
    atomic_json(root/'selection.json',selected)
    summary,seed_rows,ticker_rows = aggregate(rows)
    controls = ('matched_baseline','best_global','best_per_ticker',SERIES[2]+'_constant')
    lookup = {row['method']:row for row in summary}
    counts = []
    for row in summary:
        for control in controls:
            for metric in ('mse','mae'):
                for period in ('Q4','test'):
                    key = period+'_'+metric
                    row[f'{key}_difference_vs_{control}'] = row[key]-lookup[control][key]
                    row[f'{key}_delta_pct_vs_{control}'] = percent(row[key],lookup[control][key])
            a = np.asarray([r['test_mse'] for r in ticker_rows if r['method']==row['method']])
            b = np.asarray([r['test_mse'] for r in ticker_rows if r['method']==control])
            counts.append(dict(method=row['method'],control=control,**practical_counts(a-b,b,args.tie_atol,args.tie_rtol)))
    effects = []
    for seed,mode in ((s,m) for s in source.SEEDS for m in source.MODES):
        for before,after,effect in ((SERIES[0],SERIES[1],'initial_admission'),(SERIES[1],SERIES[2],'budget_and_patience')):
            a = next(r for r in seed_rows if r['method']==before+'_'+mode and r['seed']==seed)
            b = next(r for r in seed_rows if r['method']==after+'_'+mode and r['seed']==seed)
            effects.append(dict(mode=mode,seed=seed,effect=effect,
                                **{key+'_difference':b[key]-a[key] for key in ('Q4_mse','test_mse','test_mae')}))
    intervals = []
    for mode in ('price','price_text'):
        method = SERIES[2]+'_'+mode
        for control in (SERIES[2]+'_constant','best_global'):
            paired = {}
            for seed in source.SEEDS:
                a,b = daily[method,seed],daily[control,seed]
                for i,(date,ticker_id) in enumerate(zip(a['dates'],a['ticker'].tolist())):
                    paired.setdefault((source.TICKERS[ticker_id],date),[]).append(float(a['mse'][i]-b['mse'][i]))
            for length in (10,20,40):
                intervals.append(dict(method=method,control=control,**bootstrap(paired,length)))
    with (args.source_root/'reports/final/summary.csv').open(encoding='utf-8',newline='') as handle:
        source_summary = list(csv.DictReader(handle))
    original_baseline = next(row for row in source_summary if row['method']=='matched_baseline')
    if not np.isclose(float(original_baseline['test_mse']),lookup['matched_baseline']['test_mse'],rtol=1e-4,atol=1e-6):
        raise ValueError('Cached baseline Test MSE differs from the original saved summary')
    for filename,values in (('summary',summary),('per_ticker_seed_period',rows),('per_seed',seed_rows),
                            ('per_ticker',ticker_rows),('per_fit',metadata),('effects',effects),
                            ('wins_losses_ties',counts),('quarter_exclusions',excluded),
                            ('paired_block_intervals',intervals),('weights_by_date',weights_rows)):
        write_csv(root/(filename+'.csv'),values)
    weight_stats = []
    for method in sorted({r['method'] for r in weights_rows}):
        for seed in source.SEEDS:
            for split in ('validation','test'):
                values = [r for r in weights_rows if r['method']==method and r['seed']==seed and r['split']==split]
                row = dict(method=method,seed=seed,split=split,windows=len(values))
                for key in ('entropy','effective_experts','max_weight','weight_l1_change'):
                    available = [r[key] for r in values if r[key] is not None]
                    row[key+'_mean'] = float(np.mean(available)) if available else None
                    row[key+'_std'] = float(np.std(available)) if available else None
                weight_stats.append(row)
    write_csv(root/'weight_statistics.csv',weight_stats)
    plot_curves(root,curves)
    for method in lookup:
        if method.startswith('C_'):
            for seed in source.SEEDS:
                values = [r for r in weights_rows if r['method']==method and r['seed']==seed and r['split']=='test']
                plot_weight_stats(root,method,seed,values)
    write_report(root,summary,metadata,counts,intervals,args,audit,seed_rows,rows)
    atomic_json(root/'completed.json',dict(signature=report_signature(args,audit),
        artifacts={p.name:file_hash(p) for p in root.iterdir() if p.is_file() and p.name!='completed.json'}))
    print(f'REPORT {root/"REPORT.md"}',flush=True)


def aggregate(rows):
    methods = sorted({r['method'] for r in rows})
    periods = ('Q4','test','validation_2022_descriptive')
    seeds,tickers,summary = [],[],[]
    for method in methods:
        seed_values = []
        for seed in source.SEEDS:
            item = dict(method=method,seed=seed)
            for period in periods:
                selected = [r for r in rows if r['method']==method and r['seed']==seed and r['period']==period]
                if len(selected)!=10 or any(r['mse'] is None for r in selected):
                    raise ValueError('All ten stocks are required for each seed/overall period')
                for metric in ('mse','mae'):
                    item[period+'_'+metric] = float(np.mean([r[metric] for r in selected]))
            seeds.append(item)
            seed_values.append(item)
        row = dict(method=method,test_previously_inspected=True)
        for period in periods:
            for metric in ('mse','mae'):
                key = period+'_'+metric
                row[key] = float(np.mean([r[key] for r in seed_values]))
                row[key+'_seed_sample_std'] = float(np.std([r[key] for r in seed_values],ddof=1))
        summary.append(row)
        for ticker in source.TICKERS:
            item = dict(method=method,ticker=ticker)
            for period in periods:
                selected = [r for r in rows if r['method']==method and r['ticker']==ticker and r['period']==period]
                for metric in ('mse','mae'):
                    item[period+'_'+metric] = float(np.mean([r[metric] for r in selected]))
            tickers.append(item)
    summary.sort(key=lambda row:row['Q4_mse'])
    return summary,seeds,tickers


def plot_weight_stats(root,method,seed,rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import MaxNLocator
    dates = sorted({r['date'] for r in rows})
    fig = Figure(figsize=(10,6),tight_layout=True)
    FigureCanvasAgg(fig)
    for axis,key in zip(fig.subplots(3,1),('entropy','max_weight','weight_l1_change')):
        values = [[r[key] for r in rows if r['date']==date and r[key] is not None] for date in dates]
        axis.plot(dates,[np.mean(v) if v else np.nan for v in values])
        axis.set(ylabel=key)
        axis.tick_params(axis='x',labelrotation=45,labelsize=6)
        axis.xaxis.set_major_locator(MaxNLocator(8))
    fig.savefig(root/f'weights_{method}_seed{seed}.png',dpi=120)


def report_signature(args,audit):
    return dict(provenance=audit['provenance'],max_epochs=args.max_epochs,patience=args.patience,
                tie_atol=args.tie_atol,tie_rtol=args.tie_rtol,
                selected_files={str(args.output/'router'/mode/f'seed{seed}'/'completed.json'):
                    file_hash(args.output/'router'/mode/f'seed{seed}'/'completed.json')
                    for mode in source.MODES for seed in source.SEEDS})


def verify_report(args,audit):
    root = args.output/'reports'/args.report_name
    marker = source.read_json(root/'completed.json')
    if marker['signature']!=report_signature(args,audit):
        raise ValueError('Completed report provenance/settings differ')
    for name,digest in marker['artifacts'].items():
        if file_hash(root/name)!=digest:
            raise ValueError('Completed report artifact differs')
    selection = source.read_json(root/'selection.json')
    for seed,digest in selection['test_cache_sha256'].items():
        if file_hash(args.source_root/'cache'/f'seed{seed}_test.pt')!=digest:
            raise ValueError('Completed report test cache changed')


def write_report(root,summary,metadata,counts,intervals,args,audit,seed_rows,period_rows):
    lines = ['# Проверка бюджета маршрутизаторов','',
             'A — исходный 200/20. B — тот же fit с допуском шага 0 по Q4. '
             f'C — новый fit {args.max_epochs}/{args.patience} с допуском шага 0.',
             'Выбор сделан по Q4 до чтения test. Полный validation-2022 является описательной метрикой.',
             'test_previously_inspected=true. Отрицательный Δ% означает снижение ошибки.',
             f'Практическая ничья: |ΔMSE| ≤ {args.tie_atol} + {args.tie_rtol} × |MSE контроля|.','',
             '| Вариант | Q4 MSE | Test MSE | Test MAE | Δ% к matched baseline |',
             '| --- | --- | --- | --- | --- |']
    for row in summary:
        delta = row["test_mse_delta_pct_vs_matched_baseline"]
        delta_text = '—' if delta is None else f'{delta:.3f}%'
        lines.append(f'| {row["method"]} | {row["Q4_mse"]:.8f} | {row["test_mse"]:.8f} | '
                     f'{row["test_mae"]:.8f} | {delta_text} |')
    lookup = {row['method']:row for row in summary}
    lines += ['', '## Допуск initial и бюджет','']
    for mode in source.MODES:
        a,b,c = [lookup[series+'_'+mode] for series in SERIES]
        admitted = sum(r['series']==SERIES[1] and r['mode']==mode and r['best_step']==0 for r in metadata)
        lines.append(f'- {mode}: B выбрал шаг 0 у {admitted}/3 seeds. '
                     f'A→B ΔQ4 MSE={b["Q4_mse"]-a["Q4_mse"]:.8f}; '
                     f'B→C ΔQ4 MSE={c["Q4_mse"]-b["Q4_mse"]:.8f}, '
                     f'ΔTest MSE={c["test_mse"]-b["test_mse"]:.8f}.')
    lines += ['', '## Адаптивность и fixed global','']
    for method in (SERIES[2]+'_price',SERIES[2]+'_price_text'):
        for control in (SERIES[2]+'_constant','best_global'):
            record = next(r for r in intervals if r['method']==method and r['control']==control and r['block_rows']==20)
            tally = next(r for r in counts if r['method']==method and r['control']==control)
            conclusion = ('интервал ниже нуля' if record['ci975']<0 else 'интервал выше нуля'
                          if record['ci025']>0 else 'интервал включает ноль')
            lines.append(f'- {method} против {control}: ΔMSE={record["mean_mse_difference"]:.8f}, '
                         f'95% интервал [{record["ci025"]:.8f}, {record["ci975"]:.8f}], {conclusion}. '
                         f'По акциям побед/поражений/ничьих {tally["wins"]}/{tally["losses"]}/{tally["ties"]}.')
            a = np.asarray([r['test_mse'] for r in seed_rows if r['method']==method])
            b = np.asarray([r['test_mse'] for r in seed_rows if r['method']==control])
            seed_counts = practical_counts(a-b,b,args.tie_atol,args.tie_rtol)
            quarter_a,quarter_b = [],[]
            for quarter in ('2023Q1','2023Q2','2023Q3','2023Q4'):
                left = [r['mse'] for r in period_rows if r['method']==method and r['period']==quarter and r['mse'] is not None]
                right = [r['mse'] for r in period_rows if r['method']==control and r['period']==quarter and r['mse'] is not None]
                if len(left)==len(right)==30:
                    quarter_a.append(np.mean(left))
                    quarter_b.append(np.mean(right))
            qa,qb = np.asarray(quarter_a),np.asarray(quarter_b)
            quarters = practical_counts(qa-qb,qb,args.tie_atol,args.tie_rtol)
            stable = (tally['wins']==10 and seed_counts['wins']==3 and quarters['wins']==len(qa) and len(qa)>0)
            lines.append(f'  По seed побед/поражений/ничьих {seed_counts["wins"]}/{seed_counts["losses"]}/{seed_counts["ties"]}; '
                         f'по полным кварталам {quarters["wins"]}/{quarters["losses"]}/{quarters["ties"]}. '
                         + ('Снижение однонаправленно по этим разрезам.' if stable else
                            'Устойчивое снижение по всем акциям, seed и полным кварталам не подтверждено.'))
    old_limits = sum(r['series']==SERIES[0] and r['best_step']==200 for r in metadata)
    limited = [r for r in metadata if r['series']==SERIES[2] and r['reached_step_limit']]
    lines += ['',f'У исходных fit best_step=200 встречается {old_limits}/9 раз.',
              f'Новый лимит достигнут у {len(limited)}/9 fit. Для них сходимость не установлена.',
              'Интервалы условны на текущем пуле моделей и акций. Сначала усреднены seed в ticker/date, '
              'затем совместно пересэмплированы временные блоки по акциям. 5000 выборок, seed=7, '
              'основная длина 20, чувствительность 10/40.',
              'Устойчивость по акциям, seed и кварталам отражена в per_ticker.csv, per_seed.csv и '
              'per_ticker_seed_period.csv. Пропуски квартальных назначений отражены в quarter_exclusions.csv.',
              'Изменение весов не доказывает пользу. pi не является вероятностью правильности эксперта '
              'или причинной важностью текстовой группы.',
              f'Проверены сохранённые артефакты. Raw-data/экспертный forward replay: '
              f'{audit["snapshot"]["verification"]["raw_data_replay"]}/'
              f'{audit["snapshot"]["verification"]["expert_inference_replay"]}.','',
              '## Кривые обучения','']
    lines += [f'![{mode}, seed {seed}](curves_{mode}_seed{seed}.png)' for mode in source.MODES for seed in source.SEEDS]
    (root/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
