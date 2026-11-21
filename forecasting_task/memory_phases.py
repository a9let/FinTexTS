"""Train-only preparation and resumable epoch boundaries for memory models."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from forecasting_task.models.GroupText import TEXT_GROUPS

MEMTO_EPOCHS = (5, 10)
XTIME_SPECIALIST_EPOCHS = 10
XTIME_ROUTER_EPOCHS = 5


class InsufficientTrainData(ValueError):
    """The requested K or rarity classes cannot be fitted on this train split."""


def save_phase(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def read_phase(path, signature, device):
    if not path.exists():
        return None
    state = torch.load(path, map_location=device, weights_only=True)
    if state['signature'] != signature:
        raise ValueError(f'Incompatible phase checkpoint: {path}')
    return state


def training_only(dataset):
    if dataset.base.set_type != 0:
        raise ValueError('Preparation accepts only the train split')


def loader(dataset, seed, batch_size, indices=None, shuffle=False):
    source = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(source, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def batch_text(batch, device, window):
    # Dataset_Custom returns (price_x, price_y, text_x, x_mark, y_mark, index).
    return batch[2][:, -window:].float().to(device)


def epoch_phase(path, signature, module, optimizer, epochs, step, device):
    saved = read_phase(path, signature, device)
    elapsed, traces, completed = 0., [], 0
    if saved:
        module.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        elapsed, traces, completed = saved['seconds'], saved['orders'], saved['epoch']
    for epoch in range(completed, epochs):
        started = time.perf_counter()
        traces.append(step(epoch))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter()-started
        save_phase(path, dict(signature=signature, epoch=epoch+1, seconds=elapsed, orders=traces,
                              model=module.state_dict(), optimizer=optimizer.state_dict()))
        print(f'PHASE {path.stem} epoch={epoch+1}/{epochs} seconds={elapsed:.2f} '
              f'ETA_seconds={elapsed/(epoch+1)*(epochs-epoch-1):.2f}', flush=True)
    return {'epochs':epochs, 'seconds':elapsed, 'orders':traces}


def order_update(digest, batch):
    indices = np.asarray(batch[5], dtype='<i8')
    digest.update(np.asarray([len(indices)], dtype='<i8').tobytes())
    digest.update(indices.tobytes())


def initialize_prototypes(pool, dataset, args, device):
    training_only(dataset)
    queries = {g:[] for g in TEXT_GROUPS}
    with torch.no_grad():
        for batch in loader(dataset,args.seed,args.batch_size):
            text = batch_text(batch,device,args.text_window)
            present = text.abs().sum(-1) >= 1e-6
            for i,g in enumerate(TEXT_GROUPS):
                queries[g].append(pool[g].encode(text[:,:,i],present[:,:,i])[present[:,:,i]].cpu())
        for g in TEXT_GROUPS:
            values = torch.cat(queries[g]).numpy()
            count = len(pool[g].bank)
            if len(values) < count or len(np.unique(values,axis=0)) < count:
                raise InsufficientTrainData(f'{g}: fewer than {count} distinct available train queries')
            cluster = KMeans(n_clusters=count, n_init=10, random_state=args.seed).fit(values)
            pool[g].bank.copy_(torch.as_tensor(cluster.cluster_centers_,device=device))


def prepare_memto(model, dataset, args, device):
    training_only(dataset)
    root = Path(args.memory_cache)
    pool = model.temporal_pool
    budget = (1,1) if args.memory_smoke else MEMTO_EPOCHS
    signature = dict(code=args.code_commit, data=args.source_data_evidence['train'], seed=args.seed,
                     window=args.text_window, count=args.family_hidden_size, budget=list(budget),
                     batch_size=args.batch_size, lr=args.lr, method='gated_mean_bank_euclidean',
                     parquet=args.source_data_evidence['parquet_sha256'])
    final = root/'prepared.pt'
    saved = read_phase(final,signature,device)
    if saved:
        pool.load_state_dict(saved['model'])
        return {**saved['report'], 'prepared_sha256':hashlib.sha256(final.read_bytes()).hexdigest()}
    pool.requires_grad_(True)
    report = {}
    for phase, epochs in zip(('random_bank','clustered_bank'),budget):
        path = root/(phase+'.pt')
        if phase == 'clustered_bank':
            cluster_path = root/'kmeans.pt'
            clustered = read_phase(cluster_path,signature,device)
            if clustered:
                pool.load_state_dict(clustered['model'])
                report['kmeans_seconds'] = clustered['seconds']
            else:
                started = time.perf_counter()
                initialize_prototypes(pool,dataset,args,device)
                report['kmeans_seconds'] = time.perf_counter()-started
                save_phase(cluster_path,dict(signature=signature,model=pool.state_dict(),seconds=report['kmeans_seconds']))
        optimizer = torch.optim.Adam(pool.parameters(),lr=args.lr)

        def step(epoch):
            digest = hashlib.sha256()
            for batch in loader(dataset,args.seed+epoch,args.batch_size,shuffle=True):
                order_update(digest,batch)
                text = batch_text(batch,device,args.text_window)
                masks = text.abs().sum(-1) >= 1e-6
                optimizer.zero_grad()
                updates, losses = [], []
                counts = masks.sum((0,1))
                for i,g in enumerate(TEXT_GROUPS):
                    present = masks[:,:,i]
                    if present.any():
                        loss, candidates = pool[g].reconstruction(text[:,:,i],present)
                        losses.append(loss*counts[i])
                        updates.append((pool[g],candidates,present))
                if losses:
                    (sum(losses)/counts.sum()).backward()
                    optimizer.step()
                    for group,candidates,present in updates:
                        group.commit(candidates.detach(),present)
            return digest.hexdigest()

        report[phase] = epoch_phase(path,signature,pool,optimizer,epochs,step,device)
    started = time.perf_counter()
    with torch.no_grad():
        stats = {g:[] for g in TEXT_GROUPS}
        for batch in loader(dataset,args.seed,args.batch_size):
            text = batch_text(batch,device,args.text_window)
            present = text.abs().sum(-1) >= 1e-6
            for i,g in enumerate(TEXT_GROUPS):
                _, scalars, _ = pool[g].features(text[:,:,i],present[:,:,i])
                stats[g].append(scalars[present[:,:,i].any(1)])
        for g in TEXT_GROUPS:
            values = torch.cat(stats[g])
            pool[g].center.copy_(values.mean(0))
            pool[g].scale.copy_(values.std(0,unbiased=False).clamp_min(1e-6))
    pool.requires_grad_(False)
    report['prototype_selection'] = 'minimum_squared_euclidean'
    report['standardization_seconds'] = time.perf_counter()-started
    save_phase(final,dict(signature=signature, model=pool.state_dict(), report=report))
    return {**report, 'prepared_sha256':hashlib.sha256(final.read_bytes()).hexdigest()}


def returns_for_windows(dataset):
    close = dataset.base.data_price_x[:,3]*dataset.scaler.scale_[3]+dataset.scaler.mean_[3]
    end = np.arange(len(dataset))+dataset.seq_len-1
    if not np.isfinite(close).all() or (close[end] <= 0).any():
        raise ValueError('Close must be finite and positive at the forecast origin')
    return (close[end[:,None]+np.arange(1,dataset.pred_len+1)]-close[end,None])/close[end,None]


def train_classes(dataset, *, require_all_classes=True):
    training_only(dataset)
    returns = returns_for_windows(dataset)
    severity = np.abs(returns).max(1)
    p90,p97 = np.quantile(severity,[.90,.97])
    classes = (severity>p90).astype(np.int64)+(severity>p97)
    counts = np.bincount(classes,minlength=3)
    if require_all_classes and (p90 == p97 or not counts.all()):
        raise InsufficientTrainData(f'Train rarity classes are empty or thresholds coincide: {counts.tolist()}, {p90}, {p97}')
    origin_dates = [dataset.window_end_date(i) for i in range(len(dataset))]
    return classes, {'p90':float(p90), 'p97':float(p97), 'scale':max(float(returns.std()),1e-8),
                     'counts':counts.tolist(), 'first_origin':dataset.window_end_date(0),
                     'last_origin':dataset.window_end_date(len(dataset)-1), 'last_target':str(dataset.row_dates[-1]),
                     'origin_dates_by_class': {name:[date for date,label in zip(origin_dates,classes) if label==i]
                                               for i,name in enumerate(('normal','moderate','extreme'))}}


def student_loss(student, teacher, target, true_returns, predicted_returns, threshold, scale, expert, beta):
    u = true_returns.sign()*(true_returns-predicted_returns)/scale
    factor = torch.where(u >= 0,1.,.5 if expert == 1 else .25)
    tail = torch.expm1((u.abs()*factor).clamp_max(5))
    tail = (tail*(true_returns.abs()>threshold)).mean()
    loss = F.mse_loss(student,target)+.1*tail
    if beta:
        d = student-teacher.detach()
        loss = loss+beta*(d/(1+d.abs())).square().mean()
    return loss


def prepare_xtime(film, model, dataset, args, device, prepare_batch):
    training_only(dataset)
    normal = torch.load(Path(args.memory_cache)/'normal'/'output'/'best_checkpoint.pt',map_location=device,weights_only=True)
    previous = normal['args']
    for key in ('seed','code_commit','seq_len','label_len','pred_len','batch_size','group_weight_condition','text_window','lr','data_path',
                'source_data_evidence','window_evidence','split_policy'):
        if previous[key] != getattr(args,key):
            raise ValueError(f'xTime normal checkpoint differs in {key}')
    if previous.get('memory_stage') != 'normal' or previous.get('text_family') != 'xtime_group':
        raise ValueError('Expected a completed normal-stage checkpoint')
    if previous['memory_smoke'] != args.memory_smoke or previous['num_epoch'] != (1 if args.memory_smoke else 50):
        raise ValueError('Normal-stage budget differs from the final run')
    film.load_state_dict(normal['model_state_dict'])
    model.load_state_dict(normal['text_model_state_dict'])
    film.requires_grad_(False).eval()
    model.requires_grad_(False).eval()
    classes, thresholds = train_classes(dataset)
    expert_pool = model.temporal_pool
    root = Path(args.memory_progress) if args.memory_progress else Path(args.logdir).parent/'phases'
    signature = dict(code=args.code_commit, seed=args.seed, beta=args.kd_beta, thresholds=thresholds,
                     window=args.text_window, weights=args.group_weight_condition, data=args.source_data_evidence['train'],
                     normal_hash=hashlib.sha256((Path(args.memory_cache)/'normal'/'output'/'best_checkpoint.pt').read_bytes()).hexdigest(),
                     smoke=args.memory_smoke, lr=args.lr, batch_size=args.batch_size)
    normal_summary = json.loads((Path(args.memory_cache)/'normal'/'output'/'summary.json').read_text())
    report = {'normal_metrics':{k:normal['metrics'][k] for k in ('val_mse','test_mse','test_mae')},
              'normal_seconds':normal_summary['total_seconds'], 'normal_orders':normal_summary['train_batch_orders'],
              'normal_checkpoint_sha256':signature['normal_hash'], 'thresholds':thresholds}
    scale = float(dataset.scaler.scale_[3])
    mean = float(dataset.scaler.mean_[3])
    raw_returns = returns_for_windows(dataset)
    raw_origin = dataset.base.data_price_x[dataset.seq_len-1:dataset.seq_len-1+len(dataset),3]*scale+mean

    def forecasts(batch):
        x,text,xm,ym,decoder,target = prepare_batch(batch,device,args.pred_len,args.label_len)
        with torch.no_grad():
            base = film(x,xm,decoder,ym)
        text = text[:,-args.text_window:]
        present = text.float().abs().sum(-1)>=1e-6
        metadata = {k:batch[6][k].float().to(device) for k in ('novelty','pair_available')}
        _, predictions = model.expert_forecasts(base,x[:,-1],text[:,-1],text,present,
                                                   dict(price_window=x,**metadata))
        return target,predictions

    for phase, expert in (('moderate',1),('extreme',2),('router',None)):
        if expert is None:
            module = expert_pool.router
            indices = None
            epochs = XTIME_ROUTER_EPOCHS
        else:
            module = expert_pool.experts[expert]
            module.load_state_dict(expert_pool.experts[expert-1].state_dict())
            indices = np.flatnonzero(classes==expert).tolist()
            epochs = XTIME_SPECIALIST_EPOCHS
        module.requires_grad_(True)
        optimizer = torch.optim.Adam(module.parameters(),lr=args.lr)

        def step(epoch):
            digest = hashlib.sha256()
            for batch in loader(dataset,args.seed+epoch,args.batch_size,indices,shuffle=True):
                order_update(digest,batch)
                optimizer.zero_grad()
                target,predictions = forecasts(batch)
                if expert is None:
                    logits = module(predictions.detach().flatten(1))
                    loss = F.cross_entropy(logits,torch.as_tensor(classes[np.asarray(batch[5])],device=device))
                else:
                    windows = np.asarray(batch[5])
                    close = torch.as_tensor(raw_origin[windows,None],device=device)
                    true = torch.as_tensor(raw_returns[windows],device=device)
                    predicted = (predictions[:,expert,:,3].double()*scale+mean-close)/close
                    loss = student_loss(predictions[:,expert],predictions[:,expert-1],target,true,predicted,
                                        thresholds['p90' if expert==1 else 'p97'],thresholds['scale'],expert,args.kd_beta)
                loss.backward()
                optimizer.step()
            return digest.hexdigest()

        report[phase] = epoch_phase(root/(phase+'.pt'),signature,module,optimizer,1 if args.memory_smoke else epochs,step,device)
        module.requires_grad_(False)
    expert_pool.normal_only.fill_(False)
    return report, normal['metrics']['train_batch_order']
