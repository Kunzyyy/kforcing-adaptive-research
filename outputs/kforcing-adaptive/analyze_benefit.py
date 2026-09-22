"""Locked train/dev ridge model, then article-clustered held-out diagnostics."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def rank(x):
    x=np.asarray(x)
    return np.array([(x<v).sum()+.5*((x==v).sum()-1) for v in x],dtype=float)


def corr(a,b):
    a,b=rank(a),rank(b)
    if min(a.std(),b.std())==0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def selection_lift(scores,y):
    n=math.ceil(len(y)/2)
    indices=np.argsort(-scores,kind='stable')[:n]
    return float(y[indices].mean()-y.mean())


def prepare(root):
    raw=[json.loads(x) for x in (root/'scored.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for r in raw:
        groups[(r['prompt_id'],r['state_index'])].append(r)
    splits=defaultdict(list)
    excluded=defaultdict(int)
    for pair in groups.values():
        assert len(pair)==2 and {r['seed'] for r in pair}=={1121,2213}
        r=pair[0]
        assert pair[0]['features']==pair[1]['features']
        if not all('gain' in item for item in pair):
            excluded[r['split']]+=1
            continue
        splits[r['split']].append(dict(prompt_id=r['prompt_id'],article_id=r['article_id'],
            state_index=r['state_index'],features=r['features'],gain=float(np.mean([x['gain'] for x in pair])),
            branch_gains=[x['gain'] for x in pair]))
    names=json.loads((root/'environment.json').read_text())['feature_names']
    for split in ('train','dev','test'):
        assert len(splits[split])>=16,f'Insufficient usable states: {split}'
    articles={s:{r['article_id'] for r in rows} for s,rows in splits.items()}
    assert not articles['train']&articles['dev'] and not (articles['train']|articles['dev'])&articles['test']
    return splits,names,dict(excluded),raw


def array(rows,names):
    return np.array([[r['features'][n] for n in names] for r in rows]),np.array([r['gain'] for r in rows])


def fit(x,y,alpha):
    mean,std=x.mean(0),x.std(0)
    std=np.where(std<1e-8,1.,std)
    z=(x-mean)/std
    beta=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-y.mean()))
    return dict(mean=mean.tolist(),std=std.tolist(),coef=beta.tolist(),intercept=float(y.mean()),alpha=alpha)


def predict(model,x):
    return (x-np.array(model['mean']))/np.array(model['std'])@np.array(model['coef'])+model['intercept']


def metrics(scores,y):
    return dict(spearman=corr(scores,y),mse=float(np.mean((scores-y)**2)),
        selected_half_gain_lift=selection_lift(scores,y))


def evaluate(rows,x,y,model,names,bootstrap=2000):
    predictions={'ridge':predict(model,x),'negative_mean_margin':-x[:,names.index('margin_mean')],
                 'positive_mean_margin':x[:,names.index('margin_mean')],'oracle':y.copy()}
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[r['article_id']].append(i)
    keys=list(groups)
    rng=np.random.default_rng(987231)
    intervals={k:[] for k in predictions}
    paired={k:[] for k in ('negative_mean_margin','positive_mean_margin')}
    for _ in range(bootstrap):
        idx=np.concatenate([groups[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        lifts={name:selection_lift(s[idx],y[idx]) for name,s in predictions.items()}
        for name,v in lifts.items():
            intervals[name].append(v)
        for name in paired:
            paired[name].append(lifts['ridge']-lifts[name])
    result={name:dict(spearman=corr(s,y),selected_half_gain_lift=selection_lift(s,y),
        lift_95_article_bootstrap=np.quantile(intervals[name],[.025,.975]).tolist()) for name,s in predictions.items()}
    result['ridge']['mse']=float(np.mean((predictions['ridge']-y)**2))
    return dict(states=len(rows),prompts=len({r['prompt_id'] for r in rows}),articles=len(keys),
        mean_short_gain=float(y.mean()),positive_gain_fraction=float(np.mean(y>0)),
        mean_predictor_mse=float(np.mean((model['intercept']-y)**2)),methods=result,
        ridge_minus_baseline_lift={name:dict(point=result['ridge']['selected_half_gain_lift']-result[name]['selected_half_gain_lift'],
            interval95=np.quantile(v,[.025,.975]).tolist()) for name,v in paired.items()}),predictions


def main():
    root=Path(sys.argv[1])
    if (root/'locked_model.json').exists():
        raise RuntimeError('Locked model exists; do not refit on already examined test data')
    splits,names,excluded,raw=prepare(root)
    x,y=array(splits['train'],names)
    xd,yd=array(splits['dev'],names)
    candidates=[]
    models=[]
    for alpha in (.1,1.,10.,100.):
        model=fit(x,y,alpha)
        models.append(model)
        candidates.append(dict(alpha=alpha,dev=metrics(predict(model,xd),yd)))
    best=min(range(len(candidates)),key=lambda i:candidates[i]['dev']['mse'])
    model=models[best]
    model.update(features=names,selection='minimum development MSE; train-only fit',
        candidates=candidates,data_sha256=hashlib.sha256((root/'scored.jsonl').read_bytes()).hexdigest())
    # Persist the selected model before computing any held-out performance.
    dump(root/'locked_model.json',model)
    xt,yt=array(splits['test'],names)
    evaluation,predictions=evaluate(splits['test'],xt,yt,model,names)
    evaluation.update(sample_counts={s:dict(states=len(rows),prompts=len({r['prompt_id'] for r in rows}),
        articles=len({r['article_id'] for r in rows})) for s,rows in splits.items()},
        excluded_states=excluded,total_branches=len(raw),eos_branches=sum(r['eos_excluded'] for r in raw),
        first_two_mismatches=sum(not r['first_two_match'] for r in raw),chosen_alpha=model['alpha'],
        scope='Offline top-half ranking on EOS-filtered fixed4 states; no online quality or speed claim.')
    dump(root/'benefit_analysis.json',evaluation)
    dump(root/'heldout_predictions.json',[dict(**r,scores={name:float(v[i]) for name,v in predictions.items()})
                                          for i,r in enumerate(splits['test'])])
    print(json.dumps(evaluation,indent=2),flush=True)


if __name__=='__main__':
    main()
