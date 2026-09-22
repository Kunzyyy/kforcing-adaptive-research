"""One fixed ridge target-transfer follow-up, separate from the mechanism diagnostic."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/local-target-transfer'
DATA='results/features-014/development_data.jsonl'
FEATURES='results/decision-features/features.jsonl'
LOCAL='results/local-mechanism/local_scores.jsonl'
OOF='results/attention-features/oof_predictions.jsonl'
SEEDS=[14112001,14112002,14112003]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    assert not p.exists(),f'Preserve {p}'
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def half(pred):
    selected=np.zeros(len(pred));selected[np.argsort(-pred,kind='stable')[:len(pred)//2]]=1;return selected

def freeze():
    assert not OUT.exists();OUT.mkdir()
    audit=read(ROOT/'results/local-mechanism/integrity_audit.json')
    assert audit['passed'] and audit['local_target_followup_signal']
    assert audit['analysis_sha256']==sha(ROOT/'results/local-mechanism/analysis.json')
    names=[DATA,FEATURES,LOCAL,OOF,'results/local-mechanism/analysis.json','results/local-mechanism/integrity_audit.json',
        'LOCAL_TARGET_TRANSFER_PLAN.md','transfer_local_target.py']
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs={n:sha(ROOT/n) for n in names},
        states=446,sources=253,feature_dimensions=19,alpha=1000,fold_seeds=SEEDS,bootstrap_seed=22242002,
        one_candidate=True,retrospective_development=True,old_test_inputs=False,
        supervision='local AR teacher NLL gain in training fold only',evaluation='original full-continuation mean8 GPT-2 NLL gain'))
    print('Frozen one local-target19 candidate after the completed mechanism diagnostic.',flush=True)

def evaluate():
    m=read(OUT/'manifest.json')
    for n,h in m['inputs'].items():assert sha(ROOT/n)==h,n
    data=rows(ROOT/DATA);features=rows(ROOT/FEATURES);local=rows(ROOT/LOCAL);old=rows(ROOT/OOF)
    ids=[r['state_id'] for r in data]
    assert len(ids)==446 and ids==sorted(ids)
    for records in [features,local,old]:assert ids==[r['state_id'] for r in records]
    x=np.array([r['features'] for r in features]);g=np.array([r['gain'] for r in data]);n=len(data)
    targets={'full_target19':g,'local_target19':np.array([r['local_gain'] for r in local])}
    pred={k:np.zeros((3,n)) for k in targets};selection={k:np.zeros((3,n)) for k in targets};constant={k:np.zeros((3,n)) for k in targets}
    fraction=np.zeros((3,n));heuristic=np.zeros((3,n));fits=[]
    sources=sorted({r['prompt_id'] for r in data});members=[[i for i,r in enumerate(data) if r['prompt_id']==s] for s in sources]
    for repeat,seed in enumerate(SEEDS):
        assignment={s:i%5 for i,s in enumerate(np.random.default_rng(seed).permutation(sources).tolist())}
        for fold in range(5):
            valid=np.array([i for i,r in enumerate(data) if assignment[r['prompt_id']]==fold]);train=np.setdiff1d(np.arange(n),valid)
            mean=x[train].mean(0);std=x[train].std(0);std[std<1e-8]=1.;z=(x[train]-mean)/std;v=(x[valid]-mean)/std
            fraction[repeat,valid]=(len(valid)//2)/len(valid);heuristic[repeat,valid]=half(-x[valid,2:4].mean(1))
            for name,y in targets.items():
                intercept=float(y[train].mean());coef=np.linalg.solve(z.T@z+1000*np.eye(19),z.T@(y[train]-intercept));p=v@coef+intercept
                pred[name][repeat,valid]=p;selection[name][repeat,valid]=half(p);constant[name][repeat,valid]=intercept
                fits.append(dict(repeat=repeat,fold=fold,method=name,train=train.tolist(),valid=valid.tolist(),mean=mean.tolist(),std=std.tolist(),
                    coef=coef.tolist(),intercept=intercept,predictions=p.tolist(),selected=half(p).astype(int).tolist()))
    np.testing.assert_allclose(pred['full_target19'].T,[r['predictions']['combined19'] for r in old],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(selection['full_target19'].T,[r['selected']['combined19'] for r in old])
    counts=np.array([len(g) for g in members]);draws=np.random.default_rng(22242002).integers(0,len(sources),(4000,len(sources)))
    def interval(values,coverage=.95):
        sums=np.array([values[m].sum() for m in members]);sample=sums[draws].sum(1)/counts[draws].sum(1);tail=(1-coverage)/2
        return dict(estimate=float(values.mean()),interval=np.quantile(sample,[tail,1-tail]).tolist(),coverage=coverage)
    methods={}
    for name,y in targets.items():
        methods[name]=dict(own_target_oof_mse=float(np.mean((pred[name]-y)**2)),
            own_target_fold_mean_mse=float(np.mean((constant[name]-y)**2)),
            gain_vs_random=interval(((selection[name]-fraction)*g).mean(0)))
    primary={name:interval(((selection['local_target19']-control)*g).mean(0),.975)
        for name,control in [('vs_full_target19',selection['full_target19']),('vs_heuristic',heuristic)]}
    gate=all(r['interval'][0]>0 for r in primary.values())
    dump(OUT/'fits.json',fits)
    records=[dict(state_id=r['state_id'],prompt_id=r['prompt_id'],full_gain=float(g[i]),local_gain=float(targets['local_target19'][i]),
        predictions={k:pred[k][:,i].tolist() for k in targets},selected={k:selection[k][:,i].tolist() for k in targets},
        fraction=fraction[:,i].tolist(),heuristic=heuristic[:,i].tolist()) for i,r in enumerate(data)]
    path=OUT/'oof_predictions.jsonl';assert not path.exists();path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    report=dict(states=n,sources=len(sources),fits=30,methods=methods,primary=primary,
        development_screen_passed=bool(gate),prior19_predictions_reproduced=True,
        one_candidate=True,retrospective_development=True,independent_confirmation=False,old_test_inputs=False,
        new_online_timing=False,new_human_quality_result=False,
        outputs={n:sha(OUT/n) for n in ['fits.json','oof_predictions.jsonl']})
    dump(OUT/'analysis.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate']);args=p.parse_args();globals()[args.action]()
