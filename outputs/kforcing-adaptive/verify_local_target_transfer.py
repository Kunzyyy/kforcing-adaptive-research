"""Independent least-squares and bootstrap audit for the fixed target-transfer follow-up."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/local-target-transfer'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,atol=1e-11):np.testing.assert_allclose(a,b,rtol=0,atol=atol)

def main(output=None):
    m=read(OUT/'manifest.json');a=read(OUT/'analysis.json')
    expected={'results/features-014/development_data.jsonl','results/decision-features/features.jsonl',
        'results/local-mechanism/local_scores.jsonl','results/attention-features/oof_predictions.jsonl',
        'results/local-mechanism/analysis.json','results/local-mechanism/integrity_audit.json',
        'LOCAL_TARGET_TRANSFER_PLAN.md','transfer_local_target.py'}
    assert set(m['inputs'])==expected
    for name,digest in m['inputs'].items():assert sha(ROOT/name)==digest,name
    for name,digest in a['outputs'].items():assert sha(OUT/name)==digest,name
    seeds=[14112001,14112002,14112003]
    assert m['alpha']==1000 and m['feature_dimensions']==19 and m['fold_seeds']==seeds and m['bootstrap_seed']==22242002
    assert m['one_candidate'] and m['retrospective_development'] and not m['old_test_inputs']
    diagnostic=read(ROOT/'results/local-mechanism/analysis.json')
    assert diagnostic['local_target_followup_signal'] and diagnostic['new_model_fit'] is False
    data=rows(ROOT/'results/features-014/development_data.jsonl');features=rows(ROOT/'results/decision-features/features.jsonl')
    local=rows(ROOT/'results/local-mechanism/local_scores.jsonl');old=rows(ROOT/'results/attention-features/oof_predictions.jsonl')
    oof=rows(OUT/'oof_predictions.jsonl');fits=read(OUT/'fits.json')
    ids=[r['state_id'] for r in data];assert len(set(ids))==446 and ids==sorted(ids)
    for records in [features,local,old,oof]:
        assert [r['state_id'] for r in records]==ids
        assert [r['prompt_id'] for r in records]==[r['prompt_id'] for r in data]
    assert {r['source_split'] for r in data}=={'train','dev'}
    x=np.array([r['features'] for r in features]);assert x.shape==(446,19)
    g=np.array([r['gain'] for r in data]);close(g,[np.mean(r['gains']) for r in data])
    targets={'full_target19':g,'local_target19':np.array([r['local_gain'] for r in local])}
    lookup={(r['repeat'],r['fold'],r['method']):r for r in fits};assert len(lookup)==len(fits)==30
    predictions={name:np.full((3,446),np.nan) for name in targets};choices={name:np.zeros((3,446)) for name in targets}
    constants={name:np.zeros((3,446)) for name in targets};random=np.zeros((3,446));heuristic=np.zeros((3,446))
    sources=sorted({r['prompt_id'] for r in data});assert len(sources)==253
    def choose(scores,valid):
        order=sorted(range(len(valid)),key=lambda j:(-scores[j],ids[valid[j]]));v=np.zeros(len(valid))
        v[order[:len(valid)//2]]=1;return v
    max_prediction_error=0.
    for repeat,seed in enumerate(seeds):
        shuffled=np.random.default_rng(seed).permutation(sources).tolist()
        for fold in range(5):
            validation_sources=set(shuffled[fold::5])
            valid=np.array([i for i,r in enumerate(data) if r['prompt_id'] in validation_sources])
            train=np.array([i for i,r in enumerate(data) if r['prompt_id'] not in validation_sources])
            assert not {data[i]['prompt_id'] for i in train}&{data[i]['prompt_id'] for i in valid}
            random[repeat,valid]=(len(valid)//2)/len(valid)
            heuristic[repeat,valid]=choose(-(x[valid,2]+x[valid,3])/2,valid)
            mean=x[train].mean(axis=0);std=x[train].std(axis=0);std[std<1e-8]=1
            z=(x[train]-mean)/std;v=(x[valid]-mean)/std
            augmented=np.concatenate([z,np.sqrt(1000)*np.eye(19)],axis=0)
            for name,y in targets.items():
                r=lookup[repeat,fold,name]
                assert r['train']==train.tolist() and r['valid']==valid.tolist()
                intercept=float(y[train].mean());target=np.concatenate([y[train]-intercept,np.zeros(19)])
                coef=np.linalg.lstsq(augmented,target,rcond=None)[0];p=v@coef+intercept;selected=choose(p,valid)
                for actual,saved in [(mean,r['mean']),(std,r['std']),(intercept,r['intercept']),(coef,r['coef']),(p,r['predictions'])]:close(actual,saved)
                close(selected,r['selected'],atol=0)
                max_prediction_error=max(max_prediction_error,float(np.max(abs(p-r['predictions']))))
                predictions[name][repeat,valid]=p;choices[name][repeat,valid]=selected;constants[name][repeat,valid]=intercept
    for name in targets:assert np.isfinite(predictions[name]).all()
    close(predictions['full_target19'].T,[r['predictions']['combined19'] for r in old])
    close(choices['full_target19'].T,[r['selected']['combined19'] for r in old],atol=0)
    for i,r in enumerate(oof):
        close(r['full_gain'],g[i]);close(r['local_gain'],targets['local_target19'][i])
        close(r['fraction'],random[:,i]);close(r['heuristic'],heuristic[:,i],atol=0)
        for name in targets:
            close(r['predictions'][name],predictions[name][:,i]);close(r['selected'][name],choices[name][:,i],atol=0)
    source_index={s:i for i,s in enumerate(sources)};membership=np.array([source_index[r['prompt_id']] for r in data])
    source_sizes=np.bincount(membership,minlength=253);rng=np.random.default_rng(22242002)
    weights=np.array([np.bincount(rng.integers(253,size=253),minlength=253) for _ in range(4000)])
    denominator=weights@source_sizes;checked=0
    def interval(value,record,coverage):
        nonlocal checked
        totals=np.bincount(membership,weights=value,minlength=253);boot=weights@totals/denominator
        tail=(1-coverage)/2;bounds=np.quantile(boot,[tail,1-tail])
        assert record['coverage']==coverage
        close(value.mean(),record['estimate']);close(bounds,record['interval']);checked+=1
        return bounds
    for name,y in targets.items():
        r=a['methods'][name]
        close(r['own_target_oof_mse'],np.mean((predictions[name]-y)**2))
        close(r['own_target_fold_mean_mse'],np.mean((constants[name]-y)**2))
        interval(np.mean((choices[name]-random)*g,axis=0),r['gain_vs_random'],.95)
    bounds=[]
    for key,control in [('vs_full_target19',choices['full_target19']),('vs_heuristic',heuristic)]:
        bounds.append(interval(np.mean((choices['local_target19']-control)*g,axis=0),a['primary'][key],.975))
    gate=all(b[0]>0 for b in bounds);assert a['development_screen_passed']==gate
    assert a['states']==446 and a['sources']==253 and a['fits']==30
    assert a['prior19_predictions_reproduced'] and a['one_candidate'] and a['retrospective_development']
    for key in ['independent_confirmation','old_test_inputs','new_online_timing','new_human_quality_result']:assert a[key] is False
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'),analysis_sha256=sha(OUT/'analysis.json'),
        states=446,sources=253,augmented_least_squares_fits=30,oof_predictions_recomputed=2676,
        max_prediction_error=max_prediction_error,intervals_recomputed=checked,prior19_predictions_reproduced=True,
        grouped_source_separation_verified=True,training_fold_only_targets_verified=True,
        development_screen_passed=gate,old_test_inputs=False,
        scope='Independent numerical audit of the single local-target19 candidate; reused development data, no independent confirmation.')
    if output is not None:output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);main(p.parse_args().output)
