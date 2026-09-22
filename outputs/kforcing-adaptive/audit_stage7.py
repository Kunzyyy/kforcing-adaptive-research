"""CPU-only independent data-isolation, frozen-model and prediction audit."""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b):
    assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10),(a,b)


def main():
    root=Path(sys.argv[1])
    manifest=read(root/'data_manifest.json')
    prefixes=read(root/'prefixes.json')
    flat=sum(prefixes.values(),[])
    assert len(flat)==448 and len({p['id'] for p in flat})==448
    assert len({tuple(p['ids']) for p in flat})==448
    assert len({p['source_sha256'] for p in flat})==448
    assert len({p['source_row'] for p in flat})==448
    old=read(root.parent/'lm1b-004/prefixes.json')
    assert not {p['source_row'] for p in old}&{p['source_row'] for p in flat}
    assert not {p['source_sha256'] for p in old}&{p['source_sha256'] for p in flat}
    old_prefixes={tuple(p['ids']) for p in old}
    for folder in ('pilot-001','benefit-002'):
        for group in read(root.parent/folder/'prefixes.json').values():
            old_prefixes.update(tuple(p['ids']) for p in group)
    old_prefixes.update(tuple(p['ids']) for p in read(root.parent/'current-003/fresh_prefixes.json'))
    assert not old_prefixes&{tuple(p['ids']) for p in flat}
    assert sha(root/'prefixes.json')==manifest['prefix_sha256']
    assert sha(root/'projection.npy')==manifest['projection_sha256']
    projection=np.load(root/'projection.npy',allow_pickle=False)
    assert projection.shape==(768,64) and projection.dtype==np.float32 and np.isfinite(projection).all()
    all_rows={}
    for split,count in (('train',256),('dev',64),('test',128)):
        assert len(prefixes[split])==count
        pmap={p['id']:p for p in prefixes[split]}
        data=rows(root/(split+'.jsonl'))
        all_rows[split]=data
        expected=set(itertools.product(pmap,manifest['seeds']))
        assert len(data)==len(expected)==count*8
        assert {(r['prompt_id'],r['seed']) for r in data}==expected
        for r in data:
            assert r['prefix']==pmap[r['prompt_id']]['ids']
            assert len(r['prefix'])==6 and r['prefix'][0]==101
            assert r['noise_seed']==r['seed']+pmap[r['prompt_id']]['index']*100003
            assert len(r['features'])==82 and all(math.isfinite(x) for x in r['features'])
            assert r['features'][-4:]==r['noise'] and all(0<=x<1 for x in r['noise'])
            for key in ('fixed4','split22','ar_reference'):
                assert len(r[key])==4 and all(isinstance(x,int) and 0<=x<30522 for x in r[key])
            assert r['gain']==sum(a==b for a,b in zip(r['split22'],r['ar_reference']))-sum(a==b for a,b in zip(r['fixed4'],r['ar_reference']))
            assert r['no_eos_all_branches']==all(102 not in r[k] for k in ('fixed4','split22','ar_reference'))
        checks=read(root/(split+'_checks.json'))
        assert checks['rows']==len(data) and checks['batched_confidence_features_match_old']
        assert checks['first_two_mismatches']==sum(sum(x!=y for x,y in zip(r['fixed4'][:2],r['split22'][:2])) for r in data)
    model=read(root/'locked_model.json')
    assert model['train_sha256']==sha(root/'train.jsonl') and model['dev_sha256']==sha(root/'dev.jsonl')
    assert model['prefix_sha256']==manifest['prefix_sha256'] and model['projection_sha256']==manifest['projection_sha256']
    assert read(root/'test_checks.json')['locked_model_sha256']==sha(root/'locked_model.json')
    candidates=read(root/'dev_candidates.json')
    assert len(candidates)==6
    assert {(r['features'],r['alpha']) for r in candidates}==set(itertools.product(('confidence14','hidden82'),(1.,10.,100.)))
    winner=sorted(candidates,key=lambda m:(-m['dev_improvement']['mean'],m['name']))[0]
    assert winner['name']==model['name']
    n=14 if model['features']=='confidence14' else 82
    x=np.array([r['features'][:n] for r in all_rows['train']])
    y=np.array([r['gain'] for r in all_rows['train']])
    mean,std=x.mean(0),x.std(0)
    std[std<1e-8]=1
    np.testing.assert_allclose(mean,model['mean'],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(std,model['std'],atol=1e-10,rtol=1e-10)
    z=(x-mean)/std
    coef=np.array(model['coef'])
    # Verify ridge normal equations rather than call the fitting implementation.
    np.testing.assert_allclose((z.T@z+model['alpha']*np.eye(n))@coef,z.T@(y-y.mean()),atol=1e-7,rtol=1e-8)
    close(model['intercept'],y.mean())
    test=all_rows['test']
    predictions=rows(root/'test_predictions.jsonl')
    assert len(predictions)==len(test)==1024
    scores=((np.array([r['features'][:n] for r in test])-mean)/std)@coef+model['intercept']
    for i,(r,p) in enumerate(zip(test,predictions)):
        assert (r['prompt_id'],r['seed'],r['gain'])==(p['prompt_id'],p['seed'],p['gain'])
        close(scores[i],p['score'])
        assert p['locked_threshold']==(scores[i]>model['dev_threshold'])
    analysis=read(root/'test_analysis.json')
    gains=np.array([r['gain'] for r in test])
    base=np.array([sum(a==b for a,b in zip(r['fixed4'],r['ar_reference'])) for r in test])
    close(analysis['fixed4_match_fraction'],base.mean()/4)
    close(analysis['all_split_match_fraction'],(base+gains).mean()/4)
    close(analysis['model_mse'],np.mean((scores-gains)**2))
    close(analysis['constant_train_mean_mse'],np.mean((model['intercept']-gains)**2))
    for policy,s in analysis['policies'].items():
        mask=np.array([p[policy] for p in predictions])
        assert s['split_windows']==int(mask.sum())
        close(s['split_fraction'],mask.mean())
        if policy.endswith('_half'):
            assert mask.sum()==512
        close(s['teacher_token_match_fraction'],np.mean(base+mask*gains)/4)
        close(s['improvement_vs_matched_random_expectation']['mean'],np.mean((mask.astype(float)-mask.mean())*gains)/4)
    # Reconstruct deterministic score order, including documented tie breaking.
    for policy,values in (('learned_half',scores),('confidence_half',np.array([-np.mean(r['features'][2:4]) for r in test])),('oracle_half',gains)):
        order=sorted(range(len(test)),key=lambda i:(-float(values[i]),test[i]['prompt_id'],test[i]['seed']))
        expected=set(order[:512])
        assert {i for i,p in enumerate(predictions) if p[policy]}==expected
    assert analysis['online_experiment_gate_passed']==(model['dev_gate_passed'] and analysis['policies']['learned_half']['improvement_vs_matched_random_expectation']['interval95'][0]>0)
    report=dict(passed=True,unique_prefixes=448,train_windows=2048,dev_windows=512,test_windows=1024,
        prior_prefix_overlap=0,split_prefix_overlap=0,split_sentence_hash_overlap=0,
        train_only_normalization_and_ridge_equations_verified=True,frozen_model_hash_verified=True,
        prediction_and_aggregate_recomputed=True,online_gate=analysis['online_experiment_gate_passed'],
        limitation='CPU record audit, not independent model inference or bootstrap reconstruction. Original document-level isolation is unavailable.')
    (root/'integrity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
