"""CPU-only statistical verification of the research handoff; NumPy is the only dependency.

This checks recorded outputs/scores/timings. It does not regenerate model outputs,
rerun GPT-2, validate the original GPU environment, or establish human quality.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import numpy as np

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(a,b): np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)

def safe_file(root,name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name:
        raise ValueError('Unsafe payload path: '+name)
    path=root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Path outside evidence root')
    return path

def verify_payload(root):
    manifest=read(root/'PAYLOAD_SHA256.json')
    for name,digest in manifest.items():
        if sha(safe_file(root,name))!=digest: raise ValueError('SHA256 mismatch: '+name)
    for claim in read(root/'research_claims.json')['claims']:
        for ref in claim['evidence']:
            obj=read(safe_file(root,ref['file']))
            for part in ref['pointer'].strip('/').split('/'):
                obj=obj[part.replace('~1','/').replace('~0','~')]
            if obj!=ref['value']: raise ValueError('Claim reference mismatch: '+claim['id'])
    return len(manifest)

def baseline(root):
    directory=root/'results/baseline-016'; expected=read(directory/'analysis.json')
    assert sha(directory/'scores.jsonl')==expected['scores_sha256']
    data=[r for r in rows(directory/'scores.jsonl') if r['cohort']=='new']
    ids=[p['prompt_id'] for p in read(directory/'prefixes.json')]
    assert len(ids)==len(set(ids))==1024 and len(data)==2048
    draws=np.random.default_rng(16112004).integers(0,1024,(4000,1024));boot={};result={}
    for method in ['ar','fixed4']:
        source={r['prompt_id']:r for r in data if r['method']==method};assert set(source)==set(ids)
        loss=np.array([source[i]['gpt2_nll_sum'] for i in ids]);count=np.array([source[i]['gpt2_scored_tokens'] for i in ids])
        for r in source.values(): assert r['excluded_short']==(r['gpt2_scored_tokens']==0)
        mean=float(loss.sum()/count.sum());resamples=loss[draws].sum(1)/count[draws].sum(1);boot[method]=resamples
        result[method]=dict(gen_ppl=math.exp(mean),interval95=np.exp(np.quantile(resamples,[.025,.975])).tolist())
        recorded=expected['methods'][method]
        check(mean,recorded['nll']);check(result[method]['gen_ppl'],recorded['gen_ppl'])
        check(result[method]['interval95'],recorded['gen_ppl_interval95'])
        assert count.sum()==recorded['scored_tokens'] and int((count==0).sum())==recorded['excluded_short']
    check(expected['fixed4_over_ar_ppl_ratio']['estimate'],result['fixed4']['gen_ppl']/result['ar']['gen_ppl'])
    check(expected['fixed4_over_ar_ppl_ratio']['interval95'],np.exp(np.quantile(boot['fixed4']-boot['ar'],[.025,.975])))
    return dict(passed=True,sources=1024,new_score_records=2048,methods=result,scope='recorded score arithmetic only; paper configuration not aligned')

def online(root):
    directory=root/'results/early-010';expected=read(directory/'analysis.json');lock=read(directory/'locked_calibration.json')
    assert sha(directory/'calibration.jsonl')==lock['calibration_sha256'] and lock['quality_labels_used'] is False
    calibration=rows(directory/'calibration.jsonl');totals=defaultdict(lambda:[0,0])
    for r in calibration:
        totals[r['method']][0]+=r['forward_calls'];totals[r['method']][1]+=r['visible_new_tokens']
    reference=totals['learned'][0]/totals['learned'][1]
    best=min((m for m in totals if m.startswith('random_')),key=lambda m:abs(math.log((totals[m][0]/totals[m][1])/reference)))
    assert best==lock['chosen']['method']=='random_06' and lock['probabilities']['random_cal']==.6
    samples=rows(directory/'samples.jsonl');scores=rows(directory/'scores.jsonl');timings=rows(directory/'timings.jsonl')
    key=lambda r:(r['prompt_id'],r['seed'],r['method'])
    sample_map={key(r):r for r in samples};score_map={key(r):r for r in scores};times=defaultdict(dict)
    assert len(sample_map)==len(samples)==len(score_map)==len(scores)==1536
    assert set(sample_map)==set(score_map) and len(timings)==4608
    for r in timings:
        assert r['repetition'] not in times[key(r)] and r['seconds']>0
        times[key(r)][r['repetition']]=r['seconds']
    assert set(times)==set(sample_map) and all(set(t)=={0,1,2} for t in times.values())
    ids=sorted({r['prompt_id'] for r in samples});assert len(ids)==96
    assert {r['prompt_id'] for r in calibration}.isdisjoint(ids)
    methods=sorted({r['method'] for r in samples});arrays={};summary={}
    for method in methods:
        source=[]
        for ident in ids:
            batch=[r for r in samples if r['method']==method and r['prompt_id']==ident];assert len(batch)==2 and len({r['seed'] for r in batch})==2
            total=np.zeros(5)
            for r in batch:
                s=score_map[key(r)];duration=math.fsum(times[key(r)].values())/3
                assert s['visible_new_tokens']==r['visible_new_tokens']==len(r['ids'])-6
                assert 102 not in r['ids'][6:-1] and r['stopped_eos']==(r['ids'][-1]==102)
                assert r['forward_calls']==sum(v['forward_calls'] for v in r['steps'])
                total+=np.array([s['gpt2_nll_sum'],s['gpt2_scored_tokens'],duration,r['visible_new_tokens'],r['forward_calls']])
                if method=='learned':
                    replay=sample_map[(ident,r['seed'],'learned_replay')]
                    assert r['ids']==replay['ids'] and r['forward_calls']==replay['forward_calls']
            source.append(total)
        a=np.array(source);arrays[method]=a;total=a.sum(0)
        summary[method]=dict(gen_ppl=float(np.exp(total[0]/total[1])),visible_tokens_per_second=float(total[3]/total[2]))
        check(summary[method]['gen_ppl'],expected['summary'][method]['gpt2_gen_ppl'])
        check(summary[method]['visible_tokens_per_second'],expected['summary'][method]['visible_tokens_per_second'])
    draws=np.random.default_rng(10102004).integers(0,96,size=(4000,96))
    a,b=arrays['learned'],arrays['random_cal'];aa,bb=a.sum(0),b.sum(0);ab,bbt=a[draws].sum(1),b[draws].sum(1)
    point={'ppl_ratio':np.exp(aa[0]/aa[1]-bb[0]/bb[1]),'visible_throughput_ratio':(aa[3]/aa[2])/(bb[3]/bb[2])}
    boot={'ppl_ratio':np.exp(ab[:,0]/ab[:,1]-bbt[:,0]/bbt[:,1]),'visible_throughput_ratio':(ab[:,3]/ab[:,2])/(bbt[:,3]/bbt[:,2])}
    primary={}
    for field in point:
        bounds=np.quantile(boot[field],[.0125,.9875]);check(point[field],expected['primary'][field]['estimate']);check(bounds,expected['primary'][field]['interval'])
        primary[field]=dict(estimate=float(point[field]),interval=bounds.tolist(),coverage=.975)
    ratio=(aa[4]/aa[3])/(bb[4]/bb[3]);check(ratio,expected['budget_calls_per_token_ratio'])
    gate=.95<=ratio<=1.05 and primary['ppl_ratio']['interval'][1]<1 and primary['visible_throughput_ratio']['interval'][0]>1
    assert bool(gate)==expected['primary_budget_and_performance_gate']==False
    return dict(passed=True,sources=96,timings=4608,primary=primary,advance_gate_passed=bool(gate),
                model='earlier hidden82, not the later combined19 model',recorded_timing_reaggregated=True,new_hardware_timing=False)

def offline(root):
    directory=root/'results/decision-confirm';expected=read(directory/'analysis.json');pred=read(directory/'frozen_predictions.json')
    models=read(directory/'locked_model.json')['models']
    assert sha(directory/'locked_model.json')==pred['locked_model_sha256']
    assert sha(directory/'test_states.jsonl')==pred['states_sha256']
    prediction_checks=0
    for state in rows(directory/'test_states.jsonl'):
        if state['status']!='paired': continue
        for name,model in models.items():
            dimension=len(model['mean']);features=np.array(state['decision_features'][:dimension])
            value=float((features-np.array(model['mean']))/np.array(model['std'])@np.array(model['coef'])+model['intercept'])
            check(value,pred['scores'][state['state_id']][name]);prediction_checks+=1
    grouped=defaultdict(list)
    for r in rows(directory/'test_labels.jsonl'):
        assert r['valid_label']; grouped[r['pair_id'].rsplit('-r',1)[0]].append(r)
    ids=sorted(grouped);assert len(ids)==681
    for r in grouped.values(): assert len(r)==8
    gain=np.array([math.fsum(r['gain'] for r in grouped[s])/8 for s in ids])
    sources=sorted({grouped[s][0]['prompt_id'] for s in ids});assert len(sources)==380
    membership=[[i for i,s in enumerate(ids) if grouped[s][0]['prompt_id']==p] for p in sources]
    counts=np.array([len(g) for g in membership]);draws=np.random.default_rng(21192004).integers(0,380,(8000,380))
    masks={}
    for name in ['combined19','confidence14','heuristic']:
        ranking=sorted(ids,key=lambda s:(-pred['scores'][s][name],s));assert ranking==pred['rankings'][name]
        selected=set(ranking[:340]);masks[name]=np.array([float(s in selected) for s in ids])
    primary={};tail=(.05/3)/2
    for name,base in [('vs_random',340/681),('vs_confidence14',masks['confidence14']),('vs_heuristic',masks['heuristic'])]:
        values=(masks['combined19']-base)*gain;sums=np.array([values[g].sum() for g in membership])
        boot=sums[draws].sum(1)/counts[draws].sum(1);bounds=np.quantile(boot,[tail,1-tail])
        check(values.mean(),expected['primary'][name]['estimate']);check(bounds,expected['primary'][name]['interval'])
        primary[name]=dict(estimate=float(values.mean()),interval=bounds.tolist(),coverage=1-.05/3)
    gate=len(sources)>=128 and all(v['interval'][0]>0 for v in primary.values())
    assert bool(gate)==expected['advance_gate_passed']==False
    return dict(passed=True,states=681,sources=380,primary=primary,advance_gate_passed=bool(gate),fixed_ranks=True,stored_model_predictions_recomputed=prediction_checks)

def cost(root):
    # Reuse the already audited verifier unchanged, on a temporary copy so evidence stays read-only.
    manifest=read(root/'results/compute-budget/manifest.json')
    names=set(manifest['input_hashes'])|{'verify_compute_budget.py','results/compute-budget/manifest.json'}
    names|={'results/compute-budget/'+name for name in manifest['outputs']}
    with tempfile.TemporaryDirectory(prefix='kforcing-evidence-') as temp:
        tmp=Path(temp)
        assert tmp.resolve().parent==Path(tempfile.gettempdir()).resolve() and tmp.name.startswith('kforcing-evidence-')
        for name in names:
            destination=tmp/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(safe_file(root,name),destination)
        run=subprocess.run([sys.executable,'-B',str(tmp/'verify_compute_budget.py')],cwd=tmp,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,check=True)
        audit=json.loads(run.stdout);assert audit['passed'] and audit['intervals_checked']==116
    return audit

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--hashes-only',action='store_true');args=parser.parse_args();root=args.root.resolve()
    count=verify_payload(root)
    if args.hashes_only:
        print(json.dumps(dict(passed=True,files_verified=count,scope='hashes and claim references only'),indent=2));return
    result=dict(passed=True,files_verified=count,baseline=baseline(root),online=online(root),offline_confirmation=offline(root),
        compute_budget=cost(root),human_quality='not_evaluated',new_model_generation=False,new_gpu_timing=False,
        verification_scope='record integrity and statistical reproduction; not a rerun of the neural models')
    assert verify_payload(root)==count
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__': main()
