"""One fixed attention-feature hypothesis on existing train/dev data only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/attention-features'
DATA='results/features-014/development_data.jsonl';PRIOR='results/decision-features/features.jsonl'
SEEDS=[14112001,14112002,14112003];GROUPS={'confidence14':14,'combined19':19,'attention25':25}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    assert not p.exists(),f'Preserve {p}'
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def lines(p,x):
    assert not p.exists()
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in x),encoding='utf-8')

def freeze():
    assert not OUT.exists();OUT.mkdir()
    names=[DATA,PRIOR,'results/decision-features/analysis.json','results/decision-features/manifest.json',
        'results/averaged-013/train_states.jsonl','results/averaged-013/dev_states.jsonl',
        'ATTENTION_FEATURE_PLAN.md','attention_features.py','diagnose_attention_features.py',
        'core.py','current_features.py','upstream/models/pflm.py','upstream/models/transformer.py']
    previous=read(ROOT/'results/decision-features/manifest.json')
    for name,h in previous['input_whitelist'].items():
        if name in names:assert sha(ROOT/name)==h
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        inputs={name:sha(ROOT/name) for name in names},checkpoint_sha256='3b013f0cf9a3acab4c20a1bb748beee6525432ef3f56a74df4957b898480a2c2',
        states=446,sources=253,alpha=1000,fold_seeds=SEEDS,bootstrap_seed=22232001,
        retrospective_development=True,old_test_inputs=False,one_candidate=True))
    print('Frozen one candidate and train/dev-only inputs.',flush=True)

def verify():
    m=read(OUT/'manifest.json')
    for name,h in m['inputs'].items():assert sha(ROOT/name)==h,name
    return m

def collect():
    import torch
    from core import load_model
    from current_features import current_features_tensor
    from attention_features import from_qkv,NAMES
    manifest=verify();assert not (OUT/'features.jsonl').exists()
    checkpoint=ROOT.parents[1]/'work/checkpoints/pflm_lm1b_k4.ckpt';assert sha(checkpoint)==manifest['checkpoint_sha256']
    data=rows(ROOT/DATA);prior=rows(ROOT/PRIOR)
    assert len(data)==len(prior)==446 and [r['state_id'] for r in data]==[r['state_id'] for r in prior]
    # Only current history/noise/initial candidates may enter model extraction.
    keep=['state_id','prompt_id','offset','history_ids','noise','initial_candidates']
    current={r['state_id']:{k:r[k] for k in keep} for split in ['train','dev']
             for r in rows(ROOT/f'results/averaged-013/{split}_states.jsonl') if r['status']=='paired'}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(checkpoint,'pflm','cuda');block=model.blocks[-1]
    selected=set(np.linspace(0,len(data)-1,12,dtype=int).tolist());probes={};attentions={};features=[]
    max_delta=0.;max_output_error=0.;hook_checks=0
    with torch.inference_mode():
        for i,(d,old) in enumerate(zip(data,prior)):
            s=current[d['state_id']];h=len(s['history_ids']);assert h==6+s['offset']
            history=torch.tensor([s['history_ids']],device='cuda');noise=torch.tensor(s['noise'],device='cuda').reshape(1,4,1);tau=torch.ones(1,1,1,device='cuda')
            base=model(history,noise,tau,mode='inference') if i<12 else None
            captured={}
            def qkv_hook(_module,_input,output):captured['qkv']=output.detach()
            def attn_input(_module,inputs):captured['actual']=inputs[0].detach()
            a=block.attn_qkv.register_forward_hook(qkv_hook);b=block.attn_out.register_forward_pre_hook(attn_input)
            try:logits=model(history,noise,tau,mode='inference')
            finally:a.remove();b.remove()
            if base is not None:assert torch.equal(base,logits);hook_checks+=1
            assert logits.argmax(-1)[0].tolist()==s['initial_candidates']==old['initial_candidates']
            conf=current_features_tensor(logits).cpu().numpy();np.testing.assert_allclose(conf,old['features'][:14],atol=.002,rtol=.002)
            extras=[s['offset']/8]+(logits[0,2:,102]-logits[0,2:].max(-1).values).cpu().tolist()+[int(v==102) for v in s['initial_candidates'][2:]]
            np.testing.assert_allclose(extras,old['features'][14:19],atol=1e-5,rtol=0)
            six,attention,output=from_qkv(captured['qkv'],model.rotary_emb.inv_freq,block.n_heads,h)
            delta=float((output-captured['actual'][0,-2:]).abs().max());max_output_error=max(delta,max_output_error)
            torch.testing.assert_close(output,captured['actual'][0,-2:],atol=2e-4,rtol=2e-4)
            vector=six.cpu().double().numpy();assert np.isfinite(vector).all() and (vector>=0).all() and (vector<=1.00001).all()
            features.append(dict(state_id=s['state_id'],prompt_id=s['prompt_id'],offset=s['offset'],source_split=d['source_split'],history_length=h,
                features=old['features']+vector.tolist(),last_block=len(model.blocks)-1,heads=block.n_heads))
            attentions[s['state_id']]=attention.cpu().numpy()
            if i in selected:
                probes[f'{i}_qkv']=captured['qkv'].cpu().numpy();probes[f'{i}_actual']=captured['actual'][0,-2:].cpu().numpy()
            max_delta=max(max_delta,float(np.max(np.abs(conf-np.array(old['features'][:14])))))
            if (i+1)%100==0:print('Current-state feature replay',i+1,'/446',flush=True)
    probes['indices']=np.array(sorted(selected));probes['inv_freq']=model.rotary_emb.inv_freq.cpu().numpy()
    lines(OUT/'features.jsonl',features);np.savez_compressed(OUT/'attention_probabilities.npz',**attentions);np.savez_compressed(OUT/'numeric_probes.npz',**probes)
    dump(OUT/'collection.json',dict(passed=True,states=446,candidate_positions=1784,unchanged_hook_logit_cases=hook_checks,
        attention_sdpa_comparisons=446,max_attention_output_absolute_error=max_output_error,max_confidence_difference=max_delta,
        independent_probe_indices=sorted(selected),feature_names=NAMES,last_block=len(model.blocks)-1,heads=block.n_heads,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),new_future_generations=0,
        outputs={n:sha(OUT/n) for n in ['features.jsonl','attention_probabilities.npz','numeric_probes.npz']}))
    print('Feature extraction complete; no future branches generated.',flush=True)

def half(p):
    selected=np.zeros(len(p));selected[np.argsort(-p,kind='stable')[:len(p)//2]]=1;return selected

def evaluate():
    verify();collection=read(OUT/'collection.json')
    for name,h in collection['outputs'].items():assert sha(OUT/name)==h
    data=rows(ROOT/DATA);f=rows(OUT/'features.jsonl');x=np.array([r['features'] for r in f]);y=np.array([r['gain'] for r in data]);n=len(data)
    assert [r['state_id'] for r in f]==[r['state_id'] for r in data]==sorted(r['state_id'] for r in data)
    sources=sorted({r['prompt_id'] for r in data});pred={k:np.zeros((3,n)) for k in GROUPS};chosen={k:np.zeros((3,n)) for k in GROUPS}
    fractions=np.zeros((3,n));heuristic=np.zeros((3,n));fits=[]
    for repeat,seed in enumerate(SEEDS):
        assignment={s:i%5 for i,s in enumerate(np.random.default_rng(seed).permutation(sources).tolist())}
        for fold in range(5):
            valid=np.array([i for i,r in enumerate(data) if assignment[r['prompt_id']]==fold]);train=np.setdiff1d(np.arange(n),valid)
            assert not {data[i]['prompt_id'] for i in train}&{data[i]['prompt_id'] for i in valid}
            fractions[repeat,valid]=(len(valid)//2)/len(valid);heuristic[repeat,valid]=half(-x[valid,2:4].mean(1))
            for name,d in GROUPS.items():
                a=x[train,:d];mean=a.mean(0);std=a.std(0);std[std<1e-8]=1.;z=(a-mean)/std;intercept=float(y[train].mean())
                coef=np.linalg.solve(z.T@z+1000*np.eye(d),z.T@(y[train]-intercept));p=(x[valid,:d]-mean)/std@coef+intercept
                pred[name][repeat,valid]=p;chosen[name][repeat,valid]=half(p)
                fits.append(dict(repeat=repeat,fold=fold,method=name,train=train.tolist(),valid=valid.tolist(),mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),intercept=intercept,
                    predictions=p.tolist(),selected=half(p).astype(int).tolist(),train_mse=float(np.mean((z@coef+intercept-y[train])**2))))
    members=[[i for i,r in enumerate(data) if r['prompt_id']==s] for s in sources];counts=np.array([len(g) for g in members])
    draws=np.random.default_rng(22232001).integers(0,len(sources),(4000,len(sources)))
    def interval(values,coverage=.95):
        sums=np.array([values[g].sum() for g in members]);b=sums[draws].sum(1)/counts[draws].sum(1);tail=(1-coverage)/2
        return dict(estimate=float(values.mean()),interval=np.quantile(b,[tail,1-tail]).tolist(),coverage=coverage)
    methods={}
    for name in GROUPS:
        methods[name]=dict(dimensions=GROUPS[name],oof_mse=float(np.mean((pred[name]-y[None])**2)),
            train_mse=float(np.average([v['train_mse'] for v in fits if v['method']==name],weights=[len(v['train']) for v in fits if v['method']==name])),
            gain_vs_random=interval(((chosen[name]-fractions)*y[None]).mean(0)),
            gain_vs_confidence14=interval(((chosen[name]-chosen['confidence14'])*y[None]).mean(0)),
            gain_vs_combined19=interval(((chosen[name]-chosen['combined19'])*y[None]).mean(0)),
            gain_vs_heuristic=interval(((chosen[name]-heuristic)*y[None]).mean(0)))
    primary={k:interval(((chosen['attention25']-control)*y[None]).mean(0),.975) for k,control in [('vs_combined19',chosen['combined19']),('vs_heuristic',heuristic)]}
    old=read(ROOT/'results/decision-features/analysis.json')['methods']
    for name in ['confidence14','combined19']:
        assert abs(methods[name]['oof_mse']-old[name]['oof_mse'])<1e-12
        assert abs(methods[name]['gain_vs_random']['estimate']-old[name]['gain_vs_random']['estimate'])<1e-12
    gate=all(v['interval'][0]>0 for v in primary.values()) and methods['attention25']['oof_mse']<methods['combined19']['oof_mse']
    dump(OUT/'fits.json',fits)
    lines(OUT/'oof_predictions.jsonl',[dict(state_id=r['state_id'],prompt_id=r['prompt_id'],gain=float(y[i]),
        predictions={k:pred[k][:,i].tolist() for k in GROUPS},selected={k:chosen[k][:,i].astype(int).tolist() for k in GROUPS},
        heuristic=heuristic[:,i].astype(int).tolist(),fraction=fractions[:,i].tolist()) for i,r in enumerate(data)])
    report=dict(states=n,sources=len(sources),fits=len(fits),methods=methods,primary=primary,
        development_screen_passed=bool(gate),old_controls_reproduced=True,retrospective_development=True,independent_test=False,
        old_test_inputs=False,new_human_quality_result=False,new_online_timing=False,
        feature_ranges=[dict(min=float(x[:,j].min()),max=float(x[:,j].max()),mean=float(x[:,j].mean()),std=float(x[:,j].std())) for j in range(19,25)],
        output_hashes={name:sha(OUT/name) for name in ['features.jsonl','fits.json','oof_predictions.jsonl']})
    dump(OUT/'analysis.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','collect','evaluate']);args=parser.parse_args();globals()[args.action]()
