"""Train/dev-only controlled full versus common-horizon reward training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/training-target'
SRC=ROOT/'results/averaged-013'
DEV=ROOT/'results/features-014/development_data.jsonl'
SEEDS=[14112001,14112002,14112003]

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def dump(p,v):
    assert not p.exists(),f'Preserve {p}'
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def lines(p,v):
    assert not p.exists(),f'Preserve {p}'
    with p.open('w',encoding='utf-8') as f:
        for r in v:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[DEV,ROOT/'results/features-014/analysis.json']
    for split in ('train','dev'):
        paths.extend(SRC/f'{split}_{suffix}' for suffix in ('states.jsonl','pairs.jsonl','labels.jsonl','scoring.json'))
    historical=read(ROOT/'decision_features_file_hashes.json')
    inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    for name,h in inputs.items():assert historical[name]==h
    data=rows(DEV)
    assert len(data)==446 and len({r['prompt_id'] for r in data})==253
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_whitelist=inputs,code_hashes={p:sha(ROOT/p) for p in
            ['TRAINING_TARGET_PLAN.md','compare_training_targets.py','diagnose_decision_features.py']},
        evaluator_manifest_sha256=sha(Path('work/gpt2-large/manifest.json')),
        states=446,sources=253,alpha=1000,features='confidence14',fold_seeds=SEEDS,
        bootstrap_seed=21292026,bootstrap_reps=4000,no_test_inputs=True,
        candidate='common_target',evaluation_target='original_full_completion'))
    print('Frozen train/dev whitelist and single training-target change.',flush=True)


def verify():
    m=read(OUT/'manifest.json')
    for name,h in {**m['input_whitelist'],**m['code_hashes']}.items():assert sha(ROOT/name)==h,name
    assert sha(Path('work/gpt2-large/manifest.json'))==m['evaluator_manifest_sha256']
    return m


def score():
    import torch
    from transformers import GPT2TokenizerFast,GPT2LMHeadModel
    verify();assert not (OUT/'token_losses.jsonl').exists()
    model_path=Path('work/gpt2-large');em=read(model_path/'manifest.json')
    for f in em['files']:assert sha(model_path/f['name'])==f['sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=GPT2TokenizerFast.from_pretrained(str(model_path),local_files_only=True)
    model=GPT2LMHeadModel.from_pretrained(str(model_path),local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    def loss(x,mask):
        logits=model(x,attention_mask=mask,use_cache=False).logits
        return torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
            x[:,1:].reshape(-1),reduction='none').view_as(x[:,1:])
    all_rows=[];checks={}
    with torch.inference_mode():
        probe=tokenizer.encode('A training target should improve the original decision objective.',return_tensors='pt').cuda()
        lp=loss(probe,torch.ones_like(probe))
        torch.testing.assert_close(lp.mean(),model(probe,labels=probe,use_cache=False).loss,rtol=1e-6,atol=1e-6)
        short=probe[:,:probe.shape[1]//2];ls=loss(short,torch.ones_like(short))
        torch.testing.assert_close(lp[:,:ls.shape[1]],ls,rtol=1e-5,atol=1e-5)
        truncation=float((lp[:,:ls.shape[1]]-ls).abs().max())
        for split in ('train','dev'):
            pairs=rows(SRC/f'{split}_pairs.jsonl');labels={r['pair_id']:r for r in rows(SRC/f'{split}_labels.jsonl')}
            expected={}
            for p in pairs:
                for b in ('keep4','split22'):
                    l=labels[p['pair_id']][b];t=p[b]['completion'];v=(l['gpt2_nll_sum'],l['gpt2_scored_tokens'])
                    assert t not in expected or expected[t]==v
                    expected[t]=v
            texts=sorted(expected);encoded=[tokenizer.encode(t,add_special_tokens=False) for t in texts]
            assert max(map(len,encoded))<=1024
            max_old=max_vector=0.
            for start in range(0,len(texts),4):
                bt,ids=texts[start:start+4],encoded[start:start+4]
                width=max(2,max(map(len,ids)))
                x=torch.full((len(bt),width),tokenizer.eos_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(x)
                for j,tokens in enumerate(ids):
                    if tokens:x[j,:len(tokens)]=torch.tensor(tokens,device='cuda');mask[j,:len(tokens)]=1
                    else:mask[j,0]=1
                ll=loss(x,mask);sums=(ll*mask[:,1:]).sum(1).cpu().tolist()
                for j,t in enumerate(bt):
                    count=max(0,len(ids[j])-1);values=ll[j,:count].cpu().tolist();es,en=expected[t]
                    assert count==en and abs(sums[j]-es)<=1e-4
                    np.testing.assert_allclose(math.fsum(values),es,atol=1e-4,rtol=1e-6)
                    max_old=max(max_old,abs(sums[j]-es));max_vector=max(max_vector,abs(math.fsum(values)-es))
                    all_rows.append(dict(split=split,text=t,text_sha256=hashlib.sha256(t.encode()).hexdigest(),
                        input_ids=ids[j],token_nll=values,torch_nll_sum=sums[j],scored_tokens=count))
                if (start+4)%512==0:print(split,'token losses',start+4,'/',len(texts),'old max difference',max_old,flush=True)
            checks[split]=dict(unique_texts=len(texts),pairs=len(pairs),max_old_sum_difference=max_old,max_float64_vector_sum_difference=max_vector)
    lines(OUT/'token_losses.jsonl',all_rows)
    dump(OUT/'scoring_checks.json',dict(passed=True,splits=checks,builtin_loss_matches=True,
        causal_truncation_max_difference=truncation,evaluator_revision=em['revision'],
        token_losses_sha256=sha(OUT/'token_losses.jsonl')))
    print('Both development splits scored, original totals checked.',flush=True)


def targets():
    verify();checks=read(OUT/'scoring_checks.json')
    assert checks['passed'] and checks['token_losses_sha256']==sha(OUT/'token_losses.jsonl')
    token={(r['split'],r['text']):r for r in rows(OUT/'token_losses.jsonl')}
    data=rows(DEV);whitelist={r['state_id']:r for r in data};grouped=defaultdict(dict)
    pair_output=[]
    for split in ('train','dev'):
        labels={r['pair_id']:r for r in rows(SRC/f'{split}_labels.jsonl')}
        for p in rows(SRC/f'{split}_pairs.jsonl'):
            if p['state_id'] not in whitelist:continue
            l=labels[p['pair_id']]
            k,s=[token[(split,p[b]['completion'])]['token_nll'] for b in ('keep4','split22')]
            m=min(len(k),len(s));assert m>0 and l['valid_label']
            common=(math.fsum(k[:m])-math.fsum(s[:m]))/m
            full=l['gain'];teacher=l['keep4']['teacher_nll']-l['split22']['teacher_nll']
            r=dict(state_id=p['state_id'],pair_id=p['pair_id'],prompt_id=p['prompt_id'],split=split,
                replicate=p['replicate'],common_tokens=m,full_gain=full,common_gain=common,teacher_gain=teacher)
            assert p['replicate'] not in grouped[p['state_id']]
            grouped[p['state_id']][p['replicate']]=r;pair_output.append(r)
    output=[]
    for old in data:
        pairs=grouped[old['state_id']];assert set(pairs)==set(range(8))
        r={k:old[k] for k in ('state_id','prompt_id','source_split','offset')}
        r['features']=old['features'][:14]
        for target in ('full','common','teacher'):
            yy=[pairs[j][target+'_gain'] for j in range(8)]
            r[target+'_gains']=yy;r[target+'_gain']=float(np.mean(yy))
        assert abs(r['full_gain']-old['gain'])<1e-12 and r['full_gains']==old['gains']
        output.append(r)
    assert len(output)==446 and len(pair_output)==3568
    lines(OUT/'pair_targets.jsonl',pair_output);lines(OUT/'state_targets.jsonl',output)
    dump(OUT/'target_checks.json',dict(passed=True,states=len(output),sources=len({r['prompt_id'] for r in output}),
        valid_pairs=len(pair_output),original_target_unchanged=True,state_targets_sha256=sha(OUT/'state_targets.jsonl'),
        pair_targets_sha256=sha(OUT/'pair_targets.jsonl')))
    print('Two labels assembled on the identical 446 development states.',flush=True)


def evaluate():
    from diagnose_decision_features import fit,predict,half
    verify();check=read(OUT/'target_checks.json')
    assert check['passed'] and check['state_targets_sha256']==sha(OUT/'state_targets.jsonl')
    data=rows(OUT/'state_targets.jsonl');x=np.array([r['features'] for r in data])
    y={key:np.array([r[key+'_gain'] for r in data]) for key in ('full','common','teacher')}
    names=('full_target','common_target');sources=sorted({r['prompt_id'] for r in data});n=len(data)
    pred={key:np.zeros((3,n)) for key in names};mask={key:np.zeros((3,n)) for key in names}
    heuristic=np.zeros((3,n));fractions=np.zeros((3,n));fits=[]
    for rep,seed in enumerate(SEEDS):
        order=np.random.default_rng(seed).permutation(sources);fold_map={s:i%5 for i,s in enumerate(order)}
        for fold in range(5):
            valid=np.array([i for i,r in enumerate(data) if fold_map[r['prompt_id']]==fold]);train=np.setdiff1d(np.arange(n),valid)
            assert not {data[i]['prompt_id'] for i in train}&{data[i]['prompt_id'] for i in valid}
            heuristic[rep,valid]=half(-x[valid,2:4].mean(1));fractions[rep,valid]=(len(valid)//2)/len(valid)
            for name in names:
                target=name.split('_')[0];model=fit(x[train],y[target][train]);p=predict(model,x[valid])
                pred[name][rep,valid]=p;mask[name][rep,valid]=half(p)
                fits.append(dict(repeat=rep,fold=fold,method=name,model=model,
                    train_ids=[data[i]['state_id'] for i in train],valid_ids=[data[i]['state_id'] for i in valid],
                    predictions=p.tolist(),selected=half(p).astype(int).tolist()))
    groups={s:[i for i,r in enumerate(data) if r['prompt_id']==s] for s in sources}
    draws=np.random.default_rng(21292026).integers(0,len(sources),(4000,len(sources)))
    def interval(v):
        sums=np.array([v[groups[s]].sum() for s in sources]);counts=np.array([len(groups[s]) for s in sources])
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        return dict(estimate=float(v.mean()),interval95=np.quantile(boot,[.025,.975]).tolist())
    metrics={}
    for name in names:
        metrics[name]={}
        for target in ('full','common','teacher'):
            metrics[name][target]=dict(mse=float(np.mean((pred[name]-y[target][None,:])**2)),
                correlation=float(np.corrcoef(pred[name].ravel(),np.tile(y[target],3))[0,1]),
                gain_vs_random=interval(((mask[name]-fractions)*y[target][None,:]).mean(0)),
                gain_vs_full_target=interval(((mask[name]-mask['full_target'])*y[target][None,:]).mean(0)),
                gain_vs_heuristic=interval(((mask[name]-heuristic)*y[target][None,:]).mean(0)))
    primary={key:metrics['common_target']['full'][key] for key in ('gain_vs_full_target','gain_vs_heuristic')}
    gate=all(v['interval95'][0]>0 for v in primary.values())
    diagnostics={}
    for key in ('full','common'):
        yy=np.array([r[key+'_gains'] for r in data])
        diagnostics[key]=dict(mean_gain=float(y[key].mean()),state_variance=float(y[key].var(ddof=1)),
            split_half_correlation=float(np.corrcoef(yy[:,:4].mean(1),yy[:,4:].mean(1))[0,1]))
    diagnostics['full_common_correlation']=float(np.corrcoef(y['full'],y['common'])[0,1])
    diagnostics['state_mean_sign_disagreements']=int(np.sum(np.sign(y['full'])!=np.sign(y['common'])))
    dump(OUT/'fits.json',fits)
    lines(OUT/'oof_predictions.jsonl',[dict(state_id=r['state_id'],prompt_id=r['prompt_id'],
        predictions={k:pred[k][:,i].tolist() for k in names},selected={k:mask[k][:,i].astype(int).tolist() for k in names},
        heuristic_selected=heuristic[:,i].astype(int).tolist(),selection_fraction=fractions[:,i].tolist()) for i,r in enumerate(data)])
    dump(OUT/'analysis.json',dict(retrospective=True,no_test_inputs=True,states=n,sources=len(sources),
        fits=len(fits),primary=primary,common_target_warrants_fresh_confirmation=bool(gate),
        metrics=metrics,target_diagnostics=diagnostics,state_targets_sha256=sha(OUT/'state_targets.jsonl'),
        fits_sha256=sha(OUT/'fits.json'),oof_predictions_sha256=sha(OUT/'oof_predictions.jsonl'),
        limitations=['Old train/dev data, repeatedly explored; not independent test.',
            'Primary ranking evaluation retains original full-completion target.',
            'Bootstrap conditions on fitted fold models and fixed rankings.',
            'GPT-2 is the optimized proxy; no online/human-quality conclusion.']))
    print(json.dumps(dict(primary=primary,metrics=metrics,target_diagnostics=diagnostics,warrants_fresh_confirmation=bool(gate)),indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','score','targets','evaluate'])
    globals()[ap.parse_args().action]()
