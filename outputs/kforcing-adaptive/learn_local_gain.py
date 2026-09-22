"""Separated prepare/collect/fit/test experiment on frozen local branch gains."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from core import load_model
from coupled_teacher_audit import inverse_cdf
from current_features import current_features_tensor
from collect_benefit import dump,verify_checkpoints

SEEDS=[761,769,773,787,797,809,811,821]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    assert not (args.out/'prefixes.json').exists()
    data=Path('work/lm1b-data/test.parquet')
    assert sha(data)=='d3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    parent=args.out.parent
    old=read(parent/'lm1b-004/prefixes.json')
    seen={tuple(p['ids']) for p in old}
    texts_seen={p['source_sha256'] for p in old}
    excluded_rows={p['source_row'] for p in old}
    for folder in ('pilot-001','benefit-002'):
        for group in read(parent/folder/'prefixes.json').values():
            seen.update(tuple(p['ids']) for p in group)
    seen.update(tuple(p['ids']) for p in read(parent/'current-003/fresh_prefixes.json'))
    texts=pq.read_table(data,columns=['text'])['text'].to_pylist()
    indices=list(range(len(texts)))
    random.Random(7071701).shuffle(indices)
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    chosen=[]
    for index in indices:
        if index in excluded_rows:
            continue
        digest=hashlib.sha256(texts[index].encode()).hexdigest()
        if digest in texts_seen:
            continue
        ids=bert.encode(texts[index],add_special_tokens=True,truncation=True,max_length=128)
        if len(ids)<12 or tuple(ids[:6]) in seen:
            continue
        pi=len(chosen)
        chosen.append(dict(id=f's7-{pi:03}',index=pi,ids=ids[:6],source_row=index,source_sha256=digest))
        seen.add(tuple(ids[:6])); texts_seen.add(digest)
        if len(chosen)==448:
            break
    assert len(chosen)==448
    dump(args.out/'prefixes.json',dict(train=chosen[:256],dev=chosen[256:320],test=chosen[320:]))
    projection=(np.random.default_rng(7071702).normal(size=(768,64))/math.sqrt(768)).astype('float32')
    np.save(args.out/'projection.npy',projection,allow_pickle=False)
    dump(args.out/'data_manifest.json',dict(source_sha256=sha(data),prefix_sha256=sha(args.out/'prefixes.json'),
        projection_sha256=sha(args.out/'projection.npy'),split_counts=dict(train=256,dev=64,test=128),
        seeds=SEEDS,excluded_prior_lm1b_rows=len(excluded_rows),split_unit='Unique six-token prefix and raw sentence hash; document identity unavailable'))
    print('Frozen fresh prefixes: train 256 / dev 64 / test 128',flush=True)


def feature_tensor(logits,hidden,noise,projection):
    x=logits.float()
    top=x.topk(2,dim=-1).values
    margins=top[:,:,0]-top[:,:,1]
    logp=x.log_softmax(-1)
    entropy=-(logp.exp()*logp).sum(-1)
    probabilities=logp.max(-1).values.exp()
    common=torch.cat((margins,entropy,probabilities,
        (margins[:,2:].mean(-1)-margins[:,:2].mean(-1))[:,None],
        (entropy[:,2:].mean(-1)-entropy[:,:2].mean(-1))[:,None]),-1)
    return torch.cat((common,hidden[:,2:].float().mean(1)@projection,noise.squeeze(-1)),1)


@torch.inference_mode()
def collect(args):
    output=args.out/(args.split+'.jsonl')
    assert not output.exists(),'Preserve collected labels'
    if args.split=='test':
        assert (args.out/'locked_model.json').exists(),'Freeze selection before test collection'
    verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    prefixes=read(args.out/'prefixes.json')[args.split]
    manifest=read(args.out/'data_manifest.json')
    assert sha(args.out/'projection.npy')==manifest['projection_sha256']
    projection=torch.from_numpy(np.load(args.out/'projection.npy',allow_pickle=False)).cuda()
    cases=[]
    for p in prefixes:
        for seed in SEEDS:
            noise=torch.rand(4,generator=torch.Generator().manual_seed(seed+p['index']*100003))
            cases.append(dict(prompt_id=p['id'],seed=seed,noise_seed=seed+p['index']*100003,prefix=p['ids'],noise=noise.tolist()))
    mismatches=0
    with output.open('w',encoding='utf-8') as dest:
        for start in range(0,len(cases),16):
            batch=cases[start:start+16]
            context=torch.tensor([r['prefix'] for r in batch],device='cuda')
            noise=torch.tensor([r['noise'] for r in batch],device='cuda').unsqueeze(-1)
            tau=torch.ones(len(batch),1,1,device='cuda')
            captured=[]
            handle=model.output_layer.linear.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach()))
            try:
                logits4=model(context,noise,tau,mode='inference')
            finally:
                handle.remove()
            assert len(captured)==1 and captured[0].shape==(len(batch),4,768)
            features=feature_tensor(logits4,captured[0],noise,projection)
            assert features.shape==(len(batch),82) and torch.isfinite(features).all()
            if start==0:
                torch.testing.assert_close(features[0,:14],current_features_tensor(logits4[:1]))
            fixed=logits4.argmax(-1)
            first_logits,cache=model(context,noise[:,:2],tau,mode='inference',return_kv=True)
            first=first_logits.argmax(-1)
            mismatches+=int((first!=fixed[:,:2]).sum())
            second,cache2=model(torch.cat((context,first),1),noise[:,2:],tau,mode='inference',kv_caches=cache,return_kv=True)
            assert all(c[0].shape[1]==6 for c in cache) and all(c[0].shape[1]==8 for c in cache2)
            split=torch.cat((first,second.argmax(-1)),1)
            history=context
            reference=[]
            for j in range(4):
                next_token=inverse_cdf(teacher.forward_high_precision(history)[:,-1],noise[:,j])
                reference.append(next_token)
                history=torch.cat((history,next_token),1)
            reference=torch.cat(reference,1)
            for r,a,b,t,f in zip(batch,fixed.cpu().tolist(),split.cpu().tolist(),reference.cpu().tolist(),features.cpu().tolist()):
                r.update(fixed4=a,split22=b,ar_reference=t,features=f,
                    gain=sum(x==z for x,z in zip(b,t))-sum(x==z for x,z in zip(a,t)),
                    no_eos_all_branches=all(102 not in x for x in (a,b,t)))
                dest.write(json.dumps(r,allow_nan=False)+'\n')
            dest.flush()
            if (start+16)%256==0:
                print(args.split,'windows',start+16,'/',len(cases),flush=True)
    dump(args.out/(args.split+'_checks.json'),dict(rows=len(cases),first_two_mismatches=mismatches,
         batched_confidence_features_match_old=True,projection_sha256=manifest['projection_sha256'],
         locked_model_sha256=sha(args.out/'locked_model.json') if args.split=='test' else None))


def interval(values,prompts):
    groups=defaultdict(list)
    for v,p in zip(values,prompts):
        groups[p].append(v)
    values=np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng=np.random.default_rng(7071704)
    draws=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(2000)])
    return dict(mean=float(values.mean()),interval95=np.quantile(draws,[.025,.975]).tolist(),prefix_clusters=len(values))


def half_selection(scores,records):
    selected=np.zeros(len(records),dtype=bool)
    order=sorted(range(len(records)),key=lambda i:(-float(scores[i]),records[i]['prompt_id'],records[i]['seed']))
    selected[order[:len(records)//2]]=True
    return selected


def predict(model,records):
    n=14 if model['features']=='confidence14' else 82
    x=np.array([r['features'][:n] for r in records],dtype=np.float64)
    return ((x-np.array(model['mean']))/np.array(model['std']))@np.array(model['coef'])+model['intercept']


def fit(args):
    assert not (args.out/'locked_model.json').exists()
    assert not (args.out/'test.jsonl').exists(),'Do not select with test labels available'
    train,dev=rows(args.out/'train.jsonl'),rows(args.out/'dev.jsonl')
    y=np.array([r['gain'] for r in train],dtype=np.float64)
    yd=np.array([r['gain'] for r in dev],dtype=np.float64)
    candidates=[]
    for name,n in (('confidence14',14),('hidden82',82)):
        x=np.array([r['features'][:n] for r in train],dtype=np.float64)
        mean,std=x.mean(0),x.std(0)
        std[std<1e-8]=1.
        z=(x-mean)/std
        for alpha in (1.,10.,100.):
            coef=np.linalg.solve(z.T@z+alpha*np.eye(n),z.T@(y-y.mean()))
            model=dict(name=f'{name}_alpha{int(alpha):03}',features=name,alpha=alpha,mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),intercept=float(y.mean()))
            scores=predict(model,dev)
            selected=half_selection(scores,dev)
            improvement=interval((selected.astype(float)-.5)*yd/4,[r['prompt_id'] for r in dev])
            model.update(dev_improvement=improvement,dev_mse=float(np.mean((scores-yd)**2)),dev_threshold=float(np.median(scores)))
            candidates.append(model)
    chosen=sorted(candidates,key=lambda m:(-m['dev_improvement']['mean'],m['name']))[0]
    chosen.update(train_sha256=sha(args.out/'train.jsonl'),dev_sha256=sha(args.out/'dev.jsonl'),
        prefix_sha256=sha(args.out/'prefixes.json'),projection_sha256=sha(args.out/'projection.npy'),
        dev_gate_passed=chosen['dev_improvement']['interval95'][0]>0,
        selection='Predeclared 6 ridge candidates; dev half-budget gain; ties by name; no test labels present')
    dump(args.out/'dev_candidates.json',candidates)
    dump(args.out/'locked_model.json',chosen)
    print(json.dumps({k:chosen[k] for k in ('name','dev_improvement','dev_mse','dev_threshold','dev_gate_passed')},indent=2),flush=True)


def evaluate(args):
    assert not (args.out/'test_analysis.json').exists(),'One frozen test evaluation only'
    model=read(args.out/'locked_model.json')
    for split in ('train','dev'):
        assert sha(args.out/(split+'.jsonl'))==model[split+'_sha256']
    test=rows(args.out/'test.jsonl')
    assert read(args.out/'test_checks.json')['locked_model_sha256']==sha(args.out/'locked_model.json')
    y=np.array([r['gain'] for r in test],dtype=np.float64)
    scores=predict(model,test)
    selected=half_selection(scores,test)
    baseline=half_selection([-np.mean(r['features'][2:4]) for r in test],test)
    oracle=half_selection(y,test)
    random_selection=np.zeros(len(test),dtype=bool)
    random_selection[np.random.default_rng(7071703).permutation(len(test))[:len(test)//2]]=True
    threshold=scores>model['dev_threshold']
    prompts=[r['prompt_id'] for r in test]
    policies=dict(learned_half=selected,confidence_half=baseline,random_half=random_selection,oracle_half=oracle,locked_threshold=threshold)
    base=np.array([sum(a==t for a,t in zip(r['fixed4'],r['ar_reference'])) for r in test])
    result={}
    for name,mask in policies.items():
        p=float(mask.mean())
        result[name]=dict(split_windows=int(mask.sum()),split_fraction=p,
            teacher_token_match_fraction=float(np.mean(base+mask*y)/4),
            improvement_vs_matched_random_expectation=interval((mask.astype(float)-p)*y/4,prompts))
    gate=model['dev_gate_passed'] and result['learned_half']['improvement_vs_matched_random_expectation']['interval95'][0]>0
    report=dict(chosen_model=model['name'],test_windows=len(test),test_prefixes=len(set(prompts)),
        fixed4_match_fraction=float(base.mean()/4),all_split_match_fraction=float((base+y).mean()/4),
        model_mse=float(np.mean((scores-y)**2)),constant_train_mean_mse=float(np.mean((model['intercept']-y)**2)),
        policies=result,learned_vs_random_realization=interval((selected.astype(float)-random_selection.astype(float))*y/4,prompts),
        dev_gate_passed=model['dev_gate_passed'],online_experiment_gate_passed=bool(gate),
        limitation='Only fresh six-token contexts and AR noise-coupled token agreement. No generation quality or throughput claim; test must not be reused for model selection.')
    with (args.out/'test_predictions.jsonl').open('w',encoding='utf-8') as f:
        for i,r in enumerate(test):
            f.write(json.dumps(dict(prompt_id=r['prompt_id'],seed=r['seed'],gain=r['gain'],score=float(scores[i]),
                **{name:bool(mask[i]) for name,mask in policies.items()}))+'\n')
    dump(args.out/'test_analysis.json',report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','collect','fit','evaluate'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--split',choices=['train','dev','test'])
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    dict(prepare=prepare,collect=collect,fit=fit,evaluate=evaluate)[args.action](args)


if __name__=='__main__':
    main()
