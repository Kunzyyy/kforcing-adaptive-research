"""No-fit confirmation of the locked stage-7 selector on new contexts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from core import load_model
from coupled_teacher_audit import inverse_cdf
from learn_local_gain import feature_tensor,predict,half_selection
from current_features import current_features_tensor
from collect_benefit import dump,verify_checkpoints

SEEDS=[829,839,853,857,859,863,877,881]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prior_sources(root):
    old=read(root/'lm1b-004/prefixes.json')
    for group in read(root/'learned-007/prefixes.json').values():
        old+=group
    prefix_set={tuple(p['ids'][:6]) for p in old}
    for folder in ('pilot-001','benefit-002'):
        for group in read(root/folder/'prefixes.json').values():
            prefix_set.update(tuple(p['ids'][:6]) for p in group)
    prefix_set.update(tuple(p['ids'][:6]) for p in read(root/'current-003/fresh_prefixes.json'))
    return old,prefix_set


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    assert not (args.out/'contexts.json').exists(),'Preserve frozen contexts'
    source=args.out.parent/'learned-007'
    shutil.copyfile(source/'locked_model.json',args.out/'frozen_model.json')
    shutil.copyfile(source/'projection.npy',args.out/'projection.npy')
    old,seen=prior_sources(args.out.parent)
    seen_rows={p['source_row'] for p in old}
    seen_text={p['source_sha256'] for p in old}
    path=Path('work/lm1b-data/test.parquet')
    assert sha(path)=='d3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts=pq.read_table(path,columns=['text'])['text'].to_pylist()
    order=list(range(len(texts)))
    random.Random(8081701).shuffle(order)
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    sources=[]
    contexts=[]
    for index in order:
        if index in seen_rows:
            continue
        digest=hashlib.sha256(texts[index].encode()).hexdigest()
        if digest in seen_text:
            continue
        ids=bert.encode(texts[index],add_special_tokens=True,truncation=True,max_length=128)
        required=12 if len(sources)<256 else 36
        if len(ids)<required or tuple(ids[:6]) in seen:
            continue
        si=len(sources)
        source_id=f's8-{si:03}'
        sources.append(dict(source_id=source_id,source_index=si,source_row=index,source_sha256=digest,reference_ids=ids))
        for length in ([6] if si<256 else [16,32]):
            assert 102 not in ids[:length]
            contexts.append(dict(prompt_id=f'{source_id}-L{length}',source_id=source_id,source_index=si,
                                 cohort='short' if length==6 else 'long',length=length,ids=ids[:length]))
        seen.add(tuple(ids[:6])); seen_text.add(digest)
        if len(sources)==384:
            break
    assert len(sources)==384 and len(contexts)==512
    dump(args.out/'sources.json',sources)
    dump(args.out/'contexts.json',contexts)
    dump(args.out/'manifest.json',dict(sources=384,contexts=512,windows=4096,seeds=SEEDS,
        source_dataset_sha256=sha(path),sources_sha256=sha(args.out/'sources.json'),contexts_sha256=sha(args.out/'contexts.json'),
        model_sha256=sha(args.out/'frozen_model.json'),projection_sha256=sha(args.out/'projection.npy'),
        stage7_model_sha256=sha(source/'locked_model.json'),stage7_projection_sha256=sha(source/'projection.npy'),
        feature_code_sha256=sha(Path(__file__).parent/'learn_local_gain.py'),
        scope='New source sentences; long contexts are real text, not sampled online histories. No refitting.'))
    print('Frozen: 256 short sentences, 128 longer sentences at two lengths; 4096 windows.',flush=True)


@torch.inference_mode()
def branch_nll(teacher,context,tokens):
    full=torch.cat((context,tokens),1)
    logits=teacher.forward_high_precision(full[:,:-1])[:,context.shape[1]-1:].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens.reshape(-1),reduction='none').view(-1,4).sum(-1)


@torch.inference_mode()
def collect(args):
    path=args.out/'windows.jsonl'
    assert not path.exists(),'Preserve observed windows'
    manifest=read(args.out/'manifest.json')
    assert sha(args.out/'frozen_model.json')==manifest['model_sha256']==manifest['stage7_model_sha256']
    assert sha(args.out/'projection.npy')==manifest['projection_sha256']==manifest['stage7_projection_sha256']
    assert sha(Path(__file__).parent/'learn_local_gain.py')==manifest['feature_code_sha256']
    verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    selector=read(args.out/'frozen_model.json')
    projection=torch.from_numpy(np.load(args.out/'projection.npy',allow_pickle=False)).cuda()
    contexts=read(args.out/'contexts.json')
    disagreements=0
    count=0
    with path.open('w',encoding='utf-8') as out:
        for length in (6,16,32):
            cases=[]
            for p in contexts:
                if p['length']!=length:
                    continue
                for seed in SEEDS:
                    noise_seed=80800000+seed+p['source_index']*100003
                    noise=torch.rand(4,generator=torch.Generator().manual_seed(noise_seed))
                    cases.append(dict(prompt_id=p['prompt_id'],source_id=p['source_id'],length=length,
                        prefix=p['ids'],seed=seed,noise_seed=noise_seed,noise=noise.tolist()))
            for start in range(0,len(cases),16):
                batch=cases[start:start+16]
                context=torch.tensor([r['prefix'] for r in batch],device='cuda')
                noise=torch.tensor([r['noise'] for r in batch],device='cuda').unsqueeze(-1)
                tau=torch.ones(len(batch),1,1,device='cuda')
                captured=[]
                handle=model.output_layer.linear.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach()))
                try:
                    logits=model(context,noise,tau,mode='inference')
                finally:
                    handle.remove()
                assert len(captured)==1 and captured[0].shape==(len(batch),4,768)
                features=feature_tensor(logits,captured[0],noise,projection)
                if start==0:
                    torch.testing.assert_close(features[0,:14],current_features_tensor(logits[:1]))
                assert features.shape==(len(batch),82) and torch.isfinite(features).all()
                fixed=logits.argmax(-1)
                first_logits,cache=model(context,noise[:,:2],tau,mode='inference',return_kv=True)
                first=first_logits.argmax(-1)
                disagreements+=int((first!=fixed[:,:2]).sum())
                tail,cache2=model(torch.cat((context,first),1),noise[:,2:],tau,mode='inference',kv_caches=cache,return_kv=True)
                assert all(c[0].shape[1]==length for c in cache) and all(c[0].shape[1]==length+2 for c in cache2)
                split=torch.cat((first,tail.argmax(-1)),1)
                reference=[]
                history=context
                for j in range(4):
                    token=inverse_cdf(teacher.forward_high_precision(history)[:,-1],noise[:,j])
                    reference.append(token); history=torch.cat((history,token),1)
                reference=torch.cat(reference,1)
                nll4=branch_nll(teacher,context,fixed).cpu().tolist()
                nll2=branch_nll(teacher,context,split).cpu().tolist()
                for r,a,b,t,f,na,nb in zip(batch,fixed.cpu().tolist(),split.cpu().tolist(),reference.cpu().tolist(),features.cpu().tolist(),nll4,nll2):
                    r.update(fixed4=a,split22=b,ar_reference=t,features=f,
                        gain=sum(x==z for x,z in zip(b,t))-sum(x==z for x,z in zip(a,t)),
                        teacher_nll_fixed4=na,teacher_nll_split22=nb,nll_gain=na-nb,
                        no_eos_all_branches=all(102 not in tokens for tokens in (a,b,t)))
                scores=predict(selector,batch)
                for r,score in zip(batch,scores):
                    r['score']=float(score)
                    out.write(json.dumps(r,allow_nan=False)+'\n')
                out.flush(); count+=len(batch)
                if count%256==0:
                    print('Confirmation windows',count,'/4096; context',length,flush=True)
    dump(args.out/'collection_checks.json',dict(windows=count,first_two_disagreements=disagreements,
        confidence_features_crosschecked=True,cache_lengths_checked=True,model_sha256=sha(args.out/'frozen_model.json'),
        projection_sha256=sha(args.out/'projection.npy'),feature_code_sha256=manifest['feature_code_sha256']))


def cluster_interval(values,records,coverage=.95):
    groups=defaultdict(list)
    for v,r in zip(values,records):
        groups[r['source_id']].append(v)
    means=np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng=np.random.default_rng(8081704)
    boot=np.array([means[rng.integers(0,len(means),len(means))].mean() for _ in range(4000)])
    tail=(1-coverage)/2
    return dict(mean=float(means.mean()),coverage=coverage,interval=np.quantile(boot,[tail,1-tail]).tolist(),source_clusters=len(means))


def analyze(args):
    assert not (args.out/'analysis.json').exists(),'One frozen confirmation analysis only'
    data=rows(args.out/'windows.jsonl')
    model=read(args.out/'frozen_model.json')
    length_results={}
    for length in (6,16,32):
        group=[r for r in data if r['length']==length]
        scores=np.array([r['score'] for r in group])
        y=np.array([r['gain'] for r in group])
        policies=dict(learned_half=half_selection(scores,group),
            confidence_half=half_selection([-np.mean(r['features'][2:4]) for r in group],group),
            oracle_half=half_selection(y,group),locked_threshold=scores>model['dev_threshold'])
        random_mask=np.zeros(len(group),dtype=bool)
        random_mask[np.random.default_rng(8081703).permutation(len(group))[:len(group)//2]]=True
        policies['random_half']=random_mask
        base=np.array([sum(a==t for a,t in zip(r['fixed4'],r['ar_reference'])) for r in group])
        nll_gain=np.array([r['nll_gain'] for r in group])
        summary={}
        for name,mask in policies.items():
            p=float(mask.mean())
            benefit=(mask.astype(float)-p)*y/4
            summary[name]=dict(split_windows=int(mask.sum()),split_fraction=p,
                match_fraction=float(np.mean(base+mask*y)/4),
                versus_random_expectation=cluster_interval(benefit,group),
                teacher_nll_reduction_vs_random_per_token=cluster_interval((mask.astype(float)-p)*nll_gain/4,group))
            for r,selected in zip(group,mask):
                r[name]=bool(selected)
        length_results[str(length)]=dict(windows=len(group),source_sentences=len(group)//8,
            fixed4_match=float(base.mean()/4),all_split_match=float((base+y).mean()/4),policies=summary)
    primary={}
    for name,lengths in (('short',[6]),('long',[16,32])):
        group=[r for r in data if r['length'] in lengths]
        values=[(r['learned_half']-.5)*r['gain']/4 for r in group]
        primary[name]=cluster_interval(values,group,coverage=.975)
        primary[name]['passes']=primary[name]['interval'][0]>0
    with (args.out/'selections.jsonl').open('w',encoding='utf-8') as out:
        for r in data:
            out.write(json.dumps(r,allow_nan=False)+'\n')
    report=dict(primary=primary,by_length=length_results,
        local_confirmation_passed=all(v['passes'] for v in primary.values()),
        original_stage7_dev_gate_passed=model['dev_gate_passed'],frozen_model_sha256=sha(args.out/'frozen_model.json'),
        notes='Two 97.5% primary intervals, clustered by sentence; secondary 95% intervals exploratory. No refitting or deployment. Long contexts are real source text. Threshold secondary intervals condition on observed selection rate; ranking selections held fixed within bootstrap.')
    dump(args.out/'analysis.json',report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','collect','analyze'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    dict(prepare=prepare,collect=collect,analyze=analyze)[args.action](args)


if __name__=='__main__':
    main()
