"""Single-intervention counterfactual rollouts; features precede either outcome."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from core import load_model
from learn_local_gain import feature_tensor
from collect_benefit import dump,verify_checkpoints
from confirm_local_gain import read,rows,sha,prior_sources
from run_frozen_online import append

ROOT=Path(__file__).resolve().parent
SEEDS=[1103,1109,1117,1123]
OFFSETS=[0,8]


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    assert not (args.out/'prefixes.json').exists()
    old,seen=prior_sources(args.out.parent)
    for path in ('online-009/prefixes.json','early-010/prefixes.json'):
        old+=read(args.out.parent/path)
    old += [dict(p,ids=p['reference_ids'][:6]) for p in read(args.out.parent/'confirm-008/sources.json')]
    seen.update(tuple(p['ids'][:6]) for p in old)
    seen_rows={p['source_row'] for p in old};seen_hash={p['source_sha256'] for p in old}
    dataset=Path('work/lm1b-data/test.parquet')
    assert sha(dataset)=='d3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts=pq.read_table(dataset,columns=['text'])['text'].to_pylist()
    order=list(range(len(texts)));random.Random(11112001).shuffle(order)
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    chosen=[]
    for index in order:
        digest=hashlib.sha256(texts[index].encode()).hexdigest()
        if index in seen_rows or digest in seen_hash:continue
        ids=bert.encode(texts[index],add_special_tokens=True,truncation=True,max_length=128)
        if len(ids)<12 or tuple(ids[:6]) in seen:continue
        i=len(chosen)
        chosen.append(dict(prompt_id=f's11-{i:03}',index=i,ids=ids[:6],source_row=index,source_sha256=digest,
            split='train' if i<192 else ('dev' if i<256 else 'test')))
        seen.add(tuple(ids[:6]));seen_hash.add(digest)
        if len(chosen)==384:break
    assert len(chosen)==384
    dump(args.out/'prefixes.json',chosen)
    shutil.copyfile(args.out.parent/'learned-007/projection.npy',args.out/'projection.npy')
    shutil.copyfile(args.out.parent/'learned-007/locked_model.json',args.out/'old_model.json')
    dump(args.out/'manifest.json',dict(sources=384,splits=dict(train=192,dev=64,test=128),seeds=SEEDS,offsets=OFFSETS,
        prefixes_sha256=sha(args.out/'prefixes.json'),projection_sha256=sha(args.out/'projection.npy'),
        old_model_sha256=sha(args.out/'old_model.json'),protocol_sha256=sha(ROOT/'STAGE11_PROTOCOL.md'),
        feature_code_sha256=sha(ROOT/'learn_local_gain.py'),dataset_sha256=sha(dataset)))
    print('Frozen 192 train / 64 dev / 128 test sources; max 3072 intervention pairs.',flush=True)


def setup(args):
    m=read(args.out/'manifest.json')
    for name,key in [('prefixes.json','prefixes_sha256'),('projection.npy','projection_sha256'),('old_model.json','old_model_sha256')]:
        assert sha(args.out/name)==m[key]
    assert sha(ROOT/'STAGE11_PROTOCOL.md')==m['protocol_sha256']
    assert sha(ROOT/'learn_local_gain.py')==m['feature_code_sha256']
    verify_checkpoints(args)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    projection=torch.from_numpy(np.load(args.out/'projection.npy')).cuda()
    return model,projection


@torch.inference_mode()
def pair(model,prefix,noise_seed,offset,projection,max_new=122):
    device=prefix.device
    tape=torch.rand((1,max_new,1),generator=torch.Generator().manual_seed(noise_seed)).to(device)
    tau=torch.ones(1,1,1,device=device)
    context=prefix;cache=None;produced=0;history_calls=0
    while produced<offset:
        logits,cache=model(context,tape[:,produced:produced+4],tau,mode='inference',kv_caches=cache,return_kv=True)
        assert all(c[0].shape[1]==context.shape[1] for c in cache)
        ids=logits.argmax(-1)[0].cpu().tolist();history_calls+=1
        if 102 in ids:
            return dict(status='unreachable_eos',history_ids=context[0].cpu().tolist()+ids[:ids.index(102)+1])
        context=torch.cat((context,logits.argmax(-1)),1);produced+=4
    captured=[]
    hook=model.output_layer.linear.register_forward_pre_hook(lambda mod,inputs:captured.append(inputs[0].detach()))
    try:
        logits4,at_cache=model(context,tape[:,offset:offset+4],tau,mode='inference',kv_caches=cache,return_kv=True)
    finally:hook.remove()
    assert len(captured)==1 and captured[0].shape==(1,4,768)
    initial=logits4.argmax(-1)[0].cpu().tolist()
    if 102 in initial[:2]:
        return dict(status='head_eos',history_ids=context[0].cpu().tolist(),initial_candidates=initial)
    features=feature_tensor(logits4,captured[0],tape[:,offset:offset+4],projection)[0].cpu().tolist()
    assert len(features)==82 and np.isfinite(features).all()
    head=torch.tensor([initial[:2]],device=device)
    split_context=torch.cat((context,head),1)
    tail,tail_cache=model(split_context,tape[:,offset+2:offset+4],tau,mode='inference',kv_caches=at_cache,return_kv=True)
    split=initial[:2]+tail.argmax(-1)[0].cpu().tolist()
    assert all(c[0].shape[1]==context.shape[1] for c in at_cache)
    assert all(c[0].shape[1]==context.shape[1]+2 for c in tail_cache)
    def finish(first,start_cache,extra):
        if 102 in first:
            seq=context[0].cpu().tolist()+first[:first.index(102)+1]
            return dict(ids=seq,calls=history_calls+1+extra,computed_candidates=offset+4+2*extra,stopped_eos=True)
        hist=torch.cat((context,torch.tensor([first],device=device)),1)
        position=offset+4;kv=start_cache;calls=history_calls+1+extra;candidates=offset+4+2*extra
        while position<max_new:
            width=min(4,max_new-position)
            logits,kv=model(hist,tape[:,position:position+width],tau,mode='inference',kv_caches=kv,return_kv=True)
            assert all(c[0].shape[1]==hist.shape[1] for c in kv)
            ids=logits.argmax(-1)[0].cpu().tolist();calls+=1;candidates+=width
            kept=ids[:ids.index(102)+1] if 102 in ids else ids
            hist=torch.cat((hist,torch.tensor([kept],device=device)),1);position+=len(kept)
            if 102 in ids:break
        seq=hist[0].cpu().tolist()
        return dict(ids=seq,calls=calls,computed_candidates=candidates,stopped_eos=seq[-1]==102)
    a=finish(initial,at_cache,0);b=finish(split,tail_cache,1)
    return dict(status='paired',history_ids=context[0].cpu().tolist(),noise=tape[0,offset:offset+4,0].cpu().tolist(),
                initial_candidates=initial,split_candidates=split,features=features,keep4=a,split22=b)


@torch.inference_mode()
def preflight(args):
    from early_stop_decoder import generate
    from learn_local_gain import feature_tensor as original_feature
    model,projection=setup(args)
    old=[r for r in rows(args.out.parent/'early-010/samples.jsonl') if r['method']=='fixed4'][:12]
    cases=[];paired=0
    for r in old:
        prefix=torch.tensor([r['ids'][:6]],device='cuda')
        for offset in OFFSETS:
            p=pair(model,prefix,r['noise_seed'],offset,projection)
            if p['status']!='paired':continue
            assert p['keep4']['ids']==r['ids']
            schedule=[False]*30;schedule[offset//4]=True
            intervention=generate(model,prefix,'replay',seed=r['noise_seed'],schedule=schedule)
            assert p['split22']['ids']==intervention['ids']
            assert p['split22']['calls']==intervention['forward_calls']
            assert p['keep4']['calls']==r['forward_calls']
            # Independent no-cache feature recomputation at the identical decision history.
            hist=torch.tensor([p['history_ids']],device='cuda');noise=torch.tensor(p['noise'],device='cuda').reshape(1,4,1)
            hidden=[];hook=model.output_layer.linear.register_forward_pre_hook(lambda m,x:hidden.append(x[0].detach()))
            try:logits=model(hist,noise,torch.ones(1,1,1,device='cuda'),mode='inference')
            finally:hook.remove()
            f=original_feature(logits,hidden[0],noise,projection)[0].cpu().numpy()
            np.testing.assert_allclose(f,p['features'],rtol=2e-3,atol=2e-3)
            paired+=1;cases.append(dict(prompt_id=r['prompt_id'],seed=r['seed'],offset=offset))
    assert paired>=12
    dump(args.out/'preflight.json',dict(passed=True,paired_old_cases=paired,cases=cases,
        keep4_matches=True,single_intervention_matches=True,feature_recompute_matches=True,
        collector_sha256=sha(Path(__file__))))
    print('Old trajectory / intervention / feature checks passed:',paired,flush=True)


@torch.inference_mode()
def collect(args):
    from transformers import BertTokenizerFast
    path=args.out/(args.split+'_pairs.jsonl');assert not path.exists()
    if args.split=='test':assert (args.out/'locked_model.json').exists(),'Lock before seeing test outcomes'
    if args.split=='dev':assert (args.out/'train_labels.jsonl').exists()
    assert read(args.out/'preflight.json')['passed']
    assert sha(Path(__file__))==read(args.out/'preflight.json')['collector_sha256']
    model,projection=setup(args)
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    prefixes=[p for p in read(args.out/'prefixes.json') if p['split']==args.split]
    counts={};kept=0;seen_base={}
    with path.open('w',encoding='utf-8') as dest:
        for pi,p in enumerate(prefixes):
            prefix=torch.tensor([p['ids']],device='cuda')
            for seed in SEEDS:
                noise_seed=111000000+seed+p['index']*100003
                for offset in OFFSETS:
                    r=pair(model,prefix,noise_seed,offset,projection)
                    r.update(pair_id=f"{p['prompt_id']}-{seed}-{offset}",prompt_id=p['prompt_id'],seed=seed,noise_seed=noise_seed,offset=offset)
                    counts[r['status']]=counts.get(r['status'],0)+1
                    if r['status']=='paired':
                        for branch in ('keep4','split22'):
                            r[branch]['completion']=bert.decode(r[branch]['ids'][6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
                        ident=(p['prompt_id'],seed)
                        if ident in seen_base:assert r['keep4']['ids']==seen_base[ident]
                        else:seen_base[ident]=r['keep4']['ids']
                        kept+=1
                    append(dest,r)
            if (pi+1)%16==0:print(args.split,'sources',pi+1,'/',len(prefixes),'paired',kept,flush=True)
    dump(args.out/(args.split+'_collection.json'),dict(passed=True,status_counts=counts,pairs_sha256=sha(path),
        collector_sha256=sha(Path(__file__)),locked_model_sha256=sha(args.out/'locked_model.json') if args.split=='test' else None,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','collect'])
    ap.add_argument('--split',choices=['train','dev','test'])
    ap.add_argument('--out',type=Path,default=ROOT/'results/quality-011')
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);globals()[args.action](args)


if __name__=='__main__':main()
