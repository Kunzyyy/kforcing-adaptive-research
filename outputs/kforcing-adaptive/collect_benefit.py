"""Fresh article-separated local benefit data; see STAGE2_PROTOCOL.md."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import BertTokenizerFast
from core import generate, load_model, teacher_metrics
from run_pilot import pretrained_checks
from models.transformer import HELPER_BACKEND

FEATURES=['margin_mean','margin_min','margin_last','entropy_mean','entropy_last',
          'top1_mean','top1_last','block_unique_fraction','context_length',
          'margin_change','entropy_change','has_earlier_block']


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def articles(path):
    groups=defaultdict(list)
    article=None
    for text in pq.read_table(path,columns=['text'])['text'].to_pylist():
        s=text.strip()
        if re.match(r'^= [^=].*[^=] =$',s):
            article=hashlib.sha256(s.encode()).hexdigest()
        elif article and not s.startswith('=') and len(s.split())>=20:
            groups[article].append(s)
    return groups


def prepare(args):
    tokenizer=BertTokenizerFast(vocab_file=str(args.data/'vocab.txt'),do_lower_case=True)
    old=json.loads((args.old/'prefixes.json').read_text(encoding='utf-8'))
    old_rows=old['validation']+old['test']
    seen={tuple(p['ids']) for p in old_rows}
    seen_text={p['source_sha256'] for p in old_rows}
    val,test=articles(args.data/'validation.parquet'),articles(args.data/'test.parquet')
    assert not set(val)&set(test),'Article overlap across source splits'
    keys=sorted(val)
    random.Random(92841).shuffle(keys)
    split=int(.75*len(keys))
    sources={'train':{k:val[k] for k in keys[:split]},'dev':{k:val[k] for k in keys[split:]},'test':test}
    result={}
    for index,(name,groups) in enumerate(sources.items()):
        candidates=[]
        for article,texts in sorted(groups.items()):
            local=[]
            for text in texts:
                digest=hashlib.sha256(text.encode()).hexdigest()
                ids=[101]+tokenizer.encode(text,add_special_tokens=False)[:5]
                if len(ids)==6 and tuple(ids) not in seen and digest not in seen_text:
                    local.append(dict(article_id=article,source_sha256=digest,ids=ids,prefix=tokenizer.decode(ids)))
            random.Random(article).shuffle(local)
            candidates.extend(local[:4])
        random.Random(86410+index).shuffle(candidates)
        count={'train':96,'dev':32,'test':48}[name]
        chosen=[]
        for p in candidates:
            if tuple(p['ids']) in seen or p['source_sha256'] in seen_text:
                continue
            seen.add(tuple(p['ids']))
            seen_text.add(p['source_sha256'])
            p.update(id=f'{name}-{len(chosen):03}',split=name)
            chosen.append(p)
            if len(chosen)==count:
                break
        if len(chosen)!=count:
            raise RuntimeError(f'Insufficient {name} paragraphs: {len(chosen)} of {count}; article groups={len(groups)}')
        result[name]=chosen
    dump(args.out/'prefixes.json',result)
    return result


def features(logits,ids,context_length,previous):
    x=logits.float()[0]
    top=x.topk(2,dim=-1).values
    margin=top[:,0]-top[:,1]
    logp=x.log_softmax(-1)
    prob=logp.exp()
    entropy=-(prob*logp).sum(-1)
    top1=prob.max(-1).values
    mean_m,mean_e=float(margin.mean()),float(entropy.mean())
    values=[mean_m,float(margin.min()),float(margin[-1]),mean_e,float(entropy[-1]),
            float(top1.mean()),float(top1[-1]),len(set(ids))/len(ids),context_length,
            mean_m-previous['margin_mean'] if previous else 0.,
            mean_e-previous['entropy_mean'] if previous else 0.,float(previous is not None)]
    return dict(zip(FEATURES,values))


def verify_checkpoints(args):
    for item in json.loads((Path(__file__).parent/'checkpoint_manifest.json').read_text()):
        h=hashlib.sha256()
        with (args.checkpoints/item['filename']).open('rb') as f:
            while block:=f.read(8*1024*1024):
                h.update(block)
        assert h.hexdigest()==item['sha256'],'Checkpoint hash mismatch'


@torch.inference_mode()
def collect(args):
    if (args.out/'branches.jsonl').exists():
        raise RuntimeError('Use a new output directory')
    prefixes=prepare(args)
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    first=torch.tensor([prefixes['train'][0]['ids']],device='cuda')
    dump(args.out/'pretrained_checks.json',pretrained_checks(model,first))
    dump(args.out/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,python=sys.version,precision='fp32',helper_backend=HELPER_BACKEND,
        feature_names=FEATURES,trajectory_seed=31713,branch_seeds=[1121,2213],frequency_penalty=0.,
        protocol_sha256=hashlib.sha256((Path(__file__).parent/'STAGE2_PROTOCOL.md').read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    stats=defaultdict(int)
    started=time.perf_counter()
    with (args.out/'branches.jsonl').open('w',encoding='utf-8') as dest:
        for split,prompts in prefixes.items():
            for p in prompts:
                context=torch.tensor([p['ids']],device='cuda')
                noise=torch.rand(1,8,1,generator=torch.Generator().manual_seed(31713)).to('cuda')
                previous=None
                for state_index in range(2):
                    logits=model(context,noise[:,state_index*4:(state_index+1)*4],torch.ones(1,1,1,device='cuda'),mode='inference')
                    block=logits.argmax(-1)
                    ids=block[0].cpu().tolist()
                    if 102 in ids:
                        stats[split+'_trajectory_eos']+=1
                        break
                    context=torch.cat((context,block),dim=1)
                    f=features(logits,ids,context.shape[1],previous)
                    previous=f
                    for seed in (1121,2213):
                        common=dict(seed=seed,max_new=4,precision='fp32',frequency_penalty=0.,stop_eos=False)
                        a=generate(model,context,policy='fixed4',**common)
                        b=generate(model,context,policy='fixed2',**common)
                        ta,tb=a['ids'][-4:],b['ids'][-4:]
                        row=dict(split=split,prompt_id=p['id'],article_id=p['article_id'],state_index=state_index,
                            seed=seed,features=f,context_ids=context[0].cpu().tolist(),tokens4=ta,tokens22=tb,
                            first_two_match=ta[:2]==tb[:2],eos_excluded=102 in ta or 102 in tb)
                        dest.write(json.dumps(row,allow_nan=False)+'\n')
                        stats[split+'_branches']+=1
                    dest.flush()
                stats[split+'_prompts']+=1
                if stats[split+'_prompts']%16==0:
                    print(f'{split}: {stats[split+"_prompts"]}/{len(prompts)} prompts; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    dump(args.out/'collection_summary.json',dict(stats=stats,elapsed_seconds=time.perf_counter()-started))
    print('Collection complete',dict(stats),flush=True)


def score(args):
    if (args.out/'scored.jsonl').exists():
        raise RuntimeError('Scored file exists; do not overwrite')
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    rows=[json.loads(line) for line in (args.out/'branches.jsonl').read_text().splitlines()]
    valid=0
    with (args.out/'scored.jsonl').open('w',encoding='utf-8') as dest:
        for i,r in enumerate(rows):
            if not r['eos_excluded'] and r['first_two_match']:
                context=r['context_ids']
                a=teacher_metrics(teacher,context+r['tokens4'],len(context))
                b=teacher_metrics(teacher,context+r['tokens22'],len(context))
                r.update(nll4=a['teacher_nll'],nll22=b['teacher_nll'],gain=a['teacher_nll']-b['teacher_nll'])
                valid+=1
            dest.write(json.dumps(r,allow_nan=False)+'\n')
            if (i+1)%100==0:
                dest.flush()
                print(f'Scored {i+1}/{len(rows)} branches',flush=True)
    dump(args.out/'scoring_summary.json',dict(branches=len(rows),valid_branches=valid,helper_backend=HELPER_BACKEND))
    print('Scoring complete',len(rows),valid,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['collect','score'])
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path('work/pilot-data'))
    ap.add_argument('--old',type=Path,default=Path(__file__).parent/'results/pilot-001')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    verify_checkpoints(args)
    (collect if args.mode=='collect' else score)(args)


if __name__=='__main__':
    main()
