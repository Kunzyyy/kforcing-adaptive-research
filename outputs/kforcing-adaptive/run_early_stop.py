"""Stage 10: disjoint budget calibration and frozen early-stop evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from collect_benefit import dump,verify_checkpoints
from confirm_local_gain import read,rows,sha,prior_sources
from core import load_model
from early_stop_decoder import generate,controller_on_device
from run_frozen_online import append

ROOT=Path(__file__).resolve().parent
SEEDS=[1009,1013]
METHODS=['fixed2','fixed3','fixed4','learned','random_low','random_cal','random_high','learned_replay']


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    assert not (args.out/'prefixes.json').exists()
    old,seen=prior_sources(args.out.parent)
    old+=read(args.out.parent/'online-009/prefixes.json')
    old += [dict(p,ids=p['reference_ids'][:6]) for p in read(args.out.parent/'confirm-008/sources.json')]
    seen.update(tuple(p['ids'][:6]) for p in old)
    seen_rows={p['source_row'] for p in old}
    seen_hash={p['source_sha256'] for p in old}
    dataset=Path('work/lm1b-data/test.parquet')
    assert sha(dataset)=='d3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts=pq.read_table(dataset,columns=['text'])['text'].to_pylist()
    indices=list(range(len(texts)));random.Random(10102001).shuffle(indices)
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    chosen=[]
    for index in indices:
        digest=hashlib.sha256(texts[index].encode()).hexdigest()
        if index in seen_rows or digest in seen_hash:
            continue
        ids=bert.encode(texts[index],add_special_tokens=True,truncation=True,max_length=128)
        if len(ids)<12 or tuple(ids[:6]) in seen:
            continue
        i=len(chosen)
        chosen.append(dict(prompt_id=f's10-{i:03}',index=i,ids=ids[:6],reference_ids=ids,
                           source_row=index,source_sha256=digest,split='calibration' if i<32 else 'test'))
        seen.add(tuple(ids[:6]));seen_hash.add(digest)
        if len(chosen)==128:
            break
    assert len(chosen)==128
    dump(args.out/'prefixes.json',chosen)
    for src,dst in [('locked_model.json','frozen_model.json'),('projection.npy','projection.npy')]:
        shutil.copyfile(args.out.parent/'learned-007'/src,args.out/dst)
    dump(args.out/'manifest.json',dict(calibration_sources=32,test_sources=96,seeds=SEEDS,methods=METHODS,
        test_samples=1536,timing_runs=4608,prefixes_sha256=sha(args.out/'prefixes.json'),
        model_sha256=sha(args.out/'frozen_model.json'),projection_sha256=sha(args.out/'projection.npy'),
        protocol_sha256=sha(ROOT/'STAGE10_PROTOCOL.md'),feature_code_sha256=sha(ROOT/'learn_local_gain.py'),
        dataset_sha256=sha(dataset)))
    print('Frozen 32 calibration and 96 test sources.',flush=True)


def setup(args):
    m=read(args.out/'manifest.json')
    for f,k in [('prefixes.json','prefixes_sha256'),('frozen_model.json','model_sha256'),('projection.npy','projection_sha256')]:
        assert sha(args.out/f)==m[k]
    assert m['model_sha256']=='caeeec7c2327e2473dbbcac6d64fa596e7db70d74a8d780035cb06a8bc991aae'
    assert sha(ROOT/'STAGE10_PROTOCOL.md')==m['protocol_sha256']
    assert sha(ROOT/'learn_local_gain.py')==m['feature_code_sha256']
    verify_checkpoints(args)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    controller=controller_on_device(read(args.out/'frozen_model.json'),np.load(args.out/'projection.npy'),'cuda')
    return model,controller


def schedule_of(r):
    schedule=[False]*30
    for s in r['steps']:
        if s['width']==4:
            schedule[(s['position']-6)//4]=s['split']
    return schedule


def eos_unit_cases():
    class Mock:
        mask_index=103
        def __init__(self,answers):
            self.answers=answers;self.calls=0
        def __call__(self,context,noise,tau,**kwargs):
            ids=self.answers[self.calls];self.calls+=1
            assert len(ids)==noise.shape[1]
            logits=torch.full((1,len(ids),200),-100.)
            for j,t in enumerate(ids):logits[0,j,t]=100.
            cache=[(torch.zeros(1,context.shape[1],1,1),torch.zeros(1,context.shape[1],1,1))]
            return logits,cache
    prefix=torch.tensor([[101,10,11,12,13,14]])
    head=Mock([[11,102,12,13]])
    r=generate(head,prefix,'replay',max_new=8,schedule=[True,True])
    assert r['visible_new_tokens']==2 and head.calls==1 and not r['steps'][0]['eligible']
    removed=Mock([[11,12,102,14],[13,14],[102,15,16,17]])
    r=generate(removed,prefix,'replay',max_new=8,schedule=[True,False])
    assert r['ids'][6:]==[11,12,13,14,102] and removed.calls==3
    tail=Mock([[11,12,13,14],[102,15]])
    r=generate(tail,prefix,'replay',max_new=8,schedule=[True,True])
    assert r['ids'][6:]==[11,12,102] and tail.calls==2
    partial=Mock([[11,102,13]])
    r=generate(partial,prefix,'replay',max_new=3,schedule=[])
    assert r['visible_new_tokens']==2 and partial.calls==1
    return dict(head_eos_skips_decision_and_tail=True,discarded_tail_eos_does_not_stop=True,
                recomputed_eos_stops=True,partial_final_window=True)


@torch.inference_mode()
def preflight(args):
    checks=eos_unit_cases()
    model,controller=setup(args)
    old=rows(args.out.parent/'online-009/samples.jsonl')
    old=[r for r in old if r['prompt_id'] in [f's9-{i:03}' for i in range(4)] and r['method'] in ('fixed2','fixed3','fixed4','learned')]
    for r in old:
        prefix=torch.tensor([r['ids'][:6]],device='cuda')
        x=generate(model,prefix,r['method'],controller,seed=r['noise_seed'])
        assert x['ids']==r['visible_ids'],(r['prompt_id'],r['seed'],r['method'])
        if r['method']=='learned':
            replay=generate(model,prefix,'replay',seed=r['noise_seed'],schedule=schedule_of(x))
            assert x['ids']==replay['ids']
    for r in [r for r in old if r['method']=='learned'][:3]:
        prefix=torch.tensor([r['ids'][:6]],device='cuda')
        a=generate(model,prefix,'learned',controller,seed=r['noise_seed'])
        b=generate(model,prefix,'learned',controller,seed=r['noise_seed'],use_cache=False)
        assert a['ids']==b['ids'] and [s['split'] for s in a['steps']]==[s['split'] for s in b['steps']]
    dump(args.out/'preflight.json',dict(passed=True,synthetic_eos_cases=checks,old_visible_outputs=len(old),
        replay_equivalence_cases=8,cache_recompute_cases=3,decoder_sha256=sha(ROOT/'early_stop_decoder.py')))
    print('Preflight passed: four EOS edge cases, 32 old outputs, eight replays, three cache cases.',flush=True)


@torch.inference_mode()
def calibrate(args):
    assert not (args.out/'calibration.jsonl').exists() and not (args.out/'samples.jsonl').exists()
    assert read(args.out/'preflight.json')['passed']
    model,controller=setup(args)
    assert sha(ROOT/'early_stop_decoder.py')==read(args.out/'preflight.json')['decoder_sha256']
    prefixes=[p for p in read(args.out/'prefixes.json') if p['split']=='calibration']
    candidates=[('learned',None)]+[(f'random_{i:02}',i/10) for i in range(11)]
    counts={name:dict(calls=0,tokens=0,candidates=0) for name,p in candidates}
    with (args.out/'calibration.jsonl').open('w',encoding='utf-8') as dest:
        for p in prefixes:
            prefix=torch.tensor([p['ids']],device='cuda')
            for seed in SEEDS:
                noise_seed=101000000+seed+p['index']*100003
                for name,probability in candidates:
                    r=generate(model,prefix,'learned' if name=='learned' else 'random',controller,
                               seed=noise_seed,probability=probability)
                    r.update(prompt_id=p['prompt_id'],seed=seed,noise_seed=noise_seed,method=name,probability=probability)
                    append(dest,r)
                    c=counts[name];c['calls']+=r['forward_calls'];c['tokens']+=r['visible_new_tokens'];c['candidates']+=r['computed_candidates']
            if (p['index']+1)%4==0:print('Calibration sources',p['index']+1,'/32',flush=True)
    target=counts['learned']['calls']/counts['learned']['tokens']
    ranking=[]
    for name,probability in candidates[1:]:
        cost=counts[name]['calls']/counts[name]['tokens']
        ranking.append(dict(method=name,probability=probability,calls_per_token=cost,
                            absolute_log_cost_ratio=abs(math.log(cost/target))))
    choice=min(ranking,key=lambda r:(r['absolute_log_cost_ratio'],r['probability']))
    probability=choice['probability']
    dump(args.out/'locked_calibration.json',dict(learned_calls_per_token=target,counts=counts,ranking=ranking,
        chosen=choice,probabilities=dict(random_low=round(max(0,probability-.1),1),random_cal=probability,
                                       random_high=round(min(1,probability+.1),1)),
        calibration_sha256=sha(args.out/'calibration.jsonl'),prefixes_sha256=sha(args.out/'prefixes.json'),
        decoder_sha256=sha(ROOT/'early_stop_decoder.py'),runner_sha256=sha(Path(__file__)),
        quality_labels_used=False,test_generated=False))
    print('Locked random probabilities:',read(args.out/'locked_calibration.json')['probabilities'],flush=True)


@torch.inference_mode()
def run(args):
    from transformers import BertTokenizerFast
    assert not (args.out/'samples.jsonl').exists()
    lock=read(args.out/'locked_calibration.json')
    assert sha(args.out/'calibration.jsonl')==lock['calibration_sha256']
    assert sha(ROOT/'early_stop_decoder.py')==lock['decoder_sha256']
    assert sha(Path(__file__))==lock['runner_sha256']
    model,controller=setup(args)
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    prefixes=[p for p in read(args.out/'prefixes.json') if p['split']=='test']
    warm=torch.tensor([read(args.out.parent/'confirm-008/contexts.json')[0]['ids']],device='cuda')
    for policy in ('fixed2','fixed3','fixed4','learned','random','replay'):
        generate(model,warm,policy,controller,seed=1010123,probability=.5,schedule=[False]*30)
    dump(args.out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        precision='fp32',threads=4,tf32=False,decoder_sha256=sha(ROOT/'early_stop_decoder.py'),
        runner_sha256=sha(Path(__file__)),locked_calibration_sha256=sha(args.out/'locked_calibration.json')))
    rng=random.Random(10102005);done=0
    with (args.out/'samples.jsonl').open('w',encoding='utf-8') as dest, \
         (args.out/'timings.jsonl').open('w',encoding='utf-8') as timing, \
         (args.out/'replay_discovery.jsonl').open('w',encoding='utf-8') as discovery:
        for p in prefixes:
            prefix=torch.tensor([p['ids']],device='cuda')
            for seed in SEEDS:
                noise_seed=101000000+seed+p['index']*100003
                pilot=generate(model,prefix,'learned',controller,seed=noise_seed)
                schedule=schedule_of(pilot)
                append(discovery,dict(prompt_id=p['prompt_id'],seed=seed,noise_seed=noise_seed,
                    ids=pilot['ids'],schedule=schedule,discovery_seconds=pilot['seconds']))
                first={}
                for rep in range(3):
                    order=METHODS.copy();rng.shuffle(order)
                    for oi,name in enumerate(order):
                        policy='random' if name.startswith('random_') else ('replay' if name=='learned_replay' else name)
                        probability=lock['probabilities'].get(name)
                        r=generate(model,prefix,policy,controller,seed=noise_seed,probability=probability,
                                   schedule=schedule if policy=='replay' else None)
                        if name in ('learned','learned_replay'):assert r['ids']==pilot['ids']
                        if rep:
                            for key in ('ids','steps','forward_calls','computed_candidates'):
                                assert r[key]==first[name][key],(p['prompt_id'],seed,name,key)
                        else:
                            first[name]=r
                            r.update(prompt_id=p['prompt_id'],seed=seed,noise_seed=noise_seed,method=name,
                                probability=probability,completion=bert.decode(r['ids'][6:],skip_special_tokens=True,
                                                                             clean_up_tokenization_spaces=True))
                            append(dest,r)
                        append(timing,dict(prompt_id=p['prompt_id'],seed=seed,method=name,repetition=rep,order_index=oi,
                            seconds=r['seconds'],ids_sha256=hashlib.sha256(json.dumps(r['ids']).encode()).hexdigest()))
                done+=1
                if done%16==0:print('Early-stop requests',done,'/192; outputs',done*8,flush=True)
    dump(args.out/'generation_checks.json',dict(passed=True,outputs=1536,timing_runs=4608,
        repeated_tokens_and_decisions_identical=True,learned_replay_tokens_identical=True,
        locked_calibration_sha256=sha(args.out/'locked_calibration.json')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','calibrate','run'])
    ap.add_argument('--out',type=Path,default=ROOT/'results/early-010')
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);globals()[args.action](args)


if __name__=='__main__':main()
