"""Current-candidate commitment with true feature/waste costs and lean controls."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from transformers import BertTokenizerFast
from core import load_model,teacher_metrics,summarize
from collect_benefit import verify_checkpoints,dump
from commit_decoder import generate_commit
from models.transformer import HELPER_BACKEND


def load_controller(root):
    path=root/'locked_controller.json'
    c=json.loads(path.read_text())
    assert c['passed_development_gate']
    return c,hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(args):
    out=args.root/'online_calibration.json'
    if out.exists():
        raise RuntimeError('Online calibration is already frozen')
    controller,digest=load_controller(args.root)
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    prompts=json.loads((args.root.parent/'benefit-002/prefixes.json').read_text(encoding='utf-8'))['dev']
    rows=[]
    for p in prompts:
        for seed in (401,409):
            r=generate_commit(model,torch.tensor([p['ids']],device='cuda'),controller=controller,seed=seed)
            r.update(prompt_id=p['id'],article_id=p['article_id'])
            rows.append(r)
    eligible=[s for r in rows for s in r['steps'] if s['eligible']]
    short=sum(s['requested_k']==2 for s in eligible)
    report=dict(controller_sha256=digest,threshold=controller['threshold'],prompts=len(prompts),seeds=[401,409],
        eligible_steps=len(eligible),short_steps=short,p_short=short/len(eligible),uses_test=False,
        rule='Match eligible-step short probability observed on development rollouts; threshold unchanged')
    dump(out,report)
    dump(args.root/'online_calibration_traces.json',rows)
    print(json.dumps(report,indent=2),flush=True)


def run(args):
    if (args.root/'online_timings.jsonl').exists():
        raise RuntimeError('Online benchmark exists; use preserved results')
    controller,digest=load_controller(args.root)
    calibration=json.loads((args.root/'online_calibration.json').read_text())
    assert calibration['controller_sha256']==digest
    p_short=calibration['p_short']
    prompts=json.loads((args.root/'fresh_prefixes.json').read_text(encoding='utf-8'))
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    policies=['fixed2','fixed3','fixed4','random_commit','current']
    prefix=torch.tensor([prompts[0]['ids']],device='cuda')
    # Real-checkpoint cached/full recomputation check in the new decision path.
    a=generate_commit(model,prefix,controller=controller,max_new=24,seed=841)
    b=generate_commit(model,prefix,controller=controller,max_new=24,seed=841,use_cache=False)
    assert a['ids']==b['ids']
    dump(args.root/'online_preflight.json',dict(real_checkpoint_cached_uncached_tokens_match=True,
        tested_seed=841,tested_max_new=24,precision='fp32'))
    for policy in policies:
        generate_commit(model,prefix,policy=policy,controller=controller,max_new=16,seed=842)
    metadata=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        python=sys.version,helper_backend=HELPER_BACKEND,precision='fp32',frequency_penalty=0.,
        seeds=[503,509],repetitions=3,prompts=len(prompts),articles=len({p['article_id'] for p in prompts}),
        controller_sha256=digest,random_p_short=p_short,policies=policies,max_new=64,batch_size=1,
        timing='Prefill, forward, current-only features, transfers, policy, KV, EOS, discarded-candidate work included. Model loading, noise preparation and offline scoring excluded.',
        quality='Teacher NLL/repetition/length are proxies. Repeated timings are not independent quality samples.')
    dump(args.root/'online_environment.json',metadata)
    jobs=[(p,seed) for p in prompts for seed in (503,509)]
    ordering=random.Random(914315)
    ordering.shuffle(jobs)
    grouped={}
    with (args.root/'online_timings.jsonl').open('w',encoding='utf-8') as dest:
        for i,(p,seed) in enumerate(jobs):
            order=[(policy,repeat) for policy in policies for repeat in range(3)]
            ordering.shuffle(order)
            for policy,repeat in order:
                r=generate_commit(model,torch.tensor([p['ids']],device='cuda'),policy=policy,
                    controller=controller,seed=seed,p_short=p_short)
                r.update(prompt_id=p['id'],article_id=p['article_id'],repeat=repeat)
                key=(p['id'],seed,policy)
                if key in grouped:
                    assert r['ids']==grouped[key][0]['ids'],'Nonidentical repeated outputs; investigate before averaging'
                grouped.setdefault(key,[]).append(r)
                dest.write(json.dumps(r,allow_nan=False)+'\n')
            dest.flush()
            if (i+1)%8==0:
                print('Online timing',i+1,'/',len(jobs),'prompt/seed groups',flush=True)
    # Teacher is loaded only after all timed generation has finished.
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    tokenizer=BertTokenizerFast(vocab_file=str(args.data/'vocab.txt'),do_lower_case=True)
    merged=[]
    with (args.root/'online_samples.jsonl').open('w',encoding='utf-8') as dest:
        for runs in grouped.values():
            r=dict(runs[0])
            r.pop('repeat')
            r['timing_repetitions']=[v['seconds'] for v in runs]
            r['seconds']=float(np.mean(r['timing_repetitions']))
            r['tokens_per_second']=r['new_tokens']/r['seconds']
            r.update(teacher_metrics(teacher,r['ids'],r['prefix_length']))
            r['text']=tokenizer.decode(r['ids'],skip_special_tokens=False)
            dest.write(json.dumps(r,allow_nan=False)+'\n')
            merged.append(r)
    summary=summarize(merged)
    for policy,stats in summary.items():
        rows=[r for r in merged if r['policy']==policy]
        stats.update(computed_candidates=sum(r['computed_candidates'] for r in rows),
            policy_discarded_candidates=sum(r['policy_discarded'] for r in rows),
            eos_discarded_candidates=sum(r['eos_discarded'] for r in rows),
            candidates_per_emitted_token=sum(r['computed_candidates'] for r in rows)/stats['generated_tokens'])
    dump(args.root/'online_summary.json',summary)
    print('Online benchmark and teacher scoring completed; unique samples',len(merged),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['calibrate','run'])
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path('work/pilot-data'))
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    verify_checkpoints(args)
    (calibrate if args.mode=='calibrate' else run)(args)


if __name__=='__main__':
    main()
