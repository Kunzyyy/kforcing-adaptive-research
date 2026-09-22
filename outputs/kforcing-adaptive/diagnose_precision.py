"""Frozen-noise precision and cache diagnostics on released k=4 sampler."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import BertTokenizerFast
from core import load_model, amp_context
from audit_official import quiet_sampler, official_cli
from collect_benefit import verify_checkpoints, dump
import models.pflm as pflm_module


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def diff(a,b):
    a,b=a.float(),b.float()
    # Mask token's sentinel is not a predictive logit.
    d=(a-b).abs()
    d[...,103]=0
    return dict(max_abs_logit_difference=float(d.max()),mean_abs_logit_difference=float(d.sum()/(d.numel()-a.shape[0]*a.shape[1])),
                argmax_disagreements=int((a.argmax(-1)!=b.argmax(-1)).sum()),positions=a.shape[0]*a.shape[1])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=Path('outputs/kforcing-adaptive/results/lm1b-004'))
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    ap.add_argument('--vocab',type=Path,default=Path('work/pilot-data/vocab.txt'))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'precision_samples.jsonl').exists(),'Preserve recorded outputs'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    verify_checkpoints(args)
    prompts=json.loads((args.source/'prefixes.json').read_text(encoding='utf-8'))
    old={(r['prompt_id'],r['seed']):r for r in read_lines(args.source/'samples.jsonl') if r['method']=='fixed4'}
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    quiet_sampler()
    cli=official_cli()
    bert=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=True)
    tape_records=[]
    matches=0
    with (args.out/'precision_samples.jsonl').open('w',encoding='utf-8') as dest:
        for seed in (617,619):
            for bi in range(32):
                selected=prompts[bi*4:(bi+1)*4]
                prefix=torch.tensor([p['ids'] for p in selected],device='cuda')
                torch.manual_seed(seed+bi*101)
                tape=[torch.rand(4,4,1,device='cuda') for _ in range(31)]
                digest=hashlib.sha256(torch.stack(tape).cpu().numpy().tobytes()).hexdigest()
                tape_records.append(dict(seed=seed,batch_index=bi,sha256=digest,shape=[31,4,4,1]))
                for precision in ('fp32','bf16','fp16'):
                    cursor=0
                    def frozen_noise(*shape,**kwargs):
                        nonlocal cursor
                        assert shape==(4,4,1) and cursor<31
                        result=tape[cursor]
                        cursor+=1
                        return result
                    with patch.object(pflm_module.torch,'rand',side_effect=frozen_noise),amp_context('cuda',precision):
                        generated=model.sample_next_k_tokens_with_kv_caches(prefix,torch.ones(4,device='cuda'),122,k=4,frequency_penalty=0.)
                    assert cursor==31
                    for p,full_ids in zip(selected,generated.cpu().tolist()):
                        if precision=='fp32':
                            assert full_ids==old[p['id'],seed]['full_ids'],'fp32 control failed stage-4 replay'
                            matches+=1
                        ids=cli.truncate_at_eos(full_ids,102,6)
                        r=dict(prompt_id=p['id'],seed=seed,batch_index=bi,precision=precision,method='fixed4_'+precision,
                               full_ids=full_ids,ids=ids,prefix_length=6,new_tokens=len(ids)-6,noise_sha256=digest,
                               completion=bert.decode(ids[6:],skip_special_tokens=True,clean_up_tokenization_spaces=True),
                               full_text=bert.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=True))
                        dest.write(json.dumps(r,ensure_ascii=False)+'\n')
                dest.flush()
                if (bi+1)%8==0:
                    print('Precision generation',seed,bi+1,'/32 batches',flush=True)
    dump(args.out/'noise_manifest.json',tape_records)
    print('All 256 fp32 sequences exactly replay stage 4',flush=True)
    diagnostics=[]
    for pi,p in enumerate(prompts[:32]):
        trajectory=old[p['id'],617]['full_ids']
        cache=None
        for length in (6,10,14,30,70):
            context=torch.tensor([trajectory[:length]],device='cuda')
            noise=torch.rand(1,4,1,generator=torch.Generator().manual_seed(55000+pi*100+length)).to('cuda')
            tau=torch.ones(1,1,1,device='cuda')
            full=model(context,noise,tau,mode='inference')
            cached,cache=model(context,noise,tau,mode='inference',kv_caches=cache,return_kv=True)
            assert all(c[0].shape[1]==length for c in cache)
            r=dict(prompt_id=p['id'],length=length,cache_vs_full=diff(cached,full))
            for precision in ('bf16','fp16'):
                with amp_context('cuda',precision):
                    altered=model(context,noise,tau,mode='inference')
                r[precision+'_vs_fp32']=diff(altered,full)
            if pi<8 and length in (6,10):
                noises=noise[:,None].expand(1,length,4,1).contiguous()
                taus=tau.expand(1,length,1).contiguous()
                train=model(context,noises,taus,mode='train')[:,-1]
                r['train_last_vs_inference']=diff(train,full)
            diagnostics.append(r)
        if (pi+1)%8==0:
            print('Same-context diagnostic',pi+1,'/32 prefixes',flush=True)
    dump(args.out/'logit_diagnostics.json',diagnostics)
    summary={}
    for name in ('cache_vs_full','bf16_vs_fp32','fp16_vs_fp32','train_last_vs_inference'):
        records=[r[name] for r in diagnostics if name in r]
        summary[name]=dict(cases=len(records),positions=sum(r['positions'] for r in records),
                           argmax_disagreements=sum(r['argmax_disagreements'] for r in records),
                           max_abs_logit_difference=max(r['max_abs_logit_difference'] for r in records))
    dump(args.out/'precision_checks.json',dict(fp32_exact_stage4_replays=matches,samples=768,
        noise_batches=len(tape_records),diagnostics=summary,gpu=torch.cuda.get_device_name(),
        precision_note='FP32 weights retained; precision names indicate outer autocast. Explicit fp32 projections and fp64 sinusoidal features remain as released.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
