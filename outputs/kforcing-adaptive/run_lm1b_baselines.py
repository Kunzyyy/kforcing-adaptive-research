"""Released AR/PFLM sampling, fixed 122-token work, BS=4/16, local hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from transformers import BertTokenizerFast
from core import load_model,teacher_metrics
from collect_benefit import verify_checkpoints,dump
from audit_official import official_cli,quiet_sampler
from models.transformer import HELPER_BACKEND


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    ap.add_argument('--vocab',type=Path,default=Path('work/pilot-data/vocab.txt'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if (args.out/'batch_timings.jsonl').exists():
        raise RuntimeError('Baseline run already exists')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    verify_checkpoints(args)
    official=official_cli()
    quiet_sampler()
    prompts=json.loads((args.out/'prefixes.json').read_text(encoding='utf-8'))
    tokenizer=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=True)
    ar=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    pflm=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    methods=['ar','fixed2','fixed3','fixed4']
    def sample(method,prefix):
        if method=='ar':
            return official.generate_ar_cached(ar,prefix,122)
        k=int(method[-1])
        pflm.max_k=k
        return pflm.sample_next_k_tokens_with_kv_caches(prefix,torch.ones(prefix.shape[0],device='cuda'),122,k=k,frequency_penalty=0.)
    env=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        helper_backend=HELPER_BACKEND,ar_precision='released internal fp16 autocast',
        pflm_precision='released CLI: no outer autocast; fp32 parameters',batch_sizes=[4,16],
        timing_seed=617,timing_repeats=3,quality_seeds=[617,619],quality_batch_size=4,
        prompts=len(prompts),prefix_length=6,fixed_new_tokens=122,eos_stops_computation=False,
        freq_penalty=0.,tau=1.,timing='Synchronized decoder-call wall time; released tqdm/loguru suppressed. Data transfer, text decoding and file output excluded.',
        original_cli_note='Published CLI total-throughput includes JSON serialization; our decoder-call times match its per-batch timing scope.',
        limitation='Local RTX/SDPA/CLI precision, fewer prefixes/repeats than paper; not an H100/bf16/FlashAttention reproduction.')
    dump(args.out/'environment.json',env)
    for bs in (4,16):
        prefix=torch.tensor([p['ids'] for p in prompts[:bs]],device='cuda')
        for method in methods:
            torch.manual_seed(8000+bs)
            sample(method,prefix)
    print('Official-path warmups complete',flush=True)
    # Three repeated timing passes at seed 617, plus seed 619 quality at BS4 only.
    passes=[(bs,617,rep) for bs in (4,16) for rep in range(3)]+[(4,619,0)]
    order=random.Random(440041)
    order.shuffle(passes)
    timings=[]
    quality=[]
    fingerprints={}
    mismatches=0
    with (args.out/'batch_timings.jsonl').open('w',encoding='utf-8') as dest:
        for bs,seed,rep in passes:
            for bi in range(len(prompts)//bs):
                batch=prompts[bi*bs:(bi+1)*bs]
                prefix=torch.tensor([p['ids'] for p in batch],device='cuda')
                methods_order=methods.copy()
                order.shuffle(methods_order)
                for method in methods_order:
                    sampler_seed=seed+bi*101
                    torch.manual_seed(sampler_seed)
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    result=sample(method,prefix)
                    torch.cuda.synchronize()
                    seconds=time.perf_counter()-start
                    ids=result.cpu().tolist()
                    assert all(len(x)==128 for x in ids)
                    key=(bs,seed,bi,method)
                    fingerprint=digest_ids(ids)
                    if key in fingerprints:
                        mismatches+=fingerprints[key]!=fingerprint
                    else:
                        fingerprints[key]=fingerprint
                    row=dict(batch_size=bs,seed=seed,sampler_seed=sampler_seed,repeat=rep,batch_index=bi,
                        method=method,seconds=seconds,fixed_generated_positions=bs*122,output_sha256=fingerprint)
                    dest.write(json.dumps(row)+'\n')
                    timings.append(row)
                    if bs==4 and rep==0:
                        for p,sequence in zip(batch,ids):
                            truncated=official.truncate_at_eos(sequence,102,6)
                            quality.append(dict(prompt_id=p['id'],seed=seed,sampler_seed=sampler_seed,method=method,
                                full_ids=sequence,ids=truncated,prefix_length=6,new_tokens=len(truncated)-6,
                                stopped_eos=truncated[-1]==102,prefix=p['prefix'],
                                completion=tokenizer.decode(truncated[6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)))
                dest.flush()
            print(f'Completed BS={bs}, seed={seed}, repeat={rep}',flush=True)
    assert len(quality)==1024
    with (args.out/'samples.jsonl').open('w',encoding='utf-8') as dest:
        for i,r in enumerate(quality):
            r.update(teacher_metrics(ar,r['ids'],6))
            dest.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            if (i+1)%256==0:
                print('Teacher side scores',i+1,'/',len(quality),flush=True)
    references=[]
    for p in prompts:
        r=dict(prompt_id=p['id'],method='reference',ids=p['reference_ids'],prefix_length=6,
            new_tokens=len(p['reference_ids'])-6,completion=tokenizer.decode(p['reference_ids'][6:],skip_special_tokens=True,clean_up_tokenization_spaces=True))
        r.update(teacher_metrics(ar,r['ids'],6))
        references.append(r)
    dump(args.out/'references.json',references)
    summary={}
    for bs in (4,16):
        for method in methods:
            group=[r for r in timings if r['batch_size']==bs and r['method']==method and r['seed']==617]
            per_repeat=[sum(r['fixed_generated_positions'] for r in group if r['repeat']==rep)/sum(r['seconds'] for r in group if r['repeat']==rep) for rep in range(3)]
            summary[f'{method}_bs{bs}']=dict(mean_fixed_positions_per_second=float(np.mean(per_repeat)),
                repeat_throughputs=per_repeat,total_decoder_seconds=sum(r['seconds'] for r in group))
    dump(args.out/'throughput.json',summary)
    dump(args.out/'run_checks.json',dict(quality_samples=len(quality),batch_timing_records=len(timings),
        repeated_batch_fingerprint_mismatches=mismatches,all_full_sequences_length_128=True))
    print(json.dumps(summary,indent=2),flush=True)
    print('LM1B baseline complete',len(quality),'quality samples',flush=True)


if __name__=='__main__':
    main()
