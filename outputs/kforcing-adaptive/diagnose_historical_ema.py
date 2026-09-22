"""Fixed 128-source retrospective comparison of released versus historical EMA weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from core import load_model

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/historical-ema'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8*1024*1024): h.update(data)
    return h.hexdigest()
def append(f,x): f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()


def freeze():
    OUT.mkdir(exist_ok=False)
    prefixes=read(ROOT/'results/baseline-016/prefixes.json')[:128]
    dump(OUT/'prefixes.json',prefixes)
    files=['HISTORICAL_EMA_PLAN.md','diagnose_historical_ema.py','extract_historical_ema.py',
           'checkpoint_pickle_metadata.py','core.py','external_gpt2_score.py','audit_official.py',
           'upstream/models/pflm.py','upstream/models/transformer.py','upstream/models/autoregressive.py',
           'upstream/batch_inference_with_prefix.py','results/baseline-016/samples.jsonl',
           'results/baseline-016/scores.jsonl','results/baseline-016/prefixes.json']
    extractions={m:read(ROOT/f'results/public-history/historical_{m}/ema_extraction.json') for m in ['ar','pflm']}
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=128,
        input_code_hashes={f:sha(ROOT/f) for f in files},prefixes_sha256=sha(OUT/'prefixes.json'),
        extraction_manifest_hashes={m:sha(ROOT/f'results/public-history/historical_{m}/ema_extraction.json') for m in extractions},
        ema_checkpoint_hashes={m:v['checkpoint_sha256'] for m,v in extractions.items()},
        ordinary_checkpoint_hashes={m:v['current_checkpoint_sha256'] for m,v in extractions.items()},
        tokenizer_sha256=sha(Path('work/pilot-data/vocab.txt')),
        evaluator_manifest_sha256=sha(Path('work/gpt2-large/manifest.json')),
        retrospective=True,new_test=False,bootstrap_seed=21212026,bootstrap_repeats=4000,
        mapping_status='Shape/order verified against published model; original EMA training parameter order unconfirmed.'))
    print('Frozen 128 old sources, historical EMA hashes and fixed generation/scoring protocol.',flush=True)


def verify():
    m=read(OUT/'manifest.json')
    for f,h in m['input_code_hashes'].items(): assert sha(ROOT/f)==h,f
    assert sha(OUT/'prefixes.json')==m['prefixes_sha256']
    assert sha(Path('work/pilot-data/vocab.txt'))==m['tokenizer_sha256']
    assert sha(Path('work/gpt2-large/manifest.json'))==m['evaluator_manifest_sha256']
    for model,h in m['ema_checkpoint_hashes'].items(): assert sha(Path('work/historical-ema')/model/'ema.ckpt')==h
    for model,h in m['ordinary_checkpoint_hashes'].items():
        assert sha(Path('work/checkpoints')/('ar_best_lm1b.ckpt' if model=='ar' else 'pflm_lm1b_k4.ckpt'))==h
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    return m


@torch.inference_mode()
def generate():
    from transformers import BertTokenizerFast
    from audit_official import official_cli,quiet_sampler
    verify(); assert not (OUT/'samples.jsonl').exists()
    official=official_cli(); quiet_sampler()
    tokenizer=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    prefixes=read(OUT/'prefixes.json'); wanted={p['prompt_id'] for p in prefixes}
    old=[r for r in rows(ROOT/'results/baseline-016/samples.jsonl') if r['prompt_id'] in wanted]
    lookup={(r['prompt_id'],r['method']):r for r in old}
    ema={m:load_model(Path('work/historical-ema')/m/'ema.ckpt',m,'cuda') for m in ['ar','pflm']}
    replay=[]
    context=torch.tensor([p['ids'] for p in prefixes[:4]],device='cuda')
    for model,filename,method in [('ar','ar_best_lm1b.ckpt','ar'),('pflm','pflm_lm1b_k4.ckpt','fixed4')]:
        baseline=load_model(Path('work/checkpoints')/filename,model,'cuda')
        torch.manual_seed(16120000)
        tokens=(official.generate_ar_cached(baseline,context,122) if model=='ar' else
                baseline.sample_next_k_tokens_with_kv_caches(context,torch.ones(4,device='cuda'),122,k=4,frequency_penalty=0.))
        for p,ids in zip(prefixes[:4],tokens.tolist()):
            assert ids==lookup[p['prompt_id'],method]['full_ids']
            replay.append(dict(prompt_id=p['prompt_id'],method=method,full_128_tokens_match=True))
        del baseline; torch.cuda.empty_cache()
    with (OUT/'samples.jsonl').open('w',encoding='utf-8') as f:
        for r in old: append(f,dict(r,weights='ordinary',cohort='old_control'))
        for start in range(0,128,4):
            batch=prefixes[start:start+4]; context=torch.tensor([p['ids'] for p in batch],device='cuda')
            seed=16120000+(start//4)*101
            for model,method in [('ar','ar'),('pflm','fixed4')]:
                torch.manual_seed(seed)
                full=(official.generate_ar_cached(ema[model],context,122) if model=='ar' else
                      ema[model].sample_next_k_tokens_with_kv_caches(context,torch.ones(4,device='cuda'),122,k=4,frequency_penalty=0.))
                for p,sequence in zip(batch,full.tolist()):
                    assert len(sequence)==128
                    ids=official.truncate_at_eos(sequence,102,6)
                    text=tokenizer.decode(ids[6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    append(f,dict(prompt_id=p['prompt_id'],method=method,weights='historical_ema',cohort='diagnostic',
                        sampler_seed=seed,batch_index=start//4,prefix_length=6,full_ids=sequence,ids=ids,
                        new_tokens=len(ids)-6,stopped_eos=ids[-1]==102,completion=text))
            if (start+4)%16==0: print('EMA diagnostic sources',start+4,'/128, both models',flush=True)
    dump(OUT/'generation_checks.json',dict(passed=True,samples=512,ema_generated=256,ordinary_reused=256,
        ordinary_replay=replay,samples_sha256=sha(OUT/'samples.jsonl'),manifest_sha256=sha(OUT/'manifest.json'),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),timing_claim=False))


@torch.inference_mode()
def score():
    from transformers import GPT2TokenizerFast,GPT2LMHeadModel
    from external_gpt2_score import masked_token_nll
    verify(); assert not (OUT/'scores.jsonl').exists()
    assert read(OUT/'generation_checks.json')['samples_sha256']==sha(OUT/'samples.jsonl')
    path=Path('work/gpt2-large'); manifest=read(path/'manifest.json')
    for item in manifest['files']: assert sha(path/item['name'])==item['sha256']
    tok=GPT2TokenizerFast.from_pretrained(str(path),local_files_only=True)
    model=GPT2LMHeadModel.from_pretrained(str(path),local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    probe=tok.encode('An independent reference can help locate implementation differences.',return_tensors='pt').cuda()
    builtin=model(probe,labels=probe,use_cache=False)
    sums,counts=masked_token_nll(builtin.logits,probe,torch.ones_like(probe))
    torch.testing.assert_close(sums/counts,builtin.loss.reshape(1),atol=1e-6,rtol=1e-6)
    data=rows(OUT/'samples.jsonl')
    old={(r['prompt_id'],r['method']):r for r in rows(ROOT/'results/baseline-016/scores.jsonl') if r['cohort']=='new'}
    differences=[]
    with (OUT/'scores.jsonl').open('w',encoding='utf-8') as f:
        for start in range(0,len(data),4):
            batch=data[start:start+4]; tokens=[tok.encode(r['completion'],add_special_tokens=False) for r in batch]
            assert max(map(len,tokens))<=1024
            width=max(2,max(map(len,tokens)))
            x=torch.full((len(batch),width),tok.eos_token_id,dtype=torch.long,device='cuda'); mask=torch.zeros_like(x)
            for j,t in enumerate(tokens):
                if t: x[j,:len(t)]=torch.tensor(t,device='cuda'); mask[j,:len(t)]=1
                else: mask[j,0]=1
            sums,counts=masked_token_nll(model(x,attention_mask=mask,use_cache=False).logits,x,mask)
            for j,r in enumerate(batch):
                item=dict(prompt_id=r['prompt_id'],method=r['method'],weights=r['weights'],
                    gpt2_input_tokens=len(tokens[j]),gpt2_scored_tokens=int(counts[j]),
                    gpt2_nll_sum=float(sums[j]) if counts[j] else 0.,excluded_short=not bool(counts[j]))
                if r['weights']=='ordinary':
                    control=old[r['prompt_id'],r['method']]
                    assert item['gpt2_scored_tokens']==control['gpt2_scored_tokens']
                    assert math.isclose(item['gpt2_nll_sum'],control['gpt2_nll_sum'],rel_tol=1e-5,abs_tol=.005)
                    differences.append(abs(item['gpt2_nll_sum']-control['gpt2_nll_sum']))
                append(f,item)
            if (start+4)%64==0: print('EMA diagnostic scores',start+4,'/512',flush=True)
    dump(OUT/'scoring_checks.json',dict(passed=True,records=512,old_control_records=len(differences),
        max_old_nll_abs_difference=max(differences),builtin_loss_crosscheck=True,
        scores_sha256=sha(OUT/'scores.jsonl'),samples_sha256=sha(OUT/'samples.jsonl')))


def analyze():
    m=verify(); assert not (OUT/'analysis.json').exists()
    data=rows(OUT/'scores.jsonl'); samples=rows(OUT/'samples.jsonl')
    ids=[r['prompt_id'] for r in read(OUT/'prefixes.json')]
    index=np.random.default_rng(m['bootstrap_seed']).integers(0,len(ids),(m['bootstrap_repeats'],len(ids)))
    methods={}
    for method in ['ar','fixed4']:
        groups={}; arrays={}
        for weight in ['ordinary','historical_ema']:
            records={r['prompt_id']:r for r in data if r['method']==method and r['weights']==weight}
            assert set(records)==set(ids)
            nll=np.array([records[i]['gpt2_nll_sum'] for i in ids]); counts=np.array([records[i]['gpt2_scored_tokens'] for i in ids])
            avg=float(nll.sum()/counts.sum()); bootstrap=nll[index].sum(1)/counts[index].sum(1)
            arrays[weight]=(avg,bootstrap)
            subset=[r for r in samples if r['method']==method and r['weights']==weight]
            groups[weight]=dict(gen_ppl=math.exp(avg),mean_token_nll=avg,scored_tokens=int(counts.sum()),
                gen_ppl_interval95=np.exp(np.quantile(bootstrap,[.025,.975])).tolist(),
                excluded_short=sum(r['excluded_short'] for r in records.values()),
                mean_bert_new_tokens=sum(r['new_tokens'] for r in subset)/len(subset),
                stopped_eos=sum(r['stopped_eos'] for r in subset))
        delta=arrays['historical_ema'][0]-arrays['ordinary'][0]
        bs=arrays['historical_ema'][1]-arrays['ordinary'][1]
        groups['ema_minus_ordinary_token_nll']=dict(estimate=delta,interval95=np.quantile(bs,[.025,.975]).tolist())
        groups['ema_over_ordinary_ppl']=dict(estimate=math.exp(delta),interval95=np.exp(np.quantile(bs,[.025,.975])).tolist())
        methods[method]=groups
    result=dict(sources=128,retrospective=True,new_adaptive_test=False,methods=methods,
        bootstrap_seed=m['bootstrap_seed'],bootstrap_repeats=m['bootstrap_repeats'],
        manifest_sha256=sha(OUT/'manifest.json'),scores_sha256=sha(OUT/'scores.jsonl'),
        paper_baseline_reproduced=False,ema_parameter_mapping_confirmed_by_author=False,
        interpretation='Paired old-source diagnostic of a positional historical EMA reconstruction; no paper replication, new adaptive performance, or original training order confirmation.')
    dump(OUT/'analysis.json',result); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['freeze','generate','score','analyze'])
    args=ap.parse_args(); globals()[args.mode]()
