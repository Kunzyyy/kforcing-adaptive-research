"""Frozen precision diagnostic with explicit, separately reported scoring conventions."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2TokenizerFast,GPT2LMHeadModel,BertTokenizerFast
from external_gpt2_score import masked_token_nll
from collect_benefit import dump
from diagnose_precision import read_lines
from analyze_lm1b_audit import paired


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=Path('outputs/kforcing-adaptive/results/lm1b-004'))
    ap.add_argument('--model',type=Path,default=Path('work/gpt2-large'))
    args=ap.parse_args()
    assert not (args.root/'scores.jsonl').exists(),'Preserve original score records'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((args.model/'manifest.json').read_text())
    for item in manifest['files']:
        h=hashlib.sha256()
        with (args.model/item['name']).open('rb') as src:
            while chunk:=src.read(8*1024*1024):
                h.update(chunk)
        assert h.hexdigest()==item['sha256']
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    tok=GPT2TokenizerFast.from_pretrained(str(args.model),local_files_only=True)
    records=read_lines(args.root/'precision_samples.jsonl')
    for r in read_lines(args.source/'samples.jsonl'):
        if r['method']=='ar':
            r['full_text']=bert.decode(r['ids'],skip_special_tokens=True,clean_up_tokenization_spaces=True)
            records.append(r)
    assert len(records)==1024
    rows=[]
    for r in records:
        for variant in ('completion','full_text','completion_bos'):
            tokens=tok.encode(r['full_text'] if variant=='full_text' else r['completion'],add_special_tokens=False)
            if variant=='completion_bos':
                tokens=[tok.eos_token_id]+tokens
            assert len(tokens)<=1024
            rows.append(dict(prompt_id=r['prompt_id'],seed=r['seed'],method=r['method'],variant=variant,
                             bert_new_tokens=r['new_tokens'],text=r['full_text'] if variant=='full_text' else r['completion'],
                             gpt2_input_tokens=len(tokens),_tokens=tokens))
    model=GPT2LMHeadModel.from_pretrained(str(args.model),local_files_only=True,use_safetensors=True,
                                        torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    print('Scoring',len(rows),'records, 3 explicit conventions',flush=True)
    with (args.root/'scores.jsonl').open('w',encoding='utf-8') as dest:
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]
            width=max(2,max(len(r['_tokens']) for r in batch))
            ids=torch.full((len(batch),width),tok.eos_token_id,device='cuda',dtype=torch.long)
            mask=torch.zeros_like(ids)
            for j,r in enumerate(batch):
                tokens=r['_tokens']
                if tokens:
                    ids[j,:len(tokens)]=torch.tensor(tokens,device='cuda')
                    mask[j,:len(tokens)]=1
                else:
                    mask[j,0]=1
            sums,counts=masked_token_nll(model(ids,attention_mask=mask,use_cache=False).logits,ids,mask)
            for j,r in enumerate(batch):
                r.pop('_tokens')
                r.update(gpt2_scored_tokens=int(counts[j]),gpt2_nll_sum=float(sums[j]) if counts[j] else 0.,excluded_short=not bool(counts[j]))
                dest.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            dest.flush()
            if (start+4)%256==0:
                print('Scored',start+4,'/',len(rows),flush=True)
    groups=defaultdict(list)
    for r in rows:
        groups[r['variant']+'/'+r['method']].append(r)
    summary={}
    for key,group in sorted(groups.items()):
        n=sum(r['gpt2_scored_tokens'] for r in group)
        nll=sum(r['gpt2_nll_sum'] for r in group)/n
        summary[key]=dict(samples=len(group),scored_tokens=n,gpt2_nll=nll,gpt2_gen_ppl=math.exp(nll),
                          excluded_short=sum(r['excluded_short'] for r in group),
                          mean_bert_new_tokens=np.mean([r['bert_new_tokens'] for r in group]))
    previous={(r['prompt_id'],r['seed'],r['method']):r for r in read_lines(args.source/'external_scores.jsonl') if r['dataset']=='lm1b'}
    diffs=[]
    for r in rows:
        if r['variant']=='completion' and r['method'] in ('ar','fixed4_fp32'):
            old=previous[r['prompt_id'],r['seed'],'ar' if r['method']=='ar' else 'fixed4']
            assert r['gpt2_scored_tokens']==old['gpt2_scored_tokens']
            error=abs(r['gpt2_nll_sum']-old['gpt2_nll_sum'])
            assert math.isclose(r['gpt2_nll_sum'],old['gpt2_nll_sum'],rel_tol=1e-5,abs_tol=.005)
            diffs.append(error)
    comparisons={}
    for variant in ('completion','full_text','completion_bos'):
        for a,b in (('fixed4_bf16','fixed4_fp32'),('fixed4_fp16','fixed4_fp32'),('fixed4_fp32','ar')):
            comparisons[variant+'/'+a+'_vs_'+b]=paired(groups[variant+'/'+a],groups[variant+'/'+b])
    dump(args.root/'score_summary.json',summary)
    dump(args.root/'comparisons.json',comparisons)
    dump(args.root/'scoring_checks.json',dict(model_revision=manifest['revision'],model_sha256_verified=True,
        previous_score_checks=len(diffs),max_previous_nll_sum_abs_difference=max(diffs),
        rows=len(rows),conventions=dict(completion='Stage 4 main metric: completion only, no BOS, first token unscored',
        full_text='Prefix plus continuation, all shifted tokens scored, no BOS; sensitivity only',
        completion_bos='GPT-2 EOS inserted as BOS; all completion tokens scored; sensitivity only'),
        dtype='fp32',attention='SDPA',batch_size=4))
    print(json.dumps(summary,indent=2),flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__=='__main__':
    main()
