"""Independent GPT-2-Large completion-only Gen-PPL, including short-output accounting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from transformers import BertTokenizerFast,GPT2TokenizerFast,GPT2LMHeadModel
from collect_benefit import dump


def masked_token_nll(logits,input_ids,attention_mask):
    target=input_ids[:,1:]
    loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
        target.reshape(-1),reduction='none').view_as(target)
    valid=attention_mask[:,1:].bool()
    return (loss*valid).sum(1),valid.sum(1)


def sources(root,bert):
    result=[]
    for line in (root/'samples.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        result.append(dict(dataset='lm1b',prompt_id=r['prompt_id'],seed=r['seed'],method=r['method'],
            text=r['completion'],teacher_nll_sum=r['teacher_nll_sum'],teacher_token_count=r['teacher_token_count'],
            bert_new_tokens=r['new_tokens'],stopped_eos=r['stopped_eos']))
    for r in json.loads((root/'references.json').read_text(encoding='utf-8')):
        result.append(dict(dataset='lm1b_reference',prompt_id=r['prompt_id'],seed=None,method='reference',
            text=r['completion'],teacher_nll_sum=r['teacher_nll_sum'],teacher_token_count=r['teacher_token_count'],
            bert_new_tokens=r['new_tokens']))
    for line in (root/'recommended_samples.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        result.append(dict(dataset='lm1b_recommended',prompt_id=r['prompt_id'],seed=r['seed'],method=r['method'],
            frequency_penalty=r['frequency_penalty'],text=r['completion'],teacher_nll_sum=r['teacher_nll_sum'],
            teacher_token_count=r['teacher_token_count'],bert_new_tokens=r['new_tokens'],stopped_eos=r['stopped_eos']))
    old=root.parent/'current-003/online_samples.jsonl'
    for line in old.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        text=bert.decode(r['ids'][r['prefix_length']:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
        result.append(dict(dataset='wikitext_stage3',prompt_id=r['prompt_id'],seed=r['seed'],method=r['policy'],
            text=text,teacher_nll_sum=r['teacher_nll_sum'],teacher_token_count=r['teacher_token_count'],bert_new_tokens=r['new_tokens']))
    return result


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=Path('work/gpt2-large'))
    ap.add_argument('--vocab',type=Path,default=Path('work/pilot-data/vocab.txt'))
    args=ap.parse_args()
    out=args.root/'external_scores.jsonl'
    if out.exists():
        raise RuntimeError('External scores already exist; preserve original evaluation')
    manifest=json.loads((args.model/'manifest.json').read_text())
    assert len(manifest['files'])==6
    for item in manifest['files']:
        h=hashlib.sha256()
        with (args.model/item['name']).open('rb') as src:
            while chunk:=src.read(8*1024*1024):
                h.update(chunk)
        assert h.hexdigest()==item['sha256'],'Evaluator file hash mismatch'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=GPT2TokenizerFast.from_pretrained(str(args.model),local_files_only=True)
    bert=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=True)
    rows=sources(args.root,bert)
    for r in rows:
        r['_tokens']=tokenizer.encode(r['text'],add_special_tokens=False)
        if len(r['_tokens'])>1024:
            raise RuntimeError('Output exceeds evaluator context; do not silently truncate')
    print('Loading verified GPT-2-Large in fp32',flush=True)
    model=GPT2LMHeadModel.from_pretrained(str(args.model),local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
    metadata=dict(model_repo=manifest['repo'],revision=manifest['revision'],precision='fp32',
        scoring='Completion only; strip BERT special tokens, retokenize with GPT-2, no BOS, shifted CE excluding first token; corpus token weighting.',
        batch_size=4,attention='SDPA',rows=len(rows),parameter_count=sum(p.numel() for p in model.parameters()),
        implementation='Explicit masked per-token cross-entropy; empty and one-token outputs retained as exclusions')
    dump(args.root/'external_evaluator.json',metadata)
    # Verify our batched masking against the evaluator's built-in shifted loss.
    probe=tokenizer.encode('This is a short independent scoring check.',return_tensors='pt').to('cuda')
    result=model(probe,labels=probe,use_cache=False)
    sums,counts=masked_token_nll(result.logits,probe,torch.ones_like(probe))
    torch.testing.assert_close(sums/counts,result.loss.reshape(1),rtol=1e-6,atol=1e-6)
    padded=torch.full((2,probe.shape[1]+3),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
    mask=torch.zeros_like(padded)
    padded[:,:probe.shape[1]]=probe
    mask[:,:probe.shape[1]]=1
    ps,pc=masked_token_nll(model(padded,attention_mask=mask,use_cache=False).logits,padded,mask)
    torch.testing.assert_close(ps/pc,(sums/counts).expand_as(ps),rtol=1e-5,atol=1e-5)
    dump(args.root/'external_scoring_checks.json',dict(builtin_loss_matches=True,right_padding_invariance=True,
        probe_nll=float(result.loss),scored_probe_tokens=int(counts[0])))
    with out.open('w',encoding='utf-8') as dest:
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]
            width=max(2,max(len(r['_tokens']) for r in batch))
            x=torch.full((len(batch),width),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
            mask=torch.zeros_like(x)
            for j,r in enumerate(batch):
                ids=r['_tokens']
                if ids:
                    x[j,:len(ids)]=torch.tensor(ids,device='cuda')
                    mask[j,:len(ids)]=1
                else:
                    # A dummy visible token prevents a fully masked SDPA row;
                    # the first token is never counted by shifted scoring.
                    mask[j,0]=1
            sums,counts=masked_token_nll(model(x,attention_mask=mask,use_cache=False).logits,x,mask)
            for j,r in enumerate(batch):
                tokens=r.pop('_tokens')
                r.update(gpt2_input_tokens=len(tokens),gpt2_scored_tokens=int(counts[j]),
                    gpt2_nll_sum=float(sums[j]) if counts[j] else 0.,excluded_short=len(tokens)<2)
                dest.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            dest.flush()
            if (start+4)%128==0:
                print('External scores',min(start+4,len(rows)),'/',len(rows),flush=True)
    summary={}
    for dataset,method in sorted({(r['dataset'],r['method']) for r in rows}):
        group=[r for r in rows if r['dataset']==dataset and r['method']==method]
        n=sum(r['gpt2_scored_tokens'] for r in group)
        nll=sum(r['gpt2_nll_sum'] for r in group)/n
        summary[dataset+'/'+method]=dict(samples=len(group),scored_tokens=n,
            excluded_short=sum(r['excluded_short'] for r in group),gpt2_nll=nll,gpt2_gen_ppl=math.exp(nll),
            teacher_nll=sum(r['teacher_nll_sum'] for r in group)/sum(r['teacher_token_count'] for r in group),
            mean_bert_new_tokens=sum(r['bert_new_tokens'] for r in group)/len(group))
    dump(args.root/'external_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
