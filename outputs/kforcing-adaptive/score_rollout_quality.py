"""Offline per-sequence GPT-2 counterfactual targets, with exact duplicate reuse."""
import argparse
import gc
import hashlib
from pathlib import Path
import torch
from core import load_model,teacher_metrics
from collect_benefit import dump,verify_checkpoints
from confirm_local_gain import read,rows,sha
from external_gpt2_score import masked_token_nll
from run_frozen_online import append


@torch.inference_mode()
def score(args):
    from transformers import GPT2TokenizerFast,GPT2LMHeadModel
    path=args.out/(args.split+'_labels.jsonl');assert not path.exists()
    checks=read(args.out/(args.split+'_collection.json'))
    assert checks['passed'] and checks['pairs_sha256']==sha(args.out/(args.split+'_pairs.jsonl'))
    if args.split=='test':
        assert sha(args.out/'locked_model.json')==checks['locked_model_sha256']
    data=[r for r in rows(args.out/(args.split+'_pairs.jsonl')) if r['status']=='paired']
    verify_checkpoints(args)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    teacher_cache={}
    for i,r in enumerate(data):
        for name in ('keep4','split22'):
            branch=r[name];key=tuple(branch['ids'])
            if key not in teacher_cache:teacher_cache[key]=teacher_metrics(teacher,branch['ids'],6)
            branch['_teacher']=teacher_cache[key]
        if (i+1)%256==0:print(args.split,'teacher pairs',i+1,'/',len(data),flush=True)
    del teacher;gc.collect();torch.cuda.empty_cache()
    model_path=Path('work/gpt2-large');manifest=read(model_path/'manifest.json')
    for item in manifest['files']:
        h=hashlib.sha256()
        with (model_path/item['name']).open('rb') as f:
            while chunk:=f.read(8*1024*1024):h.update(chunk)
        assert h.hexdigest()==item['sha256']
    tokenizer=GPT2TokenizerFast.from_pretrained(str(model_path),local_files_only=True)
    model=GPT2LMHeadModel.from_pretrained(str(model_path),local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    probe=tokenizer.encode('A complete continuation can change after one local decision.',return_tensors='pt').cuda()
    result=model(probe,labels=probe,use_cache=False)
    ss,nn=masked_token_nll(result.logits,probe,torch.ones_like(probe))
    torch.testing.assert_close(ss/nn,result.loss.reshape(1),rtol=1e-6,atol=1e-6)
    texts=sorted({r[b]['completion'] for r in data for b in ('keep4','split22')})
    token_lists=[tokenizer.encode(t,add_special_tokens=False) for t in texts]
    assert all(len(t)<=1024 for t in token_lists)
    scored={}
    for start in range(0,len(texts),4):
        bt=texts[start:start+4];ids=token_lists[start:start+4]
        width=max(2,max(len(x) for x in ids))
        x=torch.full((len(bt),width),tokenizer.eos_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(x)
        for j,t in enumerate(ids):
            if t:x[j,:len(t)]=torch.tensor(t,device='cuda');mask[j,:len(t)]=1
            else:mask[j,0]=1
        sums,counts=masked_token_nll(model(x,attention_mask=mask,use_cache=False).logits,x,mask)
        for j,t in enumerate(bt):
            count=int(counts[j]);nll=float(sums[j]) if count else 0.
            scored[t]=dict(gpt2_input_tokens=len(ids[j]),gpt2_scored_tokens=count,gpt2_nll_sum=nll,
                gpt2_mean_nll=nll/count if count else None,excluded_short=count==0)
        if (start+4)%256==0:print(args.split,'unique GPT-2 texts',start+4,'/',len(texts),flush=True)
    valid=identical=0
    with path.open('w',encoding='utf-8') as f:
        for r in data:
            result={k:r[k] for k in ('pair_id','prompt_id','seed','offset')}
            for b in ('keep4','split22'):
                result[b]=dict(scored[r[b]['completion']],**r[b]['_teacher'],visible_new_tokens=len(r[b]['ids'])-6,
                    forward_calls=r[b]['calls'],computed_candidates=r[b]['computed_candidates'])
            ok=not result['keep4']['excluded_short'] and not result['split22']['excluded_short']
            same=r['keep4']['completion']==r['split22']['completion']
            gain=result['keep4']['gpt2_mean_nll']-result['split22']['gpt2_mean_nll'] if ok else None
            result.update(valid_label=ok,gain=gain,identical_decoded_text=same)
            if ok:valid+=1
            if same:
                identical+=1
                if ok:assert gain==0.
            append(f,result)
    dump(args.out/(args.split+'_scoring.json'),dict(passed=True,pairs=len(data),valid_labels=valid,
        excluded_unscorable=len(data)-valid,identical_text_pairs=identical,unique_texts=len(texts),
        unique_teacher_sequences=len(teacher_cache),builtin_loss_crosscheck=True,evaluator_files_verified=True,
        evaluator_repo=manifest['repo'],evaluator_revision=manifest['revision'],labels_sha256=sha(path),
        collector_sha256=checks['collector_sha256'],scorer_sha256=sha(Path(__file__)),
        pairs_sha256=checks['pairs_sha256'],locked_model_sha256=checks['locked_model_sha256']))
    print(args.split,'valid labels',valid,'/',len(data),'identical pairs',identical,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','dev','test'],required=True)
    ap.add_argument('--out',type=Path,default=Path('outputs/kforcing-adaptive/results/quality-011'))
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'));score(ap.parse_args())
