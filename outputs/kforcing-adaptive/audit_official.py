"""Same-noise check against the released sampler, not a second handwritten reference."""
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import BertTokenizerFast
from core import load_model,generate
from collect_benefit import verify_checkpoints,dump
import models.pflm as pflm_module


def official_cli():
    path=Path(__file__).parent/'upstream/batch_inference_with_prefix.py'
    spec=importlib.util.spec_from_file_location('official_batch_cli',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quiet_sampler():
    pflm_module.logger.disable('models.pflm')
    pflm_module.tqdm=lambda iterable,**kwargs:iterable


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    ap.add_argument('--vocab',type=Path,default=Path('work/pilot-data/vocab.txt'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    verify_checkpoints(args)
    official=official_cli()
    tokenizer=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=True)
    asset=Path(__file__).parent/'upstream/assets/prefix_lm1b_examples.jsonl'
    prefixes=official.load_prefixes(str(asset),tokenizer,6)
    assert len(prefixes)==10 and all(p[0]==101 for p in prefixes)
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    quiet_sampler()
    matches=[]
    for index,p in enumerate(prefixes[:3]):
        for k in (2,3,4):
            for penalty in (0.,.5):
                seed=761+index
                tape=torch.rand(1,17,1,generator=torch.Generator().manual_seed(seed)).to('cuda')
                cursor=0
                def fixed_noise(*shape,**kwargs):
                    nonlocal cursor
                    assert shape==(1,k,1)
                    noise=tape[:,cursor:cursor+k]
                    cursor+=k
                    return noise
                prefix=torch.tensor([p],device='cuda')
                model.max_k=k
                with patch.object(pflm_module.torch,'rand',side_effect=fixed_noise):
                    actual=model.sample_next_k_tokens_with_kv_caches(prefix,torch.ones(1,device='cuda'),13,k=k,frequency_penalty=penalty)
                model.max_k=4
                custom=generate(model,prefix,policy=f'fixed{k}',max_new=13,seed=seed,
                    precision='fp32',frequency_penalty=penalty,stop_eos=False)
                match=actual[0].cpu().tolist()==custom['ids']
                assert match
                matches.append(dict(prefix=index,k=k,frequency_penalty=penalty,all_13_tokens_match=match))
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    checks=[]
    with torch.inference_mode():
        for p in prefixes[:3]:
            prefix=torch.tensor([p],device='cuda')
            _,cache=teacher.infer_next_token(prefix,offset=0)
            next_token=torch.tensor([[2003]],device='cuda')
            cached,_=teacher.infer_next_token(next_token,kv_caches=cache,offset=len(p))
            full,_=teacher.infer_next_token(torch.cat((prefix,next_token),1),offset=0)
            delta=float((cached[:,-1]-full[:,-1]).abs().max())
            checks.append(dict(max_logprob_difference=delta,top1_matches=bool(torch.equal(cached[:,-1].argmax(-1),full[:,-1].argmax(-1)))))
    report=dict(official_prefixes=10,cls_prefixes=10,reference_noise='Position-indexed tape injected only into torch.rand calls in released sampler',
        sampler_cases=matches,all_sampler_cases_match=all(m['all_13_tokens_match'] for m in matches),
        ar_cached_vs_full=checks,ar_precision='Released infer_next_token internally uses fp16; numeric shape effects recorded, not required bitwise equal',
        implementation='Released sampler and CLI helper, portable RoPE/eager residual helpers; loading remains strict/weights_only=True')
    dump(args.out/'official_semantic_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
