"""Recheck the validation branch whose first two bf16 tokens differed."""
import argparse
import gc
import json
from pathlib import Path
import torch
from core import load_model, generate, amp_context


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    diagnostic=json.loads((args.pilot/'signal_diagnostic.json').read_text())
    prefixes=json.loads((args.pilot/'prefixes.json').read_text())['validation']
    reports=[]
    for row in diagnostic['rows']:
        if row['first_two_match']:
            continue
        p=next(p for p in prefixes if p['id']==row['prompt_id'])
        trajectory=generate(model,torch.tensor([p['ids']],device='cuda'),policy='fixed4',seed=11)
        context=torch.tensor([trajectory['ids'][:row['position']]],device='cuda')
        noise=torch.rand(1,8,1,generator=torch.Generator().manual_seed(row['seed'])).to('cuda')[:,:4]
        tau=torch.ones(1,1,1,device='cuda')
        item={k:row[k] for k in ('prompt_id','position','seed')}
        for precision in ('bf16','fp32'):
            with torch.inference_mode(),amp_context('cuda',precision):
                a=model(context,noise,tau,mode='inference')[:,:2].float()
                b=model(context,noise[:,:2],tau,mode='inference').float()
            item[precision]=dict(max_abs_logit_difference=float((a-b).abs().max()),
                first_two_match=bool(torch.equal(a.argmax(-1),b.argmax(-1))),
                tokens_width4=a.argmax(-1).cpu().tolist(),tokens_width2=b.argmax(-1).cpu().tolist(),
                top_margin_width4=(a.topk(2).values[:,:,0]-a.topk(2).values[:,:,1]).cpu().tolist(),
                top_margin_width2=(b.topk(2).values[:,:,0]-b.topk(2).values[:,:,1]).cpu().tolist())
        reports.append(item)
    (args.pilot/'numerical_audit.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2),flush=True)
    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print('Explicit CUDA cleanup completed',flush=True)


if __name__=='__main__':
    main()
