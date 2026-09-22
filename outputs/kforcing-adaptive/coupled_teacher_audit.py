"""Local same-noise comparison to an explicitly defined AR inverse-CDF reference."""
import argparse
import json
from pathlib import Path
import torch
from core import load_model
from collect_benefit import verify_checkpoints,dump


def inverse_cdf(logits,noise):
    probabilities=torch.softmax(logits.double(),dim=-1)
    cumulative=probabilities.cumsum(-1)
    return torch.searchsorted(cumulative.contiguous(),noise.double().reshape(-1,1).contiguous(),right=False).clamp_max(logits.shape[-1]-1)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=Path('outputs/kforcing-adaptive/results/lm1b-004'))
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'windows.jsonl').exists(),'Preserve recorded windows'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    verify_checkpoints(args)
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    prefixes=json.loads((args.source/'prefixes.json').read_text(encoding='utf-8'))
    seeds=(701,709,719,727,733,739,743,751)
    cases=[]
    for pi,p in enumerate(prefixes):
        for seed in seeds:
            u=torch.rand(4,generator=torch.Generator().manual_seed(seed+pi*100003))
            cases.append(dict(prompt_id=p['id'],seed=seed,noise_seed=seed+pi*100003,prefix=p['ids'],noise=u.tolist()))
    # An independent simple distribution verifies interval boundaries.
    probe=torch.tensor([[.2,.3,.5]]*4,dtype=torch.float64).log()
    assert inverse_cdf(probe,torch.tensor([.1,.19,.3,.9],dtype=torch.float64)).flatten().tolist()==[0,0,1,2]
    first_two_mismatches=0
    reference_checks=0
    with (args.out/'windows.jsonl').open('w',encoding='utf-8') as dest:
        for start in range(0,len(cases),16):
            batch=cases[start:start+16]
            context=torch.tensor([r['prefix'] for r in batch],device='cuda')
            noise=torch.tensor([r['noise'] for r in batch],device='cuda').unsqueeze(-1)
            tau=torch.ones(len(batch),1,1,device='cuda')
            logits4=model(context,noise,tau,mode='inference')
            tokens4=logits4.argmax(-1)
            top=logits4.topk(2,dim=-1).values
            margins=(top[:,:,0]-top[:,:,1]).cpu().tolist()
            logits2,cache=model(context,noise[:,:2],tau,mode='inference',return_kv=True)
            first=logits2.argmax(-1)
            assert all(c[0].shape[1]==6 for c in cache)
            second,cache2=model(torch.cat((context,first),1),noise[:,2:],tau,mode='inference',kv_caches=cache,return_kv=True)
            assert all(c[0].shape[1]==8 for c in cache2)
            split=torch.cat((first,second.argmax(-1)),1)
            first_two_mismatches+=int((first!=tokens4[:,:2]).sum())
            ar_context=context
            reference=[]
            for position in range(4):
                logits=teacher.forward_high_precision(ar_context)[:,-1]
                token=inverse_cdf(logits,noise[:,position])
                if start==0 and position==0:
                    original=model.generate_ar_teacher_sequence(teacher,context,
                        noise[:,None].expand(-1,6,4,1).contiguous(),torch.ones(len(batch),device='cuda'))
                    assert torch.equal(token[:,0],original[:,-1,0])
                    reference_checks=len(batch)
                reference.append(token)
                ar_context=torch.cat((ar_context,token),1)
            reference=torch.cat(reference,1).cpu().tolist()
            for row,a,b,t,m in zip(batch,tokens4.cpu().tolist(),split.cpu().tolist(),reference,margins):
                row.update(fixed4=a,split22=b,ar_reference=t,margins=m,
                           tail_margin=(m[2]+m[3])/2,
                           no_eos_all_branches=all(102 not in x for x in (a,b,t)))
                dest.write(json.dumps(row,allow_nan=False)+'\n')
            dest.flush()
            if (start+16)%128==0:
                print('Coupled windows',start+16,'/',len(cases),flush=True)
    dump(args.out/'collection_checks.json',dict(windows=len(cases),prefixes=128,seeds=list(seeds),
        first_two_argmax_disagreements=first_two_mismatches,official_first_step_reference_checks=reference_checks,
        inverse_cdf_simple_distribution_check=True,precision='fp32 model; float64 CDF; tau=1',
        gpu=torch.cuda.get_device_name(),notes='AR reference is a chosen noise coupling, not the unknown intermediate k2 distillation teacher. No online quality or speed inference.'))


if __name__=='__main__':
    main()
