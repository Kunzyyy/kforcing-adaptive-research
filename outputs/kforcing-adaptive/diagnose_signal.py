"""Validation-only branch diagnostic: same context/noise, four outputs via 4 or 2+2.

This tests association with a teacher-NLL proxy, NOT counterfactual human quality.
EOS branches are excluded and counted. Frequency penalty is disabled for these
local branches to isolate the window effect. No threshold fitting or test reuse.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from core import load_model, generate, teacher_metrics


def rank(x):
    x=np.asarray(x)
    return np.array([((x<v).sum()+.5*((x==v).sum()-1)) for v in x])


def correlation(x,y):
    a,b=rank(x),rank(y)
    if np.std(a)==0 or np.std(b)==0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--branch-precision',choices=['bf16','fp32'],default='fp32')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda')
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm',device)
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar',device)
    val=json.loads((args.pilot/'prefixes.json').read_text(encoding='utf-8'))['validation']
    threshold=json.loads((args.pilot/'calibration.json').read_text())['threshold']
    rows=[]
    for p in val:
        prefix=torch.tensor([p['ids']],device=device)
        trajectory=generate(model,prefix,policy='fixed4',seed=11,max_new=64)
        for step in trajectory['steps'][1:4]:
            context=torch.tensor([trajectory['ids'][:step['position']]],device=device)
            for seed in (101,202):
                a=generate(model,context,policy='fixed4',seed=seed,max_new=4,
                           frequency_penalty=0.,stop_eos=False,precision=args.branch_precision)
                b=generate(model,context,policy='fixed2',seed=seed,max_new=4,
                           frequency_penalty=0.,stop_eos=False,precision=args.branch_precision)
                ta,tb=a['ids'][-4:],b['ids'][-4:]
                row=dict(prompt_id=p['id'],position=step['position'],seed=seed,
                    previous_margin=step['previous_margin'],low_margin=step['previous_margin']<threshold,
                    first_two_match=ta[:2]==tb[:2],eos_excluded=102 in ta or 102 in tb,
                    tokens_fixed4=ta,tokens_two_plus_two=tb)
                if not row['eos_excluded']:
                    ma=teacher_metrics(teacher,a['ids'],context.shape[1])
                    mb=teacher_metrics(teacher,b['ids'],context.shape[1])
                    row['nll_fixed4']=ma['teacher_nll']
                    row['nll_two_plus_two']=mb['teacher_nll']
                    row['gain_from_short']=ma['teacher_nll']-mb['teacher_nll']
                rows.append(row)
        print('Diagnostic',p['id'],flush=True)
    keep=[r for r in rows if not r['eos_excluded']]
    x=[r['previous_margin'] for r in keep]
    y=[r['gain_from_short'] for r in keep]
    groups={str(low):[r['gain_from_short'] for r in keep if r['low_margin']==low] for low in (True,False)}
    report=dict(total_branches=len(rows),retained_branches=len(keep),eos_excluded=len(rows)-len(keep),
        first_two_match_fraction=float(np.mean([r['first_two_match'] for r in rows])),
        spearman_previous_margin_vs_short_gain=correlation(x,y),
        mean_teacher_nll_gain_from_short=float(np.mean(y)) if y else None,
        groups={g:dict(count=len(v),mean_gain=float(np.mean(v)) if v else None,
            positive_gain_fraction=float(np.mean(np.array(v)>0)) if v else None) for g,v in groups.items()},
        scope='Validation only; same four output positions; no frequency penalty; EOS branches excluded. Expected useful sign is negative correlation. Small dependent branch sample, exploratory only.')
    report['branch_precision']=args.branch_precision
    report['trajectory_precision']='bf16'
    output=args.pilot/f'signal_diagnostic_{args.branch_precision}.json'
    if output.exists():
        raise RuntimeError('Diagnostic output already exists; preserve earlier results.')
    output.write_text(json.dumps(dict(summary=report,rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    del model, teacher
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__=='__main__':
    main()
