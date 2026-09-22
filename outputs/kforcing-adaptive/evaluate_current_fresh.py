"""Frozen current-candidate score on fresh article-separated local branches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from core import load_model,generate,teacher_metrics
from collect_benefit import verify_checkpoints,dump
from current_features import current_features_tensor,current_score_tensor,prepare_controller
from analyze_benefit import corr,selection_lift


@torch.inference_mode()
def collect(args):
    output=args.root/'fresh_branches.jsonl'
    if output.exists():
        raise RuntimeError('Fresh branches already exist')
    controller=json.loads((args.root/'locked_controller.json').read_text())
    assert controller['passed_development_gate']
    controller=prepare_controller(controller,'cuda')
    prompts=json.loads((args.root/'fresh_prefixes.json').read_text(encoding='utf-8'))
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    rows=0
    with output.open('w',encoding='utf-8') as dest:
        for i,p in enumerate(prompts):
            context=torch.tensor([p['ids']],device='cuda')
            noise=torch.rand(1,8,1,generator=torch.Generator().manual_seed(31713)).to('cuda')
            tau=torch.ones(1,1,1,device='cuda')
            for state in range(2):
                prev=model(context,noise[:,state*4:(state+1)*4],tau,mode='inference').argmax(-1)
                if 102 in prev[0].cpu().tolist():
                    break
                context=torch.cat((context,prev),dim=1)
                for seed in (1121,2213):
                    branch_noise=torch.rand(1,8,1,generator=torch.Generator().manual_seed(seed)).to('cuda')[:,:4]
                    logits=model(context,branch_noise,tau,mode='inference')
                    ta=logits.argmax(-1)[0].cpu().tolist()
                    b=generate(model,context,policy='fixed2',seed=seed,max_new=4,
                        precision='fp32',frequency_penalty=0.,stop_eos=False)
                    tb=b['ids'][-4:]
                    row=dict(prompt_id=p['id'],article_id=p['article_id'],state_index=state,seed=seed,
                        context_ids=context[0].cpu().tolist(),tokens4=ta,tokens22=tb,
                        current_features=current_features_tensor(logits).cpu().tolist(),
                        score=float(current_score_tensor(logits,controller)),
                        eos_excluded=102 in ta or 102 in tb,first_two_match=ta[:2]==tb[:2])
                    dest.write(json.dumps(row,allow_nan=False)+'\n')
                    rows+=1
            dest.flush()
            if (i+1)%8==0:
                print('Fresh local collection',i+1,'/',len(prompts),flush=True)
    print('Collected fresh branches',rows,flush=True)


def score(args):
    output=args.root/'fresh_scored.jsonl'
    if output.exists():
        raise RuntimeError('Fresh scores already exist')
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    rows=[json.loads(x) for x in (args.root/'fresh_branches.jsonl').read_text().splitlines()]
    with output.open('w',encoding='utf-8') as dest:
        for r in rows:
            if not r['eos_excluded'] and r['first_two_match']:
                c=r['context_ids']
                a=teacher_metrics(teacher,c+r['tokens4'],len(c))
                b=teacher_metrics(teacher,c+r['tokens22'],len(c))
                r.update(nll4=a['teacher_nll'],nll22=b['teacher_nll'],gain=a['teacher_nll']-b['teacher_nll'])
            dest.write(json.dumps(r,allow_nan=False)+'\n')
    print('Fresh local scoring complete',len(rows),flush=True)


def analyze(args):
    output=args.root/'fresh_local_analysis.json'
    if output.exists():
        raise RuntimeError('Fresh analysis already exists')
    raw=[json.loads(x) for x in (args.root/'fresh_scored.jsonl').read_text().splitlines()]
    rows=[r for r in raw if 'gain' in r]
    y=np.array([r['gain'] for r in rows])
    scores=np.array([r['score'] for r in rows])
    articles=list(dict.fromkeys(r['article_id'] for r in rows))
    by_article={a:[i for i,r in enumerate(rows) if r['article_id']==a] for a in articles}
    rng=np.random.default_rng(443010)
    boot=[]
    for _ in range(2000):
        idx=np.concatenate([by_article[articles[i]] for i in rng.integers(0,len(articles),len(articles))])
        boot.append(selection_lift(scores[idx],y[idx]))
    report=dict(total_branches=len(raw),valid_branches=len(rows),articles=len(articles),
        excluded_eos=sum(r['eos_excluded'] for r in raw),first_two_mismatches=sum(not r['first_two_match'] for r in raw),
        spearman=corr(scores,y),selected_half_gain_lift=selection_lift(scores,y),
        lift95=np.quantile(boot,[.025,.975]).tolist(),oracle_lift=selection_lift(y,y),
        mean_short_gain=float(y.mean()),scope='Locked score; fresh articles; EOS-filtered local proxy only, not online quality.')
    dump(output,report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['collect','score','analyze'])
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args()
    if args.mode!='analyze':
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        verify_checkpoints(args)
    globals()[args.mode](args)


if __name__=='__main__':
    main()
