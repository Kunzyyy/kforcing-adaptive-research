"""Train/dev-only current-candidate screening, with a predeclared go/no-go gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from core import load_model
from collect_benefit import verify_checkpoints
from current_features import CURRENT_FEATURES,current_features_tensor
from analyze_benefit import fit,predict,corr,selection_lift,evaluate,dump


@torch.inference_mode()
def collect(args):
    path=args.out/'train_dev_features.jsonl'
    if path.exists():
        raise RuntimeError('Feature output already exists')
    source=args.source/'scored.jsonl'
    rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['split'] in ('train','dev')]
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    match=0
    with path.open('w',encoding='utf-8') as dest:
        for i,r in enumerate(rows):
            context=torch.tensor([r['context_ids']],device='cuda')
            noise=torch.rand(1,8,1,generator=torch.Generator().manual_seed(r['seed'])).to('cuda')[:,:4]
            logits=model(context,noise,torch.ones(1,1,1,device='cuda'),mode='inference')
            ids=logits.argmax(-1)[0].cpu().tolist()
            assert ids==r['tokens4'],'Replay mismatch; do not attach old labels to new candidates'
            match+=1
            r['current_features']=dict(zip(CURRENT_FEATURES,current_features_tensor(logits).cpu().tolist()))
            dest.write(json.dumps(r,allow_nan=False)+'\n')
            if (i+1)%100==0:
                dest.flush()
                print('Feature replay',i+1,'/',len(rows),flush=True)
    dump(args.out/'replay_checks.json',dict(rows=len(rows),exact_candidate_matches=match,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),uses_prior_test=False,
        precision='fp32',protocol_sha256=hashlib.sha256((Path(__file__).parent/'STAGE3_PROTOCOL.md').read_bytes()).hexdigest()))
    print('Feature collection complete',match,flush=True)


def screen(args):
    if (args.out/'screening.json').exists():
        raise RuntimeError('Screening exists; no repeated selection on viewed development results')
    rows=[json.loads(x) for x in (args.out/'train_dev_features.jsonl').read_text().splitlines()]
    groups={s:[r for r in rows if r['split']==s and 'gain' in r] for s in ('train','dev')}
    array=lambda rs:(np.array([[r['current_features'][n] for n in CURRENT_FEATURES] for r in rs]),np.array([r['gain'] for r in rs]))
    x,y=array(groups['train'])
    xd,yd=array(groups['dev'])
    models=[fit(x,y,a) for a in (.1,1.,10.,100.)]
    candidates=[dict(alpha=m['alpha'],dev_mse=float(np.mean((predict(m,xd)-yd)**2))) for m in models]
    model=models[min(range(4),key=lambda i:candidates[i]['dev_mse'])]
    score_ridge=predict(model,xd)
    score_simple=-xd[:,2:4].mean(1)
    scores={'ridge':score_ridge,'negative_tail_margin':score_simple}
    # Reuse the independently tested article-bootstrap machinery with explicit scores.
    article_ids=list(dict.fromkeys(r['article_id'] for r in groups['dev']))
    indices={a:[i for i,r in enumerate(groups['dev']) if r['article_id']==a] for a in article_ids}
    rng=np.random.default_rng(817320)
    boot={k:[] for k in scores}
    for _ in range(2000):
        idx=np.concatenate([indices[article_ids[i]] for i in rng.integers(0,len(article_ids),len(article_ids))])
        for k,v in scores.items():
            boot[k].append(selection_lift(v[idx],yd[idx]))
    stats={k:dict(spearman=corr(v,yd),selected_half_gain_lift=selection_lift(v,yd),
        lift95=np.quantile(boot[k],[.025,.975]).tolist()) for k,v in scores.items()}
    selected=max(scores,key=lambda k:stats[k]['selected_half_gain_lift'])
    gate=stats[selected]['lift95'][0]>0
    controller=dict(model) if selected=='ridge' else {}
    controller.update(kind=selected,features=CURRENT_FEATURES,threshold=float(np.median(scores[selected])),
        threshold_source='median development candidate scores; no online/test quality tuning',
        candidate_regularization=candidates,passed_development_gate=gate,
        train_dev_sha256=hashlib.sha256((args.out/'train_dev_features.jsonl').read_bytes()).hexdigest())
    dump(args.out/'locked_controller.json',controller)
    report=dict(train_rows=len(y),dev_rows=len(yd),dev_articles=len(article_ids),candidate_stats=stats,
        chosen=selected,passed_gate=gate,chosen_alpha=model['alpha'],
        scope='Exploratory development screening only; no fresh-test evidence yet. Branch seeds clustered by article.')
    dump(args.out/'screening.json',report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['collect','screen'])
    ap.add_argument('--source',type=Path,default=Path(__file__).parent/'results/benefit-002')
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='collect':
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        verify_checkpoints(args)
        collect(args)
    else:
        screen(args)


if __name__=='__main__':
    main()
