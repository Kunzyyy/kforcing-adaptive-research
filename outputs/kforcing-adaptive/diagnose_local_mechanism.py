"""Frozen local-change / local-teacher mechanism diagnostic; no learned candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/local-mechanism'
DATA='results/features-014/development_data.jsonl'
OOF='results/attention-features/oof_predictions.jsonl'
SEEDS=[14112001,14112002,14112003]
AR_SHA='e2463aa9de897dd2394580eeec7de23f0fe076931050b21612e8b06316ccc439'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    assert not p.exists(),f'Preserve {p}'
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def lines(p,v):
    assert not p.exists()
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in v),encoding='utf-8')
def trim(ids):return ids[:ids.index(102)+1] if 102 in ids else ids

def freeze():
    assert not OUT.exists();OUT.mkdir()
    names=[DATA,OOF,'results/attention-features/analysis.json',
        'results/averaged-013/train_states.jsonl','results/averaged-013/dev_states.jsonl',
        'LOCAL_MECHANISM_PLAN.md','diagnose_local_mechanism.py','core.py',
        'upstream/models/autoregressive.py','upstream/models/transformer.py']
    old=read(ROOT/'attention_features_file_hashes.json')
    for n in names:
        if n in old:assert sha(ROOT/n)==old[n],n
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        inputs={n:sha(ROOT/n) for n in names},ar_checkpoint_sha256=AR_SHA,
        states=446,sources=253,fold_seeds=SEEDS,bootstrap_seed=22242001,
        retrospective_development=True,old_test_inputs=False,new_model_fit=False,
        primary='Local teacher selection minus frozen combined19 OOF selection',
        teacher_probe_indices=np.linspace(0,445,12,dtype=int).tolist()))
    print('Frozen mechanism diagnostic; no learned candidate.',flush=True)

def verify():
    m=read(OUT/'manifest.json')
    for n,h in m['inputs'].items():assert sha(ROOT/n)==h,n
    return m

def score():
    import torch
    from core import load_model,teacher_metrics
    m=verify();assert not (OUT/'local_scores.jsonl').exists()
    checkpoint=ROOT.parents[1]/'work/checkpoints/ar_best_lm1b.ckpt';assert sha(checkpoint)==AR_SHA
    ids=[r['state_id'] for r in rows(ROOT/DATA)]
    allowed=['state_id','prompt_id','history_ids','initial_candidates','split_candidates','offset']
    states={r['state_id']:{k:r[k] for k in allowed} for split in ['train','dev']
        for r in rows(ROOT/f'results/averaged-013/{split}_states.jsonl') if r['status']=='paired'}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    teacher=load_model(checkpoint,'ar','cuda')
    probes={};output=[];max_prefix_error=0.;max_metric_error=0.;checked_positions=0
    with torch.inference_mode():
        for i,state_id in enumerate(ids):
            s=states[state_id];h=len(s['history_ids']);assert h==6+s['offset']
            assert s['initial_candidates'][:2]==s['split_candidates'][:2] and 102 not in s['initial_candidates'][:2]
            record=dict(state_id=state_id,prompt_id=s['prompt_id'],offset=s['offset'])
            for name,key in [('keep4','initial_candidates'),('split22','split_candidates')]:
                visible=trim(s[key]);tail=visible[2:];full=s['history_ids']+visible;start=h+2
                tokens=torch.tensor([full],device='cuda')
                logits=teacher.forward_high_precision(tokens[:,:-1])[0,start-1:].float()
                targets=tokens[0,start:];assert len(tail)==len(logits)==len(targets)
                losses=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
                selected=logits.gather(1,targets[:,None])[:,0]
                record[name]=dict(visible_candidates=visible,tail_tokens=tail,losses=losses.cpu().tolist(),
                    target_logits=selected.cpu().tolist(),logsumexp=logits.logsumexp(-1).cpu().tolist())
                if i in m['teacher_probe_indices']:
                    probes[f'{i}_{name}']=logits.cpu().numpy()
                    reference=teacher_metrics(teacher,full,start)['teacher_nll_sum']
                    error=abs(reference-float(losses.sum()));max_metric_error=max(max_metric_error,error)
                    assert error<2e-5
                    for j in range(len(tail)):
                        prefix_logits=teacher.forward_high_precision(tokens[:,:start+j])[0,-1].float()
                        err=float((prefix_logits-logits[j]).abs().max());max_prefix_error=max(max_prefix_error,err)
                        torch.testing.assert_close(prefix_logits,logits[j],atol=2e-4,rtol=2e-4)
                        checked_positions+=1
            k=record['keep4'];s2=record['split22'];common=min(len(k['losses']),len(s2['losses']))
            record.update(active=k['visible_candidates']!=s2['visible_candidates'],
                raw_candidates_changed=s['initial_candidates']!=s['split_candidates'],
                eos_in_either=102 in k['visible_candidates'] or 102 in s2['visible_candidates'],
                common_tail_tokens=common,local_gain=float(np.mean(np.array(k['losses'][:common])-s2['losses'][:common])))
            output.append(record)
            if (i+1)%100==0:print('Local teacher states',i+1,'/446',flush=True)
    lines(OUT/'local_scores.jsonl',output);np.savez_compressed(OUT/'teacher_probes.npz',**probes)
    dump(OUT/'scoring.json',dict(passed=True,states=446,branches=892,probe_states=12,
        independent_prefix_positions=checked_positions,max_causal_prefix_logit_error=max_prefix_error,
        max_teacher_metrics_sum_error=max_metric_error,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        dtype='fp32',tf32=False,new_future_generations=0,
        outputs={n:sha(OUT/n) for n in ['local_scores.jsonl','teacher_probes.npz']}))
    print('Local teacher scoring complete.',flush=True)

def half(p):
    a=np.zeros(len(p));a[np.argsort(-p,kind='stable')[:len(p)//2]]=1;return a

def analyze():
    verify();scoring=read(OUT/'scoring.json')
    for n,h in scoring['outputs'].items():assert sha(OUT/n)==h
    data=rows(ROOT/DATA);local=rows(OUT/'local_scores.jsonl');oof=rows(ROOT/OOF)
    assert [r['state_id'] for r in data]==[r['state_id'] for r in local]==[r['state_id'] for r in oof]
    ids=[r['state_id'] for r in data];assert ids==sorted(ids)
    y=np.array([r['gains'] for r in data]);gain=y.mean(1);n=len(data)
    x=np.array([r['local_gain'] for r in local]);active=np.array([r['active'] for r in local]);eos=np.array([r['eos_in_either'] for r in local])
    assert (y[~active]==0).all()
    sources=sorted({r['prompt_id'] for r in data});members=[[i for i,r in enumerate(data) if r['prompt_id']==s] for s in sources]
    q=np.zeros((3,n));changes=np.zeros((3,n));teacher=np.zeros((3,n));prior=np.array([r['selected']['combined19'] for r in oof]).T
    future_cross=np.zeros((3,n));ga=y[:,:4].mean(1);gb=y[:,4:].mean(1);fold_records=[]
    for repeat,seed in enumerate(SEEDS):
        assignment={s:i%5 for i,s in enumerate(np.random.default_rng(seed).permutation(sources).tolist())}
        for fold in range(5):
            valid=np.array([i for i,r in enumerate(data) if assignment[r['prompt_id']]==fold]);budget=len(valid)//2
            q[repeat,valid]=budget/len(valid);na=int(active[valid].sum());ni=len(valid)-na
            a=min(1.,budget/na) if na else 0.;b=max(0.,budget-na)/ni if ni else 0.
            changes[repeat,valid]=np.where(active[valid],a,b)
            teacher[repeat,valid]=half(x[valid])
            future_cross[repeat,valid]=((half(ga[valid])-q[repeat,valid])*gb[valid]+(half(gb[valid])-q[repeat,valid])*ga[valid])/2
            assert prior[repeat,valid].sum()==budget
            fold_records.append(dict(repeat=repeat,fold=fold,valid=valid.tolist(),budget=budget,active=na,inactive=ni,
                active_selection_probability=a,inactive_selection_probability=b))
    contributions={name:((selection-q)*gain).mean(0) for name,selection in
        [('perfect_change',changes),('local_teacher',teacher),('combined19',prior)]}
    contributions['cross_half_future']=future_cross.mean(0)
    draws=np.random.default_rng(22242001).integers(0,len(sources),(4000,len(sources)))
    def interval(values,mask=None):
        if mask is None:mask=np.ones(n,dtype=bool)
        sums=np.array([values[g][mask[g]].sum() for g in members]);counts=np.array([mask[g].sum() for g in members])
        denom=counts[draws].sum(1);ok=denom>0;boot=sums[draws].sum(1)[ok]/denom[ok]
        return dict(estimate=float(values[mask].mean()) if mask.any() else None,
            interval=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,coverage=.95,defined_bootstraps=int(ok.sum()))
    def corr(a,b,mask):
        aa=a[mask];bb=b[mask]
        return float(np.corrcoef(aa,bb)[0,1]) if len(aa)>1 and aa.std()>0 and bb.std()>0 else None
    categories={'inactive':~active,'active_no_eos':active&~eos,'active_eos':active&eos}
    subgroups={name:dict(states=int(mask.sum()),sources=len({data[i]['prompt_id'] for i in np.flatnonzero(mask)}),
        mean_full_gain=interval(gain,mask),mean_local_gain=interval(x,mask),
        local_full_correlation=corr(x,gain,mask)) for name,mask in categories.items()}
    comparison=interval(contributions['local_teacher']-contributions['combined19'])
    report=dict(states=n,sources=len(sources),active_states=int(active.sum()),inactive_states=int((~active).sum()),
        raw_candidate_changes=sum(r['raw_candidates_changed'] for r in local),inactive_all_eight_gains_zero=True,
        subgroups=subgroups,local_gain_sign_counts={k:int(v.sum()) for k,v in [('positive',x>0),('zero',x==0),('negative',x<0)]},
        local_full_correlation=corr(x,gain,np.ones(n,dtype=bool)),
        methods_vs_random={k:interval(v) for k,v in contributions.items()},primary_local_teacher_minus_combined19=comparison,
        local_target_followup_signal=bool(comparison['interval'][0]>0),
        retrospective_development=True,independent_confirmation=False,new_model_fit=False,deployable_policy=False,
        new_online_timing=False,new_human_quality_result=False,old_test_inputs=False)
    old=read(ROOT/'results/attention-features/analysis.json')['methods']['combined19']['gain_vs_random']['estimate']
    assert abs(report['methods_vs_random']['combined19']['estimate']-old)<1e-12
    dump(OUT/'folds.json',fold_records)
    lines(OUT/'state_statistics.jsonl',[dict(state_id=r['state_id'],prompt_id=r['prompt_id'],mean_full_gain=float(gain[i]),
        local_gain=float(x[i]),active=bool(active[i]),eos_in_either=bool(eos[i]),random_fraction=q[:,i].tolist(),
        change_probabilities=changes[:,i].tolist(),teacher_selected=teacher[:,i].astype(int).tolist(),
        prior19_selected=prior[:,i].tolist(),contributions_vs_random={k:float(v[i]) for k,v in contributions.items()}) for i,r in enumerate(data)])
    report['output_hashes']={n:sha(OUT/n) for n in ['folds.json','state_statistics.jsonl','local_scores.jsonl']}
    dump(OUT/'analysis.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','score','analyze']);args=p.parse_args();globals()[args.action]()
