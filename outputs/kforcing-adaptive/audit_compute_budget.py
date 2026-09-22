"""Retrospective cost sensitivity for frozen single-intervention rankings, not latency."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'results/decision-confirm'; OUT=ROOT/'results/compute-budget'
POLICIES=['random_half','combined19','confidence14','heuristic']
METRICS=['gain','calls','candidates','visible_tokens','direct_calls','direct_candidates','future_calls','future_candidates','identical_interventions','identical_extra_calls','identical_extra_candidates']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def normalized(gains, costs):
    gains=np.asarray(gains,dtype=float); costs=np.asarray(costs,dtype=float)
    if not np.isfinite(gains).all() or not np.isfinite(costs).all() or not (costs>0).all():
        return None
    budget=float(costs.min()); retention=budget/costs
    return dict(budget=budget,retention=retention.tolist(),gains=(retention*gains).tolist(),
                matched_costs=(retention*costs).tolist())

def main():
    assert not OUT.exists(), 'Refusing to overwrite frozen audit'
    inputs=['results/decision-confirm/'+n for n in ['test_pairs.jsonl','test_labels.jsonl','test_states.jsonl',
            'frozen_predictions.json','analysis.json','scoring_prediction_link.json','test_scoring.json']]
    inputs+=['COMPUTE_BUDGET_AUDIT_PLAN.md','audit_compute_budget.py']
    hashes={n:sha(ROOT/n) for n in inputs}
    prior=read(SRC/'analysis.json'); prediction=read(SRC/'frozen_predictions.json')
    assert not prior['advance_gate_passed']
    assert sha(SRC/'frozen_predictions.json')==prior['frozen_predictions_sha256']==read(SRC/'scoring_prediction_link.json')['frozen_predictions_sha256']
    assert sha(SRC/'test_labels.jsonl')==prior['labels_sha256']==read(SRC/'test_scoring.json')['labels_sha256']
    assert sha(SRC/'test_pairs.jsonl')==read(SRC/'test_scoring.json')['pairs_sha256']
    labels={r['pair_id']:r for r in rows(SRC/'test_labels.jsonl')}
    pair_rows=rows(SRC/'test_pairs.jsonl'); grouped=defaultdict(list)
    for pair in pair_rows: grouped[pair['state_id']].append(pair)
    assert len(labels)==len(pair_rows)==5448 and len(grouped)==681
    state_rows=[]
    for sid,batch in sorted(grouped.items()):
        assert sorted(r['replicate'] for r in batch)==list(range(8))
        assert len({r['prompt_id'] for r in batch})==len({r['offset'] for r in batch})==1
        records=[]; keep=[]
        for pair in sorted(batch,key=lambda p:p['replicate']):
            label=labels[pair['pair_id']]; assert label['valid_label']
            offset=pair['offset']; branch_values={}
            for branch,extra in [('keep4',0),('split22',1)]:
                b=pair[branch]; l=label[branch]; length=len(b['ids'])-6
                assert b['ids'][:6]==pair['history_ids'][:6] and 0<length<=122
                assert 102 not in b['ids'][6:-1] and b['stopped_eos']==(b['ids'][-1]==102)
                tail=math.ceil(max(0,length-offset-4)/4)
                assert b['calls']==l['forward_calls']==offset//4+1+extra+tail
                assert b['computed_candidates']==l['computed_candidates']==offset+4+2*extra+min(122-offset-4,4*tail)
                assert length==l['visible_new_tokens']
                branch_values[branch]=dict(calls=b['calls'],candidates=b['computed_candidates'],visible_tokens=length)
            d={k:branch_values['split22'][k]-branch_values['keep4'][k] for k in branch_values['keep4']}
            gain=label['keep4']['gpt2_mean_nll']-label['split22']['gpt2_mean_nll']
            assert abs(gain-label['gain'])<1e-12
            identical=pair['keep4']['ids']==pair['split22']['ids']
            if identical: assert gain==0 and d['calls']==1 and d['candidates']==2
            records.append(dict(gain=gain,**d,direct_calls=1,direct_candidates=2,future_calls=d['calls']-1,
                future_candidates=d['candidates']-2,identical_interventions=int(identical),
                identical_extra_calls=int(identical)*d['calls'],identical_extra_candidates=int(identical)*d['candidates']))
            keep.append(branch_values['keep4'])
        state_rows.append(dict(state_id=sid,prompt_id=batch[0]['prompt_id'],offset=batch[0]['offset'],
            deltas={k:float(np.mean([r[k] for r in records])) for k in METRICS},
            keep={k:float(np.mean([r[k] for r in keep])) for k in keep[0]},
            token_identical_replicates=sum(r['identical_interventions'] for r in records)))
    n=len(state_rows); p=(n//2)/n; ids=[r['state_id'] for r in state_rows]
    masks={'keep4':np.zeros(n),'all_split':np.ones(n),'random_half':np.full(n,p)}
    for policy in POLICIES[1:]:
        rank=sorted(ids,key=lambda sid:(-prediction['scores'][sid][policy],sid))
        assert rank==prediction['rankings'][policy]
        selected=set(rank[:n//2]); masks[policy]=np.array([float(s in selected) for s in ids])
    y=np.array([r['deltas']['gain'] for r in state_rows])
    for name,base in [('vs_random',masks['random_half']),('vs_confidence14',masks['confidence14']),('vs_heuristic',masks['heuristic'])]:
        assert abs(float(np.mean((masks['combined19']-base)*y))-prior['primary'][name]['estimate'])<1e-12
    sources=sorted({r['prompt_id'] for r in state_rows}); members=[[i for i,r in enumerate(state_rows) if r['prompt_id']==s] for s in sources]
    counts=np.array([len(g) for g in members])
    draws=np.random.default_rng(22222001).integers(0,len(sources),size=(4000,len(sources)),dtype=np.int32)
    denominators=counts[draws].sum(1)
    def samples(values):
        sums=np.array([values[g].sum() for g in members])
        return sums[draws].sum(1)/denominators
    def interval(point,boot): return dict(estimate=float(point),interval=np.quantile(boot,[.025,.975]).tolist(),coverage=.95)
    summary={}; boots={}
    keep_stats={k:np.array([r['keep'][k] for r in state_rows]) for k in state_rows[0]['keep']}
    for policy,mask in masks.items():
        entries={}; bootstrap={}
        for metric in METRICS:
            values=mask*np.array([r['deltas'][metric] for r in state_rows]); b=samples(values)
            entries[metric]=interval(values.mean(),b);bootstrap[metric]=b
        absolute={};relative={}
        for metric,base in keep_stats.items():
            base_boot=samples(base); delta=entries[metric]['estimate']
            absolute[metric]=interval(base.mean()+delta,base_boot+bootstrap[metric])
            relative[metric]=interval(100*delta/base.mean(),100*bootstrap[metric]/base_boot)
        rate=float(mask.mean())
        summary[policy]=dict(expected_intervention_probability=rate,expected_selected_states=float(mask.sum()),
            changes_vs_keep4=entries,absolute=absolute,percent_change_vs_keep4=relative,
            identical_fraction_among_interventions=entries['identical_interventions']['estimate']/rate if rate else None)
        boots[policy]=bootstrap
    normalized_results={}
    for cost in ['calls','candidates']:
        gains=np.array([summary[s]['changes_vs_keep4']['gain']['estimate'] for s in POLICIES])
        costs=np.array([summary[s]['changes_vs_keep4'][cost]['estimate'] for s in POLICIES])
        value=normalized(gains,costs)
        cboot=np.column_stack([boots[s][cost] for s in POLICIES]);gboot=np.column_stack([boots[s]['gain'] for s in POLICIES])
        feasible=np.isfinite(cboot).all(1)&(cboot>0).all(1); invalid=int((~feasible).sum())
        comparisons={}; probabilities={};gain_table={}
        if value is not None:
            if invalid==0:
                bboot=cboot.min(1); rboot=bboot[:,None]/cboot; normalized_boot=rboot*gboot
            for j,policy in enumerate(POLICIES):
                probabilities[policy]=value['retention'][j]
                gain_table[policy]=dict(estimate=value['gains'][j],
                    interval=np.quantile(normalized_boot[:,j],[.025,.975]).tolist() if invalid==0 else None,
                    matched_extra_cost=value['matched_costs'][j],
                    expected_intervention_probability=p*value['retention'][j])
            for other in ['random_half','confidence14','heuristic']:
                j=POLICIES.index(other);d=value['gains'][1]-value['gains'][j]
                comparisons['combined19_vs_'+other]=dict(estimate=d,
                    interval=np.quantile(normalized_boot[:,1]-normalized_boot[:,j],[.025,.975]).tolist() if invalid==0 else None)
        normalized_results[cost]=dict(feasible_at_point=value is not None,common_extra_cost=value['budget'] if value else None,
            retention_probability=probabilities,normalized_gain=gain_table,comparisons=comparisons,
            bootstrap_infeasible=invalid,bootstrap_repeats=4000,
            interval_status='descriptive' if value is not None and invalid==0 else 'not_reported_infeasible',
            posthoc_cost_calibration=True,deployable_policy=False)
    OUT.mkdir()
    with (OUT/'state_statistics.jsonl').open('w',encoding='utf-8') as f:
        for i,row in enumerate(state_rows):
            row['selection']={k:float(v[i]) for k,v in masks.items()};f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'bootstrap_indices.npz',source_ids=np.array(sources),draws=draws)
    report=dict(states=n,sources=len(sources),paired_futures=len(pair_rows),selected_states=n//2,
        mean_all_split_gain=float(y.mean()),policies=summary,cost_normalization=normalized_results,
        old_primary_points_reproduced=True,old_advance_gate_passed=False,new_advance_gate_defined=False,
        new_model_fit=False,new_generation=False,wall_clock_measured=False,human_ratings_received=0,
        scope='retrospective fixed-rank single-intervention cost sensitivity, optimized GPT-2 proxy',
        bootstrap=dict(seed=22222001,repeats=4000,unit='source',fixed_ranks=True,resample_future=False))
    dump(OUT/'analysis.json',report)
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input_hashes=hashes,
        outputs={n:sha(OUT/n) for n in ['state_statistics.jsonl','bootstrap_indices.npz','analysis.json']},
        plan_is_retrospective=True,blind_review_inputs_read=False))
    print(json.dumps(dict(states=n,sources=len(sources),cost_normalization=normalized_results),indent=2))

if __name__=='__main__': main()
