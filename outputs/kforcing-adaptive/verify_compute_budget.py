"""Separate reconstruction of branch work, policy averages, and source bootstrap."""
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/compute-budget'; SRC=ROOT/'results/decision-confirm'
POLICIES=['keep4','all_split','random_half','combined19','confidence14','heuristic']
FIELDS=['gain','calls','candidates','visible_tokens','direct_calls','direct_candidates','future_calls','future_candidates','identical_interventions','identical_extra_calls','identical_extra_candidates']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(a,b): np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)

def main():
    manifest=read(OUT/'manifest.json'); actual=read(OUT/'analysis.json')
    for name,h in manifest['input_hashes'].items(): assert sha(ROOT/name)==h,name
    for name,h in manifest['outputs'].items(): assert sha(OUT/name)==h,name
    pair_rows=rows(SRC/'test_pairs.jsonl'); labels={r['pair_id']:r for r in rows(SRC/'test_labels.jsonl')}
    assert len(pair_rows)==len(labels)==5448 and len({r['pair_id'] for r in pair_rows})==5448
    grouped=defaultdict(list); branch_checks=0
    for row in pair_rows:
        label=labels[row['pair_id']];assert label['valid_label']
        offset=row['offset'];assert offset in (0,8)
        for branch in ['keep4','split22']:
            item=row[branch];length=len(item['ids'])-6
            assert item['ids'][:6+offset]==row['history_ids']
            assert 102 not in item['ids'][6:-1]
            assert item['stopped_eos']==(item['ids'][-1]==102)
            assert item['stopped_eos'] or length==122
            # Walk actual scheduled window widths, independently of ceil-based formulas.
            widths=[4 for _ in range(0,offset,4)]+[4]
            if branch=='split22': widths.append(2)
            cursor=offset+4
            while cursor<length:
                width=min(4,122-cursor); assert width>0
                widths.append(width);cursor+=width
            assert len(widths)==item['calls']==label[branch]['forward_calls']
            assert sum(widths)==item['computed_candidates']==label[branch]['computed_candidates']
            assert length==label[branch]['visible_new_tokens']
            check(label[branch]['gpt2_mean_nll'],label[branch]['gpt2_nll_sum']/label[branch]['gpt2_scored_tokens'])
            branch_checks+=1
        grouped[row['state_id']].append(row)
    states=sorted(grouped);n=len(states);assert n==681
    saved={r['state_id']:r for r in rows(OUT/'state_statistics.jsonl')}
    assert set(saved)==set(states)
    source_ids=sorted({saved[s]['prompt_id'] for s in states});assert len(source_ids)==380
    values=np.empty((n,len(FIELDS)));keep=np.empty((n,3))
    prediction=read(SRC/'frozen_predictions.json');weights=np.zeros((n,6))
    weights[:,1]=1;weights[:,2]=(n//2)/n
    for j,name in enumerate(POLICIES[3:],3):
        ordered=sorted(states,key=lambda s:(-prediction['scores'][s][name],s))
        assert ordered==prediction['rankings'][name]
        selected=set(ordered[:n//2]);weights[:,j]=[s in selected for s in states]
    for i,sid in enumerate(states):
        batch=grouped[sid];assert len(batch)==8 and {r['replicate'] for r in batch}==set(range(8))
        assert {r['prompt_id'] for r in batch}=={saved[sid]['prompt_id']}
        members=[];baselines=[]
        for r in batch:
            l=labels[r['pair_id']];a,b=r['keep4'],r['split22']
            g=l['keep4']['gpt2_mean_nll']-l['split22']['gpt2_mean_nll'];check(g,l['gain'])
            calls=b['calls']-a['calls'];candidates=b['computed_candidates']-a['computed_candidates']
            ident=int(a['ids']==b['ids'])
            members.append([g,calls,candidates,len(b['ids'])-len(a['ids']),1,2,calls-1,candidates-2,ident,ident*calls,ident*candidates])
            baselines.append([a['calls'],a['computed_candidates'],len(a['ids'])-6])
        values[i]=[math.fsum(r[j] for r in members)/8 for j in range(len(FIELDS))]
        keep[i]=[math.fsum(r[j] for r in baselines)/8 for j in range(3)]
        check(values[i],[saved[sid]['deltas'][k] for k in FIELDS])
        check(keep[i],[saved[sid]['keep'][k] for k in ['calls','candidates','visible_tokens']])
        check(weights[i],[saved[sid]['selection'][k] for k in POLICIES])
    indices=np.load(OUT/'bootstrap_indices.npz',allow_pickle=False)
    np.testing.assert_array_equal(indices['source_ids'],source_ids)
    draws=indices['draws'];np.testing.assert_array_equal(draws,np.random.default_rng(22222001).integers(0,380,size=(4000,380),dtype=np.int32))
    source_rows=[[i for i,s in enumerate(states) if saved[s]['prompt_id']==source] for source in source_ids]
    counts=np.array([len(g) for g in source_rows])
    # Multinomial multiplicities and matrix multiplication, not indexed repeated-sum aggregation.
    multiplicity=np.vstack([np.bincount(draw,minlength=380) for draw in draws])
    denominator=multiplicity@counts
    contributions=weights[:,:,None]*values[:,None,:]
    totals=np.array([contributions[g].sum(0) for g in source_rows])
    boot=(multiplicity@totals.reshape(380,-1)/denominator[:,None]).reshape(4000,6,len(FIELDS))
    point=contributions.mean(0)
    base_sums=np.array([keep[g].sum(0) for g in source_rows])
    base_boot=(multiplicity@base_sums)/denominator[:,None]
    interval_checks=0
    for j,policy in enumerate(POLICIES):
        report=actual['policies'][policy]
        check(report['expected_intervention_probability'],weights[:,j].mean())
        for f,field in enumerate(FIELDS):
            check(report['changes_vs_keep4'][field]['estimate'],point[j,f])
            check(report['changes_vs_keep4'][field]['interval'],np.percentile(boot[:,j,f],[2.5,97.5]));interval_checks+=1
        for f,field in enumerate(['calls','candidates','visible_tokens']):
            delta=point[j,f+1]
            check(report['absolute'][field]['estimate'],keep[:,f].mean()+delta)
            check(report['absolute'][field]['interval'],np.percentile(base_boot[:,f]+boot[:,j,f+1],[2.5,97.5]))
            check(report['percent_change_vs_keep4'][field]['estimate'],100*delta/keep[:,f].mean())
            check(report['percent_change_vs_keep4'][field]['interval'],np.percentile(100*boot[:,j,f+1]/base_boot[:,f],[2.5,97.5]));interval_checks+=2
        check(point[j,1],point[j,4]+point[j,6]);check(point[j,2],point[j,5]+point[j,7])
        if weights[:,j].sum():check(report['identical_fraction_among_interventions'],point[j,8]/weights[:,j].mean())
        else: assert report['identical_fraction_among_interventions'] is None
    normalized_point_checks=0
    for field,cost_index in [('calls',1),('candidates',2)]:
        result=actual['cost_normalization'][field]
        cost=point[2:,cost_index];gain=point[2:,0]
        feasible=np.all(boot[:,2:,cost_index]>0,axis=1)
        assert result['bootstrap_infeasible']==int(np.count_nonzero(~feasible))
        assert result['feasible_at_point']==bool(np.all(cost>0))
        assert np.all(cost>0) and np.all(feasible)  # observed cohort: no conditional dropping
        target=min(cost); reduced=np.array([g*target/c for g,c in zip(gain,cost)])
        target_boot=np.min(boot[:,2:,cost_index],axis=1)
        reduced_boot=boot[:,2:,0]*target_boot[:,None]/boot[:,2:,cost_index]
        check(result['common_extra_cost'],target)
        for j,policy in enumerate(POLICIES[2:]):
            entry=result['normalized_gain'][policy]
            check(entry['estimate'],reduced[j]);check(entry['matched_extra_cost'],target)
            check(result['retention_probability'][policy],target/cost[j])
            assert 0<result['retention_probability'][policy]<=1
            check(entry['expected_intervention_probability'],(340/681)*target/cost[j])
            check(entry['interval'],np.percentile(reduced_boot[:,j],[2.5,97.5]));interval_checks+=1
            normalized_point_checks+=1
        for j,policy in enumerate(POLICIES[2:]):
            if policy=='combined19': continue
            entry=result['comparisons']['combined19_vs_'+policy]
            check(entry['estimate'],reduced[1]-reduced[j]);check(entry['interval'],np.percentile(reduced_boot[:,1]-reduced_boot[:,j],[2.5,97.5]));interval_checks+=1
    original=read(SRC/'analysis.json')
    for name,col in [('vs_random',2),('vs_confidence14',4),('vs_heuristic',5)]:
        check(original['primary'][name]['estimate'],point[3,0]-point[col,0])
    assert original['advance_gate_passed'] is False and actual['new_advance_gate_defined'] is False
    # Analytic synthetic cases exercise guards and scaling without any human ratings.
    from audit_compute_budget import normalized
    case=normalized([3,3,2,4],[3,6,3,12])
    check(case['gains'],[3,1.5,2,1]);check(case['matched_costs'],[3,3,3,3])
    check(normalized([3,3,2,4],[30,60,30,120])['gains'],[3,1.5,2,1])
    check(normalized([3,-1,0,4],[2,2,2,2])['gains'],[3,-1,0,4])
    assert normalized([1,1,1,1],[0,1,1,1]) is None
    assert normalized([1,1,1,1],[-1,1,1,1]) is None
    assert normalized([1,float('nan'),1,1],[1,1,1,1]) is None
    result=dict(passed=True,verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'),
        analysis_sha256=sha(OUT/'analysis.json'),branch_schedules_reconstructed=branch_checks,state_aggregates_checked=n,
        frozen_rankings_checked=3,old_primary_points_reproduced=3,bootstrap_repeats=4000,
        bootstrap_aggregation='independent multinomial matrix path',intervals_checked=interval_checks,
        normalized_policy_points_checked=normalized_point_checks,analytic_synthetic_checks=6,
        old_failure_gate_unchanged=True,wall_clock_measured=False,human_ratings_received=0)
    (OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
