"""Independent CPU reconstruction of the local-mechanism diagnostic."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/local-mechanism'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,atol=1e-11):np.testing.assert_allclose(a,b,atol=atol,rtol=0)
def visible(tokens):
    result=[]
    for token in tokens:
        result.append(token)
        if token==102:break
    return result

def main(output=None):
    manifest=read(OUT/'manifest.json');scoring=read(OUT/'scoring.json');analysis=read(OUT/'analysis.json')
    expected={'results/features-014/development_data.jsonl','results/attention-features/oof_predictions.jsonl',
        'results/attention-features/analysis.json','results/averaged-013/train_states.jsonl',
        'results/averaged-013/dev_states.jsonl','LOCAL_MECHANISM_PLAN.md','diagnose_local_mechanism.py','core.py',
        'upstream/models/autoregressive.py','upstream/models/transformer.py'}
    assert set(manifest['inputs'])==expected
    for name,digest in manifest['inputs'].items():assert sha(ROOT/name)==digest,name
    for inventory in [scoring['outputs'],analysis['output_hashes']]:
        for name,digest in inventory.items():assert sha(OUT/name)==digest,name
    seeds=[14112001,14112002,14112003]
    assert manifest['fold_seeds']==seeds and manifest['bootstrap_seed']==22242001
    assert not manifest['old_test_inputs'] and not manifest['new_model_fit']
    data=rows(ROOT/'results/features-014/development_data.jsonl');local=rows(OUT/'local_scores.jsonl')
    stats=rows(OUT/'state_statistics.jsonl');oof=rows(ROOT/'results/attention-features/oof_predictions.jsonl')
    ids=[r['state_id'] for r in data]
    assert ids==sorted(ids) and len(set(ids))==446
    for records in [local,stats,oof]:
        assert [r['state_id'] for r in records]==ids
        assert [r['prompt_id'] for r in records]==[r['prompt_id'] for r in data]
    assert {r['source_split'] for r in data}=={'train','dev'}
    states={r['state_id']:r for split in ['train','dev'] for r in rows(ROOT/f'results/averaged-013/{split}_states.jsonl') if r['status']=='paired'}
    assert set(ids).issubset(states) and set(states)-set(ids)=={'s13-122-1301-0'}
    probes=np.load(OUT/'teacher_probes.npz',allow_pickle=False)
    indices=np.linspace(0,445,12,dtype=int).tolist()
    assert indices==manifest['teacher_probe_indices']
    assert set(probes.files)=={f'{i}_{branch}' for i in indices for branch in ['keep4','split22']}
    reconstructed=[];active=[];eos=[];raw_change=[];max_nll_error=0.;probe_positions=0
    for i,(d,r) in enumerate(zip(data,local)):
        s=states[d['state_id']]
        assert s['prompt_id']==d['prompt_id'] and s['offset']==d['offset']==r['offset']
        assert len(s['history_ids'])==6+s['offset']
        assert s['initial_candidates'][:2]==s['split_candidates'][:2] and 102 not in s['initial_candidates'][:2]
        losses={};blocks={}
        for branch,key in [('keep4','initial_candidates'),('split22','split_candidates')]:
            block=visible(s[key]);record=r[branch];blocks[branch]=block
            assert record['visible_candidates']==block and record['tail_tokens']==block[2:]
            losses[branch]=np.array(record['losses'],dtype=np.float64)
            assert len(losses[branch])==len(block)-2 and len(block) in [3,4]
            assert np.isfinite(losses[branch]).all() and (losses[branch]>=0).all()
            close(losses[branch],np.array(record['logsumexp'])-record['target_logits'],atol=2e-5)
            if i in indices:
                logits=probes[f'{i}_{branch}'].astype(np.float64)
                assert logits.shape==(len(block)-2,30522)
                maximum=np.max(logits,axis=1)
                lse=maximum+np.log(np.sum(np.exp(logits-maximum[:,None]),axis=1))
                selected=logits[np.arange(len(block)-2),block[2:]]
                independent=lse-selected
                close(independent,losses[branch],atol=2e-5)
                close(selected,record['target_logits'],atol=0)
                close(lse,record['logsumexp'],atol=2e-5)
                max_nll_error=max(max_nll_error,float(np.max(abs(independent-losses[branch]))))
                probe_positions+=len(independent)
        length=min(len(losses['keep4']),len(losses['split22']))
        value=float(sum(losses['keep4'][:length]-losses['split22'][:length])/length)
        assert r['common_tail_tokens']==length;close(value,r['local_gain'])
        changed=blocks['keep4']!=blocks['split22'];has_eos=any(102 in b for b in blocks.values())
        raw=s['initial_candidates']!=s['split_candidates']
        assert r['active']==changed and r['eos_in_either']==has_eos and r['raw_candidates_changed']==raw
        active.append(changed);eos.append(has_eos);raw_change.append(raw);reconstructed.append(value)
    assert scoring['passed'] and scoring['states']==446 and scoring['branches']==892
    assert scoring['probe_states']==12 and scoring['independent_prefix_positions']==probe_positions
    assert scoring['new_future_generations']==0 and scoring['dtype']=='fp32' and not scoring['tf32']
    x=np.array(reconstructed);active=np.array(active);eos=np.array(eos)
    y=np.array([r['gains'] for r in data]);assert y.shape==(446,8)
    g=np.mean(y,axis=1);close(g,[r['gain'] for r in data])
    assert (y[~active]==0).all() and (x[~active]==0).all()
    source_names=sorted({r['prompt_id'] for r in data});assert len(source_names)==253
    source_lookup={s:i for i,s in enumerate(source_names)}
    state_source=np.array([source_lookup[r['prompt_id']] for r in data])
    folds=read(OUT/'folds.json');lookup={(r['repeat'],r['fold']):r for r in folds};assert len(lookup)==len(folds)==15
    fraction=np.zeros((3,446));change_prob=np.zeros((3,446));teacher=np.zeros((3,446));prior=np.zeros((3,446));future=np.zeros((3,446))
    first=y[:,:4].mean(axis=1);last=y[:,4:].mean(axis=1)
    def choose(values,indices):
        ranks=sorted(indices,key=lambda j:(-values[j],ids[j]));chosen=set(ranks[:len(indices)//2])
        return np.array([float(j in chosen) for j in indices])
    for repeat,seed in enumerate(seeds):
        permuted=np.random.default_rng(seed).permutation(source_names).tolist()
        for fold in range(5):
            names=set(permuted[fold::5]);valid=np.array([i for i,r in enumerate(data) if r['prompt_id'] in names])
            r=lookup[repeat,fold];assert r['valid']==valid.tolist()
            budget=len(valid)//2;na=sum(active[valid]);ni=len(valid)-na
            assert r['budget']==budget and r['active']==na and r['inactive']==ni
            pa=min(budget,na)/na if na else 0.;pi=(budget-min(budget,na))/ni if ni else 0.
            close(pa,r['active_selection_probability']);close(pi,r['inactive_selection_probability'])
            fraction[repeat,valid]=budget/len(valid)
            change_prob[repeat,valid]=[pa if active[j] else pi for j in valid]
            close(change_prob[repeat,valid].sum(),budget)
            teacher[repeat,valid]=choose(x,valid)
            predictions=np.array([r['predictions']['combined19'][repeat] for r in oof])
            prior[repeat,valid]=choose(predictions,valid)
            close(prior[repeat,valid],[oof[j]['selected']['combined19'][repeat] for j in valid],atol=0)
            q=budget/len(valid)
            future[repeat,valid]=((choose(first,valid)-q)*last[valid]+(choose(last,valid)-q)*first[valid])/2
    values={'perfect_change':((change_prob-fraction)*g).mean(axis=0),
        'local_teacher':((teacher-fraction)*g).mean(axis=0),
        'combined19':((prior-fraction)*g).mean(axis=0),'cross_half_future':future.mean(axis=0)}
    for i,r in enumerate(stats):
        close(r['mean_full_gain'],g[i]);close(r['local_gain'],x[i]);assert r['active']==bool(active[i]) and r['eos_in_either']==bool(eos[i])
        for key,array in [('random_fraction',fraction),('change_probabilities',change_prob),('teacher_selected',teacher),('prior19_selected',prior)]:
            close(r[key],array[:,i])
        for name,v in values.items():close(r['contributions_vs_random'][name],v[i])

    rng=np.random.default_rng(22242001)
    counts=np.array([np.bincount(rng.integers(253,size=253),minlength=253) for _ in range(4000)])
    checked_intervals=0
    def interval(v,record,mask=None):
        nonlocal checked_intervals
        if mask is None:mask=np.ones(446,dtype=bool)
        num=np.bincount(state_source[mask],weights=v[mask],minlength=253)
        den=np.bincount(state_source[mask],minlength=253);sizes=counts@den;ok=sizes>0
        assert record['coverage']==.95 and record['defined_bootstraps']==int(ok.sum())
        if mask.any():
            close(record['estimate'],np.mean(v[mask]));close(record['interval'],np.quantile((counts@num)[ok]/sizes[ok],[.025,.975]))
        else:assert record['estimate'] is None and record['interval'] is None
        checked_intervals+=1
    def correlation(a,b,expected):
        if len(a)<2 or np.std(a)==0 or np.std(b)==0:assert expected is None
        else:
            ac=a-a.mean();bc=b-b.mean();close(expected,np.dot(ac,bc)/np.sqrt(np.dot(ac,ac)*np.dot(bc,bc)))
    masks={'inactive':~active,'active_no_eos':active&~eos,'active_eos':active&eos}
    assert set(analysis['subgroups'])==set(masks)
    for name,mask in masks.items():
        r=analysis['subgroups'][name];assert r['states']==int(mask.sum()) and r['sources']==len(set(state_source[mask]))
        interval(g,r['mean_full_gain'],mask);interval(x,r['mean_local_gain'],mask)
        correlation(x[mask],g[mask],r['local_full_correlation'])
    correlation(x,g,analysis['local_full_correlation'])
    for name,v in values.items():interval(v,analysis['methods_vs_random'][name])
    interval(values['local_teacher']-values['combined19'],analysis['primary_local_teacher_minus_combined19'])
    assert analysis['local_target_followup_signal']==(analysis['primary_local_teacher_minus_combined19']['interval'][0]>0)
    assert analysis['active_states']==int(active.sum()) and analysis['inactive_states']==int((~active).sum())
    assert analysis['raw_candidate_changes']==sum(raw_change) and analysis['inactive_all_eight_gains_zero']
    assert analysis['local_gain_sign_counts']=={'positive':int((x>0).sum()),'zero':int((x==0).sum()),'negative':int((x<0).sum())}
    assert analysis['states']==446 and analysis['sources']==253 and analysis['retrospective_development']
    for key in ['independent_confirmation','new_model_fit','deployable_policy','new_online_timing','new_human_quality_result','old_test_inputs']:
        assert analysis[key] is False
    old=read(ROOT/'results/attention-features/analysis.json')['methods']['combined19']['gain_vs_random']['estimate']
    close(values['combined19'].mean(),old)
    report=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'),
        analysis_sha256=sha(OUT/'analysis.json'),scoring_sha256=sha(OUT/'scoring.json'),
        states_verified=446,branches_verified=892,teacher_probe_states=12,teacher_probe_token_losses=probe_positions,
        max_float64_nll_error=max_nll_error,folds_reconstructed=15,intervals_recomputed=checked_intervals,
        prior19_point_reproduced=True,train_dev_whitelist_verified=True,primary_decision_reproduced=True,
        local_target_followup_signal=analysis['local_target_followup_signal'],
        scope='Independent arithmetic on saved evidence; not independent confirmation, a deployable policy, or human quality evidence.')
    if output is not None:output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);main(parser.parse_args().output)
