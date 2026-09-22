"""Independent CPU audit of EOS traces, calibration, isolation and statistics."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return r['prompt_id'],r['seed'],r['method']


def audit(root,generation_only=False):
    code=root.parent.parent
    m=read(root/'manifest.json');prefixes=read(root/'prefixes.json');sel=read(root/'frozen_model.json')
    lock=read(root/'locked_calibration.json');env=read(root/'environment.json')
    for f,k in [('prefixes.json','prefixes_sha256'),('frozen_model.json','model_sha256'),('projection.npy','projection_sha256')]:
        assert sha(root/f)==m[k]
    assert sha(root/'frozen_model.json')==sha(root.parent/'learned-007/locked_model.json')
    assert sha(root/'projection.npy')==sha(root.parent/'learned-007/projection.npy')
    assert sha(code/'STAGE10_PROTOCOL.md')==m['protocol_sha256']
    assert sha(code/'learn_local_gain.py')==m['feature_code_sha256']
    assert sha(code/'early_stop_decoder.py')==lock['decoder_sha256']==env['decoder_sha256']
    assert sha(code/'run_early_stop.py')==lock['runner_sha256']==env['runner_sha256']
    assert sha(root/'locked_calibration.json')==env['locked_calibration_sha256']
    assert sha(root/'calibration.jsonl')==lock['calibration_sha256']
    assert sha(root/'prefixes.json')==lock['prefixes_sha256']
    assert not lock['quality_labels_used'] and not lock['test_generated']
    old=read(root.parent/'lm1b-004/prefixes.json')+read(root.parent/'online-009/prefixes.json')
    for group in read(root.parent/'learned-007/prefixes.json').values():old+=group
    old += [dict(p,ids=p['reference_ids'][:6]) for p in read(root.parent/'confirm-008/sources.json')]
    old_ids={tuple(p['ids'][:6]) for p in old}
    old_rows={p['source_row'] for p in old};old_hash={p['source_sha256'] for p in old}
    for folder in ('pilot-001','benefit-002'):
        for group in read(root.parent/folder/'prefixes.json').values():old_ids.update(tuple(p['ids'][:6]) for p in group)
    old_ids.update(tuple(p['ids'][:6]) for p in read(root.parent/'current-003/fresh_prefixes.json'))
    assert len(prefixes)==128 and len({tuple(p['ids']) for p in prefixes})==128
    for field in ('source_row','source_sha256','prompt_id'):assert len({p[field] for p in prefixes})==128
    for p in prefixes:
        assert tuple(p['ids']) not in old_ids and p['source_row'] not in old_rows and p['source_sha256'] not in old_hash
    pm={p['prompt_id']:p for p in prefixes}
    calp=[p for p in prefixes if p['split']=='calibration'];testp=[p for p in prefixes if p['split']=='test']
    assert len(calp)==32 and len(testp)==96
    data=rows(root/'samples.jsonl');cal=rows(root/'calibration.jsonl')
    discovery=rows(root/'replay_discovery.jsonl');dm={(r['prompt_id'],r['seed']):r for r in discovery}
    assert len(dm)==len(discovery)==192
    def trace(r,is_cal):
        p=pm[r['prompt_id']]
        assert p['split']==('calibration' if is_cal else 'test')
        assert r['noise_seed']==101000000+r['seed']+p['index']*100003
        assert r['ids']==r['visible_ids'] and r['ids'][:6]==p['ids']
        assert len(r['ids'])==6+r['visible_new_tokens'] and 1<=r['visible_new_tokens']<=122
        new=r['ids'][6:]
        assert 102 not in new[:-1] and r['stopped_eos']==(new[-1]==102)
        assert r['stopped_eos'] or len(new)==122
        isfixed=r['method'].startswith('fixed');israndom=r['method'].startswith('random')
        probability=r.get('probability')
        if israndom:
            assert probability==(int(r['method'].split('_')[1])/10 if is_cal else lock['probabilities'][r['method']])
        randoms=np.random.default_rng(r['noise_seed']+10102003).random(31)
        pos=6;emitted=[]
        for i,s in enumerate(r['steps']):
            width=min(int(r['method'][-1]) if isfixed else 4,128-pos)
            assert s['position']==pos and s['width']==width
            initial=s['initial_candidates'];final=s['final_candidates']
            assert len(initial)==len(final)==width
            eligible=width==4 and 102 not in initial[:2] and not isfixed
            assert s['eligible']==eligible
            if not eligible:
                assert not s['split'] and s['score'] is None and s['uniform'] is None
            elif r['method']=='learned':
                assert math.isfinite(s['score']) and s['split']==(s['score']>sel['dev_threshold'])
                assert s['uniform'] is None
            elif israndom:
                assert s['uniform']==float(randoms[(pos-6)//4])
                assert s['split']==(s['uniform']<probability) and s['score'] is None
            else:
                assert r['method']=='learned_replay' and not is_cal
                assert s['split']==dm[(r['prompt_id'],r['seed'])]['schedule'][(pos-6)//4]
                assert s['score'] is None and s['uniform'] is None
            assert (final[:2]==initial[:2]) if s['split'] else (final==initial)
            kept=final.index(102)+1 if 102 in final else width
            assert s['emitted']==kept and s['forward_calls']==1+int(s['split'])
            assert s['computed_candidates']==width+2*int(s['split'])
            emitted+=final[:kept];pos+=kept
            if 102 in final:assert i==len(r['steps'])-1
        assert emitted==new and pos==len(r['ids'])
        for field in ('forward_calls','computed_candidates'):assert r[field]==sum(s[field] for s in r['steps'])
        assert r['split_count']==sum(s['split'] for s in r['steps'])
        assert r['eligible_windows']==sum(s['eligible'] for s in r['steps'])
    calmethods=['learned']+[f'random_{i:02}' for i in range(11)]
    expectedcal={(p['prompt_id'],seed,method) for p in calp for seed in m['seeds'] for method in calmethods}
    assert len(cal)==len({key(r) for r in cal})==768 and {key(r) for r in cal}==expectedcal
    counts={}
    for r in cal:trace(r,True)
    for method in calmethods:
        group=[r for r in cal if r['method']==method]
        counts[method]=dict(calls=sum(r['forward_calls'] for r in group),tokens=sum(r['visible_new_tokens'] for r in group),
                            candidates=sum(r['computed_candidates'] for r in group))
    assert counts==lock['counts']
    target=counts['learned']['calls']/counts['learned']['tokens']
    assert target==lock['learned_calls_per_token']
    best=min(range(11),key=lambda i:(abs(math.log((counts[f'random_{i:02}']['calls']/counts[f'random_{i:02}']['tokens'])/target)),i))
    assert lock['chosen']['probability']==best/10
    assert lock['probabilities']==dict(random_low=round(max(0,best/10-.1),1),random_cal=best/10,random_high=round(min(1,best/10+.1),1))
    expected={(p['prompt_id'],seed,method) for p in testp for seed in m['seeds'] for method in m['methods']}
    index={key(r):r for r in data}
    assert len(data)==len(index)==1536 and set(index)==expected
    for r in data:trace(r,False)
    for ident,d in dm.items():
        a=index[ident+('learned',)];b=index[ident+('learned_replay',)]
        assert a['ids']==b['ids']==d['ids']
        assert a['forward_calls']==b['forward_calls'] and a['computed_candidates']==b['computed_candidates']
        wanted=[False]*30
        for s in a['steps']:
            if s['width']==4:wanted[(s['position']-6)//4]=s['split']
        assert wanted==d['schedule']
    timing=rows(root/'timings.jsonl');ts=defaultdict(list);seen=set();orders=defaultdict(set)
    for t in timing:
        k=key(t);unique=k+(t['repetition'],)
        assert k in expected and unique not in seen;seen.add(unique)
        assert math.isfinite(t['seconds']) and t['seconds']>0
        assert t['ids_sha256']==hashlib.sha256(json.dumps(index[k]['ids']).encode()).hexdigest()
        ts[k].append(t['seconds']);orders[(t['prompt_id'],t['seed'],t['repetition'])].add(t['order_index'])
    assert len(timing)==4608 and seen=={k+(rep,) for k in expected for rep in range(3)}
    assert all(order==set(range(8)) for order in orders.values())
    assert read(root/'preflight.json')['passed'] and read(root/'generation_checks.json')['passed']
    result=dict(passed=True,new_sources=128,calibration_sources=32,test_sources=96,calibration_outputs=768,
        test_outputs=1536,timing_runs=4608,no_post_eos_windows=True,eos_and_recomputation_counts=True,
        frozen_predictor=True,calibration_selection_independently_rebuilt=True,
        random_choices_independently_rebuilt=True,replay_equivalence_pairs=192,data_isolation=True)
    if not generation_only:
        scores=rows(root/'scores.jsonl');sc={key(r):r for r in scores};report=read(root/'analysis.json')
        assert len(scores)==len(sc)==1536 and set(sc)==expected
        arrays={}
        for r in scores:
            a=index[key(r)]
            assert r['teacher_token_count']==r['visible_new_tokens']==a['visible_new_tokens']
            assert r['gpt2_scored_tokens']==max(0,r['gpt2_input_tokens']-1)
            assert r['excluded_short']==(r['gpt2_input_tokens']<2)
            assert math.isfinite(r['gpt2_nll_sum']) and r['gpt2_nll_sum']>=0
        pids=sorted(p['prompt_id'] for p in testp)
        for method in m['methods']:
            a=[]
            for pid in pids:
                kk=[(pid,seed,method) for seed in m['seeds']]
                a.append([sum(sc[k]['gpt2_nll_sum'] for k in kk),sum(sc[k]['gpt2_scored_tokens'] for k in kk),
                    sum(sum(ts[k])/3 for k in kk),sum(index[k]['visible_new_tokens'] for k in kk),
                    sum(index[k]['forward_calls'] for k in kk),sum(index[k]['computed_candidates'] for k in kk)])
            arrays[method]=np.array(a,dtype=np.float64);tot=arrays[method].sum(0);s=report['summary'][method]
            np.testing.assert_allclose([math.exp(tot[0]/tot[1]),tot[3]/tot[2],tot[2]/192*1000,tot[4]/tot[3],tot[5]/tot[3]],
                [s['gpt2_gen_ppl'],s['visible_tokens_per_second'],s['mean_request_ms'],s['calls_per_visible_token'],s['candidates_per_visible_token']],rtol=1e-12)
        draws=np.random.default_rng(10102004).integers(0,96,size=(4000,96))
        weights=np.stack([np.bincount(d,minlength=96) for d in draws])
        a=weights@arrays['learned'];b=weights@arrays['random_cal']
        for field,v in dict(ppl_ratio=np.exp(a[:,0]/a[:,1]-b[:,0]/b[:,1]),
                            visible_throughput_ratio=(a[:,3]/a[:,2])/(b[:,3]/b[:,2])).items():
            np.testing.assert_allclose(np.quantile(v,[.0125,.9875]),report['primary'][field]['interval'],rtol=1e-12,atol=1e-12)
        a=arrays['learned'].sum(0);b=arrays['random_cal'].sum(0)
        ratio=(a[4]/a[3])/(b[4]/b[3]);comparable=bool(.95<=ratio<=1.05)
        assert abs(ratio-report['budget_calls_per_token_ratio'])<1e-12
        assert comparable==report['budget_within_predeclared_5pct']
        gate=report['primary']['ppl_ratio']['interval'][1]<1 and report['primary']['visible_throughput_ratio']['interval'][0]>1
        assert gate==report['primary_two_metric_gate']
        assert (gate and comparable)==report['primary_budget_and_performance_gate']
        # Replay is not a new quality sample: it must inherit identical scores.
        for pid in pids:
            for seed in m['seeds']:
                a=sc[(pid,seed,'learned')];b=sc[(pid,seed,'learned_replay')]
                assert a['gpt2_scored_tokens']==b['gpt2_scored_tokens']
                assert abs(a['gpt2_nll_sum']-b['gpt2_nll_sum'])<.002
        result.update(scoring_aggregates=True,primary_bootstrap_independently_rebuilt=True,
            replay_quality_crosscheck=True,budget_within_5pct=comparable,
            primary_budget_and_performance_gate=gate and comparable)
    path=root/('generation_audit.json' if generation_only else 'integrity_audit.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--generation-only',action='store_true')
    args=ap.parse_args();audit(args.root,args.generation_only)
