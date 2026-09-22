"""Independent standard-library audit of source versions and local comparison records."""
from collections import Counter,defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def close(a,b):
    assert math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def main():
    root=Path(sys.argv[1])
    config=read(root/'collection_checks.json')
    rows=lines(root/'windows.jsonl')
    selected=lines(root/'selection_records.jsonl')
    prefixes={p['id']:p['ids'] for p in read(root.parent/'lm1b-004/prefixes.json')}
    expected=set(itertools.product(prefixes,config['seeds']))
    assert len(rows)==len(selected)==len(expected)==1024
    assert {(r['prompt_id'],r['seed']) for r in rows}==expected
    smap={(r['prompt_id'],r['seed']):r for r in selected}
    assert set(smap)==expected
    for r in rows:
        s=smap[r['prompt_id'],r['seed']]
        assert all(s[k]==v for k,v in r.items())
        assert r['prefix']==prefixes[r['prompt_id']]
        assert len(r['noise'])==4 and all(0<=u<1 for u in r['noise'])
        assert len(r['margins'])==4 and all(math.isfinite(m) and m>=0 for m in r['margins'])
        close(r['tail_margin'],sum(r['margins'][2:])/2)
        assert r['fixed4'][:2]==r['split22'][:2]
        assert r['no_eos_all_branches']==all(102 not in r[k] for k in ('fixed4','split22','ar_reference'))
        for name in ('fixed4','split22'):
            assert len(r[name])==4 and all(isinstance(t,int) and 0<=t<30522 for t in r[name])
            mask=[a==b for a,b in zip(r[name],r['ar_reference'])]
            a=s[name+'_agreement']
            assert a==dict(matches=sum(mask),by_position=mask,exact=all(mask),matching_prefix=next((i for i,x in enumerate(mask) if not x),4))
        assert s['gain']==s['split22_agreement']['matches']-s['fixed4_agreement']['matches']
    analysis=read(root/'analysis.json')
    budget=analysis['selection_budget']
    assert budget==512
    for policy in ('signal','random','oracle'):
        assert sum(r[policy+'_split'] for r in selected)==budget
    for policy,key in (('signal',lambda r:(r['tail_margin'],r['prompt_id'],r['seed'])),
                       ('oracle',lambda r:(-r['gain'],r['prompt_id'],r['seed']))):
        expected_selected={(r['prompt_id'],r['seed']) for r in sorted(selected,key=key)[:budget]}
        assert {(r['prompt_id'],r['seed']) for r in selected if r[policy+'_split']}==expected_selected
    for name,subset in (('all_windows',selected),('no_eos_subset',[r for r in selected if r['no_eos_all_branches']])):
        for policy,s in analysis[name].items():
            chosen=[r[policy+'_agreement'] if policy in ('fixed4','split22') else r[('split22' if r[policy+'_split'] else 'fixed4')+'_agreement'] for r in subset]
            assert s['windows']==len(chosen)
            close(s['token_match_fraction'],sum(r['matches'] for r in chosen)/(4*len(chosen)))
            close(s['exact_window_fraction'],statistics.mean(r['exact'] for r in chosen))
            close(s['mean_matching_prefix'],statistics.mean(r['matching_prefix'] for r in chosen))
            for j in range(4):
                close(s['by_position_match_fraction'][j],statistics.mean(r['by_position'][j] for r in chosen))
    counts=Counter(r['gain'] for r in selected)
    assert analysis['gain_counts']=={str(i):counts[i] for i in range(-4,5)}
    values=dict(all_split_vs_fixed4=statistics.mean(r['gain']/4 for r in selected),
        signal_vs_random=statistics.mean((r['signal_split']-r['random_split'])*r['gain']/4 for r in selected),
        signal_vs_random_expectation=statistics.mean((r['signal_split']-.5)*r['gain']/4 for r in selected),
        oracle_vs_random=statistics.mean((r['oracle_split']-r['random_split'])*r['gain']/4 for r in selected))
    for name,value in values.items():
        c=analysis['comparisons'][name]
        close(c['token_match_fraction_difference'],value)
        assert len(c['interval95'])==2 and c['interval95'][0]<=c['interval95'][1]
    model=read(root/'model.json')
    commits=read(root/'commits.json')
    assert model['sha']=='16b984316bdd4585f17b11d0e61a58d236c2febc'
    assert commits[0]['sha']=='706caa332a69d509b7fc2fa53ac7819f381a8c78'
    weights=next(f for f in model['siblings'] if f['rfilename']=='pflm_lm1b_k4.ckpt')
    assert weights['lfs']['sha256']=='3b013f0cf9a3acab4c20a1bb748beee6525432ef3f56a74df4957b898480a2c2'
    assert config['first_two_argmax_disagreements']==0 and config['official_first_step_reference_checks']==16
    report=dict(passed=True,windows=1024,positions_per_branch=4096,prefix_clusters=128,
        seeds_per_prefix=8,selection_budget_each=512,no_eos_windows=sum(r['no_eos_all_branches'] for r in rows),
        current_public_code_matches=True,current_public_model_revision_and_lfs_match=True,
        api_issues_records=len(read(root/'issues.json')),
        limitation='Record and aggregate audit. Does not certify paper-checkpoint correspondence, independently rerun models or bootstrap/random selections, or establish distribution/quality gains.')
    (root/'integrity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
