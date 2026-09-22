"""Independent structural, data-isolation, controller, and timing audit."""
from collections import Counter,defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def main():
    root=Path(sys.argv[1])
    controller=load(root/'locked_controller.json')
    digest=hashlib.sha256((root/'locked_controller.json').read_bytes()).hexdigest()
    assert controller['passed_development_gate']
    calibration=load(root/'online_calibration.json')
    env=load(root/'online_environment.json')
    assert digest==calibration['controller_sha256']==env['controller_sha256']
    assert controller['threshold']==calibration['threshold']
    assert math.isclose(calibration['p_short'],calibration['short_steps']/calibration['eligible_steps'])
    training=rows(root/'train_dev_features.jsonl')
    assert {r['split'] for r in training}=={'train','dev'}
    assert hashlib.sha256((root/'train_dev_features.jsonl').read_bytes()).hexdigest()==controller['train_dev_sha256']
    fresh=load(root/'fresh_prefixes.json')
    inventory=load(root/'fresh_inventory.json')
    assert len(fresh)==32
    assert not {p['article_id'] for p in fresh}&set(inventory['excluded_article_ids'])
    assert max(Counter(p['article_id'] for p in fresh).values())<=4
    old=set()
    for folder in ('pilot-001','benefit-002'):
        for group in load(root.parent/folder/'prefixes.json').values():
            old|={tuple(r['ids']) for r in group}
    assert not old&{tuple(p['ids']) for p in fresh}
    local=rows(root/'fresh_scored.jsonl')
    assert len(local)==124
    for r in local:
        assert r['tokens4'][:2]==r['tokens22'][:2]
        if controller['kind']=='negative_tail_margin':
            expected=-(r['current_features'][2]+r['current_features'][3])/2
            assert math.isclose(expected,r['score'],abs_tol=5e-7)
        assert ('gain' in r)==(not r['eos_excluded'] and r['first_two_match'])
        if 'gain' in r:
            assert math.isclose(r['gain'],r['nll4']-r['nll22'],abs_tol=1e-12)
    raw=rows(root/'online_timings.jsonl')
    merged=rows(root/'online_samples.jsonl')
    expected=set(itertools.product([p['id'] for p in fresh],env['seeds'],env['policies'],range(3)))
    actual=[(r['prompt_id'],r['seed'],r['policy'],r['repeat']) for r in raw]
    assert len(actual)==len(set(actual))==960 and set(actual)==expected
    grouped=defaultdict(list)
    for r in raw:
        grouped[(r['prompt_id'],r['seed'],r['policy'])].append(r)
        assert r['new_tokens']==len(r['ids'])-r['prefix_length']==sum(s['emitted'] for s in r['steps'])
        assert r['computed_candidates']==sum(s['computed_k'] for s in r['steps'])
        assert r['computed_candidates']==r['new_tokens']+r['policy_discarded']+r['eos_discarded']
        assert r['seconds']>0 and math.isfinite(r['seconds'])
        if r['stopped_eos']:
            assert r['ids'][-1]==102 and 102 not in r['ids'][r['prefix_length']:-1]
        for i,s in enumerate(r['steps']):
            if r['policy']=='current' and s['eligible']:
                assert s['requested_k']==(2 if s['score']>controller['threshold'] else 4)
            if r['policy'] in ('current','random_commit') and i==0:
                assert s['requested_k']==4
            if r['policy'] in ('current','random_commit'):
                before=s['position']-r['prefix_length']
                assert s['computed_k']==min(4,64-before)
            if r['policy'].startswith('fixed'):
                assert s['score'] is None and s['policy_discarded']==0
    assert len(merged)==320 and len(grouped)==320
    keys=set()
    for r in merged:
        key=(r['prompt_id'],r['seed'],r['policy'])
        assert key not in keys
        keys.add(key)
        runs=grouped[key]
        assert len(runs)==3 and all(v['ids']==r['ids'] for v in runs)
        assert math.isclose(r['seconds'],sum(v['seconds'] for v in runs)/3,abs_tol=1e-12)
        assert r['teacher_token_count']==r['new_tokens'] and math.isfinite(r['teacher_nll_sum'])
    summary=load(root/'online_summary.json')
    budget={}
    for policy,s in summary.items():
        group=[r for r in merged if r['policy']==policy]
        n=sum(r['new_tokens'] for r in group)
        assert s['generated_tokens']==n and s['samples']==len(group)
        assert math.isclose(s['aggregate_tokens_per_second'],n/sum(r['seconds'] for r in group))
        assert math.isclose(s['teacher_nll'],sum(r['teacher_nll_sum'] for r in group)/n)
        steps=[step for r in group for step in r['steps'] if step['eligible']]
        if steps:
            budget[policy]=dict(eligible_steps=len(steps),short_eligible_steps=sum(s['requested_k']==2 for s in steps),
                short_eligible_fraction=sum(s['requested_k']==2 for s in steps)/len(steps))
    report=dict(passed=True,fresh_prompts=32,fresh_articles=len({p['article_id'] for p in fresh}),
        local_branches=len(local),timed_runs=len(raw),unique_outputs=len(merged),repeats_per_output=3,
        exact_repeat_token_matches=True,locked_controller_verified=True,discard_accounting_verified=True,
        budgets=budget,scope='Structural and arithmetic checks; not a claim of human quality or universal runtime stability.')
    (root/'integrity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
