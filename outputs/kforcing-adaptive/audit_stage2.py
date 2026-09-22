"""Independent data/provenance and eager-helper regression checks."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).parent


def read_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def audit():
    root=ROOT/'results/benefit-002'
    prefixes=json.loads((root/'prefixes.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'results/pilot-001/prefixes.json').read_text(encoding='utf-8'))
    old_ids={tuple(r['ids']) for rows in old.values() for r in rows}
    seen_ids=set()
    article_sets={}
    for split,rows in prefixes.items():
        assert len(rows)=={'train':96,'dev':32,'test':48}[split]
        article_sets[split]={r['article_id'] for r in rows}
        for r in rows:
            ids=tuple(r['ids'])
            assert ids not in old_ids and ids not in seen_ids
            seen_ids.add(ids)
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not article_sets[a]&article_sets[b]
    raw,scored=read_rows(root/'branches.jsonl'),read_rows(root/'scored.jsonl')
    assert len(raw)==len(scored)==662
    keys=set()
    for a,b in zip(raw,scored):
        key=(a['prompt_id'],a['state_index'],a['seed'])
        assert key not in keys
        keys.add(key)
        assert all(b[k]==v for k,v in a.items())
        assert len(a['tokens4'])==len(a['tokens22'])==4
        assert a['tokens4'][:2]==a['tokens22'][:2]
        assert a['features']['context_length']==len(a['context_ids'])
        assert all(math.isfinite(v) for v in a['features'].values())
        valid=not a['eos_excluded'] and a['first_two_match']
        assert ('gain' in b)==valid
        if valid:
            assert math.isclose(b['gain'],b['nll4']-b['nll22'],abs_tol=1e-12)
    model=json.loads((root/'locked_model.json').read_text())
    assert model['data_sha256']==hashlib.sha256((root/'scored.jsonl').read_bytes()).hexdigest()
    assert model['alpha']==min(model['candidates'],key=lambda c:c['dev']['mse'])['alpha']
    pairs=defaultdict(list)
    for r in scored:
        pairs[(r['prompt_id'],r['state_index'])].append(r)
    for r in json.loads((root/'heldout_predictions.json').read_text()):
        pair=pairs[(r['prompt_id'],r['state_index'])]
        assert len(pair)==2 and all(p['split']=='test' for p in pair)
        assert math.isclose(r['gain'],sum(p['gain'] for p in pair)/2)
        prediction=sum((r['features'][name]-m)/s*c for name,m,s,c in
                       zip(model['features'],model['mean'],model['std'],model['coef']))+model['intercept']
        assert math.isclose(prediction,r['scores']['ridge'],abs_tol=1e-12)
    dump(root/'integrity_audit.json',dict(passed=True,branches=662,prefixes=len(seen_ids),
        disjoint_article_splits=True,no_old_prefixes=True,locked_model_predictions_verified=True,
        limitation='Structural and arithmetic validation, not proof of quality or universal correctness.'))
    first=read_rows(ROOT/'results/pilot-001/samples.jsonl')
    recheck=read_rows(ROOT/'results/pilot-001-recheck/samples.jsonl')
    keyed=lambda rows:{(r['prompt_id'],r['seed'],r['policy']):r for r in rows}
    a,b=keyed(first),keyed(recheck)
    assert len(a)==len(b)==200 and a.keys()==b.keys()
    token_matches=sum(a[k]['ids']==b[k]['ids'] for k in a)
    max_metric=max(abs(a[k]['teacher_nll']-b[k]['teacher_nll']) for k in a)
    result=dict(samples=200,identical_token_sequences=token_matches,max_teacher_nll_difference=max_metric,
        same_data_and_seeds=True,interpretation='Regression recheck only; not independent new statistical evidence.')
    dump(ROOT/'results/pilot-001-recheck/regression_comparison.json',result)
    print(json.dumps(result,indent=2))
    print('Stage-two integrity checks passed')


if __name__=='__main__':
    audit()
