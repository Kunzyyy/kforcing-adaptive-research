"""Independent stdlib-only integrity audit; does not rerun GPU experiments."""
import itertools
import json
import math
import sys
from pathlib import Path


def main():
    root=Path(sys.argv[1])
    rows=[json.loads(line) for line in (root/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    env=json.loads((root/'environment.json').read_text())
    prefixes=json.loads((root/'prefixes.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    expected=set(itertools.product([p['id'] for p in prefixes['test']],env['arguments']['seeds'],summary))
    observed=[(r['prompt_id'],r['seed'],r['policy']) for r in rows]
    assert len(observed)==len(set(observed)) and set(observed)==expected
    for r in rows:
        assert r['new_tokens']==len(r['ids'])-r['prefix_length']==r['teacher_token_count']
        assert r['new_tokens']==sum(s['emitted'] for s in r['steps'])
        assert 0<r['new_tokens']<=env['max_new']
        assert r['seconds']>0 and math.isfinite(r['seconds']) and math.isfinite(r['teacher_nll_sum'])
        if r['stopped_eos']:
            assert r['ids'][-1]==102 and 102 not in r['ids'][r['prefix_length']:-1]
    for policy,s in summary.items():
        group=[r for r in rows if r['policy']==policy]
        n=sum(r['new_tokens'] for r in group)
        assert n==s['generated_tokens'] and len(group)==s['samples']
        assert math.isclose(n/sum(r['seconds'] for r in group),s['aggregate_tokens_per_second'])
        assert math.isclose(sum(r['teacher_nll_sum'] for r in group)/n,s['teacher_nll'])
    assert not {tuple(p['ids']) for p in prefixes['validation']} & {tuple(p['ids']) for p in prefixes['test']}
    result=dict(passed=True,records=len(rows),unique_prompt_seed_policy_combinations=len(expected),
        checks=['complete coverage','no duplicates','token and step counts','EOS truncation','finite metrics',
                'independent summary recomputation','no tokenized prefix overlap'],
        limitation='File integrity only; cannot certify runtime stability, algorithmic quality, or timing accuracy.')
    (root/'integrity_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
