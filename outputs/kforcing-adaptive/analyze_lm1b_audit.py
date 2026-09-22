"""Paired prefix bootstrap of external scoring; no controller fitting."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np


def load_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def aggregate(rows):
    n=sum(r['gpt2_scored_tokens'] for r in rows)
    return sum(r['gpt2_nll_sum'] for r in rows)/n


def paired(a,b,cluster_map=None):
    ga,gb=defaultdict(list),defaultdict(list)
    for r in a:
        ga[cluster_map[r['prompt_id']] if cluster_map else r['prompt_id']].append(r)
    for r in b:
        gb[cluster_map[r['prompt_id']] if cluster_map else r['prompt_id']].append(r)
    assert set(ga)==set(gb)
    keys=sorted(ga)
    arr=np.array([[sum(r['gpt2_nll_sum'] for r in ga[k]),sum(r['gpt2_scored_tokens'] for r in ga[k]),
                   sum(r['gpt2_nll_sum'] for r in gb[k]),sum(r['gpt2_scored_tokens'] for r in gb[k])] for k in keys])
    rng=np.random.default_rng(410873)
    differences=[]
    for _ in range(2000):
        selected=arr[rng.integers(0,len(arr),len(arr))].sum(0)
        differences.append(selected[0]/selected[1]-selected[2]/selected[3])
    diff=aggregate(a)-aggregate(b)
    interval=np.quantile(differences,[.025,.975])
    return dict(prefixes=len({r['prompt_id'] for r in a}),clusters=len(keys),
        cluster_unit='article' if cluster_map else 'prefix',gpt2_nll_difference=diff,nll_difference95=interval.tolist(),
        gen_ppl_ratio=math.exp(diff),gen_ppl_ratio95=np.exp(interval).tolist())


def main():
    root=Path(sys.argv[1])
    path=root/'external_comparisons.json'
    if path.exists():
        raise RuntimeError('Comparison already exists')
    rows=load_rows(root/'external_scores.jsonl')
    select=lambda d,m:[r for r in rows if r['dataset']==d and r['method']==m]
    results={}
    for method in ('fixed2','fixed3','fixed4'):
        results[method+'_vs_ar']=paired(select('lm1b',method),select('lm1b','ar'))
        results[method+'_recommended_vs_ar']=paired(select('lm1b_recommended',method),select('lm1b','ar'))
        results[method+'_recommended_vs_zero_penalty']=paired(select('lm1b_recommended',method),select('lm1b',method))
    articles={p['id']:p['article_id'] for p in json.loads((root.parent/'current-003/fresh_prefixes.json').read_text(encoding='utf-8'))}
    for method in ('random_commit','fixed3'):
        results['old_current_vs_'+method]=paired(select('wikitext_stage3','current'),select('wikitext_stage3',method),articles)
    report=dict(comparisons=results,scope='External completion-only GPT-2 scoring. LM1B prefix bootstrap preserves both seeds; source-document grouping unavailable in shuffled LM1B parquet. Older experiment uses article clusters and is an audit, not a new selection set.')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# LM1B sampled completions','', 'First four prefix IDs, seed 617, all configurations; no quality-based selection.','']
    prefixes={p['id']:p['prefix'] for p in json.loads((root/'prefixes.json').read_text(encoding='utf-8'))}
    for prompt in sorted({r['prompt_id'] for r in select('lm1b','ar')})[:4]:
        lines+=['## '+prompt,'','Prefix: '+prefixes[prompt],'']
        for r in sorted([r for r in rows if r['prompt_id']==prompt and r['dataset'].startswith('lm1b') and r['seed'] in (617,None)],key=lambda r:(r['dataset'],r['method'])):
            lines+=[f"### {r['dataset']} / {r['method']}",'',r['text'],'']
    (root/'sample_review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
