"""Article-clustered comparisons after averaging repeated timings per sample."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from collect_benefit import dump


def stats(rows):
    tokens=sum(r['new_tokens'] for r in rows)
    return dict(speed=tokens/sum(r['seconds'] for r in rows),
        nll=sum(r['teacher_nll_sum'] for r in rows)/tokens,
        mean_length=tokens/len(rows),repeat=float(np.mean([r['repeated_trigram_fraction'] for r in rows])))


def compare(a,b):
    a,b=stats(a),stats(b)
    return dict(speed_ratio=a['speed']/b['speed'],teacher_nll_difference=a['nll']-b['nll'],
                mean_length_difference=a['mean_length']-b['mean_length'],repetition_difference=a['repeat']-b['repeat'])


def main():
    root=Path(sys.argv[1])
    if (root/'online_analysis.json').exists():
        raise RuntimeError('Analysis already saved; preserve original analysis')
    rows=[json.loads(line) for line in (root/'online_samples.jsonl').read_text(encoding='utf-8').splitlines()]
    policies=sorted({r['policy'] for r in rows})
    groups={p:defaultdict(list) for p in policies}
    for r in rows:
        groups[r['policy']][r['article_id']].append(r)
    articles=sorted(groups['current'])
    assert all(sorted(groups[p])==articles for p in policies)
    comparisons={}
    for p in policies:
        if p=='current':
            continue
        a=[r for rs in groups['current'].values() for r in rs]
        b=[r for rs in groups[p].values() for r in rs]
        point=compare(a,b)
        rng=np.random.default_rng(598514)
        boots={key:[] for key in point}
        for _ in range(2000):
            selected=[articles[i] for i in rng.integers(0,len(articles),len(articles))]
            a=[r for k in selected for r in groups['current'][k]]
            b=[r for k in selected for r in groups[p][k]]
            for k,v in compare(a,b).items():
                boots[k].append(v)
        comparisons[p]=dict(point=point,interval95={k:np.quantile(v,[.025,.975]).tolist() for k,v in boots.items()})
    report=dict(unique_generations=len(rows),timed_runs=sum(len(r['timing_repetitions']) for r in rows),
        articles=len(articles),comparisons=comparisons,
        inference='Exploratory paired article bootstrap, 2000 resamples; repeated timings averaged before resampling. No human-quality equivalence claim.')
    dump(root/'online_analysis.json',report)
    lines=['# Fresh online samples','', 'First four prompt IDs; every policy and seed shown, without quality-based selection.','']
    for prompt in sorted({r['prompt_id'] for r in rows})[:4]:
        lines+=['## '+prompt,'']
        for r in sorted([r for r in rows if r['prompt_id']==prompt],key=lambda r:(r['seed'],r['policy'])):
            lines +=[f"### {r['policy']}, seed {r['seed']}",'',r['text'],'']
    (root/'online_sample_review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
