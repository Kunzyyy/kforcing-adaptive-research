"""Prompt-clustered paired bootstrap; no declaration of quality equivalence."""
import argparse
import json
from pathlib import Path
import numpy as np
from core import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args = ap.parse_args()
    rows = [json.loads(line) for line in (args.run/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    summary = summarize(rows)
    prompts = sorted({r['prompt_id'] for r in rows})
    grouped = {p:[r for r in rows if r['prompt_id']==p] for p in prompts}
    comparisons = {}
    rng = np.random.default_rng(683)
    for baseline in ('fixed2','fixed3','fixed4','random'):
        estimates=[]
        for _ in range(2000):
            selected=rng.choice(prompts,len(prompts),replace=True)
            sub=[r for p in selected for r in grouped[p] if r['policy'] in ('adaptive',baseline)]
            s=summarize(sub)
            a,b=s['adaptive'],s[baseline]
            estimates.append([a['aggregate_tokens_per_second']/b['aggregate_tokens_per_second'],
                              a['teacher_nll']-b['teacher_nll'],
                              a['mean_repeated_trigram_fraction']-b['mean_repeated_trigram_fraction']])
        a,b=summary['adaptive'],summary[baseline]
        comparisons[baseline]=dict(speed_ratio=a['aggregate_tokens_per_second']/b['aggregate_tokens_per_second'],
            teacher_nll_difference=a['teacher_nll']-b['teacher_nll'],
            bootstrap_95_percentile_intervals={key:list(map(float,np.percentile(np.array(estimates)[:,i],[2.5,97.5])))
                for i,key in enumerate(['speed_ratio','teacher_nll_difference','repetition_difference'])})
    result=dict(summary=summary,adaptive_vs=comparisons,cluster='prompt; seeds kept together',
                interpretation='Exploratory proxy metrics; neither non-inferiority nor quality equivalence established.')
    (args.run/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Pilot samples','', 'All samples are model-generated; selected by prompt ID, not perceived quality.','']
    for p in prompts[:4]:
        lines+=['## '+p,'']
        for r in sorted(grouped[p],key=lambda r:(r['seed'],r['policy'])):
            lines += [f'### {r["policy"]}, seed={r["seed"]}', '',r['text'],'']
    (args.run/'sample_review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
