"""Diagnostic budget-matched selection; never fits a deployable controller."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from collect_benefit import dump


def agreement(tokens,reference):
    matches=[a==b for a,b in zip(tokens,reference)]
    prefix=next((i for i,x in enumerate(matches) if not x),4)
    return dict(matches=sum(matches),by_position=matches,exact=all(matches),matching_prefix=prefix)


def main():
    root=Path(sys.argv[1])
    assert not (root/'analysis.json').exists(),'Preserve existing analysis'
    rows=[json.loads(s) for s in (root/'windows.jsonl').read_text().splitlines()]
    for r in rows:
        for key in ('fixed4','split22'):
            r[key+'_agreement']=agreement(r[key],r['ar_reference'])
        r['gain']=r['split22_agreement']['matches']-r['fixed4_agreement']['matches']
    n=len(rows)
    budget=n//2
    tie=lambda i:(rows[i]['prompt_id'],rows[i]['seed'])
    signal=set(sorted(range(n),key=lambda i:(rows[i]['tail_margin'],*tie(i)))[:budget])
    oracle=set(sorted(range(n),key=lambda i:(-rows[i]['gain'],*tie(i)))[:budget])
    random=set(np.random.default_rng(60619).permutation(n)[:budget].tolist())
    for i,r in enumerate(rows):
        r.update(signal_split=i in signal,random_split=i in random,oracle_split=i in oracle)
    def summary(selected):
        result={}
        for policy in ('fixed4','split22','signal','random','oracle'):
            data=[]
            for r in selected:
                name=policy if policy in ('fixed4','split22') else ('split22' if r[policy+'_split'] else 'fixed4')
                data.append(r[name+'_agreement'])
            result[policy]=dict(windows=len(data),token_match_fraction=sum(x['matches'] for x in data)/(4*len(data)),
                by_position_match_fraction=np.mean([x['by_position'] for x in data],0).tolist(),
                exact_window_fraction=np.mean([x['exact'] for x in data]),
                mean_matching_prefix=np.mean([x['matching_prefix'] for x in data]),
                split_windows=len(data) if policy=='split22' else 0 if policy=='fixed4' else sum(r[policy+'_split'] for r in selected))
        return result
    groups=defaultdict(list)
    for r in rows:
        groups[r['prompt_id']].append(r)
    names=['all_split_vs_fixed4','signal_vs_random','signal_vs_random_expectation','oracle_vs_random']
    cluster_values=[]
    for key in sorted(groups):
        group=groups[key]
        cluster_values.append([np.mean([r['gain']/4 for r in group]),
            np.mean([(r['signal_split']-r['random_split'])*r['gain']/4 for r in group]),
            np.mean([(r['signal_split']-.5)*r['gain']/4 for r in group]),
            np.mean([(r['oracle_split']-r['random_split'])*r['gain']/4 for r in group])])
    values=np.array(cluster_values)
    rng=np.random.default_rng(60620)
    draws=np.array([values[rng.integers(0,len(values),len(values))].mean(0) for _ in range(2000)])
    comparisons={name:dict(token_match_fraction_difference=float(values[:,i].mean()),
        interval95=np.quantile(draws[:,i],[.025,.975]).tolist()) for i,name in enumerate(names)}
    result=dict(all_windows=summary(rows),no_eos_subset=summary([r for r in rows if r['no_eos_all_branches']]),
        gain_counts={str(g):sum(r['gain']==g for r in rows) for g in range(-4,5)},
        comparisons=comparisons,selection_budget=budget,prefix_clusters=len(groups),
        notes='Paired prefix bootstrap retains 8 noise seeds. Fixed diagnostic selections are not refit in bootstrap. Random expectation is analytically 50% of per-window gain. Oracle observes reference answers; no deployed performance or distribution-distance claim.')
    with (root/'selection_records.jsonl').open('w',encoding='utf-8') as out:
        for r in rows:
            out.write(json.dumps(r,allow_nan=False)+'\n')
    dump(root/'analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
