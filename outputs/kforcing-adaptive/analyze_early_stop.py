"""Frozen EOS-stop evaluation: request latency, visible throughput, and quality."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def analyze(root):
    assert not (root/'analysis.json').exists()
    samples=rows(root/'samples.jsonl');scores=rows(root/'scores.jsonl');timings=rows(root/'timings.jsonl')
    key=lambda r:(r['prompt_id'],r['seed'],r['method'])
    ts=defaultdict(list)
    for r in timings:ts[key(r)].append(r['seconds'])
    smap={key(r):r for r in scores}
    merged=[]
    for r in samples:
        assert len(ts[key(r)])==3
        merged.append(dict(r,mean_seconds=float(np.mean(ts[key(r)])),**smap[key(r)]))
    methods=sorted({r['method'] for r in merged});pids=sorted({r['prompt_id'] for r in merged})
    arrays={};summary={}
    fields=('gpt2_nll_sum','gpt2_scored_tokens','mean_seconds','visible_new_tokens',
            'teacher_nll_sum','teacher_token_count','forward_calls','computed_candidates')
    for method in methods:
        group=[r for r in merged if r['method']==method]
        a=[]
        for pid in pids:
            pair=[r for r in group if r['prompt_id']==pid];assert len(pair)==2
            a.append([sum(r[f] for r in pair) for f in fields])
        arrays[method]=np.array(a,dtype=np.float64);total=arrays[method].sum(0)
        duration=np.array([r['mean_seconds'] for r in group])
        eligible=sum(r['eligible_windows'] for r in group)
        summary[method]=dict(samples=len(group),visible_tokens=int(total[3]),seconds=float(total[2]),
            visible_tokens_per_second=float(total[3]/total[2]),mean_request_ms=float(duration.mean()*1000),
            p50_request_ms=float(np.quantile(duration,.5)*1000),p95_request_ms=float(np.quantile(duration,.95)*1000),
            gpt2_gen_ppl=float(np.exp(total[0]/total[1])),gpt2_nll=float(total[0]/total[1]),
            gpt2_scored_tokens=int(total[1]),excluded_short=sum(r['excluded_short'] for r in group),
            teacher_nll=float(total[4]/total[5]),mean_visible_tokens=float(total[3]/len(group)),
            eos_fraction=float(np.mean([r['stopped_eos'] for r in group])),
            calls_per_visible_token=float(total[6]/total[3]),candidates_per_visible_token=float(total[7]/total[3]),
            total_forward_calls=int(total[6]),total_candidates=int(total[7]),
            eligible_windows=eligible,split_count=sum(r['split_count'] for r in group),
            split_fraction=sum(r['split_count'] for r in group)/eligible if eligible else None,
            mean_repeated_trigram_fraction=float(np.mean([r['repeated_trigram_fraction'] for r in group])))
    draws=np.random.default_rng(10102004).integers(0,len(pids),size=(4000,len(pids)))
    def metrics(a,b):
        return dict(ppl_ratio=np.exp(a[...,0]/a[...,1]-b[...,0]/b[...,1]),
            visible_throughput_ratio=(a[...,3]/a[...,2])/(b[...,3]/b[...,2]),
            request_latency_ratio=a[...,2]/b[...,2],
            teacher_nll_difference=a[...,4]/a[...,5]-b[...,4]/b[...,5],
            calls_per_token_ratio=(a[...,6]/a[...,3])/(b[...,6]/b[...,3]),
            candidates_per_token_ratio=(a[...,7]/a[...,3])/(b[...,7]/b[...,3]))
    def compare(other,coverage):
        a,b=arrays['learned'],arrays[other]
        point=metrics(a.sum(0),b.sum(0));boot=metrics(a[draws].sum(1),b[draws].sum(1))
        tail=(1-coverage)/2
        return {k:dict(estimate=float(point[k]),interval=np.quantile(v,[tail,1-tail]).tolist(),coverage=coverage)
                for k,v in boot.items()}
    primary={k:v for k,v in compare('random_cal',.975).items() if k in ('ppl_ratio','visible_throughput_ratio')}
    other={m:compare(m,.95) for m in methods if m!='learned'}
    budget_ratio=other['random_cal']['calls_per_token_ratio']['estimate']
    comparable=.95<=budget_ratio<=1.05
    raw_gate=primary['ppl_ratio']['interval'][1]<1 and primary['visible_throughput_ratio']['interval'][0]>1
    report=dict(summary=summary,primary=primary,exploratory=other,
        budget_calls_per_token_ratio=budget_ratio,budget_within_predeclared_5pct=bool(comparable),
        primary_two_metric_gate=bool(raw_gate),primary_budget_and_performance_gate=bool(raw_gate and comparable),
        source_clusters=len(pids),bootstrap_draws=4000,
        notes='No EOS-post blocks; within-block discarded work counted. Random probability chosen on independent calibration only. Compute matching is approximate, not FLOPs equality. Replay is an oracle scheduling ablation, not deployable.')
    (root/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    review=['# 提前停止生成样本','','测试集前四个编号，全部方法和种子；未按质量筛选，非盲评。','']
    for pid in pids[:4]:
        review+=['## '+pid,'']
        for r in sorted([r for r in samples if r['prompt_id']==pid],key=lambda r:(r['seed'],r['method'])):
            review += [f"### seed={r['seed']} / {r['method']}",'',r['completion'] or '(空文本)','']
    (root/'sample_review.md').write_text('\n'.join(review),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);analyze(ap.parse_args().root)
