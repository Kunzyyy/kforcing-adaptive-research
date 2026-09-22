"""Predeclared source-clustered comparison of full-generation quality and cost."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def analyze(root):
    assert not (root / 'analysis.json').exists(), 'Preserve completed analysis'
    samples = read_rows(root / 'samples.jsonl')
    scores = read_rows(root / 'scores.jsonl')
    timings = read_rows(root / 'timings.jsonl')
    times = defaultdict(list)
    for r in timings:
        times[(r['prompt_id'], r['seed'], r['method'])].append(r['seconds'])
    scored = {(r['prompt_id'], r['seed'], r['method']): r for r in scores}
    merged = []
    for r in samples:
        key = (r['prompt_id'], r['seed'], r['method'])
        assert len(times[key]) == 3
        merged.append(dict(r, **{'mean_seconds': float(np.mean(times[key]))}, **scored[key]))
    names = sorted({r['method'] for r in merged})
    prefixes = sorted({r['prompt_id'] for r in merged})
    # Per-source sufficient statistics retain both seeds together.
    arrays, summaries = {}, {}
    for method in names:
        group = [r for r in merged if r['method'] == method]
        array = []
        for prefix in prefixes:
            pair = [r for r in group if r['prompt_id'] == prefix]
            assert len(pair) == 2
            array.append([sum(r[k] for r in pair) for k in
                ('gpt2_nll_sum', 'gpt2_scored_tokens', 'mean_seconds', 'generated_positions',
                 'teacher_nll_sum', 'teacher_token_count')])
        arrays[method] = np.array(array, dtype=np.float64)
        totals = arrays[method].sum(0)
        summaries[method] = dict(samples=len(group), generated_positions=int(totals[3]),
            elapsed_seconds=float(totals[2]), fixed_position_tokens_per_second=float(totals[3] / totals[2]),
            gpt2_nll=float(totals[0] / totals[1]), gpt2_gen_ppl=float(np.exp(totals[0] / totals[1])),
            gpt2_scored_tokens=int(totals[1]), excluded_short=sum(r['excluded_short'] for r in group),
            teacher_nll=float(totals[4] / totals[5]),
            mean_visible_new_tokens=float(np.mean([r['visible_new_tokens'] for r in group])),
            eos_fraction=float(np.mean([r['stopped_eos'] for r in group])),
            repeated_trigram_fraction=float(np.mean([r['repeated_trigram_fraction'] for r in group])),
            mean_split_count=float(np.mean([r['split_count'] for r in group])),
            mean_forward_calls=float(np.mean([r['forward_calls'] for r in group])),
            mean_computed_candidates=float(np.mean([r['computed_candidates'] for r in group])))
    rng = np.random.default_rng(9092004)
    draws = rng.integers(0, len(prefixes), size=(4000, len(prefixes)))
    def compare(other, coverage):
        a, b = arrays['learned'], arrays[other]
        aa, bb = a[draws].sum(1), b[draws].sum(1)
        at, bt = a.sum(0), b.sum(0)
        tail = (1 - coverage) / 2
        values = dict(ppl_ratio=np.exp(aa[:, 0] / aa[:, 1] - bb[:, 0] / bb[:, 1]),
            throughput_ratio=(aa[:, 3] / aa[:, 2]) / (bb[:, 3] / bb[:, 2]),
            teacher_nll_difference=aa[:, 4] / aa[:, 5] - bb[:, 4] / bb[:, 5])
        points = dict(ppl_ratio=np.exp(at[0] / at[1] - bt[0] / bt[1]),
            throughput_ratio=(at[3] / at[2]) / (bt[3] / bt[2]),
            teacher_nll_difference=at[4] / at[5] - bt[4] / bt[5])
        return {k: dict(estimate=float(points[k]), interval=np.quantile(v, [tail, 1 - tail]).tolist(),
                       coverage=coverage) for k, v in values.items()}
    primary = compare('random_matched', .975)
    # Teacher comparison is supplementary, not one of the two corrected primary endpoints.
    primary.pop('teacher_nll_difference')
    exploratory = {name: compare(name, .95) for name in names if name != 'learned'}
    gate = primary['ppl_ratio']['interval'][1] < 1 and primary['throughput_ratio']['interval'][0] > 1
    report = dict(summary=summaries, primary_learned_vs_random_matched=primary,
        primary_joint_improvement_passed=bool(gate), exploratory=exploratory,
        source_clusters=len(prefixes), bootstrap_draws=4000,
        limitations=['Matched random uses a learned trajectory total budget, not a deployable controller.',
            'Fixed-position throughput includes post-EOS work; quality truncates at first EOS.',
            'Equal forward/candidate counts do not imply equal latency or FLOPs.',
            'Single GPU, batch=1; no human blind assessment; GPT-2 PPL is a proxy.',
            'Original paper baseline quality gap remains unresolved.'])
    (root / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    review = ['# 按编号展示的完整生成样本', '', '前四个前缀、两种子、全部方法；未按质量选择，非人工盲评。', '']
    for p in prefixes[:4]:
        review.extend(['## ' + p, ''])
        for r in sorted([r for r in samples if r['prompt_id'] == p], key=lambda r: (r['seed'], r['method'])):
            review.extend([f"### seed={r['seed']} / {r['method']}", '', r['completion'] or '(空文本)', ''])
    (root / 'sample_review.md').write_text('\n'.join(review), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    analyze(ap.parse_args().root)
