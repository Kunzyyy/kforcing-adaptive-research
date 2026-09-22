"""Independent CPU/file audit: exclusion, budgets, traces, scoring and bootstrap."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root, generation_only=False):
    manifest = read(root / 'manifest.json')
    prefixes = read(root / 'prefixes.json')
    selector = read(root / 'frozen_model.json')
    assert sha(root / 'prefixes.json') == manifest['prefixes_sha256']
    assert sha(root / 'frozen_model.json') == manifest['model_sha256'] == sha(root.parent / 'learned-007/locked_model.json')
    assert sha(root / 'projection.npy') == manifest['projection_sha256'] == sha(root.parent / 'learned-007/projection.npy')
    code = root.parent.parent
    assert sha(code / 'STAGE9_PROTOCOL.md') == manifest['protocol_sha256']
    assert sha(code / 'learn_local_gain.py') == manifest['feature_code_sha256']
    env = read(root / 'environment.json')
    assert sha(code / 'online_frozen_decoder.py') == env['decoder_sha256']
    assert sha(code / 'run_frozen_online.py') == env['runner_sha256']
    old = read(root.parent / 'lm1b-004/prefixes.json')
    for group in read(root.parent / 'learned-007/prefixes.json').values():
        old.extend(group)
    old.extend(dict(p, ids=p['reference_ids'][:6]) for p in read(root.parent / 'confirm-008/sources.json'))
    old_prefix = {tuple(p['ids'][:6]) for p in old}
    old_rows = {p['source_row'] for p in old}
    old_hash = {p['source_sha256'] for p in old}
    for name in ('pilot-001', 'benefit-002'):
        for group in read(root.parent / name / 'prefixes.json').values():
            old_prefix.update(tuple(p['ids'][:6]) for p in group)
    old_prefix.update(tuple(p['ids'][:6]) for p in read(root.parent / 'current-003/fresh_prefixes.json'))
    assert len(prefixes) == 64
    for key in ('source_row', 'source_sha256', 'prompt_id'):
        assert len({p[key] for p in prefixes}) == 64
    assert len({tuple(p['ids']) for p in prefixes}) == 64
    for p in prefixes:
        assert tuple(p['ids']) not in old_prefix and p['source_row'] not in old_rows and p['source_sha256'] not in old_hash
    prefix_map = {p['prompt_id']: p for p in prefixes}
    samples = rows(root / 'samples.jsonl')
    timing = rows(root / 'timings.jsonl')
    budgets = rows(root / 'budget_discovery.jsonl')
    expected = {(p['prompt_id'], seed, method) for p in prefixes for seed in manifest['seeds'] for method in manifest['methods']}
    key = lambda r: (r['prompt_id'], r['seed'], r['method'])
    data = {key(r): r for r in samples}
    assert len(samples) == len(data) == 640 and set(data) == expected
    budget_map = {(r['prompt_id'], r['seed']): r for r in budgets}
    assert len(budgets) == len(budget_map) == 128
    for r in samples:
        p = prefix_map[r['prompt_id']]
        assert r['ids'][:6] == p['ids'] and len(r['ids']) == 128
        noise_seed = 90900000 + r['seed'] + p['index'] * 100003
        assert r['noise_seed'] == noise_seed
        rest = r['ids'][6:]
        visible = rest[:rest.index(102) + 1] if 102 in rest else rest
        assert r['visible_ids'] == p['ids'] + visible
        assert r['visible_new_tokens'] == len(visible) and r['stopped_eos'] == (102 in rest)
        position = 6
        for s in r['steps']:
            assert s['position'] == position
            width = min(int(r['method'][-1]) if r['method'].startswith('fixed') else 4, 128 - position)
            assert s['width'] == width
            assert s['forward_calls'] == 1 + int(s['split'])
            assert s['computed_candidates'] == width + 2 * int(s['split'])
            if r['method'].startswith('fixed') or width != 4:
                assert not s['split'] and s['score'] is None
            elif r['method'] == 'learned':
                assert math.isfinite(s['score']) and s['split'] == (s['score'] > selector['dev_threshold'])
            position += width
        assert position == 128 and r['generated_positions'] == 122
        for field in ('forward_calls', 'computed_candidates'):
            assert r[field] == sum(s[field] for s in r['steps'])
        assert r['split_count'] == sum(s['split'] for s in r['steps'])
    for identity, b in budget_map.items():
        learned = data[identity + ('learned',)]
        random = data[identity + ('random_matched',)]
        assert b['ids'] == learned['ids'] and b['split_count'] == learned['split_count']
        assert learned['split_count'] == random['split_count'] == sum(b['schedule'])
        assert learned['forward_calls'] == random['forward_calls'] == 31 + b['split_count']
        assert learned['computed_candidates'] == random['computed_candidates'] == 122 + 2 * b['split_count']
        order = np.random.default_rng(9092003 + b['noise_seed']).permutation(30)
        wanted = [i in set(order[:b['split_count']]) for i in range(30)]
        assert b['schedule'] == wanted
        assert [s['split'] for s in random['steps'][:30]] == wanted
    tmap = defaultdict(list)
    seen_timing = set()
    for t in timing:
        k = key(t)
        assert k in expected and math.isfinite(t['seconds']) and t['seconds'] > 0
        assert t['ids_sha256'] == hashlib.sha256(json.dumps(data[k]['ids']).encode()).hexdigest()
        fullkey = k + (t['repetition'],)
        assert fullkey not in seen_timing; seen_timing.add(fullkey)
        tmap[k].append(t['seconds'])
    assert len(timing) == 1920
    assert seen_timing == {k + (rep,) for k in expected for rep in range(3)}
    assert read(root / 'preflight.json')['passed'] and read(root / 'generation_checks.json')['passed']
    result = dict(passed=True, new_sources=64, samples=640, timing_runs=1920,
        source_isolation=True, exact_forward_and_candidate_budget_pairs=128,
        traces_and_thresholds=True, repeated_token_hashes=True, frozen_artifacts=True)
    if not generation_only:
        scores = rows(root / 'scores.jsonl')
        score_map = {key(r): r for r in scores}
        assert len(scores) == len(score_map) == 640 and set(score_map) == expected
        report = read(root / 'analysis.json')
        for r in scores:
            original = data[key(r)]
            assert r['teacher_token_count'] == r['visible_new_tokens'] == original['visible_new_tokens']
            assert r['gpt2_scored_tokens'] == max(0, r['gpt2_input_tokens'] - 1)
            assert r['excluded_short'] == (r['gpt2_input_tokens'] < 2)
            for field in ('teacher_nll_sum', 'gpt2_nll_sum'):
                assert math.isfinite(r[field]) and r[field] >= 0
            grams = [tuple(original['visible_ids'][6:][i:i + 3]) for i in range(max(0, original['visible_new_tokens'] - 2))]
            rep = 1 - len(set(grams)) / len(grams) if grams else 0
            assert abs(rep - r['repeated_trigram_fraction']) < 1e-12
        arrays = {}
        pids = sorted(prefix_map)
        for method in manifest['methods']:
            groups = [[score_map[(pid, seed, method)] for seed in manifest['seeds']] for pid in pids]
            array = np.array([[sum(x['gpt2_nll_sum'] for x in pair), sum(x['gpt2_scored_tokens'] for x in pair),
                sum(sum(tmap[(pid, seed, method)]) / 3 for seed in manifest['seeds'])]
                for pid, pair in zip(pids, groups)])
            arrays[method] = array
            sums = array.sum(0)
            s = report['summary'][method]
            assert abs(math.exp(sums[0] / sums[1]) - s['gpt2_gen_ppl']) < 1e-9
            assert abs(128 * 122 / sums[2] - s['fixed_position_tokens_per_second']) < 1e-9
        # Independent bootstrap implementation: source frequency weights, not sampled-array summation.
        rng = np.random.default_rng(9092004)
        draws = rng.integers(0, 64, size=(4000, 64))
        weights = np.stack([np.bincount(draw, minlength=64) for draw in draws])
        a, b = weights @ arrays['learned'], weights @ arrays['random_matched']
        boot = dict(ppl_ratio=np.exp(a[:, 0] / a[:, 1] - b[:, 0] / b[:, 1]), throughput_ratio=b[:, 2] / a[:, 2])
        for field, v in boot.items():
            np.testing.assert_allclose(np.quantile(v, [.0125, .9875]),
                report['primary_learned_vs_random_matched'][field]['interval'], atol=1e-12, rtol=1e-12)
        endpoints = report['primary_learned_vs_random_matched']
        assert report['primary_joint_improvement_passed'] == (
            endpoints['ppl_ratio']['interval'][1] < 1 and endpoints['throughput_ratio']['interval'][0] > 1)
        result.update(scoring_counts_and_aggregates=True, primary_bootstrap_independently_rebuilt=True,
                      primary_joint_improvement_passed=report['primary_joint_improvement_passed'])
    name = 'generation_audit.json' if generation_only else 'integrity_audit.json'
    (root / name).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--generation-only', action='store_true')
    args = ap.parse_args(); audit(args.root, args.generation_only)
