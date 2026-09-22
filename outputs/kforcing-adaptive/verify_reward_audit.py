"""Independent arithmetic and source-bootstrap checks for the reward diagnostic."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'results/lowdim-015'
OUT = ROOT / 'results/reward-audit'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = read(OUT / 'manifest.json')
    for name, h in manifest['inputs'].items():
        assert sha(SRC / name) == h
    assert sha(ROOT / 'audit_reward_target.py') == manifest['code_sha256']
    assert sha(ROOT / 'REWARD_AUDIT_PLAN.md') == manifest['plan_sha256']
    checks = read(OUT / 'scoring_checks.json')
    assert checks['passed'] and checks['token_losses_sha256'] == sha(OUT / 'token_losses.jsonl')
    assert checks['max_old_sum_abs_difference'] <= 1e-4
    assert checks['causal_truncation_max_difference'] <= 1e-4
    token_rows = rows(OUT / 'token_losses.jsonl')
    tokens = {r['text']: r for r in token_rows}
    assert len(tokens) == len(token_rows) == 8810
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('work/gpt2-large', local_files_only=True)
    for r in token_rows:
        assert r['text_sha256'] == hashlib.sha256(r['text'].encode()).hexdigest()
        assert r['input_ids'] == tokenizer.encode(r['text'], add_special_tokens=False)
        assert len(r['token_nll']) == r['scored_tokens'] == max(0, len(r['input_ids'])-1)
        assert all(math.isfinite(v) and v >= 0 for v in r['token_nll'])
        assert abs(math.fsum(r['token_nll'])-r['torch_nll_sum']) < 1e-4
    pairs = rows(SRC / 'test_pairs.jsonl')
    labels = {r['pair_id']: r for r in rows(SRC / 'test_labels.jsonl')}
    predictions = {r['state_id']: r for r in rows(SRC / 'test_predictions.jsonl')}
    frozen = read(SRC / 'frozen_predictions.json')
    for name in ('linear14', 'rbf14', 'heuristic'):
        ranking = [sid for sid in frozen['rankings'][name] if sid in predictions]
        assert len(ranking) == len(predictions)
        selected = set(ranking[:len(ranking)//2])
        assert selected == {sid for sid, p in predictions.items() if p['selected'][name]}
    saved = {r['pair_id']: r for r in rows(OUT / 'pair_diagnostics.jsonl')}
    group = defaultdict(list)
    fields = ['full_gain', 'common_gain', 'remainder', 'teacher_gain', 'bert_length_change',
              'gpt2_length_change', 'calls_change', 'candidate_change']
    maximum_old_difference = 0.
    observed_texts = set()
    for p in pairs:
        l = labels[p['pair_id']]
        for b in ('keep4', 'split22'):
            text = p[b]['completion']
            observed_texts.add(text)
            t = tokens[text]
            diff = abs(math.fsum(t['token_nll'])-l[b]['gpt2_nll_sum'])
            assert diff <= 1e-4 and t['scored_tokens'] == l[b]['gpt2_scored_tokens']
            maximum_old_difference = max(maximum_old_difference, diff)
        if p['state_id'] not in predictions:
            assert p['pair_id'] not in saved
            continue
        k, s = l['keep4'], l['split22']
        lk, ls = (tokens[p[b]['completion']]['token_nll'] for b in ('keep4', 'split22'))
        m = min(len(lk), len(ls))
        assert m > 0
        common = sum(a-b for a, b in zip(lk[:m], ls[:m]))/m
        full = k['gpt2_nll_sum']/len(lk)-s['gpt2_nll_sum']/len(ls)
        vals = [full, common, full-common, k['teacher_nll']-s['teacher_nll'],
                s['visible_new_tokens']-k['visible_new_tokens'], len(ls)-len(lk),
                s['forward_calls']-k['forward_calls'], s['computed_candidates']-k['computed_candidates']]
        for key, value in zip(fields, vals):
            np.testing.assert_allclose(value, saved[p['pair_id']][key], atol=1e-12, rtol=0)
        if p['keep4']['completion'] == p['split22']['completion']:
            assert common == full == 0
        group[p['state_id']].append((p['replicate'], p['prompt_id'], vals))
    assert observed_texts == set(tokens)
    ids = sorted(group)
    assert len(ids) == 682 and len(saved) == 5456
    sources = sorted({r[0][1] for r in group.values()})
    assert len(sources) == 381
    x, owner = [], []
    state_saved = {r['state_id']: r for r in rows(OUT / 'state_diagnostics.jsonl')}
    for sid in ids:
        rs = sorted(group[sid])
        assert [r[0] for r in rs] == list(range(8))
        arr = np.array([r[2] for r in rs])
        x.append(arr.mean(0))
        owner.append(rs[0][1])
        np.testing.assert_allclose(x[-1], [state_saved[sid][key] for key in fields], atol=1e-12, rtol=0)
    x = np.array(x)
    owner = np.array(owner)
    analysis = read(OUT / 'analysis.json')
    assert analysis['retrospective'] and not analysis['refitted'] and not analysis['prior_gate_unchanged']
    np.testing.assert_allclose(x.mean(0), [analysis['means'][key]['estimate'] for key in fields], atol=1e-12, rtol=0)
    np.testing.assert_allclose(x[:, 0].mean(), read(SRC / 'analysis.json')['mean_all_split_gain']['estimate'], atol=1e-12, rtol=0)
    # Independent explicit resampling of complete source groups; no bootstrap helper import.
    source_indices = [np.flatnonzero(owner == source) for source in sources]
    rng = np.random.default_rng(21092026)
    boot = []
    for _ in range(4000):
        draw = rng.integers(len(sources), size=len(sources))
        idx = np.concatenate([source_indices[int(i)] for i in draw])
        boot.append(x[idx].mean(0))
    intervals = np.quantile(boot, [.025, .975], axis=0).T
    np.testing.assert_allclose(intervals, [analysis['means'][key]['interval'] for key in fields], atol=1e-11, rtol=0)
    # Recompute scalar frozen-policy and corpus summaries from independently derived records.
    heuristic = np.array([predictions[sid]['selected']['heuristic'] for sid in ids], dtype=float)
    for name in ('linear14', 'rbf14', 'heuristic'):
        mask = np.array([predictions[sid]['selected'][name] for sid in ids], dtype=float)
        for key, column in (('full_gain', 0), ('common_gain', 1), ('teacher_gain', 3)):
            for comparison, weight in (('vs_random', mask-.5), ('vs_heuristic', mask-heuristic)):
                estimate = np.mean(weight*x[:, column])
                np.testing.assert_allclose(estimate, analysis['frozen_policies'][name][key][comparison]['estimate'], atol=1e-12, rtol=0)
    for common in (False, True):
        branch_totals = {b: [0., 0] for b in ('keep4', 'split22')}
        for p in pairs:
            if p['pair_id'] not in saved:
                continue
            m = min(tokens[p[b]['completion']]['scored_tokens'] for b in ('keep4', 'split22'))
            for b in branch_totals:
                r = tokens[p[b]['completion']]
                n = m if common else r['scored_tokens']
                branch_totals[b][0] += math.fsum(r['token_nll'][:n])
                branch_totals[b][1] += n
        for b, (nll, n) in branch_totals.items():
            ref = analysis['corpus']['common_gpt2' if common else 'full_gpt2'][b]['mean_nll']['estimate']
            np.testing.assert_allclose(nll/n, ref, atol=1e-7, rtol=0)
    path = OUT / 'integrity_audit.json'
    assert not path.exists()
    result = dict(passed=True, historical_inputs_unchanged=True, frozen_selections_unchanged=True,
        tokenizations_verified=len(token_rows), valid_pairs=len(saved), states=len(ids), sources=len(sources),
        max_vector_sum_vs_old_score=maximum_old_difference, eight_mean_intervals_independently_recomputed=True,
        scalar_policy_and_corpus_checks=True, analysis_sha256=sha(OUT / 'analysis.json'),
        verifier_sha256=sha(Path(__file__)),
        scope='GPU scores checked by stored old totals, builtin loss and causal truncation; not a second full GPU pass.')
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
