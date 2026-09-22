"""Independent record/statistic audit, including CPU noise replay and cluster CIs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def correlation(x, z):
    x, z = x-x.mean(), z-z.mean()
    denom = np.sqrt(np.dot(x, x)*np.dot(z, z))
    return float(np.dot(x, z)/denom) if denom else None


def independent_metrics(y):
    n = len(y)
    means = y.sum(axis=1)/8
    a, b = y[:, :4].sum(1)/4, y[:, 4:].sum(1)/4
    sigma = float(((y-means[:, None])**2).sum()/(7*n))
    meanvar = float(((means-means.mean())**2).sum()/(n-1)) if n > 1 else 0.
    raw = meanvar-sigma/8
    signal = max(0., raw)
    return dict(mean_gain=float(means.mean()), within_variance=sigma, state_mean_variance=meanvar,
        between_variance_raw=raw,
        single_label_reliability=signal/(signal+sigma) if signal+sigma else None,
        mean8_reliability=signal/(signal+sigma/8) if signal+sigma else None,
        half_mean_correlation=correlation(a, b),
        mixed_positive_negative_fraction=sum(any(v < 0 for v in r) and any(v > 0 for v in r) for r in y)/n,
        half_same_strict_sign_fraction=float(((a > 0) & (b > 0) | (a < 0) & (b < 0)).sum()/n),
        half_opposite_sign_fraction=float(((a > 0) & (b < 0) | (a < 0) & (b > 0)).sum()/n),
        half_has_zero_fraction=float(((a == 0) | (b == 0)).sum()/n),
        all_zero_fraction=sum(all(v == 0 for v in r) for r in y)/n,
        exact_constant_fraction=sum(all(v == r[0] for v in r) for r in y)/n)


def close(x, y):
    if x is None or y is None:
        assert x is y
    else:
        np.testing.assert_allclose(x, y, atol=1e-11, rtol=1e-10)


def audit(root):
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    code = root.parent.parent
    manifest = read(root/'manifest.json')
    for name, digest in manifest['code_hashes'].items(): assert sha(code/name) == digest
    for name, digest in manifest['artifact_hashes'].items(): assert sha(root/name) == digest
    for dest, source in [('old_model.json', 'learned-007/locked_model.json'),
                         ('quality_model.json', 'quality-011/locked_model.json'), ('projection.npy', 'learned-007/projection.npy')]:
        assert sha(root/dest) == sha(root.parent/source)
    assert read(root/'preflight.json')['passed']
    assert read(root/'preflight.json')['collector_sha256'] == sha(code/'repeat_rollout_quality.py')
    prefixes = read(root/'prefixes.json')
    assert len(prefixes) == 96
    for f in ('source_row', 'source_sha256', 'prompt_id'): assert len({p[f] for p in prefixes}) == 96
    assert len({tuple(p['ids']) for p in prefixes}) == 96
    old = []
    for name in ['lm1b-004', 'online-009', 'early-010', 'quality-011']:
        old.extend(read(root.parent/name/'prefixes.json'))
    for group in read(root.parent/'learned-007/prefixes.json').values(): old.extend(group)
    old += [dict(p, ids=p['reference_ids'][:6]) for p in read(root.parent/'confirm-008/sources.json')]
    old_ids = {tuple(p['ids'][:6]) for p in old}
    for name in ['pilot-001', 'benefit-002']:
        for group in read(root.parent/name/'prefixes.json').values(): old_ids.update(tuple(p['ids'][:6]) for p in group)
    old_ids.update(tuple(p['ids'][:6]) for p in read(root.parent/'current-003/fresh_prefixes.json'))
    assert not {tuple(p['ids']) for p in prefixes}.intersection(old_ids)
    for f in ('source_row', 'source_sha256'): assert not {p[f] for p in prefixes}.intersection({p[f] for p in old})
    pm = {p['prompt_id']: p for p in prefixes}
    states = rows(root/'states.jsonl')
    raw = rows(root/'diagnostic_pairs.jsonl')
    labels = rows(root/'diagnostic_labels.jsonl')
    c, sc = read(root/'diagnostic_collection.json'), read(root/'diagnostic_scoring.json')
    assert c['passed'] and sc['passed'] and c['manifest_sha256'] == sha(root/'manifest.json')
    assert sha(root/'states.jsonl') == c['states_sha256']
    assert sha(root/'diagnostic_pairs.jsonl') == c['pairs_sha256'] == sc['pairs_sha256']
    assert sha(root/'diagnostic_labels.jsonl') == sc['labels_sha256']
    assert sc['scorer_sha256'] == sha(code/'score_rollout_quality.py')
    assert c['collector_sha256'] == sc['collector_sha256'] == sha(code/'repeat_rollout_quality.py')
    assert len(states) == 384
    assert {(s['prompt_id'], s['seed'], s['offset']) for s in states} == {
        (p['prompt_id'], seed, offset) for p in prefixes for seed in [1201, 1213] for offset in [0, 8]}
    assert dict(Counter(s['status'] for s in states)) == c['state_status_counts']
    sm = {s['state_id']: s for s in states}
    for s in states:
        p = pm[s['prompt_id']]
        assert s['state_id'] == f"{s['prompt_id']}-{s['seed']}-{s['offset']}"
        assert s['state_noise_seed'] == 121000000+s['seed']+p['index']*100003
        assert s['history_ids'][:6] == p['ids']
        if s['status'] == 'unreachable_eos':
            assert s['offset'] == 8 and s['history_ids'][-1] == 102 and len(s['history_ids']) <= 14
        else:
            assert len(s['history_ids']) == 6+s['offset'] and 102 not in s['history_ids'][6:]
            assert (102 in s['initial_candidates'][:2]) == (s['status'] == 'head_eos')
    rm = {r['pair_id']: r for r in raw}
    assert len(rm) == len(raw) == len(labels) == sc['pairs'] == c['pairs']
    assert set(rm) == {f"{s['state_id']}-r{rep}" for s in states if s['status'] == 'paired' for rep in range(8)}
    assert len({r['future_seed'] for r in raw}) == len(raw)
    for r in raw:
        s = sm[r['state_id']]
        assert all(r[k] == s[k] for k in ['status', 'history_ids', 'noise', 'initial_candidates', 'split_candidates', 'features'])
        assert s['status'] == 'paired' and len(r['features']) == 82 and np.isfinite(r['features']).all()
        assert r['noise_seed'] == s['state_noise_seed']
        assert r['future_seed'] == 122000000+pm[r['prompt_id']]['index']*100003+r['seed']*101+r['offset']*17+r['replicate']*10000019
        base = torch.rand((1, 122, 1), generator=torch.Generator().manual_seed(r['noise_seed'])).numpy()
        future = torch.rand((1, 122, 1), generator=torch.Generator().manual_seed(r['future_seed'])).numpy()
        combined = np.concatenate([base[:, :r['offset']+4], future[:, r['offset']+4:]], axis=1)
        assert hashlib.sha256(combined.astype('<f4').tobytes()).hexdigest() == r['noise_sha256']
        assert combined[0, r['offset']:r['offset']+4, 0].tolist() == r['noise']
        for name, candidates in [('keep4', r['initial_candidates']), ('split22', r['split_candidates'])]:
            branch = r[name]
            seq = branch['ids']
            n = len(seq)-6
            extra = int(name == 'split22')
            assert seq[:6+r['offset']] == r['history_ids']
            first = candidates[:candidates.index(102)+1] if 102 in candidates else candidates
            assert seq[6+r['offset']:6+r['offset']+len(first)] == first
            assert 102 not in seq[6:-1] and (seq[-1] == 102) == branch['stopped_eos']
            assert seq[-1] == 102 or n == 122
            assert branch['calls'] == math.ceil(n/4)+extra
            assert branch['computed_candidates'] == min(4*math.ceil(n/4), 122)+2*extra
            assert branch['completion'] == tokenizer.decode(seq[6:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text_scores = {}
    grouped = defaultdict(dict)
    for label in labels:
        r = rm[label['pair_id']]
        for name in ['keep4', 'split22']:
            score = label[name]
            assert score['visible_new_tokens'] == score['teacher_token_count'] == len(r[name]['ids'])-6
            assert score['forward_calls'] == r[name]['calls']
            assert score['computed_candidates'] == r[name]['computed_candidates']
            assert score['gpt2_scored_tokens'] == max(0, score['gpt2_input_tokens']-1)
            expected = score['gpt2_nll_sum']/score['gpt2_scored_tokens'] if score['gpt2_scored_tokens'] else None
            close(expected, score['gpt2_mean_nll'])
            text = r[name]['completion']
            signature = [score[k] for k in ['gpt2_input_tokens', 'gpt2_scored_tokens', 'gpt2_nll_sum', 'gpt2_mean_nll']]
            if text in text_scores: assert text_scores[text] == signature
            else: text_scores[text] = signature
        ok = not label['keep4']['excluded_short'] and not label['split22']['excluded_short']
        assert ok == label['valid_label']
        close(label['gain'], label['keep4']['gpt2_mean_nll']-label['split22']['gpt2_mean_nll'] if ok else None)
        assert label['identical_decoded_text'] == (r['keep4']['completion'] == r['split22']['completion'])
        if label['identical_decoded_text'] and ok: assert label['gain'] == 0.
        grouped[r['state_id']][r['replicate']] = label
    stat = rows(root/'state_statistics.jsonl')
    valid_ids = {k for k, v in grouped.items() if all(x['valid_label'] for x in v.values())}
    assert {s['state_id'] for s in stat} == valid_ids
    for s in stat:
        assert s['gains'] == [grouped[s['state_id']][i]['gain'] for i in range(8)]
        close(s['mean_gain'], sum(s['gains'])/8)
        for name, file in [('old', 'old_model.json'), ('quality', 'quality_model.json')]:
            model = read(root/file)
            score = sum((s['features'][i]-u)/v*w for i, (u, v, w) in enumerate(zip(model['mean'], model['std'], model['coef'])))+model['intercept']
            close(score, s[name+'_score'])
    analysis = read(root/'analysis.json')
    y = np.array([s['gains'] for s in stat])
    point = independent_metrics(y)
    for k, value in point.items(): close(value, analysis['overall']['metrics'][k]['estimate'])
    for name, indices in [('same_candidates', [i for i, s in enumerate(stat) if s['initial_candidates'] == s['split_candidates']]),
                          ('different_candidates', [i for i, s in enumerate(stat) if s['initial_candidates'] != s['split_candidates']]),
                          ('both_current_eos', [i for i, s in enumerate(stat) if 102 in s['initial_candidates'] and 102 in s['split_candidates']])]:
        assert len(indices) == analysis['strata'][name]['states']
        if indices:
            for k, value in independent_metrics(y[indices]).items():
                close(value, analysis['strata'][name]['metrics'][k]['estimate'])
    for name, key in [('a', y[:, :4].mean(1)), ('b', y[:, 4:].mean(1)),
                      ('old', [s['old_score'] for s in stat]), ('quality', [s['quality_score'] for s in stat])]:
        order = sorted(range(len(stat)), key=lambda i: (-float(key[i]), stat[i]['state_id']))
        selected = set(order[:len(stat)//2])
        for i, s in enumerate(stat): close(float(i in selected)-len(selected)/len(stat), s['centered_'+name if name in ['a', 'b'] else name+'_centered'])
    def gains(idx):
        yy = y[idx]
        a, b = yy[:, :4].mean(1), yy[:, 4:].mean(1)
        ac = np.array([stat[i]['centered_a'] for i in idx])
        bc = np.array([stat[i]['centered_b'] for i in idx])
        return dict(cross_half_selection_gain=float(np.mean(ac*b+bc*a)/2),
            quality_selection_gain=float(np.mean([stat[i]['quality_centered'] for i in idx]*yy.mean(1))),
            old_selection_gain=float(np.mean([stat[i]['old_centered'] for i in idx]*yy.mean(1))))
    for k, value in gains(np.arange(len(stat))).items(): close(value, analysis['overall']['metrics'][k]['estimate'])
    clusters = defaultdict(list)
    for i, s in enumerate(stat): clusters[s['prompt_id']].append(i)
    keys = sorted(clusters)
    draws = np.random.default_rng(12112004).integers(0, len(keys), size=(4000, len(keys)))
    bootstrap = defaultdict(list)
    # Independently recalculate all prespecified overall variance/sign CIs and key gains.
    for draw in draws:
        idx = [i for j in draw for i in clusters[keys[j]]]
        sample = dict(independent_metrics(y[idx]), **gains(idx))
        for k, value in sample.items():
            if value is not None and np.isfinite(value): bootstrap[k].append(value)
    for k, values in bootstrap.items():
        close(np.quantile(values, [.025, .975]), analysis['overall']['metrics'][k]['interval95'])
        assert len(values) == analysis['overall']['metrics'][k]['valid_bootstrap_draws']
    assert len(text_scores) == sc['unique_texts']
    result = dict(passed=True, sources=96, attempted_states=384, fully_scored_states=len(stat), repeated_pairs=len(raw),
        fresh_source_identity=True, frozen_models=True, exact_state_invariance=True, all_noise_hashes_replayed=True,
        paired_labels_recomputed=True, frozen_predictions_recomputed=True, overall_statistics_and_cluster_intervals_recomputed=True,
        analysis_sha256=sha(root/'analysis.json'), audit_sha256=sha(Path(__file__)))
    (root/'integrity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path, nargs='?', default=Path('outputs/kforcing-adaptive/results/stability-012'))
    audit(ap.parse_args().root)
