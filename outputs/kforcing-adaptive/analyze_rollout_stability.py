"""Prespecified repeated-future variance diagnostics; no fitting or selection."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(y):
    m = y.mean(1)
    a, b = y[:, :4].mean(1), y[:, 4:].mean(1)
    within = float(y.var(1, ddof=1).mean())
    between_means = float(m.var(ddof=1)) if len(y) > 1 else 0.
    raw = between_means-within/8
    signal = max(0., raw)
    r1 = signal/(signal+within) if signal+within > 0 else None
    r8 = signal/(signal+within/8) if signal+within > 0 else None
    return dict(mean_gain=float(m.mean()), within_variance=within, state_mean_variance=between_means,
        between_variance_raw=raw, single_label_reliability=r1, mean8_reliability=r8,
        half_mean_correlation=corr(a, b), mixed_positive_negative_fraction=float(((y.min(1) < 0) & (y.max(1) > 0)).mean()),
        half_same_strict_sign_fraction=float((a*b > 0).mean()), half_opposite_sign_fraction=float((a*b < 0).mean()),
        half_has_zero_fraction=float(((a == 0) | (b == 0)).mean()),
        all_zero_fraction=float((y == 0).all(1).mean()), exact_constant_fraction=float((y.max(1) == y.min(1)).mean()))


def half(scores, data):
    order = sorted(range(len(data)), key=lambda i: (-float(scores[i]), data[i]['state_id']))
    result = np.zeros(len(data), dtype=float)
    result[order[:len(data)//2]] = 1.
    return result


def predict(model, data):
    n = len(model['mean'])
    x = np.array([r['features'][:n] for r in data], dtype=np.float64)
    return ((x-model['mean'])/model['std'])@np.array(model['coef'])+model['intercept']


def all_metrics(y, extra):
    result = metrics(y)
    mean = y.mean(1)
    a, b = y[:, :4].mean(1), y[:, 4:].mean(1)
    # Selections and original cohort selection fractions remain fixed in bootstrap.
    result['half_a_selection_gain_on_b'] = float(np.mean(extra['centered_a']*b))
    result['half_b_selection_gain_on_a'] = float(np.mean(extra['centered_b']*a))
    result['cross_half_selection_gain'] = (result['half_a_selection_gain_on_b']+result['half_b_selection_gain_on_a'])/2
    result['same_half_selection_gain'] = float(np.mean((extra['centered_a']*a+extra['centered_b']*b)/2))
    for name in ('old', 'quality'):
        score = extra[name+'_score']
        result[name+'_prediction_correlation'] = corr(score, mean)
        result[name+'_mse_to_mean8'] = float(np.mean((score-mean)**2))
        result[name+'_selection_gain'] = float(np.mean(extra[name+'_centered']*mean))
    return result


def summarize(y, data, extra=None):
    fn = metrics if extra is None else lambda arr, ex=extra: all_metrics(arr, ex)
    point = fn(y)
    groups = defaultdict(list)
    for i, r in enumerate(data):
        groups[r['prompt_id']].append(i)
    keys = sorted(groups)
    clusters = [np.array(groups[k], dtype=int) for k in keys]
    draws = np.random.default_rng(12112004).integers(0, len(keys), size=(4000, len(keys)))
    boot = defaultdict(list)
    for draw in draws:
        idx = np.concatenate([clusters[i] for i in draw])
        sample = metrics(y[idx]) if extra is None else all_metrics(y[idx], {k: v[idx] for k, v in extra.items()})
        for k, v in sample.items():
            if v is not None and np.isfinite(v):
                boot[k].append(v)
    return dict(states=len(y), source_clusters=len(keys), metrics={k: dict(estimate=v,
        interval95=np.quantile(boot[k], [.025, .975]).tolist() if boot[k] else None,
        valid_bootstrap_draws=len(boot[k])) for k, v in point.items()})


def analyze(root):
    assert not (root/'analysis.json').exists()
    manifest = read(root/'manifest.json')
    for f, digest in manifest['code_hashes'].items():
        assert sha(root.parent.parent/f) == digest
    for f, digest in manifest['artifact_hashes'].items():
        assert sha(root/f) == digest
    collection = read(root/'diagnostic_collection.json')
    scoring = read(root/'diagnostic_scoring.json')
    assert collection['passed'] and scoring['passed']
    assert sha(root/'diagnostic_pairs.jsonl') == collection['pairs_sha256'] == scoring['pairs_sha256']
    assert sha(root/'diagnostic_labels.jsonl') == scoring['labels_sha256']
    assert sha(root/'states.jsonl') == collection['states_sha256']
    pairs = {r['pair_id']: r for r in rows(root/'diagnostic_pairs.jsonl')}
    groups = defaultdict(list)
    for label in rows(root/'diagnostic_labels.jsonl'):
        p = pairs[label['pair_id']]
        groups[p['state_id']].append((p['replicate'], label))
    data = []
    excluded = []
    for state in rows(root/'states.jsonl'):
        if state['status'] != 'paired':
            continue
        repeated = sorted(groups[state['state_id']])
        assert [i for i, _ in repeated] == list(range(8))
        if not all(r['valid_label'] for _, r in repeated):
            excluded.append(state['state_id'])
            continue
        gains = [r['gain'] for _, r in repeated]
        data.append(dict(state, gains=gains, mean_gain=float(np.mean(gains)),
            candidate_equal=state['initial_candidates'] == state['split_candidates'],
            both_current_eos=102 in state['initial_candidates'] and 102 in state['split_candidates']))
    assert len(data) > 64
    y = np.array([r['gains'] for r in data])
    extra = {}
    for name, scores in [('a', y[:, :4].mean(1)), ('b', y[:, 4:].mean(1))]:
        mask = half(scores, data)
        extra['centered_'+name] = mask-mask.mean()
    for name, file in [('old', 'old_model.json'), ('quality', 'quality_model.json')]:
        score = predict(read(root/file), data)
        mask = half(score, data)
        extra[name+'_score'] = score
        extra[name+'_centered'] = mask-mask.mean()
    report = dict(attempted_states=collection['attempted_states'], status_counts=collection['state_status_counts'],
        paired_rollouts=collection['pairs'], full_scored_states=len(data), excluded_incomplete_scoring=excluded,
        overall=summarize(y, data, extra), strata={},
        inference='Descriptive source-cluster bootstrap; unadjusted 95% intervals; eight futures held fixed within each state; no online advance gate.',
        frozen_models={f: sha(root/f) for f in ['old_model.json', 'quality_model.json']},
        labels_sha256=sha(root/'diagnostic_labels.jsonl'), code_sha256=sha(Path(__file__)))
    for name, mask in [('same_candidates', np.array([r['candidate_equal'] for r in data])),
                       ('different_candidates', np.array([not r['candidate_equal'] for r in data])),
                       ('both_current_eos', np.array([r['both_current_eos'] for r in data]))]:
        subset = [r for r, keep in zip(data, mask) if keep]
        report['strata'][name] = summarize(y[mask], subset) if subset else dict(states=0)
    with (root/'state_statistics.jsonl').open('w', encoding='utf-8') as f:
        for i, r in enumerate(data):
            r.update({k: float(v[i]) for k, v in extra.items()})
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    dump(root/'analysis.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path, nargs='?', default=Path('outputs/kforcing-adaptive/results/stability-012'))
    analyze(ap.parse_args().root)
