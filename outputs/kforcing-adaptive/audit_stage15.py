"""Stage 15 independent data, fit, frozen-ranking and four-endpoint audit."""
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
def close(x, y): np.testing.assert_allclose(x, y, atol=1e-10, rtol=1e-9)


def data_audit(root, split, prefixes, tokenizer):
    m = read(root/'manifest.json')
    collection = read(root/(split+'_collection.json'))
    scoring = read(root/(split+'_scoring.json'))
    assert collection['passed'] and scoring['passed']
    assert sha(root/(split+'_pairs.jsonl')) == collection['pairs_sha256'] == scoring['pairs_sha256']
    assert sha(root/(split+'_labels.jsonl')) == scoring['labels_sha256']
    assert sha(root/(split+'_states.jsonl')) == collection['states_sha256']
    assert collection['manifest_sha256'] == sha(root/'manifest.json')
    assert collection['collector_sha256'] == scoring['collector_sha256'] == sha(root.parent.parent/'confirm_lowdim.py')
    assert scoring['scorer_sha256'] == sha(root.parent.parent/'score_rollout_quality.py')
    expected_lock = sha(root/'locked_model.json') if split == 'test' else None
    assert collection['locked_model_sha256'] == scoring['locked_model_sha256'] == expected_lock
    pm = {p['prompt_id']: p for p in prefixes if p['split'] == split}
    states = rows(root/(split+'_states.jsonl'))
    raw = rows(root/(split+'_pairs.jsonl'))
    labels = rows(root/(split+'_labels.jsonl'))
    assert len(pm) == m['sources']
    assert len(states) == 2*len(pm) == collection['attempted_states']
    assert {(s['prompt_id'], s['offset']) for s in states} == {(p, offset) for p in pm for offset in [0, 8]}
    assert dict(Counter(s['status'] for s in states)) == collection['state_status_counts']
    sm = {s['state_id']: s for s in states}
    for s in states:
        p = pm[s['prompt_id']]
        assert s['seed'] == 1501 and s['split'] == split
        assert s['state_id'] == f"{s['prompt_id']}-1501-{s['offset']}"
        assert s['state_noise_seed'] == 151001501+p['index']*100003
        assert s['history_ids'][:6] == p['ids']
        if s['status'] == 'unreachable_eos':
            assert s['offset'] == 8 and s['history_ids'][-1] == 102 and len(s['history_ids']) <= 14
        else:
            assert len(s['history_ids']) == 6+s['offset'] and 102 not in s['history_ids'][6:]
            assert (102 in s['initial_candidates'][:2]) == (s['status'] == 'head_eos')
    rm = {r['pair_id']: r for r in raw}
    assert len(rm) == len(raw) == len(labels) == scoring['pairs'] == collection['pairs']
    assert set(rm) == {f"{s['state_id']}-r{i}" for s in states if s['status'] == 'paired' for i in range(8)}
    assert len({r['future_seed'] for r in raw}) == len(raw)
    for r in raw:
        s = sm[r['state_id']]
        assert all(r[k] == s[k] for k in ['status', 'history_ids', 'noise', 'initial_candidates', 'split_candidates', 'features', 'prompt_id', 'seed', 'offset'])
        assert len(r['features']) == 82 and np.isfinite(r['features']).all()
        assert r['initial_candidates'][:2] == r['split_candidates'][:2]
        assert r['noise_seed'] == s['state_noise_seed']
        assert r['future_seed'] == 152000000+pm[r['prompt_id']]['index']*100003+r['offset']*17+r['replicate']*100000007
        base = torch.rand((1, 122, 1), generator=torch.Generator().manual_seed(r['noise_seed'])).numpy()
        future = torch.rand((1, 122, 1), generator=torch.Generator().manual_seed(r['future_seed'])).numpy()
        combined = np.concatenate([base[:, :r['offset']+4], future[:, r['offset']+4:]], axis=1)
        assert hashlib.sha256(combined.astype('<f4').tobytes()).hexdigest() == r['noise_sha256']
        assert combined[0, r['offset']:r['offset']+4, 0].tolist() == r['noise']
        for name, candidates in [('keep4', r['initial_candidates']), ('split22', r['split_candidates'])]:
            b = r[name]
            seq, extra = b['ids'], int(name == 'split22')
            n = len(seq)-6
            assert seq[:6+r['offset']] == r['history_ids']
            first = candidates[:candidates.index(102)+1] if 102 in candidates else candidates
            assert seq[6+r['offset']:6+r['offset']+len(first)] == first
            assert 102 not in seq[6:-1] and (seq[-1] == 102) == b['stopped_eos']
            assert seq[-1] == 102 or n == 122
            assert b['calls'] == math.ceil(n/4)+extra
            assert b['computed_candidates'] == min(4*math.ceil(n/4), 122)+2*extra
            assert b['completion'] == tokenizer.decode(seq[6:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    cache, groups = {}, defaultdict(dict)
    for label in labels:
        r = rm[label['pair_id']]
        for name in ['keep4', 'split22']:
            ss, b = label[name], r[name]
            assert ss['visible_new_tokens'] == ss['teacher_token_count'] == len(b['ids'])-6
            assert ss['forward_calls'] == b['calls'] and ss['computed_candidates'] == b['computed_candidates']
            assert ss['gpt2_scored_tokens'] == max(0, ss['gpt2_input_tokens']-1)
            assert ss['excluded_short'] == (ss['gpt2_scored_tokens'] == 0)
            if ss['gpt2_scored_tokens']:
                close(ss['gpt2_mean_nll'], ss['gpt2_nll_sum']/ss['gpt2_scored_tokens'])
            else: assert ss['gpt2_mean_nll'] is None
            signature = [ss[k] for k in ['gpt2_input_tokens', 'gpt2_scored_tokens', 'gpt2_nll_sum', 'gpt2_mean_nll']]
            if b['completion'] in cache: assert cache[b['completion']] == signature
            else: cache[b['completion']] = signature
        ok = not label['keep4']['excluded_short'] and not label['split22']['excluded_short']
        assert label['valid_label'] == ok
        if ok: close(label['gain'], label['keep4']['gpt2_mean_nll']-label['split22']['gpt2_mean_nll'])
        else: assert label['gain'] is None
        assert label['identical_decoded_text'] == (r['keep4']['completion'] == r['split22']['completion'])
        if label['identical_decoded_text'] and ok: assert label['gain'] == 0.
        groups[r['state_id']][r['replicate']] = label
    data, excluded = [], []
    for s in states:
        if s['status'] != 'paired': continue
        ll = groups[s['state_id']]
        assert set(ll) == set(range(8))
        if not all(v['valid_label'] for v in ll.values()): excluded.append(s['state_id']); continue
        gains = [ll[i]['gain'] for i in range(8)]
        data.append(dict(s, gains=gains, gain=sum(gains)/8))
    assert len(cache) == scoring['unique_texts']
    assert scoring['valid_labels'] == sum(r['valid_label'] for r in labels)
    assert scoring['identical_text_pairs'] == sum(r['identical_decoded_text'] for r in labels)
    return data, excluded


def independent_predictions(model, training, data):
    x = np.array([r['features'][:14] for r in training])
    y = np.array([sum(r['gains'])/8 for r in training])
    mean = x.sum(0)/len(x)
    std = np.sqrt(((x-mean)**2).sum(0)/len(x)); std[std < 1e-8] = 1.
    z = (x-mean)/std
    v = (np.array([r['features'][:14] for r in data])-mean)/std
    close(model['mean'], mean); close(model['std'], std); close(model['intercept'], y.mean())
    if model['kind'] == 'ridge':
        assert model['alpha'] == 1000.
        coef = np.linalg.lstsq(np.vstack([z, np.sqrt(1000.)*np.eye(14)]),
            np.r_[y-y.mean(), np.zeros(14)], rcond=None)[0]
        close(model['coef'], coef)
        return v@coef+y.mean()
    assert model['alpha'] == 10. and model['gamma'] == 1/14.
    # Explicit distances and centering independent of the execution implementation.
    k = np.exp(-((z[:, None, :]-z[None, :, :])**2).sum(2)/14.)
    kv = np.exp(-((v[:, None, :]-z[None, :, :])**2).sum(2)/14.)
    h = np.eye(len(y))-np.ones((len(y), len(y)))/len(y)
    centered = h@k@h
    coef = np.linalg.solve(centered+10*np.eye(len(y)), h@y)
    close(model['train_z'], z); close(model['kernel_colmean'], k.mean(0))
    close(model['kernel_grandmean'], k.mean()); close(model['coef'], coef)
    return (kv-kv.mean(1)[:, None]-k.mean(0)[None, :]+k.mean())@coef+y.mean()


def selection(scores, data):
    chosen = set(sorted(range(len(data)), key=lambda i: (-float(scores[i]), data[i]['state_id']))[:len(data)//2])
    return np.array([int(i in chosen) for i in range(len(data))], dtype=float)


def check_interval(values, data, result, coverage):
    groups = defaultdict(list)
    for i, r in enumerate(data): groups[r['prompt_id']].append(i)
    keys = sorted(groups)
    totals = np.array([sum(values[i] for i in groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    draws = np.random.default_rng(15112004).integers(0, len(keys), (8000, len(keys)))
    boot = np.array([totals[d].sum()/counts[d].sum() for d in draws])
    close(result['estimate'], np.mean(values))
    close(result['interval'], np.quantile(boot, [(1-coverage)/2, (1+coverage)/2]))
    assert result['coverage'] == coverage and result['source_clusters'] == len(keys)


def fresh_source_audit(root, prefixes, lock, tokenizer):
    import pyarrow.parquet as pq
    import random
    old = []
    for path, digest in lock['historical_source_hashes'].items():
        assert sha(root.parent/path) == digest
        source = read(root.parent/path)
        if isinstance(source, dict):
            for group in source.values(): old.extend(group)
        else: old.extend(source)
    blocked_ids = {tuple((p['ids'] if 'ids' in p else p['reference_ids'])[:6]) for p in old}
    blocked_rows = {p['source_row'] for p in old if 'source_row' in p}
    blocked_hashes = {p['source_sha256'] for p in old if 'source_sha256' in p}
    assert len(prefixes) == 384
    assert len({tuple(p['ids']) for p in prefixes}) == 384
    assert len({p['source_sha256'] for p in prefixes}) == 384
    assert len({p['source_row'] for p in prefixes}) == 384
    assert not blocked_ids.intersection(tuple(p['ids']) for p in prefixes)
    assert not blocked_rows.intersection(p['source_row'] for p in prefixes)
    assert not blocked_hashes.intersection(p['source_sha256'] for p in prefixes)
    corpus = Path('work/lm1b-data/test.parquet')
    assert sha(corpus) == read(root/'manifest.json')['dataset_sha256']
    texts = pq.read_table(corpus, columns=['text'])['text'].to_pylist()
    order = list(range(len(texts))); random.Random(15112001).shuffle(order)
    expected = []
    for row in order:
        digest = hashlib.sha256(texts[row].encode()).hexdigest()
        if row in blocked_rows or digest in blocked_hashes: continue
        ids = tokenizer.encode(texts[row], add_special_tokens=True, truncation=True, max_length=128)
        if len(ids) < 12 or tuple(ids[:6]) in blocked_ids: continue
        p = prefixes[len(expected)]
        assert p['ids'] == ids[:6] and p['source_row'] == row and p['source_sha256'] == digest
        assert p['index'] == len(expected) and p['prompt_id'] == f's15-{len(expected):03}'
        expected.append(p)
        blocked_ids.add(tuple(ids[:6])); blocked_hashes.add(digest)
        if len(expected) == 384: break
    assert expected == prefixes


def model_audit(root, data, excluded):
    lock = read(root/'locked_model.json')
    frozen = read(root/'frozen_predictions.json')
    analysis = read(root/'analysis.json')
    manifest = read(root/'manifest.json')
    assert set(lock['models']) == {'linear14', 'rbf14'}
    assert lock['train_states'] == 446 and lock['train_sources'] == 253
    assert lock['primary_coverage'] == .9875 and lock['bootstrap_repeats'] == 8000
    assert lock['bootstrap_seed'] == 15112004
    assert sha(root/'locked_model.json') == manifest['locked_model_sha256'] == frozen['locked_model_sha256']
    assert sha(root/'test_states.jsonl') == frozen['states_sha256']
    assert sha(root/'frozen_predictions.json') == analysis['frozen_predictions_sha256']
    assert analysis['labels_sha256'] == sha(root/'test_labels.jsonl')
    assert analysis['locked_model_sha256'] == sha(root/'locked_model.json')
    link = read(root/'scoring_prediction_link.json')
    assert link['frozen_predictions_sha256'] == sha(root/'frozen_predictions.json')
    assert link['test_scoring_sha256'] == sha(root/'test_scoring.json')
    from datetime import datetime
    assert datetime.fromisoformat(lock['created_utc']) <= datetime.fromisoformat(manifest['created_utc'])
    assert datetime.fromisoformat(manifest['created_utc']) <= datetime.fromisoformat(frozen['created_utc'])
    assert datetime.fromisoformat(frozen['created_utc']) < datetime.fromisoformat(link['completed_utc'])
    assert frozen['test_labels_absent_when_frozen']
    training = rows(root.parent/lock['training_path'])
    assert sha(root.parent/lock['training_path']) == lock['training_data_sha256']
    assert len(training) == 446 and {r['source_split'] for r in training} == {'train', 'dev'}
    for r in training: close(r['gain'], sum(r['gains'])/8)
    all_data = [r for r in rows(root/'test_states.jsonl') if r['status'] == 'paired']
    all_scores = {name: independent_predictions(model, training, all_data) for name, model in lock['models'].items()}
    all_scores['heuristic'] = np.array([-(r['features'][2]+r['features'][3])/2 for r in all_data])
    for name, score in all_scores.items():
        ordered = sorted(range(len(all_data)), key=lambda i: (-score[i], all_data[i]['state_id']))
        assert frozen['rankings'][name] == [all_data[i]['state_id'] for i in ordered]
        close(score, [frozen['scores'][r['state_id']][name] for r in all_data])
    assert set(frozen['scores']) == {r['state_id'] for r in all_data}
    scores = {name: np.array([frozen['scores'][r['state_id']][name] for r in data]) for name in all_scores}
    masks = {name: selection(score, data) for name, score in scores.items()}
    y = np.array([sum(r['gains'])/8 for r in data])
    assert analysis['test_states'] == len(data)
    assert analysis['test_sources'] == len({r['prompt_id'] for r in data})
    assert analysis['excluded_incomplete'] == excluded
    enough = analysis['test_sources'] >= 128
    assert analysis['enough_sources'] == enough
    passes = {}
    for name in lock['models']:
        values = (masks[name]-masks[name].mean())*y
        check_interval(values, data, analysis['primary'][name]['vs_random'], .9875)
        check_interval((masks[name]-masks['heuristic'])*y, data, analysis['primary'][name]['vs_heuristic'], .9875)
        passes[name] = bool(enough and all(r['interval'][0] > 0 for r in analysis['primary'][name].values()))
        item = analysis['arms'][name]
        close(item['mse'], np.mean((scores[name]-y)**2))
        close(item['constant_train_mean_mse'], np.mean((lock['models'][name]['intercept']-y)**2))
        close(item['correlation'], np.corrcoef(scores[name], y)[0, 1])
        assert item['selected'] == int(masks[name].sum())
    assert passes == analysis['candidate_passes']
    assert analysis['advance_gate_passed'] == any(passes.values())
    expected_next = 'linear14' if passes['linear14'] else ('rbf14' if passes['rbf14'] else None)
    assert analysis['next_candidate'] == expected_next
    check_interval((masks['heuristic']-masks['heuristic'].mean())*y, data, analysis['heuristic_vs_random'], .95)
    check_interval((masks['rbf14']-masks['linear14'])*y, data, analysis['rbf_vs_linear_exploratory'], .95)
    check_interval(y, data, analysis['mean_all_split_gain'], .95)
    ya = np.array([sum(r['gains'][:4])/4 for r in data]); yb = np.array([sum(r['gains'][4:])/4 for r in data])
    close(analysis['split_half_target_correlation'], np.corrcoef(ya, yb)[0, 1])
    ma, mb = selection(ya, data), selection(yb, data)
    check_interval(((ma-ma.mean())*yb+(mb-mb.mean())*ya)/2, data, analysis['cross_half_outcome_informed_gain'], .95)
    predictions = rows(root/'test_predictions.jsonl')
    assert len(predictions) == len(data)
    for i, (r, p) in enumerate(zip(data, predictions)):
        assert p['state_id'] == r['state_id'] and p['prompt_id'] == r['prompt_id'] and p['gains'] == r['gains']
        close(p['mean_gain'], y[i])
        for name in scores:
            close(p['scores'][name], scores[name][i])
            assert p['selected'][name] == bool(masks[name][i])
    labels = {r['pair_id']: r for r in rows(root/'test_labels.jsonl')}
    for field, result in analysis['exploratory_other_metrics'].items():
        delta = np.array([sum(labels[f"{r['state_id']}-r{i}"]['keep4'][field]-labels[f"{r['state_id']}-r{i}"]['split22'][field] for i in range(8))/8 for r in data])
        for name in masks: check_interval((masks[name]-masks[name].mean())*delta, data, result[name], .95)


def audit(root):
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    manifest, lock = read(root/'manifest.json'), read(root/'locked_model.json')
    for f, digest in lock['code_hashes'].items(): assert sha(root.parent.parent/f) == digest, f
    for f, digest in manifest['artifact_hashes'].items(): assert sha(root/f) == digest, f
    assert sha(Path('work/pilot-data/vocab.txt')) == manifest['tokenizer_sha256']
    assert sha(root/'projection.npy') == sha(root.parent/'learned-007/projection.npy')
    assert read(root/'preflight.json')['passed']
    prefixes = read(root/'prefixes.json')
    fresh_source_audit(root, prefixes, lock, tokenizer)
    data, excluded = data_audit(root, 'test', prefixes, tokenizer)
    model_audit(root, data, excluded)
    result = dict(passed=True, sources=len(prefixes), valid_states=len(data), incomplete_states=excluded,
        source_isolation_and_sampling_replayed=True, all_noise_hashes_replayed=True,
        current_state_invariance=True, eos_and_call_counts_checked=True, labels_recomputed=True,
        augmented_least_squares_and_explicit_kernel_refit=True, all_frozen_predictions_recomputed=True,
        four_primary_cluster_intervals_recomputed=True, exploratory_intervals_recomputed=True,
        analysis_sha256=sha(root/'analysis.json'), locked_model_sha256=sha(root/'locked_model.json'),
        audit_sha256=sha(Path(__file__)))
    (root/'integrity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path, nargs='?', default=Path('outputs/kforcing-adaptive/results/lowdim-015'))
    audit(ap.parse_args().root)


