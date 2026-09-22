"""Independent CPU audit of repeated-label training and locked evaluation."""
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
    assert collection['collector_sha256'] == scoring['collector_sha256'] == sha(root.parent.parent/'collect_averaged_labels.py')
    assert scoring['scorer_sha256'] == sha(root.parent.parent/'score_rollout_quality.py')
    expected_lock = sha(root/'locked_model.json') if split == 'test' else None
    assert collection['locked_model_sha256'] == scoring['locked_model_sha256'] == expected_lock
    pm = {p['prompt_id']: p for p in prefixes if p['split'] == split}
    states = rows(root/(split+'_states.jsonl'))
    raw = rows(root/(split+'_pairs.jsonl'))
    labels = rows(root/(split+'_labels.jsonl'))
    assert len(pm) == m['splits'][split]
    assert len(states) == 2*len(pm) == collection['attempted_states']
    assert {(s['prompt_id'], s['offset']) for s in states} == {(p, offset) for p in pm for offset in [0, 8]}
    assert dict(Counter(s['status'] for s in states)) == collection['state_status_counts']
    sm = {s['state_id']: s for s in states}
    for s in states:
        p = pm[s['prompt_id']]
        assert s['seed'] == 1301 and s['split'] == split
        assert s['state_id'] == f"{s['prompt_id']}-1301-{s['offset']}"
        assert s['state_noise_seed'] == 131001301+p['index']*100003
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
        assert r['future_seed'] == 132000000+pm[r['prompt_id']]['index']*100003+r['offset']*17+r['replicate']*100000007
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


def prediction(model, data):
    return np.array([sum((r['features'][i]-u)/v*w for i, (u, v, w) in enumerate(zip(model['mean'], model['std'], model['coef'])))+model['intercept'] for r in data])


def selection(scores, data):
    indices = sorted(range(len(data)), key=lambda i: (-scores[i], data[i]['state_id']))[:len(data)//2]
    chosen = set(indices)
    return np.array([int(i in chosen) for i in range(len(data))], dtype=float)


def check_interval(values, data, result, coverage):
    groups = defaultdict(list)
    for i, r in enumerate(data): groups[r['prompt_id']].append(i)
    keys = sorted(groups)
    total = np.array([sum(values[i] for i in groups[k]) for k in keys])
    count = np.array([len(groups[k]) for k in keys])
    draws = np.random.default_rng(13112004).integers(0, len(keys), (4000, len(keys)))
    boots = np.array([total[d].sum()/count[d].sum() for d in draws])
    close(result['estimate'], np.mean(values))
    close(result['interval'], np.quantile(boots, [(1-coverage)/2, (1+coverage)/2]))
    assert result['coverage'] == coverage and result['source_clusters'] == len(keys)


def audit(root):
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    manifest = read(root/'manifest.json')
    code = root.parent.parent
    for f, h in manifest['code_hashes'].items(): assert sha(code/f) == h
    for f, h in manifest['artifact_hashes'].items(): assert sha(root/f) == h
    assert sha(root/'projection.npy') == sha(root.parent/'learned-007/projection.npy')
    assert read(root/'preflight.json')['passed']
    prefixes = read(root/'prefixes.json')
    assert len(prefixes) == 384 and len({tuple(p['ids']) for p in prefixes}) == 384
    for f in ['prompt_id', 'source_row', 'source_sha256']: assert len({p[f] for p in prefixes}) == 384
    old = []
    for folder in ['lm1b-004', 'online-009', 'early-010', 'quality-011', 'stability-012']:
        old.extend(read(root.parent/folder/'prefixes.json'))
    for group in read(root.parent/'learned-007/prefixes.json').values(): old.extend(group)
    old += [dict(p, ids=p['reference_ids'][:6]) for p in read(root.parent/'confirm-008/sources.json')]
    old_ids = {tuple(p['ids'][:6]) for p in old}
    for folder in ['pilot-001', 'benefit-002']:
        for group in read(root.parent/folder/'prefixes.json').values(): old_ids.update(tuple(p['ids'][:6]) for p in group)
    old_ids.update(tuple(p['ids'][:6]) for p in read(root.parent/'current-003/fresh_prefixes.json'))
    assert not {tuple(p['ids']) for p in prefixes}.intersection(old_ids)
    for f in ['source_row', 'source_sha256']: assert not {p[f] for p in prefixes}.intersection({p[f] for p in old})
    datasets, exclusions = {}, {}
    for split in ['train', 'dev', 'test']:
        datasets[split], exclusions[split] = data_audit(root, split, prefixes, tokenizer)
    model_audit(root, datasets, exclusions)
    result = dict(passed=True, sources=384, valid_states={k: len(v) for k, v in datasets.items()},
        source_isolation=True, all_noise_hashes_replayed=True, exact_state_invariance=True, labels_recomputed=True,
        train_standardization_and_ridge_equations=True, shared_dev_selection_recomputed=True,
        test_model_lock_verified=True, primary_cluster_intervals_recomputed=True,
        analysis_sha256=sha(root/'analysis.json'), audit_sha256=sha(Path(__file__)))
    (root/'integrity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def model_audit(root, datasets, exclusions):
    train, dev, test = [datasets[k] for k in ['train', 'dev', 'test']]
    labels = np.array([r['gains'] for r in train])
    yd = np.array([r['gain'] for r in dev])
    yt = np.array([r['gain'] for r in test])
    candidates = read(root/'dev_candidates.json')
    assert {c['name'] for c in candidates} == {f'{name}_alpha{a:03}' for name in ['confidence14', 'hidden82'] for a in [1, 10, 100]}
    for c in candidates:
        n = 14 if c['name'].startswith('confidence') else 82
        alpha = float(c['name'][-3:])
        x = np.array([r['features'][:n] for r in train])
        mean = x.sum(0)/len(x)
        std = np.sqrt(((x-mean)**2).sum(0)/len(x))
        std[std < 1e-8] = 1.
        z = (x-mean)/std
        assert set(c['models']) == {'mean8'} | {f'single{i}' for i in range(8)}
        gains = {}
        for arm, model in c['models'].items():
            target = labels.sum(1)/8 if arm == 'mean8' else labels[:, int(arm[-1])]
            assert model['feature_count'] == n and model['alpha'] == alpha and model['arm'] == arm
            close(model['mean'], mean)
            close(model['std'], std)
            close(model['intercept'], target.mean())
            lhs = (z.T@z+alpha*np.eye(n))@np.array(model['coef'])
            rhs = z.T@(target-target.mean())
            np.testing.assert_allclose(lhs, rhs, atol=1e-8, rtol=1e-9)
            score = prediction(model, dev)
            mask = selection(score, dev)
            check_interval((mask-mask.mean())*yd, dev, model['dev_gain'], .95)
            close(model['dev_mse'], np.mean((score-yd)**2))
            close(model['dev_threshold'], np.median(score))
            gains[arm] = model['dev_gain']['estimate']
        close(c['shared_dev_score'], (gains['mean8']+sum(gains[f'single{i}'] for i in range(8))/8)/2)
        close(c['single_average_dev_gain'], sum(gains[f'single{i}'] for i in range(8))/8)
        assert c['mean8_dev_gain'] == c['models']['mean8']['dev_gain']
        close(c['models']['mean8']['coef'], np.mean([c['models'][f'single{i}']['coef'] for i in range(8)], axis=0))
    chosen = sorted(candidates, key=lambda c: (-c['shared_dev_score'], c['name']))[0]
    lock = read(root/'locked_model.json')
    for k, v in chosen.items(): assert lock[k] == v
    assert lock['train_states'] == len(train) and lock['dev_states'] == len(dev)
    assert lock['excluded_incomplete'] == {k: exclusions[k] for k in ['train', 'dev']}
    assert lock['manifest_sha256'] == sha(root/'manifest.json')
    for split in ['train', 'dev']: assert lock[split+'_labels_sha256'] == sha(root/(split+'_labels.jsonl'))
    assert lock['dev_gate_passed'] == (chosen['mean8_dev_gain']['interval'][0] > 0)
    analysis = read(root/'analysis.json')
    pred = rows(root/'test_predictions.jsonl')
    assert len(pred) == len(test)
    assert analysis['locked_model_sha256'] == sha(root/'locked_model.json')
    assert analysis['test_labels_sha256'] == sha(root/'test_labels.jsonl')
    assert analysis['fit_code_sha256'] == sha(root.parent.parent/'fit_averaged_labels.py')
    assert analysis['configuration'] == lock['name']
    assert analysis['excluded_incomplete_test'] == exclusions['test']
    scores, masks = {}, {}
    for arm, model in lock['models'].items():
        scores[arm] = prediction(model, test)
        masks[arm] = selection(scores[arm], test)
        item = analysis['arms'][arm]
        assert item['selected'] == int(masks[arm].sum())
        check_interval((masks[arm]-masks[arm].mean())*yt, test, item['gain_vs_random'], .95)
        close(item['mse'], np.mean((scores[arm]-yt)**2))
        close(item['constant_train_mean_mse'], np.mean((model['intercept']-yt)**2))
        close(item['prediction_correlation'], np.corrcoef(scores[arm], yt)[0, 1])
        threshold = (scores[arm] > model['dev_threshold']).astype(float)
        assert item['threshold_selected'] == int(threshold.sum())
        check_interval((threshold-threshold.mean())*yt, test, item['threshold_gain_vs_matched_random'], .95)
    for i, (r, s) in enumerate(zip(pred, test)):
        assert r['state_id'] == s['state_id'] and r['prompt_id'] == s['prompt_id'] and r['gains'] == s['gains']
        close(r['mean_gain'], yt[i])
        for arm in scores:
            close(r['scores'][arm], scores[arm][i])
            assert r['selected'][arm] == bool(masks[arm][i])
    mean = masks['mean8']
    average = sum(masks[f'single{i}'] for i in range(8))/8
    check_interval((mean-mean.mean())*yt, test, analysis['primary']['mean8_vs_random'], .975)
    check_interval((mean-average)*yt, test, analysis['primary']['mean8_vs_average_single'], .975)
    check_interval((average-mean.mean())*yt, test, analysis['average_single_gain_vs_random'], .95)
    check_interval((mean-masks['single0'])*yt, test, analysis['mean8_vs_single0'], .95)
    mse_delta = sum((scores[f'single{i}']-yt)**2 for i in range(8))/8-(scores['mean8']-yt)**2
    check_interval(mse_delta, test, analysis['mse_reduction_vs_average_single'], .95)
    # For shared linear ridge, this nonnegative MSE difference is predictor variance.
    close(mse_delta, np.var([scores[f'single{i}'] for i in range(8)], axis=0))
    check_interval(yt, test, analysis['mean_all_split_gain'], .95)
    a = np.array([sum(s['gains'][:4])/4 for s in test])
    b = np.array([sum(s['gains'][4:])/4 for s in test])
    am, bm = selection(a, test), selection(b, test)
    close(analysis['test_split_half_correlation'], np.corrcoef(a, b)[0, 1])
    check_interval(((am-am.mean())*b+(bm-bm.mean())*a)/2, test, analysis['cross_half_outcome_informed_gain'], .95)
    assert analysis['advance_gate_passed'] == (lock['dev_gate_passed'] and all(v['interval'][0] > 0 for v in analysis['primary'].values()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path, nargs='?', default=Path('outputs/kforcing-adaptive/results/averaged-013'))
    audit(ap.parse_args().root)
