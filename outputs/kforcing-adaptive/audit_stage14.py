"""Independent direct-fit checks for nested development-only diagnosis."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a, b): np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)


def direct_fit(x, train, valid, config, targets):
    groups = dict(confidence14=list(range(14)), hidden64=list(range(14, 78)), noise4=list(range(78, 82)), hidden82=list(range(82)))
    cols = groups[config['group']]
    a, b = x[np.ix_(train, cols)], x[np.ix_(valid, cols)]
    mean = a.sum(0)/len(a)
    std = np.sqrt(((a-mean)**2).sum(0)/len(a)); std[std < 1e-8] = 1.
    z, v = (a-mean)/std, (b-mean)/std
    yy = targets[train]
    if yy.ndim == 1: yy = yy[:, None]
    center = yy.sum(0)/len(train)
    alpha = config['alpha']
    if config['kind'] == 'ridge':
        augmented_x = np.vstack([z, np.sqrt(alpha)*np.eye(len(cols))])
        augmented_y = np.vstack([yy-center, np.zeros((len(cols), yy.shape[1]))])
        coef = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
        pt, pv = center+z@coef, center+v@coef
        singular = np.linalg.svd(z, compute_uv=False)
        df = 1+np.sum(singular**2/(singular**2+alpha))
    else:
        # Explicit pairwise distances and centering matrices, unlike production operators.
        k = np.exp(-((z[:, None, :]-z[None, :, :])**2).sum(2)/len(cols))
        kv = np.exp(-((v[:, None, :]-z[None, :, :])**2).sum(2)/len(cols))
        n = len(train); c = np.eye(n)-np.ones((n, n))/n
        centered = c@k@c
        cross = (kv-k.mean(0)[None, :])@c
        dual = np.linalg.solve(centered+alpha*np.eye(n), yy-center)
        pt, pv = center+centered@dual, center+cross@dual
        eig = np.linalg.eigvalsh(centered)
        df = 1+np.sum(eig/(eig+alpha))
    return pt, pv, float(df)


def select(scores, ids, data):
    if scores.ndim == 1: scores = scores[:, None]
    mask = np.zeros_like(scores)
    for j in range(scores.shape[1]):
        best = sorted(range(len(ids)), key=lambda i: (-float(scores[i, j]), data[ids[i]]['state_id']))[:len(ids)//2]
        mask[best, j] = 1.
    return mask


def assigned_folds(data, pool, k, seed):
    sources = sorted({data[i]['prompt_id'] for i in pool})
    shuffled = np.random.default_rng(seed).permutation(sources).tolist()
    return [sorted(i for i in pool if shuffled.index(data[i]['prompt_id']) % k == fold) for fold in range(k)]


def check_ci(values, data, reported):
    groups = defaultdict(list)
    for i, r in enumerate(data): groups[r['prompt_id']].append(i)
    keys = sorted(groups)
    sums = np.array([sum(values[i] for i in groups[k]) for k in keys]); n = np.array([len(groups[k]) for k in keys])
    draws = np.random.default_rng(14114004).integers(0, len(keys), (4000, len(keys)))
    boot = [sums[d].sum()/n[d].sum() for d in draws]
    close(reported['estimate'], np.mean(values)); close(reported['interval95'], np.quantile(boot, [.025, .975]))
    assert reported['source_clusters'] == len(keys)


def audit_inputs(root):
    manifest = read(root/'manifest.json'); source = root.parent/'averaged-013'
    for f, h in manifest['code_hashes'].items(): assert sha(root.parent.parent/f) == h
    allowed = {'manifest.json', 'integrity_audit.json', 'prefixes.json'} | {f'{s}_{t}.json' for s in ['train', 'dev'] for t in ['scoring', 'collection']} | {f'{s}_{t}.jsonl' for s in ['train', 'dev'] for t in ['states', 'labels']}
    assert set(manifest['input_whitelist']) == set(manifest['input_hashes']) == allowed
    for f, h in manifest['input_hashes'].items(): assert sha(source/f) == h
    assert sha(root/'development_data.jsonl') == manifest['development_data_sha256']
    data = rows(root/'development_data.jsonl')
    pm = {p['prompt_id']: p for p in read(source/'prefixes.json')}
    expected = {}
    excluded = {}
    for split in ['train', 'dev']:
        labels = {l['pair_id']: l for l in rows(source/(split+'_labels.jsonl'))}
        excluded[split] = []
        for state in rows(source/(split+'_states.jsonl')):
            if state['status'] != 'paired': continue
            ll = [labels[f"{state['state_id']}-r{r}"] for r in range(8)]
            if not all(l['valid_label'] for l in ll): excluded[split].append(state['state_id']); continue
            expected[state['state_id']] = (state, [l['gain'] for l in ll])
    assert len(data) == len(expected) == 446 and len({r['state_id'] for r in data}) == 446
    assert [r['state_id'] for r in data] == sorted(expected)
    assert excluded == manifest['excluded_incomplete']
    for r in data:
        s, gains = expected[r['state_id']]
        assert r['gains'] == gains and r['features'] == s['features'] and r['offset'] == s['offset']
        assert r['source_split'] == pm[r['prompt_id']]['split'] in ['train', 'dev']
        for f in ['source_row', 'source_sha256']: assert r[f] == pm[r['prompt_id']][f]
        close(r['gain'], sum(gains)/8)
    assert len({r['prompt_id'] for r in data}) == manifest['source_clusters'] == 253
    return manifest, data


def audit(root):
    manifest, data = audit_inputs(root)
    analysis = read(root/'analysis.json'); folds = read(root/'folds.json'); pr = read(root/'permutation_results.json')
    for f, key in [('manifest.json', 'manifest_sha256'), ('development_data.jsonl', 'data_sha256'),
                   ('prediction_arrays.npz', 'predictions_sha256'), ('folds.json', 'folds_sha256'), ('permutation_results.json', 'permutations_sha256')]:
        assert sha(root/f) == analysis[key]
    assert sha(root.parent.parent/'diagnose_feature_signal.py') == analysis['code_sha256']
    arrays = np.load(root/'prediction_arrays.npz', allow_pickle=False)
    config = manifest['configurations']; n, count = len(data), len(config)
    assert config == analysis['configurations'] and len(config) == 18
    x = np.array([r['features'] for r in data]); y = np.array([sum(r['gains'])/8 for r in data])
    permutation = arrays['permutation_indices']; assert permutation.shape == (n, 200)
    by_source = defaultdict(list)
    for i, r in enumerate(data): by_source[r['prompt_id']].append(i)
    strata = defaultdict(list)
    for source in sorted(by_source):
        ids = sorted(by_source[source], key=lambda i: data[i]['offset']); by_source[source] = ids
        strata[data[ids[0]]['source_split'], tuple(data[i]['offset'] for i in ids)].append(source)
    rng = np.random.default_rng(14115001)
    expected = np.tile(np.arange(n)[:, None], (1, 200))
    for p in range(1, 200):
        for key in sorted(strata):
            origins = rng.permutation(strata[key]).tolist()
            for dst, src in zip(strata[key], origins): expected[by_source[dst], p] = by_source[src]
    np.testing.assert_array_equal(permutation, expected)
    for p in range(200): np.testing.assert_array_equal(np.sort(permutation[:, p]), np.arange(n))
    targets = y[permutation[:, :4]]
    fixed_gains = np.zeros((count, 4)); nested_gains = np.zeros(4)
    train_records = defaultdict(list); nested_train = []; curve_records = defaultdict(list)
    curve_names = [f'ridge_{g}_a{a:05}' for g in ['confidence14', 'hidden82'] for a in [1, 1000]]
    assert len(folds) == 15
    for record in folds:
        rep, fold = record['repeat'], record['fold']; train, valid = record['train'], record['valid']
        assert valid == assigned_folds(data, list(range(n)), 5, 14112001+rep)[fold]
        assert train == sorted(set(range(n))-set(valid))
        assert not {data[i]['prompt_id'] for i in train}.intersection(data[i]['prompt_id'] for i in valid)
        inner = assigned_folds(data, train, 4, 14113000+rep*100+fold)
        assert inner == record['inner_valid']
        inner_scores = np.zeros((count, 4))
        for iv in inner:
            it = sorted(set(train)-set(iv))
            assert not {data[i]['prompt_id'] for i in it}.intersection(data[i]['prompt_id'] for i in iv)
            for k, cfg in enumerate(config):
                _, pv, _ = direct_fit(x, it, iv, cfg, targets)
                masks = select(pv, iv, data)
                inner_scores[k] += ((masks-masks.mean(0))*targets[iv]).sum(0)
        inner_scores /= len(train)
        close(inner_scores[:, 0], record['true_inner_scores'])
        close(inner_scores[:, 1:4], record['inner_scores_first3_permutations'])
        choices = [min(range(count), key=lambda k: (-inner_scores[k, p], config[k]['name'])) for p in range(4)]
        assert choices == record['chosen_indices'][:4]
        held, masks = [], []
        metrics = []
        for k, cfg in enumerate(config):
            pt, pv, df = direct_fit(x, train, valid, cfg, targets)
            sel = select(pv, valid, data); held.append(pv); masks.append(sel)
            close(pv[:, 0], arrays['fixed_predictions'][rep, k, valid])
            np.testing.assert_array_equal(sel[:, 0], arrays['fixed_selected'][rep, k, valid])
            fixed_gains[k] += ((sel-sel.mean(0))*targets[valid]).sum(0)
            st = select(pt[:, 0], train, data)[:, 0]
            tm = dict(train_states=len(train), train_mse=float(np.mean((pt[:, 0]-y[train])**2)),
                train_gain=float(np.mean((st-st.mean())*y[train])), effective_df=df)
            for key, value in tm.items(): close(value, record['fixed_train_metrics'][k][key])
            metrics.append(tm); train_records[k].append(tm)
        for p, k in enumerate(choices):
            nested_gains[p] += np.sum((masks[k][:, p]-masks[k][:, p].mean())*targets[valid, p])
        close(held[choices[0]][:, 0], arrays['nested_predictions'][rep, valid])
        np.testing.assert_array_equal(masks[choices[0]][:, 0], arrays['nested_selected'][rep, valid])
        nested_train.append(metrics[choices[0]])
        close(arrays['fractions'][rep, valid], np.full(len(valid), (len(valid)//2)/len(valid)))
        close(arrays['baseline'][rep, valid], np.full(len(valid), y[train].mean()))
        order = np.random.default_rng(14116000+rep*100+fold).permutation(sorted({data[i]['prompt_id'] for i in train})).tolist()
        for fi, fraction in enumerate([.25, .5, 1.]):
            selected_sources = set(order[:max(1, int(len(order)*fraction))])
            subset = [i for i in train if data[i]['prompt_id'] in selected_sources]
            assert subset == record['curve_subsets'][fi]
            if fi: assert set(record['curve_subsets'][fi-1]).issubset(subset)
            close(arrays['curve_baseline'][fi, rep, valid], np.full(len(valid), y[subset].mean()))
            for j, name in enumerate(curve_names):
                k = next(k for k, cfg in enumerate(config) if cfg['name'] == name)
                pt, pv, df = direct_fit(x, subset, valid, config[k], y)
                sel = select(pv, valid, data)[:, 0]
                close(pv[:, 0], arrays['curve_predictions'][j, fi, rep, valid])
                np.testing.assert_array_equal(sel, arrays['curve_selected'][j, fi, rep, valid])
                st = select(pt, subset, data)[:, 0]
                curve_records[j, fi].append(dict(train_states=len(subset), train_sources=len(selected_sources),
                    train_mse=float(np.mean((pt[:, 0]-y[subset])**2)),
                    train_gain=float(np.mean((st-st.mean())*y[subset])), effective_df=df))
        print('audited outer fold', rep+1, fold+1, flush=True)
    fixed_gains /= 3*n; nested_gains /= 3*n
    close(fixed_gains, np.array(pr['fixed_gains'])[:, :4]); close(nested_gains, pr['nested_gains'][:4])
    audit_summaries(analysis, arrays, data, y, train_records, nested_train, curve_records, config, curve_names, pr, folds)
    result = dict(passed=True, development_states=n, source_clusters=253, no_test_outcome_inputs=True,
        source_derived_data_verified=True, all_outer_inner_source_separation=True,
        all_199_permutation_bijections_and_strata_verified=True,
        direct_fit_true_label_configurations=18, direct_fit_nested_outer_folds=15,
        independently_refitted_permutation_replicates=[1, 2, 3],
        all_learning_curve_predictions_verified=True, all_descriptive_gain_intervals_verified=True,
        permutation_p_values_recomputed=True,
        analysis_sha256=sha(root/'analysis.json'), audit_sha256=sha(Path(__file__)))
    (root/'integrity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def audit_summaries(analysis, arrays, data, y, train_records, nested_train, curve_records, config, curve_names, permutations, folds):
    def check(summary, pred, sel, base, training):
        values = ((sel-arrays['fractions'])*y[None, :]).mean(0)
        check_ci(values, data, summary['gain_vs_random'])
        mse = float(np.mean((pred-y[None, :])**2)); constant = float(np.mean((base-y[None, :])**2))
        close(mse, summary['oof_mse']); close(constant, summary['constant_fold_mean_mse']); close(1-mse/constant, summary['mse_skill'])
        close(summary['prediction_correlation'], np.corrcoef(pred.ravel(), np.tile(y, 3))[0, 1])
        weights = np.array([t['train_states'] for t in training])
        for key in ['train_mse', 'train_gain']: close(summary[key], np.average([t[key] for t in training], weights=weights))
        close(summary['mean_effective_df'], np.mean([t['effective_df'] for t in training]))
    null_max = np.array(permutations['fixed_gains'])[:, 1:].max(0)
    close(null_max, permutations['family_max_null'])
    for k, cfg in enumerate(config):
        s = analysis['fixed'][cfg['name']]
        check(s, arrays['fixed_predictions'][:, k], arrays['fixed_selected'][:, k], arrays['baseline'], train_records[k])
        close(s['family_max_permutation_p'], (1+sum(null_max >= permutations['fixed_gains'][k][0]))/200)
    check(analysis['nested'], arrays['nested_predictions'], arrays['nested_selected'], arrays['baseline'], nested_train)
    ng = permutations['nested_gains']
    close(analysis['nested']['permutation_p'], (1+sum(v >= ng[0] for v in ng[1:]))/200)
    assert analysis['nested']['selection_counts'] == dict(Counter(config[f['chosen_indices'][0]]['name'] for f in folds))
    for j, name in enumerate(curve_names):
        for fi, fraction in enumerate([.25, .5, 1.]):
            s = analysis['learning_curves'][name][fi]; tr = curve_records[j, fi]
            assert s['fraction'] == fraction
            check(s, arrays['curve_predictions'][j, fi], arrays['curve_selected'][j, fi], arrays['curve_baseline'][fi], tr)
            close(s['mean_train_sources'], np.mean([t['train_sources'] for t in tr]))
            close(s['mean_train_states'], np.mean([t['train_states'] for t in tr]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path, nargs='?', default=Path('outputs/kforcing-adaptive/results/features-014'))
    audit(ap.parse_args().root)
