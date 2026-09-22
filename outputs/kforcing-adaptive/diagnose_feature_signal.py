"""Train/dev-only grouped nested CV, conditional label permutations and learning curves."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = dict(confidence14=list(range(14)), hidden64=list(range(14, 78)), noise4=list(range(78, 82)), hidden82=list(range(82)))
SPLIT_SEEDS = [14112001, 14112002, 14112003]
PERMUTATIONS = 199


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, v): p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def configs():
    result = []
    for group, alphas in [('confidence14', [1, 10, 100, 1000, 10000]), ('hidden82', [1, 10, 100, 1000, 10000]),
                          ('hidden64', [1, 1000]), ('noise4', [1, 1000])]:
        for alpha in alphas:
            result.append(dict(name=f'ridge_{group}_a{alpha:05}', kind='ridge', group=group, alpha=alpha))
    for group in ['confidence14', 'hidden82']:
        for alpha in [1, 10]:
            result.append(dict(name=f'rbf_{group}_a{alpha:05}', kind='rbf', group=group, alpha=alpha))
    return sorted(result, key=lambda c: c['name'])


def prepare(out):
    assert not (out/'manifest.json').exists()
    source = out.parent/'averaged-013'
    allowed = ['manifest.json', 'integrity_audit.json', 'prefixes.json'] + [f'{s}_{t}.json' for s in ['train', 'dev'] for t in ['scoring', 'collection']] + [f'{s}_{t}.jsonl' for s in ['train', 'dev'] for t in ['states', 'labels']]
    assert read(source/'integrity_audit.json')['passed']
    manifest13 = read(source/'manifest.json')
    assert sha(source/'prefixes.json') == manifest13['artifact_hashes']['prefixes.json']
    prefixes = {p['prompt_id']: p for p in read(source/'prefixes.json') if p['split'] in ['train', 'dev']}
    data, exclusions = [], {}
    for split in ['train', 'dev']:
        scoring, collection = read(source/(split+'_scoring.json')), read(source/(split+'_collection.json'))
        assert sha(source/(split+'_labels.jsonl')) == scoring['labels_sha256']
        assert sha(source/(split+'_states.jsonl')) == collection['states_sha256']
        grouped = defaultdict(dict)
        for label in rows(source/(split+'_labels.jsonl')):
            state_id, rep = label['pair_id'].rsplit('-r', 1)
            assert int(rep) not in grouped[state_id]
            grouped[state_id][int(rep)] = label
        excluded = []
        for state in rows(source/(split+'_states.jsonl')):
            if state['status'] != 'paired': continue
            ll = grouped[state['state_id']]
            assert set(ll) == set(range(8))
            if not all(ll[i]['valid_label'] for i in range(8)):
                excluded.append(state['state_id']); continue
            p = prefixes[state['prompt_id']]
            assert p['split'] == split
            gains = [ll[i]['gain'] for i in range(8)]
            data.append(dict(state_id=state['state_id'], prompt_id=state['prompt_id'], source_split=split,
                offset=state['offset'], features=state['features'], gains=gains, gain=float(np.mean(gains)),
                source_row=p['source_row'], source_sha256=p['source_sha256']))
        exclusions[split] = excluded
    data.sort(key=lambda r: r['state_id'])
    assert len(data) == 446 and sum(r['source_split'] == 'train' for r in data) == 336
    with (out/'development_data.jsonl').open('w', encoding='utf-8') as f:
        for r in data: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    dump(out/'manifest.json', dict(input_hashes={f: sha(source/f) for f in allowed},
        source_directory='averaged-013', input_whitelist=allowed, development_data_sha256=sha(out/'development_data.jsonl'),
        code_hashes={f: sha(ROOT/f) for f in ['STAGE14_PROTOCOL.md', 'diagnose_feature_signal.py']},
        configurations=configs(), repeats=3, outer_folds=5, inner_folds=4, permutations=PERMUTATIONS,
        states=len(data), source_clusters=len({r['prompt_id'] for r in data}), excluded_incomplete=exclusions,
        no_test_outcome_inputs=True, retrospective_exploration=True))
    print('Frozen train/dev-only data:', len(data), 'states;', len({r['prompt_id'] for r in data}), 'sources.', flush=True)


def partition(data, pool, count, seed):
    ids = sorted({data[i]['prompt_id'] for i in pool})
    ordered = np.random.default_rng(seed).permutation(ids).tolist()
    assignment = {pid: i % count for i, pid in enumerate(ordered)}
    return [np.array([i for i in pool if assignment[data[i]['prompt_id']] == k], dtype=int) for k in range(count)]


def permutation_indices(data):
    by_source = defaultdict(list)
    for i, r in enumerate(data): by_source[r['prompt_id']].append(i)
    strata = defaultdict(list)
    for source in sorted(by_source):
        ix = sorted(by_source[source], key=lambda i: data[i]['offset'])
        by_source[source] = ix
        key = (data[ix[0]]['source_split'], tuple(data[i]['offset'] for i in ix))
        strata[key].append(source)
    result = np.tile(np.arange(len(data))[:, None], (1, PERMUTATIONS+1))
    rng = np.random.default_rng(14115001)
    for p in range(1, PERMUTATIONS+1):
        for key in sorted(strata):
            dest = strata[key]
            origins = rng.permutation(dest)
            for dst, src in zip(dest, origins): result[by_source[dst], p] = by_source[str(src)]
    description = [dict(source_split=k[0], offsets=list(k[1]), source_count=len(strata[k])) for k in sorted(strata)]
    return result, description


def operators(x, train, valid, config):
    columns = FEATURES[config['group']]
    xt, xv = x[np.ix_(train, columns)], x[np.ix_(valid, columns)]
    mean, std = xt.mean(0), xt.std(0)
    std[std < 1e-8] = 1.
    z, v = (xt-mean)/std, (xv-mean)/std
    n = len(train)
    centering = np.eye(n)-np.ones((n, n))/n
    if config['kind'] == 'ridge':
        mapping = np.linalg.solve(z.T@z+config['alpha']*np.eye(len(columns)), z.T@centering)
        ht, hv = np.ones((n, n))/n+z@mapping, np.ones((len(valid), n))/n+v@mapping
    else:
        def kernel(a, b):
            distance = np.maximum((a*a).sum(1)[:, None]+(b*b).sum(1)[None, :]-2*a@b.T, 0.)
            return np.exp(-distance/len(columns))
        kt, kv = kernel(z, z), kernel(v, z)
        colmean, grand = kt.mean(0), kt.mean()
        ct = kt-kt.mean(1)[:, None]-colmean[None, :]+grand
        cv = kv-kv.mean(1)[:, None]-colmean[None, :]+grand
        mapping = np.linalg.solve(ct+config['alpha']*np.eye(n), centering)
        ht, hv = np.ones((n, n))/n+ct@mapping, np.ones((len(valid), n))/n+cv@mapping
    return ht, hv


def select_half(pred):
    if pred.ndim == 1: pred = pred[:, None]
    # All index pools follow globally sorted state_id order; stable ties use state_id.
    order = np.argsort(-pred, axis=0, kind='stable')
    mask = np.zeros_like(pred, dtype=float)
    np.put_along_axis(mask, order[:len(pred)//2], 1., axis=0)
    return mask


def training_metrics(ht, y):
    pred = ht@y
    selected = select_half(pred)[:, 0]
    return dict(train_states=len(y), train_mse=float(np.mean((pred-y)**2)),
        train_gain=float(np.mean((selected-selected.mean())*y)), effective_df=float(np.trace(ht)))


def interval(values, data):
    groups = defaultdict(list)
    for i, r in enumerate(data): groups[r['prompt_id']].append(i)
    keys = sorted(groups)
    sums = np.array([sum(values[i] for i in groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    draw = np.random.default_rng(14114004).integers(0, len(keys), (4000, len(keys)))
    boot = sums[draw].sum(1)/counts[draw].sum(1)
    return dict(estimate=float(np.mean(values)), interval95=np.quantile(boot, [.025, .975]).tolist(), source_clusters=len(keys))


def summarize(pred, selected, fractions, baseline, y, data, train_records):
    values = ((selected-fractions)*y[None, :]).mean(0)
    mse = float(np.mean((pred-y[None, :])**2))
    base = float(np.mean((baseline-y[None, :])**2))
    weights = np.array([r['train_states'] for r in train_records])
    correlation = np.corrcoef(pred.ravel(), np.tile(y, pred.shape[0]))[0, 1]
    return dict(gain_vs_random=interval(values, data), oof_mse=mse, constant_fold_mean_mse=base,
        mse_skill=1-mse/base, prediction_correlation=float(correlation) if np.isfinite(correlation) else None,
        train_mse=float(np.average([r['train_mse'] for r in train_records], weights=weights)),
        train_gain=float(np.average([r['train_gain'] for r in train_records], weights=weights)),
        mean_effective_df=float(np.mean([r['effective_df'] for r in train_records])))


def run(out):
    manifest = read(out/'manifest.json')
    assert not (out/'analysis.json').exists()
    for f, h in manifest['code_hashes'].items(): assert sha(ROOT/f) == h
    assert sha(out/'development_data.jsonl') == manifest['development_data_sha256']
    data = rows(out/'development_data.jsonl')
    x = np.array([r['features'] for r in data]); y = np.array([r['gain'] for r in data])
    config = manifest['configurations']; n, c = len(data), len(config)
    perm, strata = permutation_indices(data); yp = y[perm]
    fixed_predictions = np.zeros((3, c, n)); fixed_selected = np.zeros_like(fixed_predictions)
    nested_predictions = np.zeros((3, n)); nested_selected = np.zeros((3, n))
    fractions = np.zeros((3, n)); baseline = np.zeros((3, n))
    fixed_gains = np.zeros((c, PERMUTATIONS+1)); nested_gains = np.zeros(PERMUTATIONS+1)
    curve_names = [f'ridge_{g}_a{a:05}' for g in ['confidence14', 'hidden82'] for a in [1, 1000]]
    curve_config = [next(k for k, v in enumerate(config) if v['name'] == name) for name in curve_names]
    curve_predictions = np.zeros((4, 3, 3, n)); curve_selected = np.zeros_like(curve_predictions)
    curve_baseline = np.zeros((3, 3, n)); curve_train = defaultdict(list)
    fixed_train = defaultdict(list); nested_train = []; folds = []
    all_ids = np.arange(n)
    for rep, seed in enumerate(SPLIT_SEEDS):
        outer = partition(data, all_ids, 5, seed)
        for fold, valid in enumerate(outer):
            train = np.setdiff1d(all_ids, valid)
            inner = partition(data, train, 4, 14113000+rep*100+fold)
            inner_scores = np.zeros((c, PERMUTATIONS+1))
            for inner_valid in inner:
                inner_train = np.setdiff1d(train, inner_valid)
                for k, cfg in enumerate(config):
                    _, op = operators(x, inner_train, inner_valid, cfg)
                    pred = op@yp[inner_train]
                    selected = select_half(pred)
                    inner_scores[k] += ((selected-selected.mean(0))*yp[inner_valid]).sum(0)
            inner_scores /= len(train)
            chosen = np.argmax(inner_scores, axis=0)
            held = np.zeros((c, len(valid), PERMUTATIONS+1)); masks = np.zeros_like(held)
            metrics = []
            for k, cfg in enumerate(config):
                ht, hv = operators(x, train, valid, cfg)
                pred = hv@yp[train]; selected = select_half(pred)
                held[k], masks[k] = pred, selected
                fixed_gains[k] += ((selected-selected.mean(0))*yp[valid]).sum(0)
                fixed_predictions[rep, k, valid] = pred[:, 0]
                fixed_selected[rep, k, valid] = selected[:, 0]
                tm = training_metrics(ht, y[train]); metrics.append(tm); fixed_train[k].append(tm)
            chosen_predictions = held[chosen, :, np.arange(PERMUTATIONS+1)].T
            chosen_masks = masks[chosen, :, np.arange(PERMUTATIONS+1)].T
            nested_gains += ((chosen_masks-chosen_masks.mean(0))*yp[valid]).sum(0)
            nested_predictions[rep, valid] = chosen_predictions[:, 0]
            nested_selected[rep, valid] = chosen_masks[:, 0]
            nested_train.append(metrics[int(chosen[0])])
            fractions[rep, valid] = (len(valid)//2)/len(valid)
            baseline[rep, valid] = y[train].mean()
            source_order = np.random.default_rng(14116000+rep*100+fold).permutation(sorted({data[i]['prompt_id'] for i in train})).tolist()
            curve_subsets = []
            for fi, fraction in enumerate([.25, .5, 1.]):
                sources = set(source_order[:max(1, int(len(source_order)*fraction))])
                subset = np.array([i for i in train if data[i]['prompt_id'] in sources])
                curve_subsets.append(subset.tolist()); curve_baseline[fi, rep, valid] = y[subset].mean()
                for j, k in enumerate(curve_config):
                    if fraction == 1.:
                        pv = fixed_predictions[rep, k, valid]; sel = fixed_selected[rep, k, valid]; tm = metrics[k]
                    else:
                        ht, hv = operators(x, subset, valid, config[k]); pv = hv@y[subset]
                        sel = select_half(pv)[:, 0]; tm = training_metrics(ht, y[subset])
                    curve_predictions[j, fi, rep, valid] = pv; curve_selected[j, fi, rep, valid] = sel
                    curve_train[j, fi].append(dict(tm, train_sources=len(sources)))
            folds.append(dict(repeat=rep, fold=fold, train=train.tolist(), valid=valid.tolist(),
                inner_valid=[v.tolist() for v in inner], chosen_indices=chosen.tolist(),
                true_inner_scores=inner_scores[:, 0].tolist(), inner_scores_first3_permutations=inner_scores[:, 1:4].tolist(),
                fixed_train_metrics=metrics, curve_subsets=curve_subsets))
            print('completed outer fold', rep+1, fold+1, 'chosen', config[int(chosen[0])]['name'], flush=True)
    fixed_gains /= 3*n; nested_gains /= 3*n
    null_max = fixed_gains[:, 1:].max(0)
    fixed = {}
    for k, cfg in enumerate(config):
        summary = summarize(fixed_predictions[:, k], fixed_selected[:, k], fractions, baseline, y, data, fixed_train[k])
        np.testing.assert_allclose(summary['gain_vs_random']['estimate'], fixed_gains[k, 0], atol=1e-12)
        summary['family_max_permutation_p'] = float((1+np.sum(null_max >= fixed_gains[k, 0]))/(PERMUTATIONS+1))
        fixed[cfg['name']] = summary
    nested = summarize(nested_predictions, nested_selected, fractions, baseline, y, data, nested_train)
    np.testing.assert_allclose(nested['gain_vs_random']['estimate'], nested_gains[0], atol=1e-12)
    nested['permutation_p'] = float((1+np.sum(nested_gains[1:] >= nested_gains[0]))/(PERMUTATIONS+1))
    nested['selection_counts'] = dict(Counter(config[f['chosen_indices'][0]]['name'] for f in folds))
    curves = {}
    for j, name in enumerate(curve_names):
        curves[name] = []
        for fi, fraction in enumerate([.25, .5, 1.]):
            item = summarize(curve_predictions[j, fi], curve_selected[j, fi], fractions, curve_baseline[fi], y, data, curve_train[j, fi])
            item.update(fraction=fraction, mean_train_sources=float(np.mean([r['train_sources'] for r in curve_train[j, fi]])),
                mean_train_states=float(np.mean([r['train_states'] for r in curve_train[j, fi]])))
            curves[name].append(item)
    np.savez_compressed(out/'prediction_arrays.npz', fixed_predictions=fixed_predictions, fixed_selected=fixed_selected,
        nested_predictions=nested_predictions, nested_selected=nested_selected, fractions=fractions, baseline=baseline,
        curve_predictions=curve_predictions, curve_selected=curve_selected, curve_baseline=curve_baseline,
        permutation_indices=perm)
    dump(out/'folds.json', folds)
    dump(out/'permutation_results.json', dict(fixed_gains=fixed_gains.tolist(), nested_gains=nested_gains.tolist(),
        family_max_null=null_max.tolist(), strata=strata, count=PERMUTATIONS, seed=14115001))
    report = dict(states=n, source_clusters=manifest['source_clusters'], fixed=fixed, nested=nested, learning_curves=curves,
        configurations=config, permutation_strata=strata, retrospective_exploration=True, no_test_reevaluation=True,
        interpretation='Train/dev-only diagnosis; conditional descriptive CIs and permutation controls; no fresh confirmation or online advance gate.',
        manifest_sha256=sha(out/'manifest.json'), data_sha256=sha(out/'development_data.jsonl'),
        predictions_sha256=sha(out/'prediction_arrays.npz'), folds_sha256=sha(out/'folds.json'),
        permutations_sha256=sha(out/'permutation_results.json'), code_sha256=sha(Path(__file__)))
    dump(out/'analysis.json', report)
    print(json.dumps(dict(nested=nested, fixed_configurations=len(fixed)), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['prepare', 'run'])
    ap.add_argument('--out', type=Path, default=ROOT/'results/features-014')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); globals()[args.action](args.out)
