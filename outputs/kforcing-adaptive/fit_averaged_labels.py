"""One shared dev-selected configuration, mean8 versus eight single-label fits."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, obj): p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def verify(root):
    m = read(root/'manifest.json')
    for name, h in m['code_hashes'].items(): assert sha(root.parent.parent/name) == h
    for name, h in m['artifact_hashes'].items(): assert sha(root/name) == h


def load(root, split):
    sc = read(root/(split+'_scoring.json'))
    assert sc['passed'] and sha(root/(split+'_labels.jsonl')) == sc['labels_sha256']
    assert sha(root/(split+'_pairs.jsonl')) == sc['pairs_sha256']
    pairs = {r['pair_id']: r for r in rows(root/(split+'_pairs.jsonl'))}
    labels = defaultdict(dict)
    for r in rows(root/(split+'_labels.jsonl')):
        p = pairs[r['pair_id']]
        labels[p['state_id']][p['replicate']] = r
    data, excluded = [], []
    for s in rows(root/(split+'_states.jsonl')):
        if s['status'] != 'paired': continue
        ll = labels[s['state_id']]
        assert set(ll) == set(range(8))
        if not all(ll[i]['valid_label'] for i in range(8)):
            excluded.append(s['state_id'])
            continue
        y = [ll[i]['gain'] for i in range(8)]
        data.append(dict(s, gains=y, gain=float(np.mean(y))))
    return data, excluded


def predict(model, data):
    x = np.array([r['features'][:len(model['mean'])] for r in data])
    return ((x-model['mean'])/model['std'])@np.array(model['coef'])+model['intercept']


def half(scores, data):
    order = sorted(range(len(data)), key=lambda i: (-float(scores[i]), data[i]['state_id']))
    mask = np.zeros(len(data))
    mask[order[:len(data)//2]] = 1.
    return mask


def interval(values, data, coverage=.95):
    groups = defaultdict(list)
    for value, r in zip(values, data): groups[r['prompt_id']].append(float(value))
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    draw = np.random.default_rng(13112004).integers(0, len(keys), size=(4000, len(keys)))
    bootstrap = sums[draw].sum(1)/counts[draw].sum(1)
    tail = (1-coverage)/2
    return dict(estimate=float(np.mean(values)), interval=np.quantile(bootstrap, [tail, 1-tail]).tolist(),
        coverage=coverage, source_clusters=len(keys))


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


def fit(root):
    verify(root)
    assert not (root/'locked_model.json').exists()
    assert not (root/'test_pairs.jsonl').exists() and not (root/'test_labels.jsonl').exists()
    train, te = load(root, 'train')
    dev, de = load(root, 'dev')
    yt = np.array([r['gains'] for r in train])
    yd = np.array([r['gain'] for r in dev])
    assert len(train) > 100 and len(dev) > 32
    candidates = []
    for name, n in [('confidence14', 14), ('hidden82', 82)]:
        x = np.array([r['features'][:n] for r in train])
        mean, std = x.mean(0), x.std(0)
        std[std < 1e-8] = 1.
        z = (x-mean)/std
        for alpha in [1., 10., 100.]:
            models = {}
            for arm, y in [('mean8', yt.mean(1))]+[(f'single{i}', yt[:, i]) for i in range(8)]:
                coef = np.linalg.solve(z.T@z+alpha*np.eye(n), z.T@(y-y.mean()))
                model = dict(name=f'{name}_alpha{int(alpha):03}', arm=arm, feature_count=n, alpha=alpha,
                    mean=mean.tolist(), std=std.tolist(), coef=coef.tolist(), intercept=float(y.mean()))
                score = predict(model, dev)
                mask = half(score, dev)
                model.update(dev_gain=interval((mask-mask.mean())*yd, dev),
                    dev_mse=float(np.mean((score-yd)**2)), dev_threshold=float(np.median(score)))
                models[arm] = model
            singles = [models[f'single{i}'] for i in range(8)]
            np.testing.assert_allclose(models['mean8']['coef'], np.mean([s['coef'] for s in singles], axis=0), atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(models['mean8']['intercept'], np.mean([s['intercept'] for s in singles]), atol=1e-12)
            single_gain = float(np.mean([s['dev_gain']['estimate'] for s in singles]))
            candidates.append(dict(name=models['mean8']['name'], shared_dev_score=(models['mean8']['dev_gain']['estimate']+single_gain)/2,
                mean8_dev_gain=models['mean8']['dev_gain'], single_average_dev_gain=single_gain, models=models))
    chosen = sorted(candidates, key=lambda c: (-c['shared_dev_score'], c['name']))[0]
    lock = dict(chosen, dev_gate_passed=chosen['mean8_dev_gain']['interval'][0] > 0,
        train_states=len(train), dev_states=len(dev), excluded_incomplete=dict(train=te, dev=de),
        train_labels_sha256=sha(root/'train_labels.jsonl'), dev_labels_sha256=sha(root/'dev_labels.jsonl'),
        manifest_sha256=sha(root/'manifest.json'), selection='Max symmetric shared dev score; ties by configuration name; no test outcomes present',
        linear_coefficient_mean_check=True)
    dump(root/'dev_candidates.json', candidates)
    dump(root/'locked_model.json', lock)
    print(json.dumps({k: lock[k] for k in ['name', 'shared_dev_score', 'mean8_dev_gain', 'single_average_dev_gain', 'dev_gate_passed', 'train_states', 'dev_states']}, indent=2), flush=True)


def evaluate(root):
    verify(root)
    assert not (root/'analysis.json').exists()
    lock = read(root/'locked_model.json')
    for split in ['train', 'dev']: assert sha(root/(split+'_labels.jsonl')) == lock[split+'_labels_sha256']
    for suffix in ['collection', 'scoring']:
        assert read(root/('test_'+suffix+'.json'))['locked_model_sha256'] == sha(root/'locked_model.json')
    data, excluded = load(root, 'test')
    y = np.array([r['gain'] for r in data])
    scores = {arm: predict(model, data) for arm, model in lock['models'].items()}
    masks = {arm: half(score, data) for arm, score in scores.items()}
    single_average_mask = np.mean([masks[f'single{i}'] for i in range(8)], axis=0)
    main = masks['mean8']
    primary = dict(mean8_vs_random=interval((main-main.mean())*y, data, .975),
        mean8_vs_average_single=interval((main-single_average_mask)*y, data, .975))
    arms = {}
    for arm, model in lock['models'].items():
        mask, score = masks[arm], scores[arm]
        threshold = (score > model['dev_threshold']).astype(float)
        arms[arm] = dict(selected=int(mask.sum()), gain_vs_random=interval((mask-mask.mean())*y, data),
            prediction_correlation=corr(score, y), mse=float(np.mean((score-y)**2)),
            constant_train_mean_mse=float(np.mean((model['intercept']-y)**2)),
            threshold_selected=int(threshold.sum()), threshold_gain_vs_matched_random=interval((threshold-threshold.mean())*y, data))
    ya = np.array([np.mean(r['gains'][:4]) for r in data])
    yb = np.array([np.mean(r['gains'][4:]) for r in data])
    ma, mb = half(ya, data), half(yb, data)
    train, _ = load(root, 'train')
    dev, _ = load(root, 'dev')
    result = dict(configuration=lock['name'], train_states=len(train), dev_states=len(dev), test_states=len(data),
        test_source_clusters=len({r['prompt_id'] for r in data}), excluded_incomplete_test=excluded,
        primary=primary, arms=arms, dev_gate_passed=lock['dev_gate_passed'],
        advance_gate_passed=bool(lock['dev_gate_passed'] and all(v['interval'][0] > 0 for v in primary.values())),
        average_single_gain_vs_random=interval((single_average_mask-main.mean())*y, data),
        mean8_vs_single0=interval((main-masks['single0'])*y, data),
        mse_reduction_vs_average_single=interval(np.mean([(scores[f'single{i}']-y)**2 for i in range(8)], axis=0)-(scores['mean8']-y)**2, data),
        mean_all_split_gain=interval(y, data), test_split_half_correlation=corr(ya, yb),
        cross_half_outcome_informed_gain=interval(((ma-ma.mean())*yb+(mb-mb.mean())*ya)/2, data),
        locked_model_sha256=sha(root/'locked_model.json'), test_labels_sha256=sha(root/'test_labels.jsonl'),
        fit_code_sha256=sha(Path(__file__)),
        limitations=['Same training states and shared configuration; mean8 labels cost eight futures per state, not equal offline data cost.',
            'Intervals condition on training data, eight fitted single-label controls, fixed test cohort rankings and sampled futures.',
            'GPT-2 is the optimized scoring proxy, not independent human quality.',
            'Single intervention with fixed4 history/future; no online speed or full-policy advantage claim.'])
    with (root/'test_predictions.jsonl').open('w', encoding='utf-8') as f:
        for i, r in enumerate(data):
            f.write(json.dumps(dict(state_id=r['state_id'], prompt_id=r['prompt_id'], gains=r['gains'], mean_gain=float(y[i]),
                scores={a: float(s[i]) for a, s in scores.items()}, selected={a: bool(m[i]) for a, m in masks.items()}), allow_nan=False)+'\n')
    dump(root/'analysis.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['fit', 'evaluate'])
    ap.add_argument('root', type=Path, nargs='?', default=Path('outputs/kforcing-adaptive/results/averaged-013'))
    args = ap.parse_args()
    globals()[args.action](args.root)
