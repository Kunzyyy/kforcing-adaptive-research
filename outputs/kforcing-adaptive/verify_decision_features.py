"""Independent least-squares and grouped-statistics checks of the development ablation."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/decision-features'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    m = read(OUT/'manifest.json')
    for name, digest in {**m['input_whitelist'], **m['code_hashes']}.items():
        assert sha(ROOT/name) == digest
    assert not any('/test_' in p or 'lowdim-015' in p for p in m['input_whitelist'])
    a, c = read(OUT/'analysis.json'), read(OUT/'collection.json')
    assert c['passed'] and c['features_sha256'] == a['features_sha256'] == sha(OUT/'features.jsonl')
    assert a['fits_sha256'] == sha(OUT/'fits.json')
    assert a['oof_predictions_sha256'] == sha(OUT/'oof_predictions.jsonl')
    data = rows(ROOT/'results/features-014/development_data.jsonl')
    features = rows(OUT/'features.jsonl')
    ids = [r['state_id'] for r in data]
    assert ids == sorted(ids) == [r['state_id'] for r in features]
    x = np.array([r['features'] for r in features]); y = np.array([r['gain'] for r in data])
    for i, r in enumerate(features):
        assert r['source_split'] in ('train', 'dev')
        np.testing.assert_array_equal(x[i, :14], data[i]['features'][:14])
        np.testing.assert_array_equal(x[i, 14:], [r['offset']/8]+
            (np.array(r['eos_logits'], dtype=np.float32)-np.array(r['tail_top_logits'], dtype=np.float32)).astype(float).tolist()+
            [int(t == 102) for t in r['initial_candidates'][2:]])
        assert len(r['history_ids']) == 6+r['offset'] and 102 not in r['history_ids']
    labels = {}
    for split in ('train', 'dev'):
        for r in rows(ROOT/f'results/averaged-013/{split}_labels.jsonl'):
            sid, repeat = r['pair_id'].rsplit('-r', 1)
            labels.setdefault(sid, {})[int(repeat)] = r
    for i, sid in enumerate(ids):
        assert set(labels[sid]) == set(range(8))
        assert all(r['valid_label'] for r in labels[sid].values())
        assert abs(y[i]-np.mean([labels[sid][j]['gain'] for j in range(8)])) < 1e-12
    sources = sorted({r['prompt_id'] for r in data})
    assert len(data) == 446 and len(sources) == 253
    groups = {'confidence14': list(range(14)), 'position15': list(range(15)),
              'eos18': list(range(14))+list(range(15, 19)), 'combined19': list(range(19))}
    p = {k: np.zeros((3, 446)) for k in groups}
    masks = {k: np.zeros((3, 446)) for k in groups}
    h, fraction = np.zeros((3, 446)), np.zeros((3, 446))
    max_fit_diff = 0.
    fits = read(OUT/'fits.json')
    assert len(fits) == 60
    assert len({(f['repeat'], f['fold'], f['method']) for f in fits}) == 60
    index = {sid: i for i, sid in enumerate(ids)}
    for f in fits:
        order = np.random.default_rng([14112001,14112002,14112003][f['repeat']]).permutation(sources)
        assignment = {source: i%5 for i, source in enumerate(order)}
        valid = [i for i, r in enumerate(data) if assignment[r['prompt_id']] == f['fold']]
        train = [i for i in range(446) if i not in set(valid)]
        assert f['train_ids'] == [ids[i] for i in train] and f['valid_ids'] == [ids[i] for i in valid]
        assert not {data[i]['prompt_id'] for i in train} & {data[i]['prompt_id'] for i in valid}
        cols = groups[f['method']]
        xt, xv = x[np.ix_(train, cols)], x[np.ix_(valid, cols)]
        mean, std = xt.mean(0), xt.std(0); std[std < 1e-8] = 1
        z, v = (xt-mean)/std, (xv-mean)/std
        # Unpenalized intercept in the augmented system, independently of normal equations.
        design = np.column_stack([np.ones(len(train)), z])
        penalty = np.column_stack([np.zeros(len(cols)), np.sqrt(1000)*np.eye(len(cols))])
        coef = np.linalg.lstsq(np.vstack([design, penalty]), np.r_[y[train], np.zeros(len(cols))], rcond=None)[0]
        pred = np.column_stack([np.ones(len(valid)), v])@coef
        np.testing.assert_allclose(pred, f['predictions'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(mean, f['model']['mean'], atol=1e-13, rtol=0)
        np.testing.assert_allclose(std, f['model']['std'], atol=1e-13, rtol=0)
        max_fit_diff = max(max_fit_diff, float(np.max(np.abs(pred-f['predictions']))))
        ordering = sorted(range(len(valid)), key=lambda j: (-pred[j], ids[valid[j]]))
        chosen = np.zeros(len(valid)); chosen[ordering[:len(valid)//2]] = 1
        np.testing.assert_array_equal(chosen, f['selected'])
        rep, name = f['repeat'], f['method']
        p[name][rep, valid], masks[name][rep, valid] = pred, chosen
        heuristic_order = sorted(valid, key=lambda j: (x[j, 2:4].mean(), ids[j]))
        h[rep, heuristic_order[:len(valid)//2]] = 1
        fraction[rep, valid] = (len(valid)//2)/len(valid)
    metrics, descriptions = [], []
    ref_error = (p['confidence14']-y)**2
    for name in groups:
        method = a['methods'][name]
        np.testing.assert_allclose(np.mean((p[name]-y)**2), method['oof_mse'], atol=1e-12, rtol=0)
        for key, values in (
            ('gain_vs_random', ((masks[name]-fraction)*y).mean(0)),
            ('gain_vs_confidence14', ((masks[name]-masks['confidence14'])*y).mean(0)),
            ('gain_vs_heuristic', ((masks[name]-h)*y).mean(0)),
            ('mse_reduction_vs_confidence14', (ref_error-(p[name]-y)**2).mean(0))):
            metrics.append(values); descriptions.append((name,key))
            np.testing.assert_allclose(values.mean(), method[key]['estimate'], atol=1e-12, rtol=0)
    values = np.array(metrics).T
    blocks = [np.array([i for i, r in enumerate(data) if r['prompt_id']==source]) for source in sources]
    rng, boots = np.random.default_rng(21192026), []
    for _ in range(4000):
        selected = np.concatenate([blocks[i] for i in rng.integers(0, 253, 253)])
        boots.append(values[selected].mean(0))
    intervals = np.quantile(boots, [.025, .975], axis=0).T
    for interval, (name, key) in zip(intervals, descriptions):
        np.testing.assert_allclose(interval, a['methods'][name][key]['interval95'], atol=1e-11, rtol=0)
    historical = read(ROOT/'results/features-014/analysis.json')['fixed']['ridge_confidence14_a01000']
    for key in ('oof_mse', 'train_mse'):
        np.testing.assert_allclose(a['methods']['confidence14'][key], historical[key], atol=1e-12, rtol=0)
    np.testing.assert_allclose(a['methods']['confidence14']['gain_vs_random']['estimate'],
                               historical['gain_vs_random']['estimate'], atol=1e-12, rtol=0)
    # Fresh GPU spot-check of EOS scores on first/last states, without any future branch.
    import torch
    from core import load_model
    checkpoint = Path('work/checkpoints/pflm_lm1b_k4.ckpt')
    assert sha(checkpoint) == m['checkpoint_sha256']
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    model = load_model(checkpoint, 'pflm', 'cuda')
    selected_indices = list(range(8))+list(range(len(data)-8, len(data)))
    with torch.inference_mode():
        for i in selected_indices:
            r = features[i]
            logits = model(torch.tensor([r['history_ids']], device='cuda'),
                torch.tensor(r['noise'], device='cuda').reshape(1,4,1), torch.ones(1,1,1,device='cuda'), mode='inference')
            np.testing.assert_allclose(logits[0,2:,102].cpu().numpy(), r['eos_logits'], atol=1e-5, rtol=0)
            assert logits.argmax(-1)[0].tolist() == r['initial_candidates']
    c = a['methods']['combined19']
    gate = c['gain_vs_confidence14']['interval95'][0]>0 and c['gain_vs_heuristic']['interval95'][0]>0 and c['oof_mse']<a['methods']['confidence14']['oof_mse']
    assert bool(gate) == a['combined19_warrants_fresh_confirmation']
    result = dict(passed=True, all_60_fits_independent_lstsq=True, max_prediction_difference=max_fit_diff,
        all_16_intervals_independently_recomputed=True, no_test_inputs=True,
        original_stage14_control_reproduced=True, eos_gpu_spot_checks=len(selected_indices),
        analysis_sha256=sha(OUT/'analysis.json'), verifier_sha256=sha(Path(__file__)))
    assert not (OUT/'integrity_audit.json').exists()
    (OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
