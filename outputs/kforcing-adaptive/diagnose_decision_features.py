"""Controlled train/dev-only position and EOS feature ablation."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/decision-features'
DEV = ROOT/'results/features-014/development_data.jsonl'
SOURCE = ROOT/'results/averaged-013'
SEEDS = [14112001, 14112002, 14112003]
ALPHA = 1000.
GROUPS = {'confidence14': list(range(14)), 'position15': list(range(15)),
          'eos18': list(range(14))+list(range(15, 19)), 'combined19': list(range(19))}
DEPENDENCIES = ['DECISION_FEATURE_PLAN.md', 'diagnose_decision_features.py', 'core.py',
                'current_features.py', 'upstream/models/pflm.py', 'upstream/models/transformer.py']


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, value):
    assert not p.exists(), f'Preserve: {p}'
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def lines(p, data):
    assert not p.exists(), f'Preserve: {p}'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in data), encoding='utf-8')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(ROOT/'results/features-014/manifest.json')
    assert old['development_data_sha256'] == sha(DEV)
    inputs = {str(DEV.relative_to(ROOT)).replace('\\', '/'): sha(DEV)}
    for split in ('train', 'dev'):
        for suffix in ('states.jsonl', 'labels.jsonl'):
            p = SOURCE/f'{split}_{suffix}'
            assert sha(p) == old['input_hashes'][p.name]
            inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    # Historical aggregate is a development-only numerical control, not a new model choice.
    p = ROOT/'results/features-014/analysis.json'
    inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    dump(OUT/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_whitelist=inputs, code_hashes={s: sha(ROOT/s) for s in DEPENDENCIES},
        checkpoint_sha256='3b013f0cf9a3acab4c20a1bb748beee6525432ef3f56a74df4957b898480a2c2',
        states=446, sources=253, groups=GROUPS, alpha=ALPHA, fold_seeds=SEEDS,
        bootstrap_seed=21192026, bootstrap_reps=4000, retrospective=True,
        no_test_inputs=True, target='Original eight-future full-completion GPT-2 mean NLL gain'))
    print('Fixed train/dev input whitelist, code and four feature groups.', flush=True)


def verify():
    m = read(OUT/'manifest.json')
    for name, h in {**m['input_whitelist'], **m['code_hashes']}.items():
        assert sha(ROOT/name) == h, name
    return m


def collect():
    import torch
    from core import load_model
    from current_features import current_features_tensor
    m = verify()
    assert not (OUT/'features.jsonl').exists()
    checkpoint = Path('work/checkpoints/pflm_lm1b_k4.ckpt')
    assert sha(checkpoint) == m['checkpoint_sha256']
    data = rows(DEV)
    # Explicit projection of source records: future split candidates never enter extraction.
    allowed = ('state_id', 'prompt_id', 'history_ids', 'noise', 'offset', 'initial_candidates', 'features')
    states = {r['state_id']: {k: r[k] for k in allowed}
              for split in ('train', 'dev') for r in rows(SOURCE/f'{split}_states.jsonl')
              if r['status'] == 'paired'}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(checkpoint, 'pflm', 'cuda')
    output, max_error, eos_counts = [], 0., Counter()
    with torch.inference_mode():
        for i, r in enumerate(data):
            s = states[r['state_id']]
            assert s['prompt_id'] == r['prompt_id'] and s['offset'] == r['offset']
            assert len(s['history_ids']) == 6+s['offset'] and s['offset'] in (0, 8)
            history = torch.tensor([s['history_ids']], device='cuda')
            noise = torch.tensor(s['noise'], device='cuda').reshape(1, 4, 1)
            logits = model(history, noise, torch.ones(1, 1, 1, device='cuda'), mode='inference')
            candidates = logits.argmax(-1)[0].cpu().tolist()
            assert candidates == s['initial_candidates']
            assert 102 not in candidates[:2]
            confidence = current_features_tensor(logits).cpu().numpy()
            np.testing.assert_allclose(confidence, r['features'][:14], rtol=.002, atol=.002)
            np.testing.assert_array_equal(r['features'], s['features'])
            max_error = max(max_error, float(np.max(np.abs(confidence-np.array(r['features'][:14])))))
            top = logits[0, 2:].max(-1).values
            gaps = (logits[0, 2:, 102]-top).cpu().tolist()
            indicators = [int(t == 102) for t in candidates[2:]]
            extras = [s['offset']/8]+gaps+indicators
            x = r['features'][:14]+extras
            assert len(x) == 19 and np.isfinite(x).all() and all(v <= 0 for v in gaps)
            eos_counts['tail0'] += indicators[0]
            eos_counts['tail1'] += indicators[1]
            eos_counts['either'] += int(any(indicators))
            output.append(dict(state_id=r['state_id'], prompt_id=r['prompt_id'], source_split=r['source_split'],
                offset=r['offset'], features=x, eos_logits=logits[0, 2:, 102].cpu().tolist(),
                tail_top_logits=top.cpu().tolist(), initial_candidates=candidates,
                history_ids=s['history_ids'], noise=s['noise']))
            if (i+1) % 100 == 0:
                print(f'Replayed current decision only: {i+1}/{len(data)}', flush=True)
    assert len(output) == 446 and len({r['prompt_id'] for r in output}) == 253
    lines(OUT/'features.jsonl', output)
    dump(OUT/'collection.json', dict(passed=True, states=len(output), matched_candidate_positions=len(output)*4,
        max_confidence_feature_absolute_difference=max_error, tail_eos_counts=dict(eos_counts),
        feature_names=[f'original_{i}' for i in range(14)]+['offset_div8', 'eos_gap_tail0',
            'eos_gap_tail1', 'argmax_eos_tail0', 'argmax_eos_tail1'],
        features_sha256=sha(OUT/'features.jsonl'), no_future_generation=True,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))
    print('Feature extraction and original-candidate replay passed.', flush=True)


def fit(x, y):
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1.
    z, intercept = (x-mean)/std, float(y.mean())
    coef = np.linalg.solve(z.T@z+ALPHA*np.eye(z.shape[1]), z.T@(y-intercept))
    return dict(mean=mean.tolist(), std=std.tolist(), coef=coef.tolist(), intercept=intercept)


def predict(model, x):
    return (x-np.array(model['mean']))/np.array(model['std'])@np.array(model['coef'])+model['intercept']


def half(pred):
    order = np.argsort(-pred, kind='stable')
    result = np.zeros(len(pred))
    result[order[:len(pred)//2]] = 1.
    return result


def evaluate():
    verify()
    check = read(OUT/'collection.json')
    assert check['passed'] and check['features_sha256'] == sha(OUT/'features.jsonl')
    data, feature = rows(DEV), rows(OUT/'features.jsonl')
    assert [r['state_id'] for r in data] == sorted(r['state_id'] for r in data)
    assert [r['state_id'] for r in data] == [r['state_id'] for r in feature]
    x = np.array([r['features'] for r in feature])
    y = np.array([r['gain'] for r in data])
    sources = sorted({r['prompt_id'] for r in data})
    n = len(data)
    prediction = {name: np.zeros((3, n)) for name in GROUPS}
    chosen = {name: np.zeros((3, n)) for name in GROUPS}
    heuristic, fractions, baseline = [np.zeros((3, n)) for _ in range(3)]
    fits = []
    for repeat, seed in enumerate(SEEDS):
        order = np.random.default_rng(seed).permutation(sources).tolist()
        assignment = {source: i % 5 for i, source in enumerate(order)}
        for fold in range(5):
            valid = np.array([i for i, r in enumerate(data) if assignment[r['prompt_id']] == fold])
            train = np.setdiff1d(np.arange(n), valid)
            assert not {data[i]['prompt_id'] for i in train} & {data[i]['prompt_id'] for i in valid}
            heuristic[repeat, valid] = half(-x[valid, 2:4].mean(1))
            fractions[repeat, valid] = (len(valid)//2)/len(valid)
            baseline[repeat, valid] = y[train].mean()
            for name, columns in GROUPS.items():
                xt, xv = x[np.ix_(train, columns)], x[np.ix_(valid, columns)]
                model = fit(xt, y[train])
                p = predict(model, xv)
                prediction[name][repeat, valid] = p
                chosen[name][repeat, valid] = half(p)
                fits.append(dict(repeat=repeat, fold=fold, method=name, columns=columns, model=model,
                    train_ids=[data[i]['state_id'] for i in train], valid_ids=[data[i]['state_id'] for i in valid],
                    predictions=p.tolist(), selected=half(p).astype(int).tolist(),
                    train_mse=float(np.mean((predict(model, xt)-y[train])**2))))
    clusters = {s: [i for i, r in enumerate(data) if r['prompt_id'] == s] for s in sources}
    draws = np.random.default_rng(21192026).integers(0, len(sources), size=(4000, len(sources)))

    def interval(v):
        sums = np.array([v[clusters[s]].sum() for s in sources])
        counts = np.array([len(clusters[s]) for s in sources])
        boot = sums[draws].sum(1)/counts[draws].sum(1)
        return dict(estimate=float(np.mean(v)), interval95=np.quantile(boot, [.025, .975]).tolist())

    methods = {}
    ref_error = (prediction['confidence14']-y[None, :])**2
    for name in GROUPS:
        error = (prediction[name]-y[None, :])**2
        methods[name] = dict(dimensions=len(GROUPS[name]), oof_mse=float(error.mean()),
            fold_mean_mse=float(np.mean((baseline-y[None, :])**2)),
            train_mse=float(np.average([r['train_mse'] for r in fits if r['method']==name],
                weights=[len(r['train_ids']) for r in fits if r['method']==name])),
            gain_vs_random=interval(((chosen[name]-fractions)*y[None, :]).mean(0)),
            gain_vs_confidence14=interval(((chosen[name]-chosen['confidence14'])*y[None, :]).mean(0)),
            gain_vs_heuristic=interval(((chosen[name]-heuristic)*y[None, :]).mean(0)),
            mse_reduction_vs_confidence14=interval((ref_error-error).mean(0)))
    c = methods['combined19']
    gate = (c['gain_vs_confidence14']['interval95'][0] > 0 and
            c['gain_vs_heuristic']['interval95'][0] > 0 and
            c['oof_mse'] < methods['confidence14']['oof_mse'])
    records = [dict(state_id=r['state_id'], prompt_id=r['prompt_id'], gain=r['gain'],
        predictions={name: prediction[name][:, i].tolist() for name in GROUPS},
        selected={name: chosen[name][:, i].astype(int).tolist() for name in GROUPS},
        heuristic_selected=heuristic[:, i].astype(int).tolist(), selection_fraction=fractions[:, i].tolist())
        for i, r in enumerate(data)]
    dump(OUT/'fits.json', fits)
    lines(OUT/'oof_predictions.jsonl', records)
    dump(OUT/'analysis.json', dict(states=n, sources=len(sources), fits=len(fits), retrospective=True,
        no_test_outcome_inputs=True, methods=methods, combined19_warrants_fresh_confirmation=bool(gate),
        gate_is_development_screen_not_confirmation=True, tail_eos_counts=check['tail_eos_counts'],
        features_sha256=sha(OUT/'features.jsonl'), fits_sha256=sha(OUT/'fits.json'),
        oof_predictions_sha256=sha(OUT/'oof_predictions.jsonl'), code_sha256=sha(Path(__file__)),
        limitations=['Previously used development sample; no independent test.',
          'Bootstrap conditions on fitted OOF models and fold ranks, no refitting.',
          'Two observed decision offsets only; EOS sample may be sparse.',
          'Original GPT-2 proxy target retained; no new quality or timing validation.',
          'No resolution of paper-baseline reproduction gap.']))
    print(json.dumps(dict(methods=methods, warrants_fresh_confirmation=bool(gate)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'collect', 'evaluate'])
    globals()[parser.parse_args().action]()
