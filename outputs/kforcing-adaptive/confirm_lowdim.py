"""Stage 15: freeze two development-selected low-dimensional models, then fresh test."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT/'results/lowdim-015'
N_SOURCES = 384
SEED = 1501
OFFSETS = [0, 8]
BOOT_SEED = 15112004
CODE_FILES = ['STAGE15_PROTOCOL.md', 'confirm_lowdim.py', 'repeat_rollout_quality.py',
    'score_rollout_quality.py', 'core.py', 'learn_local_gain.py', 'current_features.py',
    'collect_benefit.py', 'run_frozen_online.py', 'external_gpt2_score.py',
    'diagnose_feature_signal.py', 'confirm_local_gain.py', 'fit_averaged_labels.py']
OLD_FILES = ['lm1b-004/prefixes.json', 'learned-007/prefixes.json', 'pilot-001/prefixes.json',
    'benefit-002/prefixes.json', 'current-003/fresh_prefixes.json', 'confirm-008/sources.json',
    'online-009/prefixes.json', 'early-010/prefixes.json', 'quality-011/prefixes.json',
    'stability-012/prefixes.json', 'averaged-013/prefixes.json']

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, obj): p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
def now(): return datetime.now(timezone.utc).isoformat()
def write_rows(path, data):
    with path.open('w', encoding='utf-8') as f:
        for r in data: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')

def old_sources(result_root):
    old = []
    for file in OLD_FILES:
        raw = read(result_root/file)
        if isinstance(raw, dict):
            for group in raw.values(): old.extend(group)
        else: old.extend(raw)
    return old

def kernel(a, b):
    distance = np.maximum((a*a).sum(1)[:, None]+(b*b).sum(1)[None, :]-2*a@b.T, 0.)
    return np.exp(-distance/14.)

def fit_model(data, kind, alpha):
    x = np.array([r['features'][:14] for r in data])
    y = np.array([r['gain'] for r in data])
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1.
    z = (x-mean)/std
    m = dict(kind=kind, alpha=alpha, feature_count=14, mean=mean.tolist(), std=std.tolist(),
        intercept=float(y.mean()), training_states=len(y))
    if kind == 'ridge':
        m['coef'] = np.linalg.solve(z.T@z+alpha*np.eye(14), z.T@(y-y.mean())).tolist()
    else:
        k = kernel(z, z)
        colmean, grand = k.mean(0), float(k.mean())
        centered = k-k.mean(1)[:, None]-colmean[None, :]+grand
        m.update(train_z=z.tolist(), kernel_colmean=colmean.tolist(), kernel_grandmean=grand,
            coef=np.linalg.solve(centered+alpha*np.eye(len(y)), y-y.mean()).tolist(), gamma=1/14.)
    return m

def predict(model, data):
    x = np.array([r['features'][:14] for r in data])
    z = (x-model['mean'])/model['std']
    if model['kind'] == 'ridge': design = z
    else:
        k = kernel(z, np.array(model['train_z']))
        design = k-k.mean(1)[:, None]-np.array(model['kernel_colmean'])[None, :]+model['kernel_grandmean']
    return design@np.array(model['coef'])+model['intercept']

def freeze(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    out = args.out
    assert not (out/'locked_model.json').exists() and not (out/'manifest.json').exists()
    development = out.parent/'features-014'
    dm = read(development/'manifest.json')
    path = development/'development_data.jsonl'
    assert sha(path) == dm['development_data_sha256']
    assert read(development/'integrity_audit.json')['passed']
    data = rows(path)
    assert len(data) == 446 and all(r['source_split'] in ['train', 'dev'] for r in data)
    models = {'linear14': fit_model(data, 'ridge', 1000.), 'rbf14': fit_model(data, 'rbf', 10.)}
    lock = dict(created_utc=now(), models=models, train_states=len(data),
        train_sources=len({r['prompt_id'] for r in data}), training_data_sha256=sha(path),
        training_path='features-014/development_data.jsonl',
        code_hashes={f: sha(ROOT/f) for f in CODE_FILES},
        historical_source_hashes={f: sha(out.parent/f) for f in OLD_FILES},
        selection='Two configurations chosen by stage14; no further model selection or test-label input',
        primary_coverage=.9875, bootstrap_repeats=8000, bootstrap_seed=BOOT_SEED)
    dump(out/'locked_model.json', lock)
    old = old_sources(out.parent)
    seen = {tuple(p.get('ids', p.get('reference_ids'))[:6]) for p in old}
    seen_rows = {p['source_row'] for p in old if 'source_row' in p}
    seen_hash = {p['source_sha256'] for p in old if 'source_sha256' in p}
    dataset = Path('work/lm1b-data/test.parquet')
    assert sha(dataset) == 'd3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    order = list(range(len(texts)))
    random.Random(15112001).shuffle(order)
    tokenizer = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    chosen = []
    for index in order:
        digest = hashlib.sha256(texts[index].encode()).hexdigest()
        if index in seen_rows or digest in seen_hash: continue
        ids = tokenizer.encode(texts[index], add_special_tokens=True, truncation=True, max_length=128)
        if len(ids) < 12 or tuple(ids[:6]) in seen: continue
        i = len(chosen)
        chosen.append(dict(prompt_id=f's15-{i:03}', index=i, ids=ids[:6], source_row=index,
            source_sha256=digest, split='test'))
        seen.add(tuple(ids[:6])); seen_hash.add(digest)
        if len(chosen) == N_SOURCES: break
    assert len(chosen) == N_SOURCES
    dump(out/'prefixes.json', chosen)
    shutil.copyfile(out.parent/'learned-007/projection.npy', out/'projection.npy')
    dump(out/'manifest.json', dict(created_utc=now(), sources=N_SOURCES, seeds=[SEED], offsets=OFFSETS,
        repeats=8, dataset_sha256=sha(dataset), locked_model_sha256=sha(out/'locked_model.json'),
        artifact_hashes={f: sha(out/f) for f in ['prefixes.json', 'projection.npy']},
        tokenizer_sha256=sha(Path('work/pilot-data/vocab.txt'))))
    print('Locked two models trained on 446 old development states; 384 fresh test sources.', flush=True)

def verify(out):
    m, lock = read(out/'manifest.json'), read(out/'locked_model.json')
    assert sha(out/'locked_model.json') == m['locked_model_sha256']
    for f, h in m['artifact_hashes'].items(): assert sha(out/f) == h, f
    for f, h in lock['code_hashes'].items(): assert sha(ROOT/f) == h, f
    for f, h in lock['historical_source_hashes'].items(): assert sha(out.parent/f) == h, f
    assert sha(out.parent/lock['training_path']) == lock['training_data_sha256']
    assert sha(Path('work/pilot-data/vocab.txt')) == m['tokenizer_sha256']
    return lock

def setup(args):
    import torch
    from core import load_model
    from collect_benefit import verify_checkpoints
    verify(args.out); verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    return load_model(args.checkpoints/'pflm_lm1b_k4.ckpt', 'pflm', 'cuda'), torch.from_numpy(np.load(args.out/'projection.npy')).cuda()

def preflight(args):
    import torch
    from diagnose_feature_signal import operators
    from repeat_rollout_quality import pair, mixed_noise, noise_hash, STATE_KEYS
    lock = verify(args.out)
    data = rows(args.out.parent/lock['training_path'])
    x = np.array([r['features'] for r in data]); y = np.array([r['gain'] for r in data])
    ix = np.arange(len(data))
    for name, model in lock['models'].items():
        _, op = operators(x, ix, ix, dict(group='confidence14', kind=model['kind'], alpha=model['alpha']))
        actual = predict(model, data)
        np.testing.assert_allclose(actual, op@y, atol=1e-10, rtol=1e-9)
        z = (x[:, :14]-model['mean'])/model['std']
        if name == 'linear14':
            a = np.vstack([z, np.sqrt(model['alpha'])*np.eye(14)])
            b = np.concatenate([y-y.mean(), np.zeros(14)])
            coef = np.linalg.lstsq(a, b, rcond=None)[0]
            other = z@coef+y.mean()
        else:
            k = np.exp(-np.sum((z[:, None, :]-z[None, :, :])**2, axis=2)/14.)
            h = np.eye(len(y))-np.ones((len(y), len(y)))/len(y)
            centered = h@k@h
            coef = np.linalg.solve(centered+10*np.eye(len(y)), h@y)
            other = centered@coef+y.mean()
        np.testing.assert_allclose(actual, other, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(actual[:1], predict(model, data[:1]), atol=1e-10)
    model, projection = setup(args)
    old = rows(args.out.parent/'averaged-013/train_pairs.jsonl')[:16]
    with torch.inference_mode():
        for r in old:
            tape = mixed_noise(r['noise_seed'], r['future_seed'], r['offset'])
            assert noise_hash(tape) == r['noise_sha256']
            actual = pair(model, torch.tensor([r['history_ids'][:6]], device='cuda'), tape, r['offset'], projection)
            assert all(actual[k] == r[k] for k in STATE_KEYS)
            for b in ['keep4', 'split22']:
                assert all(actual[b][k] == r[b][k] for k in actual[b])
    dump(args.out/'preflight.json', dict(passed=True, time_utc=now(), models_independently_crosschecked=2,
        old_branches_replayed=len(old), locked_model_sha256=sha(args.out/'locked_model.json')))
    print('Both model implementations crosschecked; 16 prior branch pairs replayed exactly.', flush=True)

def collect(args):
    import torch
    from transformers import BertTokenizerFast
    from repeat_rollout_quality import pair, mixed_noise, noise_hash, STATE_KEYS
    from run_frozen_online import append
    out = args.out
    assert read(out/'preflight.json')['passed']
    model, projection = setup(args)
    tokenizer = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    path, state_path = out/'test_pairs.jsonl', out/'test_states.jsonl'
    assert not path.exists() and not state_path.exists()
    counts, total = {}, 0
    with torch.inference_mode(), path.open('w', encoding='utf-8') as dest, state_path.open('w', encoding='utf-8') as sf:
        for p in read(out/'prefixes.json'):
            prefix = torch.tensor([p['ids']], device='cuda')
            state_seed = 151000000+SEED+p['index']*100003
            for offset in OFFSETS:
                sid = f"{p['prompt_id']}-{SEED}-{offset}"
                anchor = None
                for rep in range(8):
                    future_seed = 152000000+p['index']*100003+offset*17+rep*100000007
                    tape = mixed_noise(state_seed, future_seed, offset)
                    r = pair(model, prefix, tape, offset, projection)
                    if rep == 0:
                        anchor = {k: r[k] for k in STATE_KEYS if k in r}
                        counts[r['status']] = counts.get(r['status'], 0)+1
                        append(sf, dict(anchor, state_id=sid, prompt_id=p['prompt_id'], seed=SEED,
                            state_noise_seed=state_seed, offset=offset, split='test'))
                        if r['status'] != 'paired': break
                    assert all(r[k] == v for k, v in anchor.items())
                    r.update(pair_id=f'{sid}-r{rep}', state_id=sid, prompt_id=p['prompt_id'], seed=SEED,
                        noise_seed=state_seed, offset=offset, replicate=rep, future_seed=future_seed,
                        noise_sha256=noise_hash(tape))
                    for b in ['keep4', 'split22']:
                        r[b]['completion'] = tokenizer.decode(r[b]['ids'][6:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    append(dest, r); total += 1
            if (p['index']+1) % 16 == 0:
                print('fresh test sources', p['index']+1, '/', N_SOURCES, 'paired futures', total, flush=True)
    dump(out/'test_collection.json', dict(passed=True, finished_utc=now(), attempted_states=2*N_SOURCES,
        state_status_counts=counts, pairs=total, pairs_sha256=sha(path), states_sha256=sha(state_path),
        collector_sha256=sha(Path(__file__)), manifest_sha256=sha(out/'manifest.json'),
        locked_model_sha256=sha(out/'locked_model.json'), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))

def rank(scores, data):
    return sorted(range(len(data)), key=lambda i: (-float(scores[i]), data[i]['state_id']))
def half(scores, data):
    mask = np.zeros(len(data)); mask[rank(scores, data)[:len(data)//2]] = 1.
    return mask

def predict_test(args):
    out = args.out
    lock = verify(out)
    assert not (out/'test_labels.jsonl').exists() and not (out/'frozen_predictions.json').exists()
    collection = read(out/'test_collection.json')
    assert collection['states_sha256'] == sha(out/'test_states.jsonl')
    data = [r for r in rows(out/'test_states.jsonl') if r['status'] == 'paired']
    scores = {name: predict(model, data) for name, model in lock['models'].items()}
    scores['heuristic'] = np.array([-np.mean(r['features'][2:4]) for r in data])
    dump(out/'frozen_predictions.json', dict(created_utc=now(), locked_model_sha256=sha(out/'locked_model.json'),
        states_sha256=sha(out/'test_states.jsonl'), scores={r['state_id']: {n: float(v[i]) for n, v in scores.items()} for i, r in enumerate(data)},
        rankings={name: [data[i]['state_id'] for i in rank(score, data)] for name, score in scores.items()},
        test_labels_absent_when_frozen=True))
    print('Frozen predictions and rankings before scoring:', len(data), 'paired states.', flush=True)

def score(args):
    from score_rollout_quality import score as original_score
    verify(args.out)
    prediction_hash = sha(args.out/'frozen_predictions.json')
    args.split = 'test'
    original_score(args)
    dump(args.out/'scoring_prediction_link.json', dict(passed=True, frozen_predictions_sha256=prediction_hash,
        test_scoring_sha256=sha(args.out/'test_scoring.json'), completed_utc=now()))

def interval(values, data, coverage=.95):
    groups = defaultdict(list)
    for i, r in enumerate(data): groups[r['prompt_id']].append(i)
    keys = sorted(groups)
    sums = np.array([sum(values[i] for i in groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(keys), (8000, len(keys)))
    boot = sums[draws].sum(1)/counts[draws].sum(1)
    tail = (1-coverage)/2
    return dict(estimate=float(np.mean(values)), interval=np.quantile(boot, [tail, 1-tail]).tolist(),
        coverage=coverage, source_clusters=len(keys))

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None

def evaluate(args):
    from fit_averaged_labels import load
    out = args.out; lock = verify(out)
    assert not (out/'analysis.json').exists()
    frozen = read(out/'frozen_predictions.json')
    link = read(out/'scoring_prediction_link.json')
    assert link['frozen_predictions_sha256'] == sha(out/'frozen_predictions.json')
    assert link['test_scoring_sha256'] == sha(out/'test_scoring.json')
    data, excluded = load(out, 'test')
    y = np.array([r['gain'] for r in data])
    scores = {n: np.array([frozen['scores'][r['state_id']][n] for r in data]) for n in ['linear14', 'rbf14', 'heuristic']}
    masks = {n: half(s, data) for n, s in scores.items()}
    for name, model in lock['models'].items(): np.testing.assert_allclose(scores[name], predict(model, data), atol=1e-10)
    for name in scores:
        valid = {r['state_id'] for r in data}
        ordered = [s for s in frozen['rankings'][name] if s in valid]
        selected_ids = {r['state_id'] for i, r in enumerate(data) if masks[name][i]}
        assert set(ordered[:len(data)//2]) == selected_ids
    primary, arms, passes = {}, {}, {}
    enough = len({r['prompt_id'] for r in data}) >= 128
    for name in ['linear14', 'rbf14']:
        primary[name] = dict(vs_random=interval((masks[name]-masks[name].mean())*y, data, .9875),
            vs_heuristic=interval((masks[name]-masks['heuristic'])*y, data, .9875))
        passes[name] = bool(enough and all(r['interval'][0] > 0 for r in primary[name].values()))
        arms[name] = dict(mse=float(np.mean((scores[name]-y)**2)), correlation=corr(scores[name], y),
            constant_train_mean_mse=float(np.mean((lock['models'][name]['intercept']-y)**2)), selected=int(masks[name].sum()))
    raw_labels = rows(out/'test_labels.jsonl')
    labels = {r['pair_id']: r for r in raw_labels}
    secondary = {}
    for field in ['teacher_nll_sum', 'visible_new_tokens', 'forward_calls', 'computed_candidates']:
        delta = np.array([np.mean([labels[f"{r['state_id']}-r{j}"]['keep4'][field]-labels[f"{r['state_id']}-r{j}"]['split22'][field] for j in range(8)]) for r in data])
        secondary[field] = {name: interval((m-m.mean())*delta, data) for name, m in masks.items()}
    ya = np.array([np.mean(r['gains'][:4]) for r in data]); yb = np.array([np.mean(r['gains'][4:]) for r in data])
    ma, mb = half(ya, data), half(yb, data)
    result = dict(stage=15, completed_utc=now(), test_states=len(data),
        test_sources=len({r['prompt_id'] for r in data}), excluded_incomplete=excluded,
        primary=primary, candidate_passes=passes, enough_sources=enough,
        advance_gate_passed=any(passes.values()),
        next_candidate='linear14' if passes['linear14'] else ('rbf14' if passes['rbf14'] else None),
        arms=arms, heuristic_vs_random=interval((masks['heuristic']-masks['heuristic'].mean())*y, data),
        rbf_vs_linear_exploratory=interval((masks['rbf14']-masks['linear14'])*y, data),
        mean_all_split_gain=interval(y, data), split_half_target_correlation=corr(ya, yb),
        cross_half_outcome_informed_gain=interval(((ma-ma.mean())*yb+(mb-mb.mean())*ya)/2, data),
        exploratory_other_metrics=secondary, locked_model_sha256=sha(out/'locked_model.json'),
        frozen_predictions_sha256=sha(out/'frozen_predictions.json'), labels_sha256=sha(out/'test_labels.jsonl'),
        limitations=['Optimized GPT-2 proxy, not independent human text quality.',
            'Offline one-intervention ranking, not online policy or wall-clock advantage.',
            'Intervals condition on trained models, frozen cohort rankings and sampled futures.',
            'Both candidates selected on old development data; fresh test used once.'])
    write_rows(out/'test_predictions.jsonl', [dict(state_id=r['state_id'], prompt_id=r['prompt_id'],
        gains=r['gains'], mean_gain=float(y[i]), scores={n: float(s[i]) for n, s in scores.items()},
        selected={n: bool(m[i]) for n, m in masks.items()}) for i, r in enumerate(data)])
    dump(out/'analysis.json', result)
    print(json.dumps({k: result[k] for k in ['test_states', 'test_sources', 'primary', 'candidate_passes', 'advance_gate_passed']}, indent=2), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['freeze', 'preflight', 'collect', 'predict_test', 'score', 'evaluate'])
    ap.add_argument('--out', type=Path, default=DEFAULT)
    ap.add_argument('--checkpoints', type=Path, default=Path('work/checkpoints'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.action](args)
