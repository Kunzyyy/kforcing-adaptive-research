"""Stage 12: fixed decision, eight paired independently resampled futures."""
import argparse
import hashlib
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from core import load_model
from learn_local_gain import feature_tensor
from collect_benefit import dump, verify_checkpoints
from confirm_local_gain import read, rows, sha, prior_sources
from run_frozen_online import append

ROOT = Path(__file__).resolve().parent
SEEDS = [1201, 1213]
OFFSETS = [0, 8]
REPEATS = 8
STATE_KEYS = ('status', 'history_ids', 'noise', 'initial_candidates', 'split_candidates', 'features')


def noise_tape(seed):
    return torch.rand((1, 122, 1), generator=torch.Generator().manual_seed(seed))


def mixed_noise(state_seed, future_seed, offset):
    base = noise_tape(state_seed)
    future = noise_tape(future_seed)
    tape = base.clone()
    tape[:, offset+4:] = future[:, offset+4:]
    assert torch.equal(tape[:, :offset+4], base[:, :offset+4])
    assert not torch.equal(tape[:, offset+4:], base[:, offset+4:])
    return tape


def noise_hash(tape):
    return hashlib.sha256(tape.cpu().numpy().astype('<f4').tobytes()).hexdigest()


@torch.inference_mode()
def pair(model, prefix, tape, offset, projection):
    # Same single-intervention semantics as rollout_quality.pair, with supplied tape.
    device = prefix.device
    tape = tape.to(device)
    tau = torch.ones(1, 1, 1, device=device)
    context = prefix
    cache = None
    produced = history_calls = 0
    while produced < offset:
        logits, cache = model(context, tape[:, produced:produced+4], tau,
                              mode='inference', kv_caches=cache, return_kv=True)
        assert all(c[0].shape[1] == context.shape[1] for c in cache)
        ids = logits.argmax(-1)[0].cpu().tolist()
        history_calls += 1
        if 102 in ids:
            return dict(status='unreachable_eos', history_ids=context[0].cpu().tolist()+ids[:ids.index(102)+1])
        context = torch.cat((context, logits.argmax(-1)), 1)
        produced += 4
    captured = []
    hook = model.output_layer.linear.register_forward_pre_hook(lambda mod, inputs: captured.append(inputs[0].detach()))
    try:
        logits4, at_cache = model(context, tape[:, offset:offset+4], tau,
                                  mode='inference', kv_caches=cache, return_kv=True)
    finally:
        hook.remove()
    assert len(captured) == 1 and captured[0].shape == (1, 4, 768)
    initial = logits4.argmax(-1)[0].cpu().tolist()
    if 102 in initial[:2]:
        return dict(status='head_eos', history_ids=context[0].cpu().tolist(), initial_candidates=initial)
    features = feature_tensor(logits4, captured[0], tape[:, offset:offset+4], projection)[0].cpu().tolist()
    assert len(features) == 82 and np.isfinite(features).all()
    split_context = torch.cat((context, torch.tensor([initial[:2]], device=device)), 1)
    tail, tail_cache = model(split_context, tape[:, offset+2:offset+4], tau,
                            mode='inference', kv_caches=at_cache, return_kv=True)
    split = initial[:2]+tail.argmax(-1)[0].cpu().tolist()
    assert all(c[0].shape[1] == context.shape[1] for c in at_cache)
    assert all(c[0].shape[1] == context.shape[1]+2 for c in tail_cache)

    def finish(first, start_cache, extra):
        if 102 in first:
            seq = context[0].cpu().tolist()+first[:first.index(102)+1]
            return dict(ids=seq, calls=history_calls+1+extra, computed_candidates=offset+4+2*extra, stopped_eos=True)
        hist = torch.cat((context, torch.tensor([first], device=device)), 1)
        position = offset+4
        kv = start_cache
        calls = history_calls+1+extra
        candidates = offset+4+2*extra
        while position < 122:
            width = min(4, 122-position)
            logits, kv = model(hist, tape[:, position:position+width], tau,
                               mode='inference', kv_caches=kv, return_kv=True)
            assert all(c[0].shape[1] == hist.shape[1] for c in kv)
            ids = logits.argmax(-1)[0].cpu().tolist()
            calls += 1
            candidates += width
            kept = ids[:ids.index(102)+1] if 102 in ids else ids
            hist = torch.cat((hist, torch.tensor([kept], device=device)), 1)
            position += len(kept)
            if 102 in ids:
                break
        seq = hist[0].cpu().tolist()
        return dict(ids=seq, calls=calls, computed_candidates=candidates, stopped_eos=seq[-1] == 102)

    return dict(status='paired', history_ids=context[0].cpu().tolist(), noise=tape[0, offset:offset+4, 0].cpu().tolist(),
                initial_candidates=initial, split_candidates=split, features=features,
                keep4=finish(initial, at_cache, 0), split22=finish(split, tail_cache, 1))


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    assert not (args.out/'manifest.json').exists()
    old, seen = prior_sources(args.out.parent)
    for path in ('online-009/prefixes.json', 'early-010/prefixes.json', 'quality-011/prefixes.json'):
        old += read(args.out.parent/path)
    old += [dict(p, ids=p['reference_ids'][:6]) for p in read(args.out.parent/'confirm-008/sources.json')]
    seen.update(tuple(p['ids'][:6]) for p in old)
    seen_rows = {p['source_row'] for p in old}
    seen_hash = {p['source_sha256'] for p in old}
    dataset = Path('work/lm1b-data/test.parquet')
    assert sha(dataset) == 'd3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    order = list(range(len(texts)))
    random.Random(12112001).shuffle(order)
    bert = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    chosen = []
    for index in order:
        digest = hashlib.sha256(texts[index].encode()).hexdigest()
        if index in seen_rows or digest in seen_hash:
            continue
        ids = bert.encode(texts[index], add_special_tokens=True, truncation=True, max_length=128)
        if len(ids) < 12 or tuple(ids[:6]) in seen:
            continue
        i = len(chosen)
        chosen.append(dict(prompt_id=f's12-{i:03}', index=i, ids=ids[:6], source_row=index, source_sha256=digest))
        seen.add(tuple(ids[:6]))
        seen_hash.add(digest)
        if len(chosen) == 96:
            break
    assert len(chosen) == 96
    dump(args.out/'prefixes.json', chosen)
    for source, dest in [('learned-007/projection.npy', 'projection.npy'),
                         ('learned-007/locked_model.json', 'old_model.json'),
                         ('quality-011/locked_model.json', 'quality_model.json')]:
        shutil.copyfile(args.out.parent/source, args.out/dest)
    files = ['STAGE12_PROTOCOL.md', 'repeat_rollout_quality.py', 'analyze_rollout_stability.py',
             'learn_local_gain.py', 'score_rollout_quality.py', 'early_stop_decoder.py']
    dump(args.out/'manifest.json', dict(sources=96, seeds=SEEDS, offsets=OFFSETS, repeats=REPEATS,
        dataset_sha256=sha(dataset), code_hashes={f: sha(ROOT/f) for f in files},
        artifact_hashes={f: sha(args.out/f) for f in ['prefixes.json', 'projection.npy', 'old_model.json', 'quality_model.json']}))
    print('Frozen 96 sources / 384 attempted states / 8 futures each.', flush=True)


def setup(args):
    m = read(args.out/'manifest.json')
    for f, digest in m['code_hashes'].items():
        assert sha(ROOT/f) == digest, f
    for f, digest in m['artifact_hashes'].items():
        assert sha(args.out/f) == digest, f
    verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    return load_model(args.checkpoints/'pflm_lm1b_k4.ckpt', 'pflm', 'cuda'), torch.from_numpy(np.load(args.out/'projection.npy')).cuda()


@torch.inference_mode()
def preflight(args):
    from early_stop_decoder import generate
    model, projection = setup(args)
    old = [r for r in rows(args.out.parent/'quality-011/test_pairs.jsonl') if r['status'] == 'paired'][:12]
    cases = []
    for i, r in enumerate(old):
        prefix = torch.tensor([r['history_ids'][:6]], device='cuda')
        base = pair(model, prefix, noise_tape(r['noise_seed']), r['offset'], projection)
        assert all(base[k] == r[k] for k in STATE_KEYS)
        for b in ('keep4', 'split22'):
            assert all(base[b][k] == r[b][k] for k in base[b])
        for future_seed in (123000001+i, 123010001+i):
            tape = mixed_noise(r['noise_seed'], future_seed, r['offset'])
            alt = pair(model, prefix, tape, r['offset'], projection)
            assert all(alt[k] == base[k] for k in STATE_KEYS)
            schedule = [False]*31
            schedule[r['offset']//4] = True
            for b, policy in [('keep4', 'fixed4'), ('split22', 'replay')]:
                other = generate(model, prefix, policy, schedule=schedule, supplied_noise=tape.cuda())
                assert alt[b]['ids'] == other['ids']
                assert alt[b]['calls'] == other['forward_calls']
                assert alt[b]['computed_candidates'] == other['computed_candidates']
            cases.append(dict(old_pair_id=r['pair_id'], future_seed=future_seed))
    dump(args.out/'preflight.json', dict(passed=True, old_cases=len(old), changed_future_cases=len(cases),
        original_branches_exact=True, changed_future_replay_exact=True, state_exact=True, cases=cases,
        collector_sha256=sha(Path(__file__))))
    print('12 original branch pairs and 24 changed-future replay pairs match exactly.', flush=True)


@torch.inference_mode()
def collect(args):
    from transformers import BertTokenizerFast
    assert read(args.out/'preflight.json')['passed']
    model, projection = setup(args)
    bert = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    path = args.out/'diagnostic_pairs.jsonl'
    state_path = args.out/'states.jsonl'
    assert not path.exists() and not state_path.exists()
    counts = {}
    paired = 0
    with path.open('w', encoding='utf-8') as dest, state_path.open('w', encoding='utf-8') as states:
        for p in read(args.out/'prefixes.json'):
            prefix = torch.tensor([p['ids']], device='cuda')
            for seed in SEEDS:
                state_seed = 121000000+seed+p['index']*100003
                for offset in OFFSETS:
                    state_id = f"{p['prompt_id']}-{seed}-{offset}"
                    anchor = None
                    for rep in range(REPEATS):
                        future_seed = 122000000+p['index']*100003+seed*101+offset*17+rep*10000019
                        tape = mixed_noise(state_seed, future_seed, offset)
                        r = pair(model, prefix, tape, offset, projection)
                        if rep == 0:
                            anchor = {k: r[k] for k in STATE_KEYS if k in r}
                            counts[r['status']] = counts.get(r['status'], 0)+1
                            append(states, dict(anchor, state_id=state_id, prompt_id=p['prompt_id'], seed=seed,
                                                state_noise_seed=state_seed, offset=offset))
                            if r['status'] != 'paired':
                                break
                        assert all(r[k] == v for k, v in anchor.items())
                        r.update(pair_id=f'{state_id}-r{rep}', state_id=state_id, prompt_id=p['prompt_id'], seed=seed,
                                 noise_seed=state_seed, offset=offset, replicate=rep, future_seed=future_seed,
                                 noise_sha256=noise_hash(tape))
                        for b in ('keep4', 'split22'):
                            r[b]['completion'] = bert.decode(r[b]['ids'][6:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        append(dest, r)
                        paired += 1
            if (p['index']+1) % 8 == 0:
                print('sources', p['index']+1, '/ 96; repeated pairs', paired, flush=True)
    dump(args.out/'diagnostic_collection.json', dict(passed=True, attempted_states=384, state_status_counts=counts,
        pairs=paired, pairs_sha256=sha(path), states_sha256=sha(state_path), collector_sha256=sha(Path(__file__)),
        locked_model_sha256=None, manifest_sha256=sha(args.out/'manifest.json'),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'preflight', 'collect', 'score'])
    ap.add_argument('--out', type=Path, default=ROOT/'results/stability-012')
    ap.add_argument('--checkpoints', type=Path, default=Path('work/checkpoints'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == 'score':
        from score_rollout_quality import score
        args.split = 'diagnostic'
        score(args)
    else:
        globals()[args.action](args)
