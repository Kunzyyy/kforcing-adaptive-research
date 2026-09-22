"""Fresh train/dev/test repeated futures for a paired training-target experiment."""
import argparse
import hashlib
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from core import load_model
from collect_benefit import dump, verify_checkpoints
from confirm_local_gain import read, rows, sha, prior_sources
from run_frozen_online import append
from repeat_rollout_quality import pair, mixed_noise, noise_hash, STATE_KEYS

ROOT = Path(__file__).resolve().parent
SEED = 1301
OFFSETS = [0, 8]


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    assert not (args.out/'manifest.json').exists()
    old, seen = prior_sources(args.out.parent)
    for name in ['online-009', 'early-010', 'quality-011', 'stability-012']:
        old += read(args.out.parent/name/'prefixes.json')
    old += [dict(p, ids=p['reference_ids'][:6]) for p in read(args.out.parent/'confirm-008/sources.json')]
    seen.update(tuple(p['ids'][:6]) for p in old)
    seen_rows = {p['source_row'] for p in old}
    seen_hashes = {p['source_sha256'] for p in old}
    dataset = Path('work/lm1b-data/test.parquet')
    assert sha(dataset) == 'd3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    order = list(range(len(texts)))
    random.Random(13112001).shuffle(order)
    tokenizer = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    chosen = []
    for index in order:
        digest = hashlib.sha256(texts[index].encode()).hexdigest()
        if index in seen_rows or digest in seen_hashes:
            continue
        ids = tokenizer.encode(texts[index], add_special_tokens=True, truncation=True, max_length=128)
        if len(ids) < 12 or tuple(ids[:6]) in seen:
            continue
        i = len(chosen)
        chosen.append(dict(prompt_id=f's13-{i:03}', index=i, ids=ids[:6], source_row=index, source_sha256=digest,
            split='train' if i < 192 else ('dev' if i < 256 else 'test')))
        seen.add(tuple(ids[:6]))
        seen_hashes.add(digest)
        if len(chosen) == 384:
            break
    assert len(chosen) == 384
    dump(args.out/'prefixes.json', chosen)
    shutil.copyfile(args.out.parent/'learned-007/projection.npy', args.out/'projection.npy')
    files = ['STAGE13_PROTOCOL.md', 'collect_averaged_labels.py', 'fit_averaged_labels.py',
             'repeat_rollout_quality.py', 'score_rollout_quality.py', 'learn_local_gain.py']
    dump(args.out/'manifest.json', dict(sources=384, splits=dict(train=192, dev=64, test=128),
        seeds=[SEED], offsets=OFFSETS, repeats=8, dataset_sha256=sha(dataset),
        code_hashes={f: sha(ROOT/f) for f in files},
        artifact_hashes={f: sha(args.out/f) for f in ['prefixes.json', 'projection.npy']}))
    print('Frozen 192 train / 64 dev / 128 test sources; up to 6144 repeated pairs.', flush=True)


def verify(args):
    m = read(args.out/'manifest.json')
    for f, h in m['code_hashes'].items(): assert sha(ROOT/f) == h, f
    for f, h in m['artifact_hashes'].items(): assert sha(args.out/f) == h, f


def setup(args):
    verify(args)
    verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    return load_model(args.checkpoints/'pflm_lm1b_k4.ckpt', 'pflm', 'cuda'), torch.from_numpy(np.load(args.out/'projection.npy')).cuda()


@torch.inference_mode()
def preflight(args):
    model, projection = setup(args)
    old = [r for r in rows(args.out.parent/'stability-012/diagnostic_pairs.jsonl') if r['replicate'] == 0][:16]
    for r in old:
        tape = mixed_noise(r['noise_seed'], r['future_seed'], r['offset'])
        assert noise_hash(tape) == r['noise_sha256']
        result = pair(model, torch.tensor([r['history_ids'][:6]], device='cuda'), tape, r['offset'], projection)
        assert all(result[k] == r[k] for k in STATE_KEYS)
        for b in ['keep4', 'split22']:
            assert all(result[b][k] == r[b][k] for k in result[b])
    dump(args.out/'preflight.json', dict(passed=True, old_states=len(old), exact_branches=True, exact_features=True,
        collector_sha256=sha(Path(__file__)), pair_implementation_sha256=sha(ROOT/'repeat_rollout_quality.py')))
    print('16 prior states reproduce exactly, both branches and all features.', flush=True)


@torch.inference_mode()
def collect(args):
    from transformers import BertTokenizerFast
    assert read(args.out/'preflight.json')['passed']
    if args.split == 'train':
        assert not (args.out/'locked_model.json').exists()
    if args.split == 'dev':
        assert (args.out/'train_labels.jsonl').exists()
    if args.split == 'test':
        assert (args.out/'locked_model.json').exists()
        lock = read(args.out/'locked_model.json')
        assert lock['manifest_sha256'] == sha(args.out/'manifest.json')
        for split in ['train', 'dev']:
            assert lock[split+'_labels_sha256'] == sha(args.out/(split+'_labels.jsonl'))
    model, projection = setup(args)
    tokenizer = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    path = args.out/(args.split+'_pairs.jsonl')
    state_path = args.out/(args.split+'_states.jsonl')
    assert not path.exists() and not state_path.exists()
    prefixes = [p for p in read(args.out/'prefixes.json') if p['split'] == args.split]
    counts = {}
    total = 0
    with path.open('w', encoding='utf-8') as dest, state_path.open('w', encoding='utf-8') as state_file:
        for pi, p in enumerate(prefixes):
            prefix = torch.tensor([p['ids']], device='cuda')
            state_seed = 131000000+SEED+p['index']*100003
            for offset in OFFSETS:
                state_id = f"{p['prompt_id']}-{SEED}-{offset}"
                anchor = None
                for rep in range(8):
                    future_seed = 132000000+p['index']*100003+offset*17+rep*100000007
                    tape = mixed_noise(state_seed, future_seed, offset)
                    r = pair(model, prefix, tape, offset, projection)
                    if rep == 0:
                        anchor = {k: r[k] for k in STATE_KEYS if k in r}
                        counts[r['status']] = counts.get(r['status'], 0)+1
                        append(state_file, dict(anchor, state_id=state_id, prompt_id=p['prompt_id'], seed=SEED,
                            state_noise_seed=state_seed, offset=offset, split=args.split))
                        if r['status'] != 'paired':
                            break
                    assert all(r[k] == v for k, v in anchor.items())
                    r.update(pair_id=f'{state_id}-r{rep}', state_id=state_id, prompt_id=p['prompt_id'], seed=SEED,
                        noise_seed=state_seed, offset=offset, replicate=rep, future_seed=future_seed,
                        noise_sha256=noise_hash(tape))
                    for b in ['keep4', 'split22']:
                        r[b]['completion'] = tokenizer.decode(r[b]['ids'][6:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    append(dest, r)
                    total += 1
            if (pi+1) % 16 == 0:
                print(args.split, 'sources', pi+1, '/', len(prefixes), 'repeated pairs', total, flush=True)
    dump(args.out/(args.split+'_collection.json'), dict(passed=True, attempted_states=2*len(prefixes),
        state_status_counts=counts, pairs=total, pairs_sha256=sha(path), states_sha256=sha(state_path),
        collector_sha256=sha(Path(__file__)), manifest_sha256=sha(args.out/'manifest.json'),
        locked_model_sha256=sha(args.out/'locked_model.json') if args.split == 'test' else None,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'preflight', 'collect', 'score'])
    ap.add_argument('--split', choices=['train', 'dev', 'test'])
    ap.add_argument('--out', type=Path, default=ROOT/'results/averaged-013')
    ap.add_argument('--checkpoints', type=Path, default=Path('work/checkpoints'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == 'score':
        from score_rollout_quality import score
        verify(args)
        score(args)
    else:
        globals()[args.action](args)
