"""Fresh full-generation evaluation; freeze before prepare, never fit."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from collect_benefit import dump, verify_checkpoints
from confirm_local_gain import prior_sources, read, rows, sha
from core import load_model, teacher_metrics
from online_frozen_decoder import generate, controller_on_device, matched_schedule

METHODS = ['fixed2', 'fixed3', 'fixed4', 'learned', 'random_matched']
SEEDS = [907, 911]
ROOT = Path(__file__).resolve().parent


def append(file, row):
    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    file.flush()


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    assert not (args.out / 'prefixes.json').exists()
    old, seen = prior_sources(args.out.parent)
    for p in read(args.out.parent / 'confirm-008/sources.json'):
        old.append(p)
        seen.add(tuple(p['reference_ids'][:6]))
    old_rows = {p['source_row'] for p in old}
    old_hash = {p['source_sha256'] for p in old}
    path = Path('work/lm1b-data/test.parquet')
    assert sha(path) == 'd3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts = pq.read_table(path, columns=['text'])['text'].to_pylist()
    indices = list(range(len(texts)))
    random.Random(9092001).shuffle(indices)
    bert = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    chosen = []
    for index in indices:
        digest = hashlib.sha256(texts[index].encode()).hexdigest()
        if index in old_rows or digest in old_hash:
            continue
        ids = bert.encode(texts[index], add_special_tokens=True, truncation=True, max_length=128)
        if len(ids) < 12 or tuple(ids[:6]) in seen:
            continue
        chosen.append(dict(prompt_id=f's9-{len(chosen):03}', index=len(chosen), ids=ids[:6],
                           reference_ids=ids, source_row=index, source_sha256=digest))
        seen.add(tuple(ids[:6])); old_hash.add(digest)
        if len(chosen) == 64:
            break
    assert len(chosen) == 64
    dump(args.out / 'prefixes.json', chosen)
    for src, dst in [('locked_model.json', 'frozen_model.json'), ('projection.npy', 'projection.npy')]:
        shutil.copyfile(args.out.parent / 'learned-007' / src, args.out / dst)
    dump(args.out / 'manifest.json', dict(prefixes=64, seeds=SEEDS, methods=METHODS,
        samples=640, timing_repetitions=3, generated_positions=122,
        prefixes_sha256=sha(args.out / 'prefixes.json'), dataset_sha256=sha(path),
        model_sha256=sha(args.out / 'frozen_model.json'), projection_sha256=sha(args.out / 'projection.npy'),
        feature_code_sha256=sha(ROOT / 'learn_local_gain.py'), protocol_sha256=sha(ROOT / 'STAGE9_PROTOCOL.md')))
    print('Frozen 64 new sources, 128 trajectories per method, 640 outputs.', flush=True)


def setup(args):
    manifest = read(args.out / 'manifest.json')
    for file, key in [('prefixes.json', 'prefixes_sha256'), ('frozen_model.json', 'model_sha256'),
                      ('projection.npy', 'projection_sha256')]:
        assert sha(args.out / file) == manifest[key]
    assert manifest['model_sha256'] == 'caeeec7c2327e2473dbbcac6d64fa596e7db70d74a8d780035cb06a8bc991aae'
    assert sha(ROOT / 'learn_local_gain.py') == manifest['feature_code_sha256']
    assert sha(ROOT / 'STAGE9_PROTOCOL.md') == manifest['protocol_sha256']
    verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(args.checkpoints / 'pflm_lm1b_k4.ckpt', 'pflm', 'cuda')
    selector = read(args.out / 'frozen_model.json')
    controller = controller_on_device(selector, np.load(args.out / 'projection.npy'), 'cuda')
    return model, selector, controller


@torch.inference_mode()
def preflight(args):
    from commit_decoder import generate_commit
    from learn_local_gain import predict
    model, selector, controller = setup(args)
    old = rows(args.out.parent / 'confirm-008/windows.jsonl')
    results = []
    # Old confirmation windows cover both branches and all three context lengths.
    for length in (6, 16, 32):
        group = [r for r in old if r['length'] == length][:4]
        for r in group:
            prefix = torch.tensor([r['prefix']], device='cuda')
            noise = torch.tensor(r['noise']).reshape(1, 4, 1)
            fixed = generate(model, prefix, 'fixed4', max_new=4, supplied_noise=noise)
            split = generate(model, prefix, 'split_all', max_new=4, supplied_noise=noise)
            learned = generate(model, prefix, 'learned', controller, max_new=4,
                               supplied_noise=noise, debug_features=True)
            assert fixed['ids'][-4:] == r['fixed4']
            assert split['ids'][-4:] == r['split22']
            f = learned['steps'][0]['features']
            np.testing.assert_allclose(f, r['features'], rtol=2e-3, atol=2e-3)
            cpu = float(predict(selector, [{'features': f}])[0])
            assert abs(cpu - learned['steps'][0]['score']) < 1e-10
            results.append(dict(prompt_id=r['prompt_id'], seed=r['seed'], length=length,
                max_feature_delta=float(np.max(np.abs(np.array(f) - r['features']))),
                online_cpu_score_delta=abs(cpu - learned['steps'][0]['score'])))
    contexts = read(args.out.parent / 'confirm-008/contexts.json')[:3]
    for i, p in enumerate(contexts):
        prefix = torch.tensor([p['ids']], device='cuda')
        for method in ('fixed2', 'fixed3', 'fixed4'):
            ours = generate(model, prefix, method, seed=9100 + i, max_new=32)
            previous = generate_commit(model, prefix, policy=method, seed=9100 + i,
                                       max_new=32, eos_id=None)
            assert ours['ids'] == previous['ids']
        full = generate(model, prefix, 'split_all', seed=9100 + i, max_new=32)
        fixed2 = generate(model, prefix, 'fixed2', seed=9100 + i, max_new=32)
        assert full['ids'] == fixed2['ids']
        cached = generate(model, prefix, 'learned', controller, seed=9100 + i, max_new=32)
        recomputed = generate(model, prefix, 'learned', controller, seed=9100 + i,
                              max_new=32, use_cache=False)
        assert cached['ids'] == recomputed['ids']
        assert [s['split'] for s in cached['steps']] == [s['split'] for s in recomputed['steps']]
    dump(args.out / 'preflight.json', dict(passed=True, old_windows=results,
        fixed_decoder_equivalence_cases=9, split_all_fixed2_cases=3, learned_cache_recompute_cases=3,
        scope='Old data only; checks are finite-case evidence, not proof for all inputs.',
        decoder_sha256=sha(ROOT / 'online_frozen_decoder.py')))
    print('Preflight passed: 12 old windows, 9 baseline cases, 3 split/cache cases.', flush=True)


@torch.inference_mode()
def run(args):
    from transformers import BertTokenizerFast
    assert not (args.out / 'samples.jsonl').exists()
    assert read(args.out / 'preflight.json')['passed']
    assert read(args.out / 'preflight.json')['decoder_sha256'] == sha(ROOT / 'online_frozen_decoder.py')
    model, selector, controller = setup(args)
    bert = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    warm = torch.tensor([read(args.out.parent / 'confirm-008/contexts.json')[0]['ids']], device='cuda')
    for method in METHODS:
        generate(model, warm, method, controller, seed=9123,
                 schedule=matched_schedule(15, 30, 9123) if method == 'random_matched' else None)
    dump(args.out / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(), precision='fp32', tf32=False,
        decoder_sha256=sha(ROOT / 'online_frozen_decoder.py'), runner_sha256=sha(Path(__file__))))
    prefixes = read(args.out / 'prefixes.json')
    rng = random.Random(9092005)
    completed = 0
    with (args.out / 'samples.jsonl').open('w', encoding='utf-8') as dest, \
         (args.out / 'timings.jsonl').open('w', encoding='utf-8') as timing, \
         (args.out / 'budget_discovery.jsonl').open('w', encoding='utf-8') as discovery:
        for p in prefixes:
            prefix = torch.tensor([p['ids']], device='cuda')
            for seed in SEEDS:
                noise_seed = 90900000 + seed + p['index'] * 100003
                pilot = generate(model, prefix, 'learned', controller, seed=noise_seed)
                schedule = matched_schedule(pilot['split_count'], 30, noise_seed)
                append(discovery, dict(prompt_id=p['prompt_id'], seed=seed, noise_seed=noise_seed,
                    split_count=pilot['split_count'], schedule=schedule, ids=pilot['ids'],
                    discovery_seconds=pilot['seconds']))
                first = {}
                for repetition in range(3):
                    order = METHODS.copy(); rng.shuffle(order)
                    for order_index, method in enumerate(order):
                        r = generate(model, prefix, method, controller, seed=noise_seed,
                                     schedule=schedule if method == 'random_matched' else None)
                        if method == 'learned':
                            assert r['ids'] == pilot['ids'] and r['split_count'] == pilot['split_count']
                        if method == 'random_matched':
                            assert r['forward_calls'] == pilot['forward_calls']
                            assert r['computed_candidates'] == pilot['computed_candidates']
                        if repetition:
                            for key in ('ids', 'steps', 'split_count', 'forward_calls', 'computed_candidates'):
                                assert r[key] == first[method][key], (p['prompt_id'], seed, method, key)
                        else:
                            first[method] = r
                            r.update(prompt_id=p['prompt_id'], seed=seed, noise_seed=noise_seed, method=method,
                                completion=bert.decode(r['visible_ids'][6:], skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True))
                            append(dest, r)
                        append(timing, dict(prompt_id=p['prompt_id'], seed=seed, method=method,
                            repetition=repetition, order_index=order_index, seconds=r['seconds'],
                            ids_sha256=hashlib.sha256(json.dumps(r['ids']).encode()).hexdigest()))
                completed += 1
                if completed % 4 == 0:
                    print('Completed trajectories', completed, '/128; outputs', completed * 5, flush=True)
    dump(args.out / 'generation_checks.json', dict(passed=True, outputs=640, timing_runs=1920,
        matched_forward_and_candidate_counts=True, repeat_tokens_and_decisions_identical=True,
        frozen_model_sha256=sha(args.out / 'frozen_model.json')))


@torch.inference_mode()
def score(args):
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    from external_gpt2_score import masked_token_nll
    assert not (args.out / 'scores.jsonl').exists()
    assert read(args.out / 'generation_checks.json')['passed']
    data = rows(args.out / 'samples.jsonl')
    verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = load_model(args.checkpoints / 'ar_best_lm1b.ckpt', 'ar', 'cuda')
    for i, r in enumerate(data):
        r.update(teacher_metrics(teacher, r['visible_ids'], r['prefix_length']))
        if (i + 1) % 128 == 0:
            print('Teacher scores', i + 1, '/640', flush=True)
    del teacher; gc.collect(); torch.cuda.empty_cache()
    model_path = Path('work/gpt2-large')
    manifest = read(model_path / 'manifest.json')
    for item in manifest['files']:
        h = hashlib.sha256()
        with (model_path / item['name']).open('rb') as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
        assert h.hexdigest() == item['sha256']
    tokenizer = GPT2TokenizerFast.from_pretrained(str(model_path), local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(str(model_path), local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation='sdpa').cuda().eval()
    probe = tokenizer.encode('A frozen predictor requires independent evaluation.', return_tensors='pt').cuda()
    result = model(probe, labels=probe, use_cache=False)
    ss, nn = masked_token_nll(result.logits, probe, torch.ones_like(probe))
    torch.testing.assert_close(ss / nn, result.loss.reshape(1), rtol=1e-6, atol=1e-6)
    dump(args.out / 'scoring_checks.json', dict(builtin_loss_matches=True, evaluator_files_verified=True,
        repo=manifest['repo'], revision=manifest['revision'], precision='fp32',
        scope='Completion only, no BOS, first token unscored; corpus token weighting.'))
    for r in data:
        r['_tokens'] = tokenizer.encode(r['completion'], add_special_tokens=False)
        assert len(r['_tokens']) <= 1024
    with (args.out / 'scores.jsonl').open('w', encoding='utf-8') as out:
        for start in range(0, len(data), 4):
            batch = data[start:start + 4]
            width = max(2, max(len(r['_tokens']) for r in batch))
            x = torch.full((len(batch), width), tokenizer.eos_token_id, dtype=torch.long, device='cuda')
            mask = torch.zeros_like(x)
            for j, r in enumerate(batch):
                ids = r['_tokens']
                if ids:
                    x[j, :len(ids)] = torch.tensor(ids, device='cuda'); mask[j, :len(ids)] = 1
                else:
                    mask[j, 0] = 1
            sums, counts = masked_token_nll(model(x, attention_mask=mask, use_cache=False).logits, x, mask)
            for j, r in enumerate(batch):
                n = len(r.pop('_tokens'))
                record = {k: r[k] for k in ('prompt_id', 'seed', 'method', 'teacher_nll_sum',
                    'teacher_token_count', 'repeated_trigram_fraction', 'visible_new_tokens', 'stopped_eos')}
                record.update(gpt2_input_tokens=n, gpt2_scored_tokens=int(counts[j]),
                    gpt2_nll_sum=float(sums[j]) if int(counts[j]) else 0., excluded_short=n < 2)
                append(out, record)
            if (start + 4) % 128 == 0:
                print('GPT-2 scores', start + 4, '/640', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'preflight', 'run', 'score'])
    ap.add_argument('--out', type=Path, default=ROOT / 'results/online-009')
    ap.add_argument('--checkpoints', type=Path, default=Path('work/checkpoints'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.action](args)


if __name__ == '__main__':
    main()
