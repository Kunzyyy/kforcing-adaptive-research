"""Exploratory evaluation, not a paper-quality benchmark or quality guarantee."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import BertTokenizerFast

from core import load_model, generate, teacher_metrics, summarize, amp_context
from models.transformer import HELPER_BACKEND


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def prefixes(path, tokenizer, count, split):
    texts = pq.read_table(path, columns=['text'])['text'].to_pylist()
    random.Random(20260915 if split=='validation' else 20260916).shuffle(texts)
    result, seen = [], set()
    for text in texts:
        if len(text.split()) < 20 or text.strip().startswith('='):
            continue
        ids = tokenizer.encode(text.strip(), add_special_tokens=False)
        ids = [tokenizer.cls_token_id]+ids[:5]
        if len(ids)!=6 or tuple(ids) in seen:
            continue
        seen.add(tuple(ids))
        result.append(dict(id=f'{split}-{len(result):03}', ids=ids,
            prefix=tokenizer.decode(ids), source_sha256=hashlib.sha256(text.encode()).hexdigest()))
        if len(result)==count:
            return result
    raise RuntimeError('Not enough unique prefixes')


@torch.inference_mode()
def pretrained_checks(model, prefix):
    device = prefix.device
    generator = torch.Generator(device='cpu').manual_seed(923)
    context = prefix.clone()
    cache = None
    max_cache_error, max_width_error = 0., 0.
    for width in (4,2,1,3):
        noise = torch.rand(1,4,1,generator=generator).to(device)
        tau = torch.ones(1,1,1,device=device)
        full = model(context, noise, tau, mode='inference')
        short = model(context, noise[:,:width], tau, mode='inference')
        cached, cache = model(context, noise[:,:width], tau, mode='inference', kv_caches=cache, return_kv=True)
        max_width_error = max(max_width_error, float((full[:,:width]-short).abs().max()))
        max_cache_error = max(max_cache_error, float((short-cached).abs().max()))
        torch.testing.assert_close(full[:,:width], short, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(short, cached, atol=3e-4, rtol=3e-4)
        assert cache[0][0].shape[1]==context.shape[1]
        context = torch.cat((context, short.argmax(-1)),dim=1)
    return dict(precision='fp32', max_cached_vs_recompute_error=max_cache_error,
        max_short_vs_full_prefix_error=max_width_error, passed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dev-count', type=int, default=8)
    ap.add_argument('--test-count', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=[17,29])
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--precision', choices=['fp32','bf16','fp16'], default='bf16')
    ap.add_argument('--checks-only', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'samples.jsonl').exists():
        raise RuntimeError('Use a fresh output directory; previous samples will not be overwritten.')
    torch.set_num_threads(4)
    # Avoid TF32 obscuring the numerical validation. Same setting for all methods.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; do not report CPU timing as GPU timing')
    device = torch.device(args.device)
    expected = json.loads((Path(__file__).parent/'checkpoint_manifest.json').read_text(encoding='utf-8'))
    for entry in expected:
        digest = hashlib.sha256()
        with (args.checkpoints/entry['filename']).open('rb') as source:
            while chunk := source.read(8*1024*1024):
                digest.update(chunk)
        if digest.hexdigest()!=entry['sha256']:
            raise RuntimeError('Checkpoint hash mismatch: '+entry['filename'])
    tokenizer = BertTokenizerFast(vocab_file=str(args.data/'vocab.txt'),do_lower_case=True)
    val = prefixes(args.data/'validation.parquet',tokenizer,args.dev_count,'validation')
    test = prefixes(args.data/'test.parquet',tokenizer,args.test_count,'test')
    # Ensure duplicate tokenized prefixes cannot cross the calibration/eval split.
    val_set = {tuple(p['ids']) for p in val}
    assert not any(tuple(p['ids']) in val_set for p in test), 'Prefix overlap across splits'
    dump(args.out/'prefixes.json',dict(validation=val,test=test))
    env = dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        device=str(device),precision=args.precision,batch_size=1,max_new=args.max_new,helper_backend=HELPER_BACKEND,
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        cuda=torch.version.cuda,git_commit='706caa332a69d509b7fc2fa53ac7819f381a8c78',
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        packages={p:importlib.metadata.version(p) for p in ['torch','transformers','einops','omegaconf','numpy','pyarrow']},
        caveats=['WikiText-2 is out of distribution for LM1B; small exploratory sample.',
                 'Teacher NLL and repetition are proxies, not human-rated quality.',
                 'Single-request portable PyTorch SDPA implementation, not paper H100 kernels.',
                 'Timing includes prefill, EOS detection, policy, cache, and common diagnostics; model/data loading and teacher scoring excluded.',
                 'Threshold chosen on validation margins without optimizing test results.'])
    dump(args.out/'environment.json',env)
    print('Loading verified PFLM',flush=True)
    model = load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm',device)
    print('Numerical preflight',flush=True)
    tensor = lambda p: torch.tensor([p['ids']],device=device)
    checks = pretrained_checks(model,tensor(val[0]))
    dump(args.out/'pretrained_checks.json',checks)
    print(checks,flush=True)
    if args.checks_only:
        return
    common = dict(max_new=args.max_new,precision=args.precision,eos_id=tokenizer.sep_token_id,
                  frequency_penalty=.5)
    # Warm all window shapes before calibration/timing.
    for policy in ('fixed1','fixed2','fixed3','fixed4','adaptive','random'):
        generate(model,tensor(val[0]),max_new=12,policy=policy,precision=args.precision)
    margins = []
    for p in val:
        r = generate(model,tensor(p),policy='fixed4',seed=11,**common)
        margins.extend(s['observed_margin'] for s in r['steps'])
    threshold = float(np.median(margins))
    dev_steps = []
    for p in val:
        r = generate(model,tensor(p),policy='adaptive',threshold=threshold,seed=11,**common)
        dev_steps.extend(r['steps'])
    p_short = float(np.mean([s['requested_k']==2 for s in dev_steps]))
    dump(args.out/'calibration.json',dict(threshold=threshold,rule='median of validation fixed4 raw top1-top2 mean margin',
        samples=len(val),margin_observations=len(margins),random_control_short_probability=p_short,
        calibration_seed=11,uses_test_data=False))
    print(f'Locked threshold={threshold:.5f}, random p_short={p_short:.4f}',flush=True)
    teacher = load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar',device)
    policies = ['fixed2','fixed3','fixed4','adaptive','random']
    ordering = random.Random(318)
    rows = []
    jobs = [(p,seed) for p in test for seed in args.seeds]
    ordering.shuffle(jobs)
    with (args.out/'samples.jsonl').open('w',encoding='utf-8') as dest:
        for index,(p,seed) in enumerate(jobs):
            order = policies.copy()
            ordering.shuffle(order)
            # Generate all policies before teacher evaluation to reduce systematic
            # timing interference from the scorer. Policy order is randomized.
            batch = []
            for policy in order:
                r = generate(model,tensor(p),policy=policy,threshold=threshold,seed=seed,
                             p_short=p_short,**common)
                r.update(prompt_id=p['id'],prefix=p['prefix'])
                batch.append(r)
            for r in batch:
                r.update(teacher_metrics(teacher,r['ids'],r['prefix_length']))
                r['text'] = tokenizer.decode(r['ids'],skip_special_tokens=False)
                dest.write(json.dumps(r,ensure_ascii=False)+'\n')
                dest.flush()
                rows.append(r)
            print(f'Completed {index+1}/{len(jobs)} prompt-seed pairs',flush=True)
    summary = summarize(rows)
    dump(args.out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
