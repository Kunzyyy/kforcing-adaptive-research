"""Stage 9 scoring semantics reused verbatim, with dynamic progress counts."""
import argparse
import gc
import hashlib
from pathlib import Path
import torch
from collect_benefit import dump,verify_checkpoints
from confirm_local_gain import read,rows
from core import load_model,teacher_metrics
from run_frozen_online import append

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
            print('Teacher scores', i + 1, '/', len(data), flush=True)
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
                print('GPT-2 scores', start + 4, '/', len(data), flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('outputs/kforcing-adaptive/results/early-010'))
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    score(ap.parse_args())
