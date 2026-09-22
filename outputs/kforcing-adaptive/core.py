"""Single-request adaptive decoding. All policy inputs precede the next forward pass."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream'))
from models.pflm import MTP
from models.autoregressive import AR
from omegaconf import OmegaConf


def load_model(path, kind, device, mask_id=103):
    # Official, SHA-256-verified checkpoints only. Never enable arbitrary pickle.
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    state = {k.removeprefix('backbone.'): v for k, v in state.items()}
    vocab, hidden = state['vocab_embed.embedding'].shape
    heads = hidden // (2 * state['rotary_emb.inv_freq'].numel())
    blocks = 1 + max(int(re.match(r'blocks\.(\d+)\.', k).group(1))
                     for k in state if k.startswith('blocks.'))
    config = OmegaConf.create({'model': dict(hidden_size=hidden, vocab_size=vocab,
        n_heads=heads, n_blocks=blocks, cond_dim=1024, dropout=0., causal=True,
        scale_by_sigma=False)})
    if kind == 'pflm':
        model = MTP(config, vocab_size=vocab, mask_index=mask_id, max_k=4)
    else:
        model = AR(config, vocab_size=vocab, mask_index=mask_id)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def amp_context(device, precision='bf16'):
    if torch.device(device).type != 'cuda' or precision == 'fp32':
        return nullcontext()
    return torch.autocast('cuda', dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[precision])


def pick_k(policy, previous_margin, threshold, rng, p_short):
    if policy.startswith('fixed'):
        return int(policy[-1])
    if policy == 'random':
        return 2 if rng.random() < p_short else 4
    if policy == 'adaptive':
        # Neutral warm start; this signal is conditional on noise, not calibrated risk.
        return 2 if previous_margin is not None and previous_margin < threshold else 4
    raise ValueError(policy)


@torch.inference_mode()
def generate(model, prefix, max_new=64, policy='fixed4', threshold=0., seed=0,
             p_short=.5, eos_id=102, precision='bf16', frequency_penalty=.5,
             use_cache=True, force_full_width=False, stop_eos=True):
    if prefix.shape[0] != 1 or max_new < 1:
        raise ValueError('This pilot supports batch=1 and positive max_new only')
    device = prefix.device
    # Position-indexed randomness makes comparisons independent of loop counts.
    # Same next token position receives the same exogenous uniform variate.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    tape = torch.rand((1, max_new + model.max_k, 1), generator=generator).to(device)
    rng = np.random.default_rng(seed + 17003)
    tau = torch.ones((1, 1, 1), device=device)
    counts = torch.zeros((1, model.vocab_size), device=device)
    ones = torch.ones((1, 1), device=device)
    tokens = torch.full((1, prefix.shape[1] + max_new), model.mask_index,
                        dtype=torch.long, device=device)
    tokens[:, :prefix.shape[1]] = prefix
    position, produced = prefix.shape[1], 0
    cache, previous_margin = None, None
    steps = []
    stopped = False
    sync(device)
    start = time.perf_counter()
    with amp_context(device, precision):
        while produced < max_new:
            requested = pick_k(policy, previous_margin, threshold, rng, p_short)
            actual = min(requested, max_new-produced)
            width = model.max_k if force_full_width else actual
            context = tokens[:, :position]
            noise = tape[:, produced:produced+width]
            logits, next_cache = model(context, noise, tau, mode='inference',
                kv_caches=cache if use_cache else None, return_kv=True)
            cache = next_cache if use_cache else None
            # Exactly as upstream, hints never enter long-term KV; generated real
            # tokens enter on the next iteration. Cache currently ends at position.
            if use_cache and cache[0][0].shape[1] != position:
                raise AssertionError('KV cache has the wrong context length')
            selected = []
            for j in range(actual):
                scores = logits[:, j].float() - frequency_penalty * counts
                tok = scores.argmax(-1, keepdim=True)
                selected.append(tok)
                counts.scatter_add_(1, tok, ones)
            selected = torch.cat(selected, dim=1)
            # One small D2H transfer per step. Included in elapsed time for every
            # policy, enabling both EOS handling and comparable diagnostic logging.
            top = logits[:, :actual].float().topk(2, dim=-1).values
            margin = (top[..., 0]-top[..., 1]).mean()
            step_values = torch.cat((selected.flatten().float(), margin.reshape(1))).cpu().tolist()
            ids, next_margin = [int(x) for x in step_values[:-1]], float(step_values[-1])
            kept = len(ids)
            if stop_eos and eos_id is not None and eos_id in ids:
                kept = ids.index(eos_id) + 1
                stopped = True
            tokens[:, position:position+kept] = selected[:, :kept]
            steps.append(dict(position=position, requested_k=requested, computed_k=width,
                emitted=kept, previous_margin=previous_margin, observed_margin=next_margin))
            previous_margin = next_margin
            position += kept
            produced += kept
            if stopped:
                break
    sync(device)
    elapsed = time.perf_counter()-start
    result_ids = tokens[0, :position].cpu().tolist()
    return dict(ids=result_ids, prefix_length=prefix.shape[1], new_tokens=produced,
        seconds=elapsed, tokens_per_second=produced/elapsed, steps=steps,
        stopped_eos=stopped, policy=policy, seed=seed)


@torch.inference_mode()
def teacher_metrics(teacher, ids, prefix_length):
    """Teacher NLL is a diagnostic, NOT a human-quality or diversity guarantee."""
    device = next(teacher.parameters()).device
    full = torch.tensor([ids], device=device)
    # Use full context without cached suffixes to avoid upstream AR's multi-token
    # cached-suffix mask ambiguity. This is offline evaluation, never policy input.
    logits = teacher.forward_high_precision(full[:, :-1])
    begin = prefix_length - 1
    scores = logits[:, begin:].float()
    target = full[:, prefix_length:]
    total = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]),
                                              target.reshape(-1), reduction='sum')
    continuation = ids[prefix_length:]
    grams = [tuple(continuation[i:i+3]) for i in range(max(0, len(continuation)-2))]
    return dict(teacher_nll_sum=float(total), teacher_token_count=target.numel(),
        teacher_nll=float(total)/target.numel(),
        repeated_trigram_fraction=1-len(set(grams))/len(grams) if grams else 0.)


def summarize(rows):
    result = {}
    for policy in sorted({r['policy'] for r in rows}):
        group = [r for r in rows if r['policy']==policy]
        n = sum(r['new_tokens'] for r in group)
        duration = sum(r['seconds'] for r in group)
        total_nll = sum(r['teacher_nll_sum'] for r in group)
        scored = sum(r['teacher_token_count'] for r in group)
        steps = [s for r in group for s in r['steps']]
        result[policy] = dict(samples=len(group), generated_tokens=n, elapsed_seconds=duration,
            aggregate_tokens_per_second=n/duration, teacher_nll=total_nll/scored,
            teacher_perplexity=math.exp(min(50, total_nll/scored)),
            mean_new_tokens=n/len(group), eos_fraction=np.mean([r['stopped_eos'] for r in group]),
            mean_repeated_trigram_fraction=np.mean([r['repeated_trigram_fraction'] for r in group]),
            short_step_fraction=np.mean([s['requested_k']==2 for s in steps]))
    return result
