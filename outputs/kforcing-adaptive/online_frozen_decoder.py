"""Frozen stage-7 predictor; four-candidate generation and optional tail recompute."""
import time
import numpy as np
import torch
from core import sync
from learn_local_gain import feature_tensor


def controller_on_device(selector, projection, device):
    return dict(projection=torch.as_tensor(projection, dtype=torch.float32, device=device),
                mean=torch.tensor(selector['mean'], dtype=torch.float64, device=device),
                std=torch.tensor(selector['std'], dtype=torch.float64, device=device),
                coef=torch.tensor(selector['coef'], dtype=torch.float64, device=device),
                intercept=selector['intercept'], threshold=selector['dev_threshold'])


@torch.inference_mode()
def generate(model, prefix, policy, controller=None, seed=0, max_new=122,
             schedule=None, use_cache=True, supplied_noise=None, debug_features=False):
    assert prefix.shape[0] == 1 and max_new > 0
    assert policy in ('fixed2', 'fixed3', 'fixed4', 'learned', 'random_matched', 'split_all')
    if policy == 'learned':
        assert controller is not None
    if policy == 'random_matched':
        assert schedule is not None and len(schedule) == max_new // 4
    device = prefix.device
    tape = (torch.rand((1, max_new, 1), generator=torch.Generator().manual_seed(seed))
            if supplied_noise is None else supplied_noise).to(device)
    assert tape.shape == (1, max_new, 1)
    tau = torch.ones(1, 1, 1, device=device)
    tokens = torch.full((1, prefix.shape[1] + max_new), model.mask_index,
                        dtype=torch.long, device=device)
    tokens[:, :prefix.shape[1]] = prefix
    cache, produced, steps = None, 0, []
    captured = []
    sync(device)
    start = time.perf_counter()
    handle = None
    if policy == 'learned':
        handle = model.output_layer.linear.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach()))
    try:
        while produced < max_new:
            position = prefix.shape[1] + produced
            width = min(int(policy[-1]) if policy.startswith('fixed') else 4,
                        max_new - produced)
            noise = tape[:, produced:produced + width]
            captured.clear()
            logits, next_cache = model(tokens[:, :position], noise, tau, mode='inference',
                                      kv_caches=cache if use_cache else None, return_kv=True)
            assert all(c[0].shape[1] == position for c in next_cache)
            selected = logits.argmax(-1)
            score, features = None, None
            split = False
            if width == 4:
                if policy == 'learned':
                    assert len(captured) == 1 and captured[0].shape == (1, 4, 768)
                    features = feature_tensor(logits, captured[0], noise, controller['projection'])
                    value = ((features.double() - controller['mean']) / controller['std']) @ controller['coef']
                    score = float(value[0] + controller['intercept'])
                    split = score > controller['threshold']
                elif policy == 'random_matched':
                    split = bool(schedule[produced // 4])
                elif policy == 'split_all':
                    split = True
            tokens[:, position:position + width] = selected
            cache = next_cache if use_cache else None
            if split:
                captured.clear()
                tail, tail_cache = model(tokens[:, :position + 2], noise[:, 2:], tau,
                    mode='inference', kv_caches=cache if use_cache else None, return_kv=True)
                assert all(c[0].shape[1] == position + 2 for c in tail_cache)
                tokens[:, position + 2:position + 4] = tail.argmax(-1)
                cache = tail_cache if use_cache else None
            step = dict(position=position, width=width, split=split, score=score,
                        forward_calls=1 + int(split), computed_candidates=width + 2 * int(split))
            if debug_features and features is not None:
                step['features'] = features[0].cpu().tolist()
            steps.append(step)
            produced += width
    finally:
        if handle is not None:
            handle.remove()
    sync(device)
    seconds = time.perf_counter() - start
    ids = tokens[0].cpu().tolist()
    continuation = ids[prefix.shape[1]:]
    visible = continuation[:continuation.index(102) + 1] if 102 in continuation else continuation
    return dict(ids=ids, visible_ids=ids[:prefix.shape[1]] + visible,
                prefix_length=prefix.shape[1], visible_new_tokens=len(visible),
                stopped_eos=102 in continuation, generated_positions=max_new,
                seconds=seconds, steps=steps, split_count=sum(s['split'] for s in steps),
                forward_calls=sum(s['forward_calls'] for s in steps),
                computed_candidates=sum(s['computed_candidates'] for s in steps))


def matched_schedule(count, windows, noise_seed):
    result = np.zeros(windows, dtype=bool)
    result[np.random.default_rng(9092003 + noise_seed).permutation(windows)[:count]] = True
    return result.tolist()
