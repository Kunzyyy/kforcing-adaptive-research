"""EOS-aware frozen selector; no model forward after the accepted EOS block."""
import time
import numpy as np
import torch
from core import sync
from learn_local_gain import feature_tensor
from online_frozen_decoder import controller_on_device


@torch.inference_mode()
def generate(model, prefix, policy, controller=None, seed=0, max_new=122,
             probability=None, schedule=None, use_cache=True, supplied_noise=None):
    assert prefix.shape[0] == 1 and max_new > 0
    assert policy in ('fixed2','fixed3','fixed4','learned','random','replay')
    if policy == 'learned':
        assert controller is not None
    if policy == 'random':
        assert 0 <= probability <= 1
    if policy == 'replay':
        assert schedule is not None
    device = prefix.device
    tape = (torch.rand((1,max_new,1),generator=torch.Generator().manual_seed(seed))
            if supplied_noise is None else supplied_noise).to(device)
    choices = np.random.default_rng(seed + 10102003).random((max_new+3)//4)
    tau = torch.ones(1,1,1,device=device)
    tokens = torch.full((1,prefix.shape[1]+max_new),model.mask_index,dtype=torch.long,device=device)
    tokens[:,:prefix.shape[1]] = prefix
    cache, produced, steps, captured = None, 0, [], []
    stopped = False
    handle = None
    sync(device)
    start = time.perf_counter()
    if policy == 'learned':
        handle = model.output_layer.linear.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach()))
    try:
        while produced < max_new:
            position = prefix.shape[1] + produced
            width = min(int(policy[-1]) if policy.startswith('fixed') else 4,max_new-produced)
            noise = tape[:,produced:produced+width]
            captured.clear()
            logits, next_cache = model(tokens[:,:position],noise,tau,mode='inference',
                kv_caches=cache if use_cache else None,return_kv=True)
            assert all(c[0].shape[1] == position for c in next_cache)
            selected = logits.argmax(-1)
            initial = selected[0].cpu().tolist()
            cache = next_cache if use_cache else None
            eligible = width == 4 and 102 not in initial[:2] and not policy.startswith('fixed')
            score, split, uniform = None, False, None
            if eligible:
                if policy == 'learned':
                    assert len(captured) == 1
                    feat = feature_tensor(logits,captured[0],noise,controller['projection'])
                    value = ((feat.double()-controller['mean'])/controller['std']) @ controller['coef']
                    score = float(value[0]+controller['intercept'])
                    split = score > controller['threshold']
                elif policy == 'random':
                    uniform = float(choices[produced//4])
                    split = uniform < probability
                else:
                    split = bool(schedule[produced//4])
            final = initial.copy()
            tokens[:,position:position+width] = selected
            if split:
                captured.clear()
                logits2, cache2 = model(tokens[:,:position+2],noise[:,2:],tau,mode='inference',
                    kv_caches=cache if use_cache else None,return_kv=True)
                assert all(c[0].shape[1] == position+2 for c in cache2)
                tail = logits2.argmax(-1)
                final[2:] = tail[0].cpu().tolist()
                tokens[:,position+2:position+4] = tail
                cache = cache2 if use_cache else None
            emitted = final.index(102)+1 if 102 in final else width
            stopped = 102 in final
            steps.append(dict(position=position,width=width,eligible=eligible,score=score,
                split=split,uniform=uniform,initial_candidates=initial,final_candidates=final,
                emitted=emitted,forward_calls=1+int(split),computed_candidates=width+2*int(split)))
            produced += emitted
            if stopped:
                break
    finally:
        if handle is not None:
            handle.remove()
    sync(device)
    elapsed = time.perf_counter()-start
    ids = tokens[0,:prefix.shape[1]+produced].cpu().tolist()
    return dict(ids=ids,visible_ids=ids,prefix_length=prefix.shape[1],visible_new_tokens=produced,
        stopped_eos=stopped,seconds=elapsed,steps=steps,split_count=sum(s['split'] for s in steps),
        eligible_windows=sum(s['eligible'] for s in steps),
        forward_calls=sum(s['forward_calls'] for s in steps),
        computed_candidates=sum(s['computed_candidates'] for s in steps))
