"""Actual 4-candidate computation followed by optional 2-token commitment."""
import time
import numpy as np
import torch
from core import sync
from current_features import current_score_tensor,prepare_controller


@torch.inference_mode()
def generate_commit(model,prefix,policy='current',controller=None,p_short=.5,seed=0,max_new=64,
                    eos_id=102,use_cache=True):
    if prefix.shape[0]!=1 or max_new<1:
        raise ValueError('Expected one prefix and positive max_new')
    if policy not in ('fixed2','fixed3','fixed4','current','random_commit','commit2'):
        raise ValueError(policy)
    if policy=='current' and controller is None:
        raise ValueError('Current policy requires a frozen controller')
    device=prefix.device
    controller=prepare_controller(controller,device) if controller is not None else None
    tape=torch.rand(1,max_new+4,1,generator=torch.Generator().manual_seed(seed)).to(device)
    rng=np.random.default_rng(seed+51001)
    tau=torch.ones(1,1,1,device=device)
    tokens=torch.full((1,prefix.shape[1]+max_new),model.mask_index,dtype=torch.long,device=device)
    tokens[:,:prefix.shape[1]]=prefix
    position,produced=prefix.shape[1],0
    cache=None
    steps=[]
    stopped=False
    sync(device)
    start=time.perf_counter()
    while produced<max_new:
        width=min(int(policy[-1]) if policy.startswith('fixed') else 4,max_new-produced)
        logits,cache_new=model(tokens[:,:position],tape[:,produced:produced+width],tau,
            mode='inference',kv_caches=cache if use_cache else None,return_kv=True)
        cache=cache_new if use_cache else None
        if use_cache and cache[0][0].shape[1]!=position:
            raise AssertionError('Uncommitted candidates leaked into the text cache')
        selected=logits.argmax(-1)[0]
        eligible=policy in ('current','random_commit') and produced>0 and width==4
        score=None
        if policy=='current' and eligible:
            packed=torch.cat((selected.float(),current_score_tensor(logits,controller))).cpu().tolist()
            ids=[int(t) for t in packed[:width]]
            score=packed[-1]
            requested=2 if score>controller['threshold'] else 4
        else:
            ids=selected.cpu().tolist()
            if policy=='random_commit' and eligible:
                requested=2 if rng.random()<p_short else 4
            elif policy=='commit2':
                requested=2
            else:
                requested=width
        chosen=min(requested,width)
        emitted=chosen
        if eos_id is not None and eos_id in ids[:chosen]:
            emitted=ids[:chosen].index(eos_id)+1
            stopped=True
        tokens[:,position:position+emitted]=selected[:emitted]
        steps.append(dict(position=position,computed_k=width,requested_k=requested,emitted=emitted,
            eligible=eligible,score=score,policy_discarded=width-chosen,eos_discarded=chosen-emitted))
        position+=emitted
        produced+=emitted
        if stopped:
            break
    sync(device)
    seconds=time.perf_counter()-start
    return dict(ids=tokens[0,:position].cpu().tolist(),prefix_length=prefix.shape[1],new_tokens=produced,
        seconds=seconds,tokens_per_second=produced/seconds,policy=policy,seed=seed,steps=steps,
        stopped_eos=stopped,computed_candidates=sum(s['computed_k'] for s in steps),
        policy_discarded=sum(s['policy_discarded'] for s in steps),eos_discarded=sum(s['eos_discarded'] for s in steps))
