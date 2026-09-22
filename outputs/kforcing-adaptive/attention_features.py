"""Six pre-recomputation last-block attention statistics; no future branch input."""
import math
import torch

NAMES=['head_noise_mass_mean_tail0','head_noise_mass_mean_tail1',
       'head_noise_mass_max_tail0','head_noise_mass_max_tail1',
       'normalized_attention_entropy_tail0','normalized_attention_entropy_tail1']

def from_attention(attention, history_length):
    # attention: heads x 2 tail queries x all (history+4) keys.
    assert attention.ndim==3 and attention.shape[1:]==(2,history_length+4)
    mass=attention[:,:,history_length:history_length+2].sum(-1)
    entropy=-(attention*attention.clamp_min(torch.finfo(attention.dtype).tiny).log()).sum(-1)
    divisor=torch.tensor([math.log(history_length+3),math.log(history_length+4)],device=attention.device,dtype=attention.dtype)
    return torch.cat([mass.mean(0),mass.max(0).values,(entropy/divisor).mean(0)])

def from_qkv(qkv, inv_freq, heads, history_length):
    # Observed, unrotated output of last block attn_qkv; full current context, no cache.
    batch,length,width=qkv.shape
    assert batch==1 and length==history_length+4 and width%(3*heads)==0
    dim=width//(3*heads)
    packed=qkv.reshape(batch,length,3,heads,dim)
    angles=torch.arange(length,device=qkv.device,dtype=inv_freq.dtype)[:,None]*inv_freq[None,:]
    angles=torch.cat([angles,angles],dim=-1)
    qk=packed[:,:,:2].float()
    rotation=torch.cat([-qk[...,dim//2:],qk[...,:dim//2]],dim=-1)
    rotated=qk*angles.cos()[None,:,None,None,:]+rotation*angles.sin()[None,:,None,None,:]
    q=rotated[0,-2:,0].permute(1,0,2);k=rotated[0,:,1].permute(1,0,2)
    value=packed[0,:,2].permute(1,0,2)
    scores=(q@k.transpose(-1,-2))/math.sqrt(dim)
    allowed=torch.arange(length,device=qkv.device)[None,:]<=torch.tensor([length-2,length-1],device=qkv.device)[:,None]
    attention=scores.masked_fill(~allowed[None],float('-inf')).softmax(-1)
    output=(attention@value).permute(1,0,2).reshape(2,heads*dim)
    return from_attention(attention,history_length),attention,output
