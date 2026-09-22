"""Weight-only full-context PFLM reference: no upstream functions, cache, or SDPA."""
import math
import torch
import torch.nn.functional as F


def normalize(x, weight, bias=None):
    centered = x-x.mean(-1, keepdim=True)
    out = centered*torch.rsqrt(centered.square().mean(-1, keepdim=True)+1e-5)
    out = out*weight
    return out if bias is None else out+bias


def linear(x, state, name):
    out = x@state[name+'.weight'].T
    return out+state[name+'.bias'] if name+'.bias' in state else out


def encode_scalar(value, width):
    # Published expansion: float32 frequency grid, double-precision phase/trig.
    freq = torch.pow(10000., -torch.arange(width//2, device=value.device).div(width//2))
    phase = value.double()*freq*2.*math.pi
    return torch.cat([phase.cos(), phase.sin()], -1).float()


@torch.inference_mode()
def reference_forward(state, context, noise, tau):
    embedding = state['vocab_embed.embedding']
    width = embedding.shape[1]
    a, b = encode_scalar(noise, width), encode_scalar(tau, width)
    if b.shape[1] == 1: b = b.expand(-1, noise.shape[1], -1)
    hints = linear(torch.cat([a, b], -1), state, 'noise_encoder.combined_proj')
    norm = normalize(hints, state['noise_encoder.norm.weight'], state['noise_encoder.norm.bias'])
    hints = hints+linear(F.gelu(linear(norm, state, 'noise_encoder.mlp.0'), approximate='none'), state, 'noise_encoder.mlp.2')
    x = torch.cat([embedding[context], hints], 1)
    batch, length, _ = x.shape
    inv = state['rotary_emb.inv_freq']
    dim = len(inv)*2; heads = width//dim
    phase = torch.arange(length, device=x.device).float()[:, None]*inv[None, :]
    cosine, sine = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]
    blocked = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
    layers = 1+max(int(key.split('.')[1]) for key in state if key.startswith('blocks.'))
    for i in range(layers):
        prefix = f'blocks.{i}'
        norm = normalize(x, state[prefix+'.norm1.weight'])
        qkv = linear(norm, state, prefix+'.attn_qkv').reshape(batch, length, 3, heads, dim)
        q, k, v = qkv.unbind(2)
        rotated = []
        for t in [q, k]:
            left, right = t.chunk(2, -1)
            rotated.append(torch.cat([left*cosine-right*sine, right*cosine+left*sine], -1).transpose(1, 2))
        q, k = rotated
        attention = (q@k.transpose(-1, -2))/math.sqrt(dim)
        attention = attention.masked_fill(blocked, float('-inf')).softmax(-1)
        output = (attention@v.transpose(1, 2)).transpose(1, 2).reshape(batch, length, width)
        x = x+linear(output, state, prefix+'.attn_out')
        norm = normalize(x, state[prefix+'.norm2.weight'])
        x = x+linear(F.gelu(linear(norm, state, prefix+'.mlp.0'), approximate='tanh'), state, prefix+'.mlp.2')
    norm = normalize(x[:, -noise.shape[1]:], state['output_layer.norm_final.weight'])
    logits = linear(norm, state, 'output_layer.linear')
    logits[:, :, 103] = -1000.
    return logits
