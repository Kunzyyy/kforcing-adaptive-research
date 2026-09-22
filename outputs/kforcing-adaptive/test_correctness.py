"""Semantic tests: RoPE math, causal prefixes, changing-width KV cache, and EOS."""
import pytest
import torch
from omegaconf import OmegaConf
from core import MTP, generate, pick_k
from models.transformer import Rotary, apply_rotary_pos_emb
from models.transformer import bias_dropout_add_scale_fused_inference


def test_inference_residual_helper_matches_arithmetic():
    x=torch.tensor([[1.,-2.,3.]])
    bias=torch.tensor([.5,1.,-1.])
    scale=torch.tensor([.25])
    residual=torch.tensor([[4.,5.,6.]])
    # Inference disables dropout even with a nonzero probability.
    actual=bias_dropout_add_scale_fused_inference(x,bias,scale,residual,.9)
    torch.testing.assert_close(actual,residual+scale*(x+bias),atol=0,rtol=0)
    torch.testing.assert_close(bias_dropout_add_scale_fused_inference(x,None,scale,None,.9),scale*x)


@pytest.fixture
def tiny():
    torch.manual_seed(741)
    config = OmegaConf.create({'model': dict(hidden_size=32, n_heads=4, n_blocks=2,
        cond_dim=32, dropout=0., causal=True, scale_by_sigma=False)})
    model = MTP(config, vocab_size=37, mask_index=36, max_k=4).eval()
    # Upstream zero-initializes the output head. Randomize to avoid trivial tests.
    torch.nn.init.normal_(model.output_layer.linear.weight, std=.15)
    torch.nn.init.normal_(model.output_layer.linear.bias, std=.1)
    return model


def test_rotary_against_complex_rotation():
    qkv = torch.randn(2, 7, 3, 4, 8)
    rotary = Rotary(8)
    cos, sin = rotary.forward_with_offset(qkv, offset=9)
    actual = apply_rotary_pos_emb(qkv.clone(), cos, sin)
    a, b = qkv[:, :, :2, :, :4], qkv[:, :, :2, :, 4:]
    z = torch.complex(a, b)
    phase = torch.polar(torch.ones(7, 4), torch.arange(9,16)[:,None]*rotary.inv_freq)
    rotated = z*phase[None,:,None,None,:]
    expected = torch.cat((rotated.real, rotated.imag), dim=-1)
    torch.testing.assert_close(actual[:,:,:2], expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(actual[:,:,2], qkv[:,:,2])


@torch.inference_mode()
def test_short_window_matches_full_prefix(tiny):
    context = torch.tensor([[2, 7, 13, 5]])
    noise = torch.rand(1, 4, 1)
    full = tiny(context, noise, torch.ones(1,1,1), mode='inference')
    for k in (1,2,3):
        short = tiny(context, noise[:,:k], torch.ones(1,1,1), mode='inference')
        torch.testing.assert_close(short, full[:,:k], rtol=2e-5, atol=2e-6)


@torch.inference_mode()
def test_changing_width_cache_matches_recomputation(tiny):
    context = torch.tensor([[2, 7, 13, 5]])
    cache = None
    for width in (4,2,1,3,2):
        noise = torch.rand(1,width,1)
        full = tiny(context, noise, torch.ones(1,1,1), mode='inference')
        cached, cache = tiny(context, noise, torch.ones(1,1,1), mode='inference',
                              kv_caches=cache, return_kv=True)
        torch.testing.assert_close(cached, full, rtol=2e-5, atol=2e-6)
        assert cache[0][0].shape[1] == context.shape[1]
        context = torch.cat((context, cached.argmax(-1)), dim=1)


def test_generation_cache_and_window_equivalence(tiny):
    prefix = torch.tensor([[2,7,13,5]])
    for policy in ('fixed2', 'fixed3', 'fixed4', 'adaptive', 'random'):
        opts = dict(max_new=13, policy=policy, threshold=.5, seed=42,
                    eos_id=None, precision='fp32')
        actual = generate(tiny, prefix, **opts)
        uncached = generate(tiny, prefix, use_cache=False, **opts)
        full_width = generate(tiny, prefix, force_full_width=True, **opts)
        assert actual['ids'] == uncached['ids'] == full_width['ids']
        assert actual['new_tokens'] == 13


def test_eos_stops_and_does_not_return_padding(tiny):
    with torch.no_grad():
        tiny.output_layer.linear.weight.zero_()
        tiny.output_layer.linear.bias.zero_()
        tiny.output_layer.linear.bias[9] = 100
    result = generate(tiny, torch.tensor([[1,2]]), max_new=8, eos_id=9, precision='fp32')
    assert result['ids'] == [1,2,9]
    assert result['new_tokens'] == 1 and result['stopped_eos']


def test_policy_uses_previous_signal():
    assert pick_k('adaptive', None, 1., None, .5) == 4
    assert pick_k('adaptive', .1, 1., None, .5) == 2
    assert pick_k('adaptive', 2., 1., None, .5) == 4
