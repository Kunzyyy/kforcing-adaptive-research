import math
import pytest
import torch
from current_features import current_features_tensor,current_score_tensor,prepare_controller


def test_current_features_have_probabilistic_units():
    probabilities=torch.tensor([.7,.2,.1])
    logits=probabilities.log()[None,None,:].repeat(1,4,1)
    f=current_features_tensor(logits)
    torch.testing.assert_close(f[:4],torch.full((4,),math.log(3.5)))
    torch.testing.assert_close(f[4:8],torch.full((4,),-(probabilities*probabilities.log()).sum()))
    torch.testing.assert_close(f[8:12],torch.full((4,),.7))
    torch.testing.assert_close(f[12:],torch.zeros(2),atol=0,rtol=0)


def test_current_features_ignore_logit_offset_per_position():
    logits=torch.randn(1,4,17)
    offsets=torch.tensor([7.,-3.,1.,-.5])[None,:,None]
    torch.testing.assert_close(current_features_tensor(logits),current_features_tensor(logits+offsets),atol=2e-6,rtol=2e-6)


def test_simple_score_uses_only_current_tail():
    logits=torch.tensor([[[2.,0.],[3.,1.],[5.,1.],[4.,2.]]])
    controller=prepare_controller(dict(kind='negative_tail_margin'),'cpu')
    assert current_score_tensor(logits,controller).item()==-3.
    logits[:,:2]+=19.
    assert current_score_tensor(logits,controller).item()==-3.
