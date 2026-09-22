import math
import torch
from external_gpt2_score import masked_token_nll


def test_shifted_scoring_counts_only_valid_targets():
    logits=torch.zeros(2,4,3)
    ids=torch.tensor([[0,1,2,0],[0,2,0,0]])
    mask=torch.tensor([[1,1,1,1],[1,1,0,0]])
    sums,counts=masked_token_nll(logits,ids,mask)
    assert counts.tolist()==[3,1]
    torch.testing.assert_close(sums,torch.tensor([3*math.log(3),math.log(3)]))


def test_first_token_and_padding_do_not_contribute_to_nll():
    logits=torch.tensor([[[2.,0.],[0.,2.],[40.,-40.]]])
    ids=torch.tensor([[1,0,1]])
    sums,counts=masked_token_nll(logits,ids,torch.tensor([[1,1,0]]))
    assert counts.item()==1
    assert abs(sums.item()-math.log1p(math.exp(-2)))<1e-6
