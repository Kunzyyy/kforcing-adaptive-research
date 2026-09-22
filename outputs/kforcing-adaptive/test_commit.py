import torch
from test_correctness import tiny
from core import generate
from commit_decoder import generate_commit


def test_native_windows_and_always_short_commit_match_reference(tiny):
    prefix=torch.tensor([[2,7,13,5]])
    for policy in ('fixed2','fixed3','fixed4','commit2'):
        reference=generate(tiny,prefix,policy='fixed2' if policy=='commit2' else policy,
            max_new=13,seed=91,eos_id=None,precision='fp32',frequency_penalty=0.)
        actual=generate_commit(tiny,prefix,policy=policy,max_new=13,seed=91,eos_id=None)
        uncached=generate_commit(tiny,prefix,policy=policy,max_new=13,seed=91,eos_id=None,use_cache=False)
        assert actual['ids']==reference['ids']==uncached['ids']
        assert actual['computed_candidates']==actual['new_tokens']+actual['policy_discarded']+actual['eos_discarded']


def test_current_and_random_commit_cache_matches_full_recomputation(tiny):
    prefix=torch.tensor([[2,7,13,5]])
    for policy in ('current','random_commit'):
        controller=dict(kind='negative_tail_margin',threshold=-.3)
        options=dict(policy=policy,controller=controller,max_new=15,seed=117,eos_id=None)
        a=generate_commit(tiny,prefix,**options)
        b=generate_commit(tiny,prefix,use_cache=False,**options)
        assert a['ids']==b['ids']
        assert a['steps'][0]['requested_k']==4
        assert a['new_tokens']==15


def test_discarded_tail_eos_does_not_stop_generation():
    class CandidateModel:
        mask_index=19
        def __call__(self,context,noise,tau,**kwargs):
            n=context.shape[1]
            width=noise.shape[1]
            ids=[3,4,9,6] if n==2 else [7,8,10,11]
            logits=torch.full((1,width,20),-100.)
            for j in range(width):
                logits[0,j,ids[j]]=100.
            cache=[(torch.zeros(1,n,1,1),torch.zeros(1,n,1,1))]
            return logits,cache
    result=generate_commit(CandidateModel(),torch.tensor([[1,2]]),policy='commit2',max_new=4,eos_id=9)
    assert result['ids']==[1,2,3,4,7,8]
    assert not result['stopped_eos']
    assert result['policy_discarded']==2


def test_committed_eos_stops_and_counts_unused_work(tiny):
    with torch.no_grad():
        tiny.output_layer.linear.weight.zero_()
        tiny.output_layer.linear.bias.zero_()
        tiny.output_layer.linear.bias[9]=100.
    r=generate_commit(tiny,torch.tensor([[1,2]]),policy='current',
        controller=dict(kind='negative_tail_margin',threshold=0.),max_new=8,eos_id=9)
    assert r['ids']==[1,2,9] and r['eos_discarded']==3 and r['computed_candidates']==4
