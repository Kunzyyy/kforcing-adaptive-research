"""Features of an already-computed four-position candidate, not future tokens."""
import torch

CURRENT_FEATURES=([f'margin_{i}' for i in range(4)]+[f'entropy_{i}' for i in range(4)]+
                  [f'top1_{i}' for i in range(4)]+['tail_head_margin','tail_head_entropy'])


def current_features_tensor(logits):
    if logits.shape[0]!=1 or logits.shape[1]!=4:
        raise ValueError('Current-block features require batch 1, width 4')
    x=logits[0].float()
    top=x.topk(2,dim=-1).values
    margins=top[:,0]-top[:,1]
    lp=x.log_softmax(-1)
    p=lp.exp()
    entropy=-(p*lp).sum(-1)
    top1=lp.max(-1).values.exp()
    return torch.cat((margins,entropy,top1,(margins[2:].mean()-margins[:2].mean()).view(1),
                      (entropy[2:].mean()-entropy[:2].mean()).view(1)))


def current_score_tensor(logits,controller):
    if controller['kind']=='negative_tail_margin':
        top=logits[0,2:4].float().topk(2,dim=-1).values
        return -(top[:,0]-top[:,1]).mean().view(1)
    # Buffers are prepared before timing by prepare_controller().
    f=current_features_tensor(logits)
    return ((f-controller['_mean'])/controller['_std']@controller['_coef']+controller['intercept']).view(1)


def prepare_controller(controller,device):
    result=dict(controller)
    if controller['kind']=='ridge':
        for name in ('mean','std','coef'):
            result['_'+name]=torch.tensor(controller[name],dtype=torch.float64,device=device)
    return result
