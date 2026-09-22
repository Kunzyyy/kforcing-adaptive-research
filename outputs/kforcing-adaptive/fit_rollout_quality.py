"""Train-only ridge models selected by dev rollout gain; one frozen test."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def load(root,split):
    checks=read(root/(split+'_scoring.json'))
    assert checks['passed'] and sha(root/(split+'_labels.jsonl'))==checks['labels_sha256']
    assert sha(root/(split+'_pairs.jsonl'))==checks['pairs_sha256']
    pairs={r['pair_id']:r for r in rows(root/(split+'_pairs.jsonl'))}
    return [dict(r,features=pairs[r['pair_id']]['features']) for r in rows(root/(split+'_labels.jsonl')) if r['valid_label']]


def predict(model,data):
    n=len(model['mean']);x=np.array([r['features'][:n] for r in data],dtype=np.float64)
    return ((x-np.array(model['mean']))/np.array(model['std']))@np.array(model['coef'])+model['intercept']


def half(scores,data):
    selection=np.zeros(len(data),dtype=bool)
    order=sorted(range(len(data)),key=lambda i:(-float(scores[i]),data[i]['pair_id']))
    selection[order[:len(data)//2]]=True
    return selection


def interval(values,data,coverage=.95):
    groups=defaultdict(list)
    for v,r in zip(values,data):groups[r['prompt_id']].append(float(v))
    keys=sorted(groups);sums=np.array([sum(groups[k]) for k in keys]);counts=np.array([len(groups[k]) for k in keys])
    idx=np.random.default_rng(11112004).integers(0,len(keys),size=(4000,len(keys)))
    boot=sums[idx].sum(1)/counts[idx].sum(1);tail=(1-coverage)/2
    return dict(estimate=float(np.mean(values)),interval=np.quantile(boot,[tail,1-tail]).tolist(),
                coverage=coverage,source_clusters=len(keys))


def fit(root):
    assert not (root/'locked_model.json').exists()
    assert not (root/'test_pairs.jsonl').exists() and not (root/'test_labels.jsonl').exists()
    train=load(root,'train');dev=load(root,'dev')
    y=np.array([r['gain'] for r in train]);yd=np.array([r['gain'] for r in dev]);candidates=[]
    assert len(train)>100 and len(dev)>64
    for name,n in [('confidence14',14),('hidden82',82)]:
        x=np.array([r['features'][:n] for r in train]);mean=x.mean(0);std=x.std(0);std[std<1e-8]=1.
        z=(x-mean)/std
        for alpha in (1.,10.,100.):
            coef=np.linalg.solve(z.T@z+alpha*np.eye(n),z.T@(y-y.mean()))
            model=dict(name=f'{name}_alpha{int(alpha):03}',features=name,alpha=alpha,mean=mean.tolist(),
                std=std.tolist(),coef=coef.tolist(),intercept=float(y.mean()),
                target='Per-sequence complete-continuation GPT-2 NLL reduction under one split, fixed4 future')
            score=predict(model,dev);selected=half(score,dev);p=float(selected.mean())
            gain=interval((selected.astype(float)-p)*yd,dev)
            model.update(dev_gain=gain,dev_mse=float(np.mean((score-yd)**2)),dev_threshold=float(np.median(score)))
            candidates.append(model)
    chosen=sorted(candidates,key=lambda c:(-c['dev_gain']['estimate'],c['name']))[0].copy()
    chosen.update(dev_gate_passed=chosen['dev_gain']['interval'][0]>0,
        train_labels_sha256=sha(root/'train_labels.jsonl'),dev_labels_sha256=sha(root/'dev_labels.jsonl'),
        prefixes_sha256=sha(root/'prefixes.json'),projection_sha256=sha(root/'projection.npy'),
        selection='Predeclared six ridge candidates; max dev half-budget per-sequence NLL gain; ties by name; no test outcomes present')
    dump(root/'dev_candidates.json',candidates);dump(root/'locked_model.json',chosen)
    print(json.dumps({k:chosen[k] for k in ('name','dev_gain','dev_gate_passed','dev_mse')},indent=2),flush=True)


def evaluate(root):
    assert not (root/'analysis.json').exists()
    model=read(root/'locked_model.json');old=read(root/'old_model.json')
    for split in ('train','dev'):assert sha(root/(split+'_labels.jsonl'))==model[split+'_labels_sha256']
    assert sha(root/'locked_model.json')==read(root/'test_collection.json')['locked_model_sha256']
    assert sha(root/'locked_model.json')==read(root/'test_scoring.json')['locked_model_sha256']
    test=load(root,'test');y=np.array([r['gain'] for r in test]);score=predict(model,test)
    masks=dict(new_half=half(score,test),old_half=half(predict(old,test),test),
        confidence_half=half([-np.mean(r['features'][2:4]) for r in test],test),
        oracle_half=half(y,test),locked_threshold=score>model['dev_threshold'])
    primary=dict(versus_random=interval((masks['new_half'].astype(float)-masks['new_half'].mean())*y,test,.975),
                 versus_old=interval((masks['new_half'].astype(float)-masks['old_half'].astype(float))*y,test,.975))
    policies={}
    an=np.array([r['keep4']['gpt2_nll_sum'] for r in test]);bn=np.array([r['split22']['gpt2_nll_sum'] for r in test])
    ac=np.array([r['keep4']['gpt2_scored_tokens'] for r in test]);bc=np.array([r['split22']['gpt2_scored_tokens'] for r in test])
    teacher=np.array([r['keep4']['teacher_nll']-r['split22']['teacher_nll'] for r in test])
    lengths=np.array([r['split22']['visible_new_tokens']-r['keep4']['visible_new_tokens'] for r in test])
    calls=np.array([r['split22']['forward_calls']-r['keep4']['forward_calls'] for r in test])
    for name,mask in masks.items():
        p=float(mask.mean());mixed_n=np.where(mask,bn,an).sum();mixed_c=np.where(mask,bc,ac).sum()
        random_n=((1-p)*an+p*bn).sum();random_c=((1-p)*ac+p*bc).sum()
        policies[name]=dict(selected=int(mask.sum()),selection_fraction=p,
            nll_gain_vs_matched_random=interval((mask.astype(float)-p)*y,test),
            teacher_nll_gain_vs_matched_random=interval((mask.astype(float)-p)*teacher,test),
            mean_length_change_vs_matched_random=float(np.mean((mask.astype(float)-p)*lengths)),
            mean_full_rollout_calls_change_vs_matched_random=float(np.mean((mask.astype(float)-p)*calls)),
            corpus_gen_ppl=float(np.exp(mixed_n/mixed_c)),
            random_expected_sums_corpus_ppl=float(np.exp(random_n/random_c)),
            corpus_ppl_ratio_to_random_expected_sums=float(np.exp(mixed_n/mixed_c-random_n/random_c)))
    gate=bool(model['dev_gate_passed'] and all(v['interval'][0]>0 for v in primary.values()))
    report=dict(model=model['name'],train_labels=len(load(root,'train')),dev_labels=len(load(root,'dev')),
        test_labels=len(test),test_source_clusters=len({r['prompt_id'] for r in test}),
        primary=primary,policies=policies,dev_gate_passed=model['dev_gate_passed'],advance_gate_passed=gate,
        label_positive=int((y>0).sum()),label_negative=int((y<0).sum()),label_zero=int((y==0).sum()),
        mean_all_split_gain=interval(y,test),new_mse=float(np.mean((score-y)**2)),
        constant_train_mean_mse=float(np.mean((model['intercept']-y)**2)),
        prediction_label_correlation=float(np.corrcoef(score,y)[0,1]),
        limitations=['Optimizes GPT-2 scoring proxy; same scorer is not independent of the optimization target.',
            'Offline half-budget ranking conditional on the observed cohort; no online decision or speed claim.',
            'Single intervention with fixed4 future and fixed4 generated histories, not learned-policy trajectories.',
            'Same intervention count does not equal same full-rollout cost; branch lengths differ.',
            'Unreachable/EOS-head/short-text exclusions recorded; raw-document and semantic-near-duplicate identity unavailable.'])
    with (root/'test_predictions.jsonl').open('w',encoding='utf-8') as f:
        for i,r in enumerate(test):
            f.write(json.dumps(dict(pair_id=r['pair_id'],prompt_id=r['prompt_id'],gain=r['gain'],score=float(score[i]),
                **{k:bool(v[i]) for k,v in masks.items()}),allow_nan=False)+'\n')
    dump(root/'analysis.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['fit','evaluate']);ap.add_argument('root',type=Path)
    args=ap.parse_args();globals()[args.action](args.root)
