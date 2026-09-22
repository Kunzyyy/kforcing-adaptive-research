"""Independent CPU audit of rollout labels, train-only fitting, and held-out gates."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root):
    code=root.parent.parent;m=read(root/'manifest.json');prefixes=read(root/'prefixes.json')
    model=read(root/'locked_model.json');old_model=read(root/'old_model.json')
    for f,k in [('prefixes.json','prefixes_sha256'),('projection.npy','projection_sha256'),('old_model.json','old_model_sha256')]:
        assert sha(root/f)==m[k]
    assert sha(code/'STAGE11_PROTOCOL.md')==m['protocol_sha256']
    assert sha(code/'learn_local_gain.py')==m['feature_code_sha256']
    assert sha(root/'old_model.json')==sha(root.parent/'learned-007/locked_model.json')
    assert sha(root/'projection.npy')==sha(root.parent/'learned-007/projection.npy')
    assert sha(code/'rollout_quality.py')==read(root/'preflight.json')['collector_sha256']
    assert read(root/'preflight.json')['passed']
    old=read(root.parent/'lm1b-004/prefixes.json')+read(root.parent/'online-009/prefixes.json')+read(root.parent/'early-010/prefixes.json')
    for group in read(root.parent/'learned-007/prefixes.json').values():old+=group
    old += [dict(p,ids=p['reference_ids'][:6]) for p in read(root.parent/'confirm-008/sources.json')]
    old_ids={tuple(p['ids'][:6]) for p in old};old_rows={p['source_row'] for p in old};old_hash={p['source_sha256'] for p in old}
    for folder in ('pilot-001','benefit-002'):
        for group in read(root.parent/folder/'prefixes.json').values():old_ids.update(tuple(p['ids'][:6]) for p in group)
    old_ids.update(tuple(p['ids'][:6]) for p in read(root.parent/'current-003/fresh_prefixes.json'))
    assert len(prefixes)==384 and len({tuple(p['ids']) for p in prefixes})==384
    for f in ('source_row','source_sha256','prompt_id'):assert len({p[f] for p in prefixes})==384
    for p in prefixes:
        assert tuple(p['ids']) not in old_ids and p['source_row'] not in old_rows and p['source_sha256'] not in old_hash
    pm={p['prompt_id']:p for p in prefixes};labeled={};counts={}
    for split,count in m['splits'].items():
        pp=[p for p in prefixes if p['split']==split];assert len(pp)==count
        data=rows(root/(split+'_pairs.jsonl'));labels=rows(root/(split+'_labels.jsonl'))
        c=read(root/(split+'_collection.json'));sc=read(root/(split+'_scoring.json'))
        assert c['passed'] and sc['passed']
        assert sha(root/(split+'_pairs.jsonl'))==c['pairs_sha256']==sc['pairs_sha256']
        assert sha(root/(split+'_labels.jsonl'))==sc['labels_sha256']
        assert sc['collector_sha256']==sha(code/'rollout_quality.py')
        assert sc['scorer_sha256']==sha(code/'score_rollout_quality.py')
        if split=='test':assert sha(root/'locked_model.json')==c['locked_model_sha256']==sc['locked_model_sha256']
        else:assert c['locked_model_sha256'] is None and sc['locked_model_sha256'] is None
        expected={(p['prompt_id'],seed,offset) for p in pp for seed in m['seeds'] for offset in m['offsets']}
        assert len(data)==len(expected)==count*8
        assert {(r['prompt_id'],r['seed'],r['offset']) for r in data}==expected
        base={};paired={};text_score={}
        for r in data:
            p=pm[r['prompt_id']];assert p['split']==split
            assert r['noise_seed']==111000000+r['seed']+p['index']*100003
            assert r['history_ids'][:6]==p['ids']
            assert r['pair_id']==f"{r['prompt_id']}-{r['seed']}-{r['offset']}"
            if r['status']=='unreachable_eos':
                assert r['offset']==8 and r['history_ids'][-1]==102 and len(r['history_ids'])<=14
                continue
            assert len(r['history_ids'])==6+r['offset'] and 102 not in r['history_ids'][6:]
            if r['status']=='head_eos':
                assert 102 in r['initial_candidates'][:2];continue
            assert r['status']=='paired' and 102 not in r['initial_candidates'][:2]
            assert len(r['features'])==82 and np.isfinite(r['features']).all()
            assert r['initial_candidates'][:2]==r['split_candidates'][:2]
            for b,candidates in [('keep4',r['initial_candidates']),('split22',r['split_candidates'])]:
                seq=r[b]['ids'];n=len(seq)-6;extra=int(b=='split22')
                assert seq[:len(r['history_ids'])]==r['history_ids']
                first=candidates[:candidates.index(102)+1] if 102 in candidates else candidates
                assert seq[6+r['offset']:6+r['offset']+len(first)]==first
                assert 102 not in seq[6:-1] and (seq[-1]==102)==r[b]['stopped_eos']
                assert seq[-1]==102 or n==122
                assert r[b]['calls']==math.ceil(n/4)+extra
                assert r[b]['computed_candidates']==min(4*math.ceil(n/4),122)+2*extra
            ident=r['prompt_id'],r['seed']
            if ident in base:assert r['keep4']['ids']==base[ident]
            else:base[ident]=r['keep4']['ids']
            paired[r['pair_id']]=r
        assert dict(Counter(r['status'] for r in data))==c['status_counts']
        assert len(labels)==len(paired)==sc['pairs'] and {r['pair_id'] for r in labels}==set(paired)
        valid=[]
        for label in labels:
            pair=paired[label['pair_id']]
            for name in ('keep4','split22'):
                r=label[name];branch=pair[name]
                assert r['visible_new_tokens']==r['teacher_token_count']==len(branch['ids'])-6
                assert r['forward_calls']==branch['calls'] and r['computed_candidates']==branch['computed_candidates']
                assert r['gpt2_scored_tokens']==max(0,r['gpt2_input_tokens']-1)
                assert r['excluded_short']==(r['gpt2_scored_tokens']==0)
                assert math.isfinite(r['teacher_nll_sum']) and r['teacher_nll_sum']>=0
                assert abs(r['teacher_nll']-r['teacher_nll_sum']/r['teacher_token_count'])<1e-12
                assert math.isfinite(r['gpt2_nll_sum']) and r['gpt2_nll_sum']>=0
                if not r['excluded_short']:assert abs(r['gpt2_mean_nll']-r['gpt2_nll_sum']/r['gpt2_scored_tokens'])<1e-12
                text=branch['completion'];value=(r['gpt2_nll_sum'],r['gpt2_scored_tokens'])
                if text in text_score:assert value==text_score[text]
                else:text_score[text]=value
            ok=not label['keep4']['excluded_short'] and not label['split22']['excluded_short']
            assert ok==label['valid_label']
            same=pair['keep4']['completion']==pair['split22']['completion']
            assert label['identical_decoded_text']==same
            if ok:
                assert abs(label['gain']-(label['keep4']['gpt2_mean_nll']-label['split22']['gpt2_mean_nll']))<1e-12
                if same:assert label['gain']==0.
                valid.append(dict(label,features=pair['features']))
            else:assert label['gain'] is None
        assert len(valid)==sc['valid_labels'];labeled[split]=valid
        counts[split]=dict(attempts=len(data),paired=len(paired),valid_labels=len(valid),statuses=c['status_counts'])
    for split in ('train','dev'):assert sha(root/(split+'_labels.jsonl'))==model[split+'_labels_sha256']
    assert sha(root/'prefixes.json')==model['prefixes_sha256'] and sha(root/'projection.npy')==model['projection_sha256']
    def prediction(m,data):
        x=np.array([r['features'][:len(m['coef'])] for r in data])
        return (x-np.array(m['mean'])) @ (np.array(m['coef'])/np.array(m['std']))+m['intercept']
    def select(scores,data):
        ids=np.array([r['pair_id'] for r in data]);order=np.lexsort((ids,-np.asarray(scores)))
        mask=np.zeros(len(data),dtype=bool);mask[order[:len(data)//2]]=True;return mask
    def boot(values,data,coverage):
        ids=sorted({r['prompt_id'] for r in data});at={p:i for i,p in enumerate(ids)}
        sums=np.zeros(len(ids));numbers=np.zeros(len(ids))
        for value,r in zip(values,data):sums[at[r['prompt_id']]]+=value;numbers[at[r['prompt_id']]]+=1
        rng=np.random.default_rng(11112004)
        weights=np.stack([np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids)) for _ in range(4000)])
        vals=(weights@sums)/(weights@numbers);tail=(1-coverage)/2
        return np.quantile(vals,[tail,1-tail])
    train=labeled['train'];dev=labeled['dev'];yt=np.array([r['gain'] for r in train]);yd=np.array([r['gain'] for r in dev])
    candidates=read(root/'dev_candidates.json');assert len(candidates)==6
    for c in candidates:
        n=len(c['coef']);x=np.array([r['features'][:n] for r in train]);mean=x.mean(0);std=x.std(0);std[std<1e-8]=1.
        np.testing.assert_allclose(c['mean'],mean,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(c['std'],std,rtol=1e-12,atol=1e-12)
        assert abs(c['intercept']-yt.mean())<1e-12
        z=(x-mean)/std;lhs=(z.T@z+c['alpha']*np.eye(n))@np.array(c['coef']);rhs=z.T@(yt-yt.mean())
        np.testing.assert_allclose(lhs,rhs,rtol=1e-8,atol=1e-8)
        pred=prediction(c,dev);mask=select(pred,dev);values=(mask.astype(float)-mask.mean())*yd
        assert abs(c['dev_gain']['estimate']-values.mean())<1e-12
        assert abs(c['dev_threshold']-np.median(pred))<1e-10
        np.testing.assert_allclose(boot(values,dev,.95),c['dev_gain']['interval'],atol=1e-12,rtol=1e-12)
    chosen=min(candidates,key=lambda c:(-c['dev_gain']['estimate'],c['name']))
    assert model['name']==chosen['name'] and model['coef']==chosen['coef']
    assert model['dev_gate_passed']==(chosen['dev_gain']['interval'][0]>0)
    test=labeled['test'];y=np.array([r['gain'] for r in test]);pred=prediction(model,test)
    masks=dict(new_half=select(pred,test),old_half=select(prediction(old_model,test),test),
        confidence_half=select([-np.mean(r['features'][2:4]) for r in test],test),oracle_half=select(y,test),
        locked_threshold=pred>model['dev_threshold'])
    saved={r['pair_id']:r for r in rows(root/'test_predictions.jsonl')};assert len(saved)==len(test)
    for i,r in enumerate(test):
        assert abs(saved[r['pair_id']]['score']-pred[i])<1e-10
        for name,mask in masks.items():assert saved[r['pair_id']][name]==bool(mask[i])
    report=read(root/'analysis.json')
    values=dict(versus_random=(masks['new_half'].astype(float)-masks['new_half'].mean())*y,
                versus_old=(masks['new_half'].astype(float)-masks['old_half'].astype(float))*y)
    for name,v in values.items():
        assert abs(report['primary'][name]['estimate']-v.mean())<1e-12
        np.testing.assert_allclose(boot(v,test,.975),report['primary'][name]['interval'],rtol=1e-12,atol=1e-12)
    for name,mask in masks.items():
        p=float(mask.mean());v=(mask.astype(float)-p)*y
        pol=report['policies'][name]
        assert int(mask.sum())==pol['selected'] and p==pol['selection_fraction']
        assert abs(pol['nll_gain_vs_matched_random']['estimate']-v.mean())<1e-12
        ns=cs=rn=rc=0.
        for select_,r in zip(mask,test):
            a,b=r['keep4'],r['split22'];branch=b if select_ else a
            ns+=branch['gpt2_nll_sum'];cs+=branch['gpt2_scored_tokens']
            rn+=(1-p)*a['gpt2_nll_sum']+p*b['gpt2_nll_sum'];rc+=(1-p)*a['gpt2_scored_tokens']+p*b['gpt2_scored_tokens']
        assert abs(pol['corpus_gen_ppl']-math.exp(ns/cs))<1e-9
        assert abs(pol['random_expected_sums_corpus_ppl']-math.exp(rn/rc))<1e-9
    gate=model['dev_gate_passed'] and all(r['interval'][0]>0 for r in report['primary'].values())
    assert gate==report['advance_gate_passed']
    result=dict(passed=True,new_sources=384,splits=counts,data_isolation=True,branch_eos_and_counts=True,
        paired_label_math=True,exact_duplicate_score_reuse=True,train_normalization_and_normal_equations=True,
        dev_candidate_choice_and_bootstrap=True,locked_test_predictions=True,primary_bootstrap_independently_rebuilt=True,
        corpus_ppl_aggregates=True,advance_gate_passed=gate)
    (root/'integrity_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);audit(ap.parse_args().root)
