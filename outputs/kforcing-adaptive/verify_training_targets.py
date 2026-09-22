"""Independent token-label, least-squares and source-bootstrap verification."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/training-target'
SRC=ROOT/'results/averaged-013'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    m=read(OUT/'manifest.json')
    for name,h in {**m['input_whitelist'],**m['code_hashes']}.items():assert sha(ROOT/name)==h,name
    assert not any('/test_' in p or 'lowdim-015' in p or 'decision-confirm' in p for p in m['input_whitelist'])
    assert sha(Path('work/gpt2-large/manifest.json'))==m['evaluator_manifest_sha256']
    check=read(OUT/'scoring_checks.json');target_check=read(OUT/'target_checks.json');a=read(OUT/'analysis.json')
    assert check['passed'] and check['token_losses_sha256']==sha(OUT/'token_losses.jsonl')
    assert check['builtin_loss_matches'] and check['causal_truncation_max_difference']<=1e-4
    assert target_check['passed'] and target_check['state_targets_sha256']==a['state_targets_sha256']==sha(OUT/'state_targets.jsonl')
    assert target_check['pair_targets_sha256']==sha(OUT/'pair_targets.jsonl')
    assert a['fits_sha256']==sha(OUT/'fits.json') and a['oof_predictions_sha256']==sha(OUT/'oof_predictions.jsonl')
    records=rows(OUT/'token_losses.jsonl');token={(r['split'],r['text']):r for r in records}
    assert len(records)==len(token)
    from transformers import GPT2TokenizerFast
    tokenizer=GPT2TokenizerFast.from_pretrained('work/gpt2-large',local_files_only=True)
    for r in records:
        assert r['split'] in ('train','dev')
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
        assert tokenizer.encode(r['text'],add_special_tokens=False)==r['input_ids']
        assert len(r['token_nll'])==r['scored_tokens']==max(0,len(r['input_ids'])-1)
        assert all(math.isfinite(v) and v>=0 for v in r['token_nll'])
        np.testing.assert_allclose(math.fsum(r['token_nll']),r['torch_nll_sum'],atol=1e-4,rtol=1e-6)
    old=rows(ROOT/'results/features-014/development_data.jsonl');old_ids={r['state_id'] for r in old}
    pair_saved={r['pair_id']:r for r in rows(OUT/'pair_targets.jsonl')}
    groups=defaultdict(dict);observed=set();max_old=0.;max_label=0.
    for split in ('train','dev'):
        labels={r['pair_id']:r for r in rows(SRC/f'{split}_labels.jsonl')}
        for p in rows(SRC/f'{split}_pairs.jsonl'):
            l=labels[p['pair_id']]
            vectors=[]
            for branch in ('keep4','split22'):
                key=(split,p[branch]['completion']);observed.add(key);t=token[key]
                assert t['scored_tokens']==l[branch]['gpt2_scored_tokens']
                error=abs(t['torch_nll_sum']-l[branch]['gpt2_nll_sum']);max_old=max(max_old,error)
                assert error<=1e-4
                vectors.append(t['token_nll'])
            if p['state_id'] not in old_ids:
                assert p['pair_id'] not in pair_saved
                continue
            k,s=vectors;n=min(len(k),len(s));assert n>0 and l['valid_label']
            common=float(np.mean(np.array(k[:n])-np.array(s[:n])))
            full=l['keep4']['gpt2_mean_nll']-l['split22']['gpt2_mean_nll']
            teacher=l['keep4']['teacher_nll']-l['split22']['teacher_nll']
            expected=pair_saved[p['pair_id']]
            assert expected['common_tokens']==n and expected['replicate']==p['replicate']
            vals=[full,common,teacher]
            np.testing.assert_allclose(vals,[expected[k] for k in ('full_gain','common_gain','teacher_gain')],atol=1e-12,rtol=0)
            max_label=max(max_label,abs(common-expected['common_gain']))
            assert p['replicate'] not in groups[p['state_id']]
            groups[p['state_id']][p['replicate']]=vals
    assert observed==set(token) and len(pair_saved)==3568
    data=rows(OUT/'state_targets.jsonl');ids=[r['state_id'] for r in data]
    assert ids==[r['state_id'] for r in old]==sorted(ids)
    x=np.array([r['features'] for r in data]);ys=[]
    for r,o in zip(data,old):
        assert r['source_split'] in ('train','dev') and r['features']==o['features'][:14]
        assert set(groups[r['state_id']])==set(range(8))
        repeat_values=np.array([groups[r['state_id']][i] for i in range(8)])
        vals=repeat_values.mean(0);ys.append(vals)
        np.testing.assert_allclose(vals,[r[k+'_gain'] for k in ('full','common','teacher')],atol=1e-12,rtol=0)
        assert r['full_gains']==o['gains'] and abs(vals[0]-o['gain'])<1e-12
    ys=np.array(ys);sources=sorted({r['prompt_id'] for r in data})
    assert len(data)==446 and len(sources)==253
    fits=read(OUT/'fits.json');assert len(fits)==30
    assert len({(r['repeat'],r['fold'],r['method']) for r in fits})==30
    predictions={k:np.zeros((3,446)) for k in ('full_target','common_target')}
    selected={k:np.zeros((3,446)) for k in predictions}
    heuristic=np.zeros((3,446));fractions=np.zeros((3,446));max_prediction=0.
    for f in fits:
        order=np.random.default_rng([14112001,14112002,14112003][f['repeat']]).permutation(sources)
        assignment={s:i%5 for i,s in enumerate(order)}
        valid=[i for i,r in enumerate(data) if assignment[r['prompt_id']]==f['fold']]
        train=[i for i in range(446) if i not in set(valid)]
        assert f['valid_ids']==[ids[i] for i in valid] and f['train_ids']==[ids[i] for i in train]
        assert not {data[i]['prompt_id'] for i in train}&{data[i]['prompt_id'] for i in valid}
        mean,std=x[train].mean(0),x[train].std(0);std[std<1e-8]=1
        np.testing.assert_array_equal(mean,f['model']['mean']);np.testing.assert_array_equal(std,f['model']['std'])
        z,v=(x[train]-mean)/std,(x[valid]-mean)/std
        target=ys[:,0 if f['method']=='full_target' else 1]
        design=np.column_stack([np.ones(len(train)),z]);penalty=np.column_stack([np.zeros(14),np.sqrt(1000)*np.eye(14)])
        beta=np.linalg.lstsq(np.vstack([design,penalty]),np.r_[target[train],np.zeros(14)],rcond=None)[0]
        pred=np.column_stack([np.ones(len(valid)),v])@beta
        np.testing.assert_allclose(pred,f['predictions'],atol=1e-11,rtol=0)
        max_prediction=max(max_prediction,float(np.max(np.abs(pred-f['predictions']))))
        rank=sorted(range(len(valid)),key=lambda j:(-pred[j],ids[valid[j]]))
        mask=np.zeros(len(valid));mask[rank[:len(valid)//2]]=1
        np.testing.assert_array_equal(mask,f['selected'])
        predictions[f['method']][f['repeat'],valid]=pred;selected[f['method']][f['repeat'],valid]=mask
        hrank=sorted(valid,key=lambda i:(x[i,2:4].mean(),ids[i]))
        heuristic[f['repeat'],hrank[:len(valid)//2]]=1;fractions[f['repeat'],valid]=(len(valid)//2)/len(valid)
    saved=rows(OUT/'oof_predictions.jsonl')
    assert [r['state_id'] for r in saved]==ids
    for i,r in enumerate(saved):
        for name in predictions:
            np.testing.assert_allclose(predictions[name][:,i],r['predictions'][name],atol=1e-11,rtol=0)
            np.testing.assert_array_equal(selected[name][:,i],r['selected'][name])
    primary=np.column_stack([((selected['common_target']-selected['full_target'])*ys[:,0]).mean(0),
                             ((selected['common_target']-heuristic)*ys[:,0]).mean(0)])
    blocks=[np.flatnonzero(np.array([r['prompt_id']==s for r in data])) for s in sources]
    rng=np.random.default_rng(21292026);boots=[]
    for _ in range(4000):
        index=np.concatenate([blocks[i] for i in rng.integers(0,253,253)])
        boots.append(primary[index].mean(0))
    bounds=np.quantile(boots,[.025,.975],axis=0).T
    for j,name in enumerate(('gain_vs_full_target','gain_vs_heuristic')):
        np.testing.assert_allclose(primary[:,j].mean(),a['primary'][name]['estimate'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(bounds[j],a['primary'][name]['interval95'],atol=1e-11,rtol=0)
    assert a['common_target_warrants_fresh_confirmation']==bool(np.all(bounds[:,0]>0))
    for name in predictions:
        for j,target in enumerate(('full','common','teacher')):
            np.testing.assert_allclose(np.mean((predictions[name]-ys[:,j])**2),a['metrics'][name][target]['mse'],atol=1e-11,rtol=0)
            gain=((selected[name]-fractions)*ys[:,j]).mean()
            np.testing.assert_allclose(gain,a['metrics'][name][target]['gain_vs_random']['estimate'],atol=1e-12,rtol=0)
    historical=read(ROOT/'results/features-014/analysis.json')['fixed']['ridge_confidence14_a01000']
    np.testing.assert_allclose(a['metrics']['full_target']['full']['mse'],historical['oof_mse'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(a['metrics']['full_target']['full']['gain_vs_random']['estimate'],historical['gain_vs_random']['estimate'],atol=1e-12,rtol=0)
    result=dict(passed=True,no_test_inputs=True,tokenization_records_verified=len(records),valid_pairs=3568,states=446,sources=253,
        max_old_sum_difference=max_old,max_common_label_difference=max_label,all_30_fits_independent_lstsq=True,
        max_prediction_difference=max_prediction,primary_intervals_independently_recomputed=2,
        original_stage14_control_reproduced=True,analysis_sha256=sha(OUT/'analysis.json'),verifier_sha256=sha(Path(__file__)),
        scope='All tokenization/sum/label mappings and primary statistics checked; no second full GPU scoring pass.')
    assert not (OUT/'integrity_audit.json').exists()
    (OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
