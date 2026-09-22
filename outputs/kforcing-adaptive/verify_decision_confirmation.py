"""Independent provenance, source isolation, paired-label and primary-test audit."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/decision-confirm'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import torch
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast,GPT2TokenizerFast
    lock,manifest=read(OUT/'locked_model.json'),read(OUT/'manifest.json')
    assert sha(OUT/'locked_model.json')==manifest['locked_model_sha256']
    for name,digest in lock['code_hashes'].items(): assert sha(ROOT/name)==digest
    for name,digest in {**lock['training_hashes'],**lock['historical_source_hashes']}.items(): assert sha(ROOT/'results'/name)==digest
    for name,digest in manifest['artifact_hashes'].items(): assert sha(OUT/name)==digest
    old=[]
    for name in lock['historical_source_hashes']:
        item=read(ROOT/'results'/name)
        old.extend([p for group in item.values() for p in group] if isinstance(item,dict) else item)
    old_prefixes={tuple(p.get('ids',p.get('reference_ids'))[:6]) for p in old}
    old_rows={p['source_row'] for p in old if 'source_row' in p}
    old_hashes={p['source_sha256'] for p in old if 'source_sha256' in p}
    dataset=Path('work/lm1b-data/test.parquet')
    assert sha(dataset)==manifest['dataset_sha256']
    assert sha(Path('work/pilot-data/vocab.txt'))==manifest['tokenizer_sha256']
    texts=pq.read_table(dataset,columns=['text'])['text'].to_pylist()
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    prefixes=read(OUT/'prefixes.json')
    assert len(prefixes)==384 and len({tuple(p['ids']) for p in prefixes})==384
    order=list(range(len(texts)));random.Random(21192001).shuffle(order)
    rebuilt=[]
    for row in order:
        digest=hashlib.sha256(texts[row].encode()).hexdigest()
        if row in old_rows or digest in old_hashes: continue
        ids=bert.encode(texts[row],add_special_tokens=True,truncation=True,max_length=128)
        if len(ids)<12 or tuple(ids[:6]) in old_prefixes: continue
        rebuilt.append((row,digest,ids[:6]));old_hashes.add(digest);old_prefixes.add(tuple(ids[:6]))
        if len(rebuilt)==384: break
    assert rebuilt==[(p['source_row'],p['source_sha256'],p['ids']) for p in prefixes]
    prefix_map={p['prompt_id']:p for p in prefixes}
    development=rows(ROOT/'results/features-014/development_data.jsonl')
    features=rows(ROOT/'results/decision-features/features.jsonl')
    target=np.array([r['gain'] for r in development])
    for name,model in lock['models'].items():
        d=len(model['mean']);x=np.array([r['features'][:d] for r in features])
        mean,std=x.mean(0),x.std(0);std[std<1e-8]=1
        np.testing.assert_array_equal(mean,model['mean']);np.testing.assert_array_equal(std,model['std'])
        z=(x-mean)/std
        design=np.column_stack([np.ones(len(x)),z]);penalty=np.column_stack([np.zeros(d),np.sqrt(1000)*np.eye(d)])
        co=np.linalg.lstsq(np.vstack([design,penalty]),np.r_[target,np.zeros(d)],rcond=None)[0]
        np.testing.assert_allclose(co,np.r_[model['intercept'],model['coef']],atol=1e-11,rtol=0)
    collection=read(OUT/'test_collection.json');scoring=read(OUT/'test_scoring.json')
    assert collection['passed'] and scoring['passed']
    assert collection['pairs_sha256']==scoring['pairs_sha256']==sha(OUT/'test_pairs.jsonl')
    assert collection['states_sha256']==sha(OUT/'test_states.jsonl')
    assert scoring['labels_sha256']==sha(OUT/'test_labels.jsonl')
    assert collection['locked_model_sha256']==scoring['locked_model_sha256']==sha(OUT/'locked_model.json')
    predictions=read(OUT/'frozen_predictions.json');link=read(OUT/'scoring_prediction_link.json')
    assert predictions['test_labels_absent_when_frozen']
    assert predictions['states_sha256']==sha(OUT/'test_states.jsonl')
    assert link['frozen_predictions_sha256']==sha(OUT/'frozen_predictions.json')
    assert link['test_scoring_sha256']==sha(OUT/'test_scoring.json')
    assert datetime.fromisoformat(lock['created_utc'])<datetime.fromisoformat(predictions['created_utc'])<datetime.fromisoformat(link['completed_utc'])
    states=rows(OUT/'test_states.jsonl');assert len(states)==768
    state_map={s['state_id']:s for s in states};assert len(state_map)==768
    labels={r['pair_id']:r for r in rows(OUT/'test_labels.jsonl')}
    pair_records=rows(OUT/'test_pairs.jsonl');assert len(pair_records)==collection['pairs']==len(labels)
    by_state=defaultdict(dict)
    gpt=GPT2TokenizerFast.from_pretrained('work/gpt2-large',local_files_only=True)
    token_counts={}
    for r in pair_records:
        state=state_map[r['state_id']];source=prefix_map[r['prompt_id']]
        assert state['status']=='paired' and r['replicate'] not in by_state[r['state_id']]
        assert r['noise_seed']==211000000+1901+source['index']*100003
        assert r['future_seed']==212000000+source['index']*100003+r['offset']*17+r['replicate']*100000007
        tape=torch.rand((1,122,1),generator=torch.Generator().manual_seed(r['noise_seed']))
        future=torch.rand((1,122,1),generator=torch.Generator().manual_seed(r['future_seed']))
        tape[:,r['offset']+4:]=future[:,r['offset']+4:]
        assert hashlib.sha256(tape.numpy().astype('<f4').tobytes()).hexdigest()==r['noise_sha256']
        assert tape[0,r['offset']:r['offset']+4,0].tolist()==r['noise']
        for key in ('history_ids','noise','initial_candidates','split_candidates','features'):
            assert r[key]==state[key]
        assert r['history_ids'][:6]==source['ids'] and len(r['history_ids'])==6+r['offset']
        label=labels[r['pair_id']]
        for branch,extra in (('keep4',0),('split22',1)):
            b,l=r[branch],label[branch];ids=b['ids'];generated=ids[6:];length=len(generated)
            assert ids[:len(r['history_ids'])]==r['history_ids']
            assert 0<length<=122
            assert b['stopped_eos']==(generated[-1]==102) and 102 not in generated[:-1]
            first=r['initial_candidates'] if branch=='keep4' else r['split_candidates']
            actual=ids[len(r['history_ids']):len(r['history_ids'])+4]
            expected=first[:first.index(102)+1] if 102 in first else first
            assert actual==expected
            tail_calls=math.ceil(max(0,length-r['offset']-4)/4)
            assert b['calls']==r['offset']//4+1+extra+tail_calls==l['forward_calls']
            assert b['computed_candidates']==r['offset']+4+2*extra+min(122-r['offset']-4,4*tail_calls)==l['computed_candidates']
            assert l['visible_new_tokens']==l['teacher_token_count']==length
            np.testing.assert_allclose(l['teacher_nll'],l['teacher_nll_sum']/length,atol=1e-12,rtol=0)
            text=bert.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            assert text==b['completion']
            if text not in token_counts: token_counts[text]=len(gpt.encode(text,add_special_tokens=False))
            count=max(0,token_counts[text]-1)
            assert count==l['gpt2_scored_tokens'] and token_counts[text]==l['gpt2_input_tokens']
            assert l['excluded_short']==(count==0)
            if count: assert abs(l['gpt2_mean_nll']-l['gpt2_nll_sum']/count)<1e-12
        valid=label['keep4']['gpt2_scored_tokens']>0 and label['split22']['gpt2_scored_tokens']>0
        assert valid==label['valid_label']
        if valid: assert abs(label['gain']-(label['keep4']['gpt2_mean_nll']-label['split22']['gpt2_mean_nll']))<1e-12
        if r['keep4']['completion']==r['split22']['completion'] and valid: assert label['gain']==0
        by_state[r['state_id']][r['replicate']]=label
    data=[];excluded=[]
    for s in states:
        p=prefix_map[s['prompt_id']]
        assert s['seed']==1901 and s['offset'] in (0,8)
        assert s['state_noise_seed']==211000000+1901+p['index']*100003
        if s['status']!='paired':
            assert s['state_id'] not in by_state
            continue
        assert set(by_state[s['state_id']])==set(range(8))
        df=s['decision_features']
        assert df[:14]==s['features'][:14] and df[14]==s['offset']/8
        assert df[17:]==[int(t==102) for t in s['initial_candidates'][2:]]
        assert all(g<=0 for g in df[15:17])
        for name,model in lock['models'].items():
            d=len(model['mean']);v=(np.array(df[:d])-model['mean'])/model['std']
            pred=float(v@np.array(model['coef'])+model['intercept'])
            assert abs(pred-predictions['scores'][s['state_id']][name])<1e-12
        assert predictions['scores'][s['state_id']]['heuristic']==-np.mean(s['features'][2:4])
        if not all(l['valid_label'] for l in by_state[s['state_id']].values()):
            excluded.append(s['state_id']);continue
        data.append(dict(s,gain=np.mean([by_state[s['state_id']][i]['gain'] for i in range(8)])))
    ids=[s['state_id'] for s in data];all_ids=[s['state_id'] for s in states if s['status']=='paired']
    assert set(predictions['scores'])==set(all_ids)
    masks={}
    for name in ('combined19','confidence14','heuristic'):
        order=sorted(all_ids,key=lambda sid:(-predictions['scores'][sid][name],sid))
        assert order==predictions['rankings'][name]
        filtered=[sid for sid in order if sid in set(ids)]
        chosen=set(filtered[:len(ids)//2]);masks[name]=np.array([float(sid in chosen) for sid in ids])
    y=np.array([r['gain'] for r in data]);groups=defaultdict(list)
    for i,r in enumerate(data): groups[r['prompt_id']].append(i)
    sources=sorted(groups);blocks=[np.array(groups[s]) for s in sources]
    values=np.column_stack([(masks['combined19']-masks['combined19'].mean())*y,
        (masks['combined19']-masks['confidence14'])*y,(masks['combined19']-masks['heuristic'])*y])
    rng=np.random.default_rng(21192004);boot=[]
    for _ in range(8000):
        indices=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        boot.append(values[indices].mean(0))
    bounds=np.quantile(boot,[.05/6,1-.05/6],axis=0).T
    a=read(OUT/'analysis.json')
    assert a['test_states']==len(data) and a['test_sources']==len(sources) and a['excluded_incomplete']==excluded
    for j,name in enumerate(('vs_random','vs_confidence14','vs_heuristic')):
        np.testing.assert_allclose(a['primary'][name]['estimate'],values[:,j].mean(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(a['primary'][name]['interval'],bounds[j],atol=1e-11,rtol=0)
    assert a['advance_gate_passed']==bool(len(sources)>=128 and np.all(bounds[:,0]>0))
    # Exact fresh-sequence replay and current-only EOS feature check on two separated states.
    from core import load_model
    from repeat_rollout_quality import pair,mixed_noise
    checkpoint=Path('work/checkpoints/pflm_lm1b_k4.ckpt')
    assert sha(checkpoint)=='3b013f0cf9a3acab4c20a1bb748beee6525432ef3f56a74df4957b898480a2c2'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(checkpoint,'pflm','cuda');projection=torch.from_numpy(np.load(OUT/'projection.npy')).cuda()
    probes=pair_records[:4]+pair_records[-4:]
    with torch.inference_mode():
        for r in probes:
            result=pair(model,torch.tensor([r['history_ids'][:6]],device='cuda'),
                mixed_noise(r['noise_seed'],r['future_seed'],r['offset']),r['offset'],projection)
            for b in ('keep4','split22'): assert all(result[b][k]==r[b][k] for k in result[b])
        for s in (data[0],data[-1]):
            logits=model(torch.tensor([s['history_ids']],device='cuda'),torch.tensor(s['noise'],device='cuda').reshape(1,4,1),
                torch.ones(1,1,1,device='cuda'),mode='inference')
            gaps=(logits[0,2:,102]-logits[0,2:].max(-1).values).cpu().tolist()
            np.testing.assert_allclose(gaps,s['decision_features'][15:17],atol=1e-5,rtol=0)
    result=dict(passed=True,source_list_independently_resampled=True,all_noise_hashes_verified=len(pair_records),
        all_eos_counts_and_label_arithmetic_verified=True,independent_primary_intervals=3,
        frozen_model_fits_and_predictions_verified=True,fresh_pairs_exactly_replayed=len(probes),
        test_states=len(data),test_sources=len(sources),advance_gate_passed=a['advance_gate_passed'],
        analysis_sha256=sha(OUT/'analysis.json'),verifier_sha256=sha(Path(__file__)),
        scoring_scope='All score arithmetic/token counts checked; scorer builtin loss check retained, no second full evaluator pass.')
    path=OUT/'integrity_audit.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
