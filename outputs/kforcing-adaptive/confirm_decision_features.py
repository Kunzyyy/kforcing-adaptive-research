"""Frozen combined19 versus confidence14, heuristic and random on fresh sources."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/decision-confirm'
N=384
SEED=1901
CODE=['DECISION_CONFIRM_PLAN.md','confirm_decision_features.py','diagnose_decision_features.py','verify_decision_features.py',
 'repeat_rollout_quality.py','score_rollout_quality.py','core.py','current_features.py',
 'learn_local_gain.py','collect_benefit.py','confirm_local_gain.py','confirm_lowdim.py',
 'run_frozen_online.py','external_gpt2_score.py','fit_averaged_labels.py',
 'upstream/models/pflm.py','upstream/models/transformer.py']

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def dump(p,v):
    assert not p.exists(),f'Preserve {p}'
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def freeze(args):
    from diagnose_decision_features import fit
    from confirm_lowdim import OLD_FILES, old_sources
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    d=ROOT/'results/decision-features'
    assert read(d/'integrity_audit.json')['passed']
    assert read(d/'analysis.json')['combined19_warrants_fresh_confirmation']
    assert read(d/'integrity_audit.json')['analysis_sha256']==sha(d/'analysis.json')
    data=rows(ROOT/'results/features-014/development_data.jsonl'); features=rows(d/'features.jsonl')
    assert [r['state_id'] for r in data]==[r['state_id'] for r in features]
    x=np.array([r['features'] for r in features]);y=np.array([r['gain'] for r in data])
    history=list(OLD_FILES)+['lowdim-015/prefixes.json','baseline-016/prefixes.json']
    training=['features-014/development_data.jsonl','decision-features/features.jsonl',
              'decision-features/analysis.json','decision-features/integrity_audit.json','averaged-013/train_pairs.jsonl']
    lock=dict(created_utc=now(),models={'combined19':fit(x,y),'confidence14':fit(x[:,:14],y)},
        training_hashes={p:sha(ROOT/'results'/p) for p in training},
        code_hashes={p:sha(ROOT/p) for p in CODE},
        historical_source_hashes={p:sha(ROOT/'results'/p) for p in history},
        primary_comparisons=['vs_random','vs_confidence14','vs_heuristic'],primary_coverage=1-.05/3)
    dump(OUT/'locked_model.json',lock)
    old=old_sources(ROOT/'results')
    old+=read(ROOT/'results/lowdim-015/prefixes.json')+read(ROOT/'results/baseline-016/prefixes.json')
    seen={tuple(p.get('ids',p.get('reference_ids'))[:6]) for p in old}
    seen_rows={p['source_row'] for p in old if 'source_row' in p}
    seen_hash={p['source_sha256'] for p in old if 'source_sha256' in p}
    path=Path('work/lm1b-data/test.parquet')
    assert sha(path)=='d3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    texts=pq.read_table(path,columns=['text'])['text'].to_pylist()
    order=list(range(len(texts)));random.Random(21192001).shuffle(order)
    tokenizer=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    chosen=[]
    for index in order:
        h=hashlib.sha256(texts[index].encode()).hexdigest()
        if index in seen_rows or h in seen_hash: continue
        ids=tokenizer.encode(texts[index],add_special_tokens=True,truncation=True,max_length=128)
        if len(ids)<12 or tuple(ids[:6]) in seen: continue
        i=len(chosen)
        chosen.append(dict(prompt_id=f'dfc-{i:03}',index=i,ids=ids[:6],source_row=index,source_sha256=h,split='test'))
        seen.add(tuple(ids[:6]));seen_hash.add(h)
        if len(chosen)==N: break
    assert len(chosen)==N
    dump(OUT/'prefixes.json',chosen)
    shutil.copyfile(ROOT/'results/learned-007/projection.npy',OUT/'projection.npy')
    dump(OUT/'manifest.json',dict(created_utc=now(),sources=N,seeds=[SEED],offsets=[0,8],repeats=8,
        dataset_sha256=sha(path),tokenizer_sha256=sha(Path('work/pilot-data/vocab.txt')),
        locked_model_sha256=sha(OUT/'locked_model.json'),
        artifact_hashes={p:sha(OUT/p) for p in ['prefixes.json','projection.npy']}))
    print('Locked combined19 and confidence14; selected 384 unused sources.',flush=True)


def verify():
    m,lock=read(OUT/'manifest.json'),read(OUT/'locked_model.json')
    assert sha(OUT/'locked_model.json')==m['locked_model_sha256']
    for p,h in m['artifact_hashes'].items(): assert sha(OUT/p)==h
    for p,h in lock['code_hashes'].items(): assert sha(ROOT/p)==h
    for p,h in {**lock['training_hashes'],**lock['historical_source_hashes']}.items(): assert sha(ROOT/'results'/p)==h
    assert sha(Path('work/pilot-data/vocab.txt'))==m['tokenizer_sha256']
    return lock


def setup(args):
    import torch
    from core import load_model
    from collect_benefit import verify_checkpoints
    verify();verify_checkpoints(args)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    return load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda'),torch.from_numpy(np.load(OUT/'projection.npy')).cuda()


def extra_features(model,state):
    import torch
    from current_features import current_features_tensor
    logits=model(torch.tensor([state['history_ids']],device='cuda'),
        torch.tensor(state['noise'],device='cuda').reshape(1,4,1),torch.ones(1,1,1,device='cuda'),mode='inference')
    assert logits.argmax(-1)[0].tolist()==state['initial_candidates']
    np.testing.assert_allclose(current_features_tensor(logits).cpu().numpy(),state['features'][:14],rtol=.002,atol=.002)
    gaps=(logits[0,2:,102]-logits[0,2:].max(-1).values).cpu().tolist()
    return state['features'][:14]+[state['offset']/8]+gaps+[int(v==102) for v in state['initial_candidates'][2:]]


def preflight(args):
    import torch
    from repeat_rollout_quality import pair,mixed_noise,noise_hash,STATE_KEYS
    from diagnose_decision_features import predict
    lock=verify()
    features=rows(ROOT/'results/decision-features/features.jsonl')
    y=np.array([r['gain'] for r in rows(ROOT/'results/features-014/development_data.jsonl')])
    for name,m in lock['models'].items():
        d=len(m['mean']);x=np.array([r['features'][:d] for r in features]);z=(x-m['mean'])/m['std']
        coef=np.linalg.lstsq(np.vstack([z,np.sqrt(1000)*np.eye(d)]),np.r_[y-y.mean(),np.zeros(d)],rcond=None)[0]
        np.testing.assert_allclose(predict(m,x),z@coef+y.mean(),atol=1e-11,rtol=0)
    model,projection=setup(args)
    old=rows(ROOT/'results/averaged-013/train_pairs.jsonl')[:16]
    with torch.inference_mode():
        for r in old:
            tape=mixed_noise(r['noise_seed'],r['future_seed'],r['offset'])
            assert noise_hash(tape)==r['noise_sha256']
            actual=pair(model,torch.tensor([r['history_ids'][:6]],device='cuda'),tape,r['offset'],projection)
            assert all(actual[k]==r[k] for k in STATE_KEYS)
            for b in ('keep4','split22'): assert all(actual[b][k]==r[b][k] for k in actual[b])
        for r in features[:8]:
            x=extra_features(model,dict(r,features=r['features']))
            np.testing.assert_allclose(x,r['features'],atol=1e-5,rtol=0)
    dump(OUT/'preflight.json',dict(passed=True,old_pairs_replayed=16,feature_checks=8,
        both_models_independent_lstsq=True,locked_model_sha256=sha(OUT/'locked_model.json')))
    print('Independent model fits, 16 historical pairs and new feature extraction checked.',flush=True)


def collect(args):
    import torch
    from transformers import BertTokenizerFast
    from repeat_rollout_quality import pair,mixed_noise,noise_hash,STATE_KEYS
    from run_frozen_online import append
    assert read(OUT/'preflight.json')['passed']
    model,projection=setup(args)
    tokenizer=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    path,sp=OUT/'test_pairs.jsonl',OUT/'test_states.jsonl'
    assert not path.exists() and not sp.exists()
    total=0;counts=defaultdict(int)
    with torch.inference_mode(),path.open('w',encoding='utf-8') as f,sp.open('w',encoding='utf-8') as sf:
        for p in read(OUT/'prefixes.json'):
            state_seed=211000000+SEED+p['index']*100003
            for offset in (0,8):
                sid=f"{p['prompt_id']}-{SEED}-{offset}"
                for rep in range(8):
                    fs=212000000+p['index']*100003+offset*17+rep*100000007
                    tape=mixed_noise(state_seed,fs,offset)
                    r=pair(model,torch.tensor([p['ids']],device='cuda'),tape,offset,projection)
                    if rep==0:
                        anchor={k:r[k] for k in STATE_KEYS if k in r}
                        counts[r['status']]+=1
                        state=dict(anchor,state_id=sid,prompt_id=p['prompt_id'],seed=SEED,
                            state_noise_seed=state_seed,offset=offset,split='test')
                        if r['status']=='paired': state['decision_features']=extra_features(model,state)
                        append(sf,state)
                        if r['status']!='paired': break
                    assert all(r[k]==v for k,v in anchor.items())
                    r.update(pair_id=f'{sid}-r{rep}',state_id=sid,prompt_id=p['prompt_id'],seed=SEED,
                        noise_seed=state_seed,offset=offset,replicate=rep,future_seed=fs,noise_sha256=noise_hash(tape))
                    for b in ('keep4','split22'):
                        r[b]['completion']=tokenizer.decode(r[b]['ids'][6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    append(f,r);total+=1
            if (p['index']+1)%16==0: print('Fresh sources',p['index']+1,'/',N,'paired futures',total,flush=True)
    dump(OUT/'test_collection.json',dict(passed=True,finished_utc=now(),attempted_states=2*N,
        state_status_counts=dict(counts),pairs=total,pairs_sha256=sha(path),states_sha256=sha(sp),
        collector_sha256=sha(Path(__file__)),locked_model_sha256=sha(OUT/'locked_model.json')))


def predict_test(args):
    from diagnose_decision_features import predict
    lock=verify();assert not (OUT/'test_labels.jsonl').exists()
    data=[r for r in rows(OUT/'test_states.jsonl') if r['status']=='paired']
    scores={name:predict(m,np.array([r['decision_features'][:len(m['mean'])] for r in data])) for name,m in lock['models'].items()}
    scores['heuristic']=np.array([-np.mean(r['features'][2:4]) for r in data])
    dump(OUT/'frozen_predictions.json',dict(created_utc=now(),test_labels_absent_when_frozen=True,
        states_sha256=sha(OUT/'test_states.jsonl'),locked_model_sha256=sha(OUT/'locked_model.json'),
        scores={r['state_id']:{name:float(s[i]) for name,s in scores.items()} for i,r in enumerate(data)},
        rankings={name:[data[i]['state_id'] for i in sorted(range(len(data)),key=lambda i:(-float(s[i]),data[i]['state_id']))]
                  for name,s in scores.items()}))
    print('Froze all predictions and rankings before label scoring:',len(data),flush=True)


def score(args):
    from score_rollout_quality import score as original
    verify();pred_sha=sha(OUT/'frozen_predictions.json')
    original(args)
    dump(OUT/'scoring_prediction_link.json',dict(passed=True,frozen_predictions_sha256=pred_sha,
        test_scoring_sha256=sha(OUT/'test_scoring.json'),completed_utc=now()))


def evaluate(args):
    from fit_averaged_labels import load
    lock=verify();link=read(OUT/'scoring_prediction_link.json');frozen=read(OUT/'frozen_predictions.json')
    assert link['frozen_predictions_sha256']==sha(OUT/'frozen_predictions.json')
    assert link['test_scoring_sha256']==sha(OUT/'test_scoring.json')
    data,excluded=load(OUT,'test');y=np.array([r['gain'] for r in data])
    masks={};scores={}
    for name in ('combined19','confidence14','heuristic'):
        scores[name]=np.array([frozen['scores'][r['state_id']][name] for r in data])
        rank=sorted(range(len(data)),key=lambda i:(-scores[name][i],data[i]['state_id']))
        masks[name]=np.zeros(len(data));masks[name][rank[:len(data)//2]]=1
    groups=defaultdict(list)
    for i,r in enumerate(data): groups[r['prompt_id']].append(i)
    keys=sorted(groups);counts=np.array([len(groups[k]) for k in keys])
    draws=np.random.default_rng(21192004).integers(0,len(keys),size=(8000,len(keys)))
    def interval(v,coverage=.95):
        sums=np.array([v[groups[k]].sum() for k in keys]);boot=sums[draws].sum(1)/counts[draws].sum(1)
        t=(1-coverage)/2
        return dict(estimate=float(v.mean()),interval=np.quantile(boot,[t,1-t]).tolist(),coverage=coverage)
    primary={}
    for name,baseline in [('vs_random',masks['combined19'].mean()),('vs_confidence14',masks['confidence14']),('vs_heuristic',masks['heuristic'])]:
        primary[name]=interval((masks['combined19']-baseline)*y,1-.05/3)
    gate=len(keys)>=128 and all(r['interval'][0]>0 for r in primary.values())
    arms={name:dict(mse=float(np.mean((p-y)**2)),correlation=float(np.corrcoef(p,y)[0,1]),
        vs_random=interval((masks[name]-masks[name].mean())*y)) for name,p in scores.items()}
    label_groups=defaultdict(list)
    for r in rows(OUT/'test_labels.jsonl'): label_groups[r['pair_id'].rsplit('-r',1)[0]].append(r)
    other={}
    for key in ('teacher_nll','visible_new_tokens','forward_calls','computed_candidates'):
        sign=1 if key=='teacher_nll' else -1
        delta=np.array([np.mean([sign*(r['keep4'][key]-r['split22'][key]) for r in label_groups[s['state_id']]]) for s in data])
        other[key]=dict(direction='keep4-minus-split22' if sign==1 else 'split22-minus-keep4',
            mean_all_split_change=interval(delta),
            combined19_vs_random=interval((masks['combined19']-masks['combined19'].mean())*delta),
            combined19_vs_confidence14=interval((masks['combined19']-masks['confidence14'])*delta))
    dump(OUT/'analysis.json',dict(completed_utc=now(),test_states=len(data),test_sources=len(keys),
        excluded_incomplete=excluded,primary=primary,advance_gate_passed=bool(gate),enough_sources=len(keys)>=128,
        arms=arms,exploratory_other_metrics=other,mean_all_split_gain=interval(y),constant_train_mean_mse=float(np.mean((y-lock['models']['combined19']['intercept'])**2)),
        locked_model_sha256=sha(OUT/'locked_model.json'),frozen_predictions_sha256=sha(OUT/'frozen_predictions.json'),
        labels_sha256=sha(OUT/'test_labels.jsonl'),
        limitations=['One frozen candidate, fresh sources, single intervention, optimized GPT-2 proxy.',
            'No online time advantage or independent human-quality conclusion.',
            'Conditional source bootstrap, fixed ranks, eight sampled futures.',
            'Paper-baseline gap remains unresolved.']))
    print(json.dumps(dict(primary=primary,arms=arms,advance_gate_passed=bool(gate),states=len(data),sources=len(keys)),indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','preflight','collect','predict_test','score','evaluate'])
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    args=ap.parse_args();args.out=OUT;args.split='test';OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.action](args)
