"""Independent metadata/storage, source preservation, sample and statistic checks."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import zipfile
import zlib
import numpy as np
import torch
from transformers import BertTokenizerFast, GPT2TokenizerFast
from checkpoint_pickle_metadata import parse, items, tensor, Symbol

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/historical-ema'; PUB=ROOT/'results/public-history'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8*1024*1024): h.update(data)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    manifest=read(OUT/'manifest.json')
    for name,h in manifest['input_code_hashes'].items(): assert sha(ROOT/name)==h,name
    prior=read(ROOT/'training_target_file_hashes.json')
    for name,h in prior.items():
        if name not in {'README.md','provenance.json'}: assert sha(ROOT/name)==h,name
    # Basic protocol-2 structures and inert unknown GLOBAL/REDUCE/BUILD behavior.
    fixture={'a':[1,2,None,True,False,3.5],'b':('text',-42),'c':{'x':9}}
    assert parse(pickle.dumps(fixture,protocol=2))==fixture
    symbolic=parse(b'\x80\x02cnot_a_real_module\nNotARealClass\n)R.')
    assert isinstance(symbolic,Symbol) and symbolic.kind=='not_a_real_module NotARealClass'
    try: parse(b'\x80\x02Punsupported\n.')
    except ValueError: pass
    else: raise AssertionError('Unsupported opcode should fail')
    model_checks={}
    for kind,filename in [('ar','ar_best_lm1b.ckpt'),('pflm','pflm_lm1b_k4.ckpt')]:
        extraction=read(PUB/f'historical_{kind}/ema_extraction.json')
        assert sha(PUB/f'historical_{kind}/data.pkl')==extraction['source_metadata_sha256']
        assert sha(Path(extraction['checkpoint']))==manifest['ema_checkpoint_hashes'][kind]==extraction['checkpoint_sha256']
        ema=torch.load(extraction['checkpoint'],map_location='cpu',weights_only=True)['state_dict']
        old=torch.load(Path('work/checkpoints')/filename,map_location='cpu',weights_only=True)['state_dict']
        with zipfile.ZipFile(Path('work/checkpoints')/filename) as z:
            raw=z.read(next(n for n in z.namelist() if n.endswith('/data.pkl')))
            symbolic=items(parse(raw)['state_dict'])
            for key,value in old.items():
                descriptor=tensor(symbolic[key])
                assert descriptor['shape']==list(value.shape) and descriptor['stride']==list(value.stride())
                member=next(n for n in z.namelist() if n.endswith('/data/'+descriptor['storage_key']))
                actual=value.contiguous().numpy().tobytes()
                assert actual==z.read(member)
        assert set(old)==set(ema)
        assert torch.equal(old['backbone.rotary_emb.inv_freq'],ema['backbone.rotary_emb.inv_freq'])
        changes=[]
        for item in extraction['tensors']:
            name='backbone.'+item['name']; value=ema[name]; raw=value.numpy().tobytes()
            assert hashlib.sha256(raw).hexdigest()==item['sha256'] and zlib.crc32(raw)==item['crc32']
            assert list(value.shape)==item['shape'] and torch.isfinite(value).all()
            changes.append(float((value-old[name]).abs().max()))
        # Record continuous-range overhead honestly; it may include non-EMA members.
        span_bytes=extraction['span_end_exclusive']-extraction['span_start']
        tensor_bytes=sum(t['bytes'] for t in extraction['tensors'])
        model_checks[kind]=dict(parameter_tensors=len(changes),changed_parameters=sum(x>0 for x in changes),
            maximum_absolute_parameter_change=max(changes),buffer_identical=True,
            ema_tensor_bytes=tensor_bytes,downloaded_span_bytes=span_bytes,range_overhead_bytes=span_bytes-tensor_bytes,
            local_metadata_parser_matches_loaded_tensors=True,extracted_tensor_sha256_and_crc_verified=True)
    # Remote metadata history agrees with complete local main ancestry.
    github=read(PUB/'github_commits.json'); history=read(PUB/'git_history.json')
    assert github['status']==200 and history['shallow']=='false'
    assert {c['sha'] for c in github['data']}=={c['commit'] for c in history['commits']}
    assert 'rel="next"' not in github['headers'].get('link','')
    hf=read(PUB/'hf_commits.json'); trees=read(PUB/'hf_historical_trees.json')
    assert {c['id'] for c in hf['data']}==set(trees)
    assert all(v['status']==200 and 'rel="next"' not in v['headers'].get('link','') for v in trees.values())
    prefixes=read(OUT/'prefixes.json'); ids=[p['prompt_id'] for p in prefixes]
    assert prefixes==read(ROOT/'results/baseline-016/prefixes.json')[:128]
    bert=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    gpt=GPT2TokenizerFast.from_pretrained('work/gpt2-large',local_files_only=True)
    samples=rows(OUT/'samples.jsonl'); scores=rows(OUT/'scores.jsonl')
    key=lambda r:(r['prompt_id'],r['method'],r['weights'])
    bysample={key(r):r for r in samples}; byscore={key(r):r for r in scores}
    expected={(i,m,w) for i in ids for m in ['ar','fixed4'] for w in ['ordinary','historical_ema']}
    assert len(samples)==len(scores)==512 and set(bysample)==set(byscore)==expected
    old={(r['prompt_id'],r['method']):r for r in rows(ROOT/'results/baseline-016/samples.jsonl')}
    for p in prefixes:
        i=ids.index(p['prompt_id'])
        for method in ['ar','fixed4']:
            for weight in ['ordinary','historical_ema']:
                r=bysample[p['prompt_id'],method,weight]; s=byscore[key(r)]
                assert len(r['full_ids'])==128 and r['full_ids'][:6]==p['ids']
                assert r['sampler_seed']==16120000+(i//4)*101 and r['batch_index']==i//4
                stop=next((n+1 for n in range(6,128) if r['full_ids'][n]==102),128)
                assert r['ids']==r['full_ids'][:stop]
                assert r['new_tokens']==stop-6 and r['stopped_eos']==(r['ids'][-1]==102)
                assert r['completion']==bert.decode(r['ids'][6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
                length=len(gpt.encode(r['completion'],add_special_tokens=False))
                assert s['gpt2_input_tokens']==length and s['gpt2_scored_tokens']==max(0,length-1)
                assert s['excluded_short']==(length<2)
                if weight=='ordinary': assert r['full_ids']==old[p['prompt_id'],method]['full_ids']
    analysis=read(OUT/'analysis.json'); independent={}
    for method in ['ar','fixed4']:
        values={w:np.asarray([[byscore[i,method,w]['gpt2_nll_sum'],byscore[i,method,w]['gpt2_scored_tokens']]
                             for i in ids],dtype=np.float64) for w in ['ordinary','historical_ema']}
        nlls={w:float(v[:,0].sum()/v[:,1].sum()) for w,v in values.items()}
        for w,avg in nlls.items(): assert math.isclose(math.exp(avg),analysis['methods'][method][w]['gen_ppl'],rel_tol=1e-12)
        rng=np.random.default_rng(21212026); boot=[]
        for _ in range(4000):
            selected=rng.integers(0,128,128)
            totals={w:v[selected].sum(axis=0) for w,v in values.items()}
            boot.append(totals['historical_ema'][0]/totals['historical_ema'][1]-totals['ordinary'][0]/totals['ordinary'][1])
        interval=np.percentile(boot,[2.5,97.5]); delta=nlls['historical_ema']-nlls['ordinary']
        published=analysis['methods'][method]['ema_minus_ordinary_token_nll']
        assert math.isclose(delta,published['estimate'],abs_tol=1e-12)
        assert np.allclose(interval,published['interval95'],rtol=0,atol=1e-12)
        independent[method]=dict(delta=delta,interval95=interval.tolist())
    scoring=read(OUT/'scoring_checks.json'); generation=read(OUT/'generation_checks.json')
    assert scoring['passed'] and generation['passed'] and len(generation['ordinary_replay'])==8
    assert scoring['scores_sha256']==sha(OUT/'scores.jsonl') and generation['samples_sha256']==sha(OUT/'samples.jsonl')
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),analysis_sha256=sha(OUT/'analysis.json'),
        verifier_sha256=sha(Path(__file__)),prior_scientific_files_unchanged=True,
        model_checks=model_checks,source_history_commits=27,hf_revisions=11,sample_and_score_checks=512,
        independent_statistics=independent,remote_pickle_code_executed=False,
        limitations='Does not verify full historical LFS digest, author EMA registration order, original hardware, or paper checkpoint identity. GPT2 forward not independently reimplemented; built-in loss and historical controls checked.')
    assert not (OUT/'integrity_audit.json').exists()
    (OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
