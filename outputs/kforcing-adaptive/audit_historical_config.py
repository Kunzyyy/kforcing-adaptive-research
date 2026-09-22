"""Retrospective checkpoint-config/tokenizer equivalence checks, no old pickle execution."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from checkpoint_pickle_metadata import parse,items,plain
from core import load_model,AR,MTP

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/config-tokenizer'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()

def initialize():
    OUT.mkdir(exist_ok=False)
    files=['CONFIG_TOKENIZER_AUDIT_PLAN.md','audit_historical_config.py','checkpoint_pickle_metadata.py',
        'core.py','audit_official.py','upstream/models/pflm.py','upstream/models/autoregressive.py',
        'upstream/models/transformer.py','upstream/batch_inference_with_prefix.py',
        'results/baseline-016/prefixes.json','results/baseline-016/samples.jsonl',
        'results/baseline-016/author_probe/probe_cases.json']
    for kind in ['ar','pflm']:
        files += [f'results/public-history/historical_{kind}/{name}' for name in ['data.pkl','metadata_summary.json']]
    inputs=['work/pilot-data/vocab.txt','work/lm1b-data/test.parquet',
            'work/checkpoints/ar_best_lm1b.ckpt','work/checkpoints/pflm_lm1b_k4.ckpt']
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        code_input_hashes={name:sha(ROOT/name) for name in files},external_input_hashes={name:sha(Path(name)) for name in inputs},
        sources=1024,samples=2048,pflm_probe_cases=160,ar_probe_cases=40,
        generation_sources=32,logit_absolute_tolerance=1e-6,
        retrospective=True,preliminary_decode_check_already_observed=True,new_test=False))
    print('Fixed historical config/tokenizer compatibility audit inputs.',flush=True)

def verify():
    manifest=read(OUT/'manifest.json')
    for name,h in manifest['code_input_hashes'].items(): assert sha(ROOT/name)==h,name
    for name,h in manifest['external_input_hashes'].items(): assert sha(Path(name))==h,name
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    return manifest

def historical_tokenizer(kind):
    from transformers import BertTokenizer
    data=parse((ROOT/f'results/public-history/historical_{kind}/data.pkl').read_bytes())
    symbolic=data['hyper_parameters']['tokenizer']; state=symbolic.state
    assert symbolic.kind=='transformers.models.bert.tokenization_bert BertTokenizer'
    vocab=items(state['vocab'])
    basic=state['basic_tokenizer'].state
    settings={k:basic[k] for k in ['do_lower_case','tokenize_chinese_chars','strip_accents','do_split_on_punc']}
    settings.update(do_basic_tokenize=state['do_basic_tokenize'],
        clean_up_tokenization_spaces=state['clean_up_tokenization_spaces'],
        model_max_length=state['model_max_length'],padding_side=state['padding_side'],truncation_side=state['truncation_side'],
        split_special_tokens=state['split_special_tokens'])
    never=basic['never_split']
    assert never.kind=='__builtin__ set' and never.args==([],)
    settings['never_split']=[]
    assert plain(state['_special_tokens_map'])=={'bos_token':'[CLS]','eos_token':'[SEP]',
        'unk_token':'[UNK]','sep_token':'[SEP]','pad_token':'[PAD]','cls_token':'[CLS]',
        'mask_token':'[MASK]','additional_special_tokens':[]}
    settings.update(plain(state['_special_tokens_map']))
    tokenizer=BertTokenizer(vocab_file='work/pilot-data/vocab.txt',**settings)
    tokenizer.wordpiece_tokenizer.max_input_chars_per_word=state['wordpiece_tokenizer'].state['max_input_chars_per_word']
    assert tokenizer.get_vocab()==vocab
    added={str(k):plain(v.state) for k,v in state['_added_tokens_decoder'].items()}
    reconstructed={str(k):v.__getstate__() for k,v in tokenizer.added_tokens_decoder.items()}
    assert added==reconstructed
    return tokenizer,dict(vocab=sorted(vocab.items(),key=lambda x:x[1]),settings=settings,
        max_input_chars_per_word=tokenizer.wordpiece_tokenizer.max_input_chars_per_word,
        added_tokens=added,original_init_kwargs=plain(state['init_kwargs']))

def tokenizer():
    import pyarrow.parquet as pq
    import transformers
    from transformers import BertTokenizerFast
    verify(); assert not (OUT/'tokenizer_checks.json').exists()
    slow,description=historical_tokenizer('pflm'); ar_slow,ar_description=historical_tokenizer('ar')
    fast=BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt',do_lower_case=True)
    assert slow.get_vocab()==ar_slow.get_vocab()==fast.get_vocab()
    prefixes=read(ROOT/'results/baseline-016/prefixes.json')
    texts=pq.read_table('work/lm1b-data/test.parquet',columns=['text'])['text'].to_pylist()
    encoding=[]; decoding=[]
    for p in prefixes:
        text=texts[p['source_row']]
        assert hashlib.sha256(text.encode()).hexdigest()==p['source_sha256']
        a=slow.encode(text,add_special_tokens=True,truncation=True,max_length=128)
        b=fast.encode(text,add_special_tokens=True,truncation=True,max_length=128)
        c=ar_slow.encode(text,add_special_tokens=True,truncation=True,max_length=128)
        encoding.append(dict(prompt_id=p['prompt_id'],historical_ids=a,current_ids=b,ar_historical_ids=c,
            full_ids_equal=a==b==c,first_six_equal=a[:6]==b[:6]==c[:6]==p['ids']))
    for r in rows(ROOT/'results/baseline-016/samples.jsonl'):
        a=slow.decode(r['ids'][6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
        b=fast.decode(r['ids'][6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
        c=ar_slow.decode(r['ids'][6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)
        decoding.append(dict(prompt_id=r['prompt_id'],method=r['method'],historical_text=a,current_text=b,
            ar_historical_text=c,stored_text=r['completion'],all_equal=a==b==c==r['completion']))
    descriptions=dict(pflm=description,ar=ar_description,current_special_map=fast.special_tokens_map,
        historical_bos_id=slow.bos_token_id,historical_eos_id=slow.eos_token_id,
        current_bos_id=fast.bos_token_id,current_eos_id=fast.eos_token_id,
        actual_current_prefix_id=101,actual_current_stop_id=102)
    dump(OUT/'tokenizer_descriptions.json',descriptions)
    dump(OUT/'encoding_records.json',encoding); dump(OUT/'decoding_records.json',decoding)
    result=dict(vocabulary_entries=len(slow.get_vocab()),vocabularies_identical=True,
        source_sentences=len(encoding),encoding_differences=sum(not x['full_ids_equal'] for x in encoding),
        prefix_differences=sum(not x['first_six_equal'] for x in encoding),
        generated_outputs=len(decoding),decode_differences=sum(not x['all_equal'] for x in decoding),
        bos_eos_attribute_difference='Current fast tokenizer attributes are None; experiments explicitly use historical-equivalent IDs 101/102.',
        transformers=transformers.__version__,original_library_version_recreated=False,
        encoding_sha256=sha(OUT/'encoding_records.json'),decoding_sha256=sha(OUT/'decoding_records.json'))
    result['passed']=result['encoding_differences']==result['prefix_differences']==result['decode_differences']==0
    dump(OUT/'tokenizer_checks.json',result); print(json.dumps(result,indent=2),flush=True)

def instantiate(kind,device):
    data=read(ROOT/f'results/public-history/historical_{kind}/metadata_summary.json')
    fields=data['hyper_parameters']['config']['model']
    # Only model fields go into OmegaConf; interpolation/resume paths are never resolved or run.
    assert all(not isinstance(v,str) or '${' not in v for v in fields.values())
    filename='ar_best_lm1b.ckpt' if kind=='ar' else 'pflm_lm1b_k4.ckpt'
    ordinary=load_model(Path('work/checkpoints')/filename,kind,device)
    cfg=OmegaConf.create({'model':fields})
    restored=(AR(cfg,vocab_size=ordinary.vocab_size,mask_index=103) if kind=='ar' else
              MTP(cfg,vocab_size=ordinary.vocab_size,mask_index=103,max_k=fields['max_k']))
    restored.load_state_dict(ordinary.state_dict(),strict=True); restored.to(device).eval()
    baseline=OmegaConf.to_container(ordinary.config)['model']
    difference={k:dict(current=baseline.get(k,'<absent>'),historical=fields.get(k,'<absent>'))
                for k in sorted(set(baseline)|set(fields)) if baseline.get(k,'<absent>')!=fields.get(k,'<absent>')}
    return ordinary,restored,difference

def compare(a,b):
    assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    return dict(max_abs_difference=float((a-b).abs().max()),argmax_disagreements=int((a.argmax(-1)!=b.argmax(-1)).sum()),
        shape=list(a.shape),bitwise_equal=torch.equal(a,b))

@torch.inference_mode()
def model():
    from audit_official import official_cli,quiet_sampler
    verify(); assert not (OUT/'model_checks.json').exists()
    cases=read(ROOT/'results/baseline-016/author_probe/probe_cases.json')['cases']
    grouped=defaultdict(list)
    for case in cases: grouped[case['context_length'],case['k']].append(case)
    config_diff={}; records=[]; caches=[]; generations=[]
    official=official_cli(); quiet_sampler()
    for kind in ['pflm','ar']:
        current,restored,difference=instantiate(kind,'cuda'); config_diff[kind]=difference
        for (length,k),batch in sorted(grouped.items()):
            if kind=='ar' and k!=4: continue
            context=torch.tensor([r['context_ids'] for r in batch],device='cuda')
            noise=torch.tensor([r['noise'] for r in batch],device='cuda',dtype=torch.float32).unsqueeze(-1)
            tau=torch.ones(len(batch),1,1,device='cuda')
            if kind=='pflm': a=current(context,noise,tau,mode='inference'); b=restored(context,noise,tau,mode='inference')
            else: a=current.infer_next_token(context)[0][:,-1:]; b=restored.infer_next_token(context)[0][:,-1:]
            for i,r in enumerate(batch):
                check=compare(a[i],b[i]); check.update(model=kind,prompt_id=r['prompt_id'],context_length=length,k=k)
                if kind=='pflm':
                    check['published_probe_argmax_match']=a[i].argmax(-1).tolist()==r['expected_argmax']
                    assert check['published_probe_argmax_match']
                records.append(check)
        cache_a=cache_b=None; last=0
        for length in [6,10,14,30,70]:
            batch=grouped[length,4]
            context=torch.tensor([r['context_ids'] for r in batch],device='cuda')
            noise=torch.tensor([r['noise'] for r in batch],device='cuda',dtype=torch.float32).unsqueeze(-1)
            if kind=='pflm':
                a,cache_a=current(context,noise,torch.ones(8,1,1,device='cuda'),mode='inference',kv_caches=cache_a,return_kv=True)
                b,cache_b=restored(context,noise,torch.ones(8,1,1,device='cuda'),mode='inference',kv_caches=cache_b,return_kv=True)
            else:
                a,cache_a=current.infer_next_token(context[:,last:],kv_caches=cache_a,offset=last)
                b,cache_b=restored.infer_next_token(context[:,last:],kv_caches=cache_b,offset=last)
            last=length
            difference=max(float((aa-bb).abs().max()) for x,y in zip(cache_a,cache_b) for aa,bb in zip(x,y))
            caches.append(dict(model=kind,context_length=length,output=compare(a,b),max_cache_abs_difference=difference))
        prefixes=read(ROOT/'results/baseline-016/prefixes.json')[:32]
        method='ar' if kind=='ar' else 'fixed4'
        old={r['prompt_id']:r for r in rows(ROOT/'results/baseline-016/samples.jsonl') if r['method']==method}
        for start in range(0,32,4):
            batch=prefixes[start:start+4]; context=torch.tensor([r['ids'] for r in batch],device='cuda')
            seed=16120000+(start//4)*101
            sequences=[]
            for instance in [current,restored]:
                torch.manual_seed(seed)
                full=(official.generate_ar_cached(instance,context,122) if kind=='ar' else
                    instance.sample_next_k_tokens_with_kv_caches(context,torch.ones(4,device='cuda'),122,k=4,frequency_penalty=0.))
                sequences.append(full.tolist())
            for r,a,b in zip(batch,*sequences):
                generations.append(dict(model=kind,prompt_id=r['prompt_id'],seed=seed,
                    current_ids=a,historical_config_ids=b,all_128_tokens_match=a==b==old[r['prompt_id']]['full_ids']))
        print('Config equivalence completed:',kind,flush=True)
        del current,restored; torch.cuda.empty_cache()
    dump(OUT/'config_differences.json',config_diff); dump(OUT/'model_records.json',records)
    dump(OUT/'cache_records.json',caches); dump(OUT/'generation_records.json',generations)
    result=dict(fixed_input_cases=len(records),pflm_candidate_positions=400,ar_last_positions=40,
        max_abs_logit_difference=max(r['max_abs_difference'] for r in records),
        argmax_disagreements=sum(r['argmax_disagreements'] for r in records),
        all_bitwise_equal=all(r['bitwise_equal'] for r in records),
        cache_batches=len(caches),max_cache_abs_difference=max(r['max_cache_abs_difference'] for r in caches),
        max_cached_output_abs_difference=max(r['output']['max_abs_difference'] for r in caches),
        generation_pairs=len(generations),full_sequence_mismatches=sum(not r['all_128_tokens_match'] for r in generations),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        source_or_hardware_recreated=False,manifest_sha256=sha(OUT/'manifest.json'),
        record_hashes={name:sha(OUT/name) for name in ['config_differences.json','model_records.json','cache_records.json','generation_records.json']})
    tolerance=read(OUT/'manifest.json')['logit_absolute_tolerance']
    result['passed']=(result['max_abs_logit_difference']<=tolerance and result['argmax_disagreements']==0
        and result['max_cache_abs_difference']<=tolerance and result['max_cached_output_abs_difference']<=tolerance
        and result['full_sequence_mismatches']==0)
    dump(OUT/'model_checks.json',result); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['initialize','tokenizer','model'])
    args=ap.parse_args(); globals()[args.mode]()
