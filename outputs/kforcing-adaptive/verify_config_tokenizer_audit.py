"""Check saved evidence, input preservation and static relevance of config fields."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf
from core import load_model,AR,MTP
import torch

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/config-tokenizer'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    m=read(OUT/'manifest.json'); tokenizer=read(OUT/'tokenizer_checks.json'); model=read(OUT/'model_checks.json')
    for name,h in m['code_input_hashes'].items(): assert sha(ROOT/name)==h,name
    for name,h in m['external_input_hashes'].items(): assert sha(Path(name))==h,name
    for name,h in read(ROOT/'historical_ema_file_hashes.json').items():
        if name not in {'README.md','provenance.json'}: assert sha(ROOT/name)==h,name
    assert tokenizer['passed'] and model['passed']
    assert sha(OUT/'encoding_records.json')==tokenizer['encoding_sha256']
    assert sha(OUT/'decoding_records.json')==tokenizer['decoding_sha256']
    for name,h in model['record_hashes'].items(): assert sha(OUT/name)==h
    descriptions=read(OUT/'tokenizer_descriptions.json')
    local_vocab=Path('work/pilot-data/vocab.txt').read_text(encoding='utf-8').splitlines()
    for kind in ['ar','pflm']:
        assert descriptions[kind]['vocab']==[[text,i] for i,text in enumerate(local_vocab)]
    assert len(local_vocab)==30522
    prefixes={r['prompt_id']:r for r in read(ROOT/'results/baseline-016/prefixes.json')}
    encoding=read(OUT/'encoding_records.json'); decoding=read(OUT/'decoding_records.json')
    assert len(encoding)==1024 and {r['prompt_id'] for r in encoding}==set(prefixes)
    for r in encoding:
        assert r['historical_ids']==r['current_ids']==r['ar_historical_ids']
        assert r['historical_ids'][:6]==prefixes[r['prompt_id']]['ids']
        assert r['full_ids_equal'] and r['first_six_equal']
    stored={(r['prompt_id'],r['method']):r for r in rows(ROOT/'results/baseline-016/samples.jsonl')}
    assert len(decoding)==2048 and {(r['prompt_id'],r['method']) for r in decoding}==set(stored)
    for r in decoding:
        assert r['historical_text']==r['current_text']==r['ar_historical_text']==r['stored_text']==stored[r['prompt_id'],r['method']]['completion']
        assert r['all_equal']
    cases=read(ROOT/'results/baseline-016/author_probe/probe_cases.json')['cases']
    expected={('pflm',r['prompt_id'],r['context_length'],r['k']) for r in cases}
    expected|={('ar',r['prompt_id'],r['context_length'],4) for r in cases if r['k']==4}
    records=read(OUT/'model_records.json')
    assert len(records)==200 and {(r['model'],r['prompt_id'],r['context_length'],r['k']) for r in records}==expected
    assert all(r['max_abs_difference']==0 and r['argmax_disagreements']==0 and r['bitwise_equal'] for r in records)
    assert sum(r['shape'][0] for r in records if r['model']=='pflm')==400
    assert sum(r['shape'][0] for r in records if r['model']=='ar')==40
    caches=read(OUT/'cache_records.json')
    assert len(caches)==10 and {(r['model'],r['context_length']) for r in caches}=={(k,n) for k in ['ar','pflm'] for n in [6,10,14,30,70]}
    assert all(r['max_cache_abs_difference']==0 and r['output']['max_abs_difference']==0 for r in caches)
    generations=read(OUT/'generation_records.json'); first=list(prefixes)[:32]
    assert len(generations)==64 and {(r['model'],r['prompt_id']) for r in generations}=={(k,p) for k in ['ar','pflm'] for p in first}
    for r in generations:
        method='ar' if r['model']=='ar' else 'fixed4'
        assert len(r['current_ids'])==128 and r['current_ids']==r['historical_config_ids']==stored[r['prompt_id'],method]['full_ids']
        assert r['seed']==16120000+(first.index(r['prompt_id'])//4)*101
        assert r['all_128_tokens_match']
    # Independent static check: cond_dim is accepted but never read by these constructors.
    tree=ast.parse((ROOT/'upstream/models/transformer.py').read_text(encoding='utf-8'))
    for name in ['DDiTBlock','DDitFinalLayer']:
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        constructor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assert 'cond_dim' in [a.arg for a in constructor.args.args]
        assert not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='cond_dim' for n in ast.walk(constructor))
    scale_reads=[]
    for filename in ['pflm.py','autoregressive.py','transformer.py']:
        t=ast.parse((ROOT/'upstream/models'/filename).read_text(encoding='utf-8'))
        for node in ast.walk(t):
            if isinstance(node,ast.Attribute) and node.attr=='scale_by_sigma' and isinstance(node.ctx,ast.Load):
                scale_reads.append(dict(file=filename,expression=ast.unparse(node)))
    assert scale_reads==[{'file':'autoregressive.py','expression':'config.model.scale_by_sigma'}]
    constructions={}
    for kind,filename in [('ar','ar_best_lm1b.ckpt'),('pflm','pflm_lm1b_k4.ckpt')]:
        fields=read(ROOT/f'results/public-history/historical_{kind}/metadata_summary.json')['hyper_parameters']['config']['model']
        modern=load_model(Path('work/checkpoints')/filename,kind,'cpu')
        old=(AR(OmegaConf.create({'model':fields}),30522,103) if kind=='ar' else
             MTP(OmegaConf.create({'model':fields}),30522,103,4))
        old.load_state_dict(modern.state_dict(),strict=True); old.eval()
        assert old.config.model.cond_dim==128 and modern.config.model.cond_dim==1024
        assert old.scale_by_sigma is True and modern.scale_by_sigma is False
        assert all(not module.training for module in old.modules())
        expected_dropout=.1 if kind=='ar' else 0.
        assert all(block.dropout==expected_dropout for block in old.blocks)
        assert all(torch.equal(v,modern.state_dict()[name]) for name,v in old.state_dict().items())
        constructions[kind]=dict(cond_dim=128,scale_by_sigma=True,dropout=expected_dropout,
            all_modules_eval=True,all_weights_identical=True,strict_load=True)
        del old,modern
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(OUT/'manifest.json'),verifier_sha256=sha(Path(__file__)),
        tokenizer_checks_sha256=sha(OUT/'tokenizer_checks.json'),model_checks_sha256=sha(OUT/'model_checks.json'),
        previous_scientific_files_unchanged=True,record_counts=dict(vocab=30522,source_encodings=1024,decodes=2048,
            fixed_input_cases=200,cache_batches=10,generation_pairs=64),
        static_cond_dim_unused_in_blocks_and_head=True,scale_by_sigma_read_expressions=scale_reads,
        independent_cpu_construction=constructions,
        limitations='Record/structure/source integrity audit; GPU forward comparisons and tokenization runs were not independently reimplemented. Current source/library only, no original training environment validation.')
    destination=OUT/'integrity_audit.json'; assert not destination.exists()
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
