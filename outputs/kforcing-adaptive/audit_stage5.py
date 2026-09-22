"""Standard-library audit of frozen-noise replay, scores and comparison summaries."""
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import sys


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def main():
    root=Path(sys.argv[1])
    prior=root.parent/'lm1b-004'
    prompts=read(prior/'prefixes.json')
    pmap={p['id']:p for p in prompts}
    old={(r['prompt_id'],r['seed'],r['method']):r for r in lines(prior/'samples.jsonl')}
    samples=lines(root/'precision_samples.jsonl')
    expected=set(itertools.product(pmap,(617,619),('fp32','bf16','fp16')))
    assert len(samples)==len(expected)==768
    assert {(r['prompt_id'],r['seed'],r['precision']) for r in samples}==expected
    sample_map={}
    noise=read(root/'noise_manifest.json')
    assert len(noise)==64
    noise_map={(r['seed'],r['batch_index']):r for r in noise}
    assert set(noise_map)==set(itertools.product((617,619),range(32)))
    export=read(root/'replay_noise.json')
    assert len(export)==64
    assert {(r['seed'],r['batch_index']) for r in export}==set(noise_map)
    for r in export:
        assert r['shape']==[31,4,4,1] and len(r['values'])==496
        payload=struct.pack('<496f',*r['values'])
        assert hashlib.sha256(payload).hexdigest()==noise_map[r['seed'],r['batch_index']]['sha256']==r['sha256']
    for r in samples:
        key=r['prompt_id'],r['seed'],r['precision']
        sample_map[key]=r
        full=r['full_ids']
        assert len(full)==128 and full[:6]==pmap[r['prompt_id']]['ids']
        end=next((i+1 for i in range(6,128) if full[i]==102),128)
        assert r['ids']==full[:end] and r['new_tokens']==end-6
        assert r['noise_sha256']==noise_map[r['seed'],r['batch_index']]['sha256']
        assert r['method']=='fixed4_'+r['precision']
        if r['precision']=='fp32':
            assert full==old[r['prompt_id'],r['seed'],'fixed4']['full_ids']
    replay=read(root/'precision_checks.json')
    assert replay['fp32_exact_stage4_replays']==256
    diagnostics=read(root/'logit_diagnostics.json')
    assert len(diagnostics)==160
    assert {(r['prompt_id'],r['length']) for r in diagnostics}==set(itertools.product([p['id'] for p in prompts[:32]],(6,10,14,30,70)))
    for name,stored in replay['diagnostics'].items():
        group=[r[name] for r in diagnostics if name in r]
        assert stored['cases']==len(group)
        assert stored['positions']==sum(r['positions'] for r in group)
        assert stored['argmax_disagreements']==sum(r['argmax_disagreements'] for r in group)
        close(stored['max_abs_logit_difference'],max(r['max_abs_logit_difference'] for r in group))
    scores=lines(root/'scores.jsonl')
    methods=('ar','fixed4_fp32','fixed4_bf16','fixed4_fp16')
    variants=('completion','full_text','completion_bos')
    expected=set(itertools.product(pmap,(617,619),methods,variants))
    assert len(scores)==len(expected)==3072
    assert {(r['prompt_id'],r['seed'],r['method'],r['variant']) for r in scores}==expected
    groups=defaultdict(list)
    for r in scores:
        assert r['gpt2_scored_tokens']==max(0,r['gpt2_input_tokens']-1)
        assert r['excluded_short']==(r['gpt2_scored_tokens']==0)
        assert math.isfinite(r['gpt2_nll_sum']) and r['gpt2_nll_sum']>=0
        if r['excluded_short']:
            assert r['gpt2_nll_sum']==0
        if r['method']!='ar':
            source=sample_map[r['prompt_id'],r['seed'],r['method'].split('_')[-1]]
            assert r['text']==source['full_text' if r['variant']=='full_text' else 'completion']
            assert r['bert_new_tokens']==source['new_tokens']
        groups[r['variant']+'/'+r['method']].append(r)
    summary=read(root/'score_summary.json')
    assert set(summary)==set(groups)
    for key,rows in groups.items():
        n=sum(r['gpt2_scored_tokens'] for r in rows)
        nll=sum(r['gpt2_nll_sum'] for r in rows)/n
        s=summary[key]
        assert s['samples']==len(rows) and s['scored_tokens']==n
        assert s['excluded_short']==sum(r['excluded_short'] for r in rows)
        close(s['gpt2_nll'],nll)
        close(s['gpt2_gen_ppl'],math.exp(nll))
        close(s['mean_bert_new_tokens'],statistics.mean(r['bert_new_tokens'] for r in rows))
    for key,c in read(root/'comparisons.json').items():
        variant,comparison=key.split('/')
        a,b=comparison.split('_vs_')
        diff=summary[variant+'/'+a]['gpt2_nll']-summary[variant+'/'+b]['gpt2_nll']
        close(c['gpt2_nll_difference'],diff)
        close(c['gen_ppl_ratio'],math.exp(diff))
        assert c['prefixes']==c['clusters']==128 and c['cluster_unit']=='prefix'
        assert len(c['nll_difference95'])==2 and c['nll_difference95'][0]<=c['nll_difference95'][1]
        for nll,ratio in zip(c['nll_difference95'],c['gen_ppl_ratio95']):
            close(ratio,math.exp(nll))
    differences={}
    for precision in ('bf16','fp16'):
        total_full_equal=0
        total_visible_equal=0
        shared=[]
        for prompt,seed in itertools.product(pmap,(617,619)):
            a=sample_map[prompt,seed,'fp32']
            b=sample_map[prompt,seed,precision]
            total_full_equal+=a['full_ids']==b['full_ids']
            total_visible_equal+=a['ids']==b['ids']
            n=next((i for i,(x,y) in enumerate(zip(a['full_ids'][6:],b['full_ids'][6:])) if x!=y),122)
            shared.append(n)
        differences[precision]=dict(samples=256,identical_full_sequences=total_full_equal,
            identical_eos_truncated_sequences=total_visible_equal,mean_shared_new_token_prefix=statistics.mean(shared))
    checks=read(root/'scoring_checks.json')
    assert checks['previous_score_checks']==512 and checks['rows']==3072 and checks['model_sha256_verified']
    report=dict(passed=True,precision_samples=768,fp32_exact_replays=256,new_precision_outputs=512,
        identical_noise_batches=64,exported_noise_values_verified=True,same_context_cases=160,scored_sequences=1024,scoring_records=3072,
        prior_scoring_checks=512,trajectory_differences=differences,
        limitations='Audits records and aggregates; does not rerun model kernels or independently reconstruct bootstrap draws. No exact-paper or adaptive-performance claim.')
    (root/'integrity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
