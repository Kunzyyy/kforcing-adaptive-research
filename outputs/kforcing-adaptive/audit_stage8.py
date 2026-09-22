"""CPU audit of isolation, frozen predictions, matched budgets and primary bootstrap."""
from collections import defaultdict,Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b):
    assert math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def main():
    root=Path(sys.argv[1])
    manifest=read(root/'manifest.json')
    sources=read(root/'sources.json')
    contexts=read(root/'contexts.json')
    assert sha(root/'sources.json')==manifest['sources_sha256']
    assert sha(root/'contexts.json')==manifest['contexts_sha256']
    assert sha(root/'frozen_model.json')==manifest['model_sha256']==sha(root.parent/'learned-007/locked_model.json')
    assert sha(root/'projection.npy')==manifest['projection_sha256']==sha(root.parent/'learned-007/projection.npy')
    assert sha(root.parent.parent/'learn_local_gain.py')==manifest['feature_code_sha256']
    assert len(sources)==384 and len(contexts)==512
    assert len({s['source_id'] for s in sources})==384
    assert len({s['source_sha256'] for s in sources})==384
    assert len({s['source_row'] for s in sources})==384
    assert len({tuple(s['reference_ids'][:6]) for s in sources})==384
    old=read(root.parent/'lm1b-004/prefixes.json')
    for group in read(root.parent/'learned-007/prefixes.json').values():
        old+=group
    assert not {p['source_row'] for p in old}&{s['source_row'] for s in sources}
    assert not {p['source_sha256'] for p in old}&{s['source_sha256'] for s in sources}
    previous={tuple(p['ids'][:6]) for p in old}
    for folder in ('pilot-001','benefit-002'):
        for group in read(root.parent/folder/'prefixes.json').values():
            previous.update(tuple(p['ids'][:6]) for p in group)
    previous.update(tuple(p['ids'][:6]) for p in read(root.parent/'current-003/fresh_prefixes.json'))
    assert not previous&{tuple(s['reference_ids'][:6]) for s in sources}
    smap={s['source_id']:s for s in sources}
    pmap={p['prompt_id']:p for p in contexts}
    assert len(pmap)==512 and Counter(p['length'] for p in contexts)=={6:256,16:128,32:128}
    for p in contexts:
        s=smap[p['source_id']]
        assert p['ids']==s['reference_ids'][:p['length']]
        assert len(p['ids'])==p['length'] and p['ids'][0]==101 and 102 not in p['ids']
        assert p['source_index']==s['source_index']
    raw=rows(root/'windows.jsonl')
    selected=rows(root/'selections.jsonl')
    expected=set(itertools.product(pmap,manifest['seeds']))
    assert len(raw)==len(selected)==len(expected)==4096
    assert {(r['prompt_id'],r['seed']) for r in raw}==expected
    selected_map={(r['prompt_id'],r['seed']):r for r in selected}
    assert set(selected_map)==expected
    model=read(root/'frozen_model.json')
    mean,std,coef=(np.array(model[k]) for k in ('mean','std','coef'))
    for r in raw:
        s=selected_map[r['prompt_id'],r['seed']]
        assert all(s[k]==v for k,v in r.items())
        p=pmap[r['prompt_id']]
        assert r['source_id']==p['source_id'] and r['length']==p['length'] and r['prefix']==p['ids']
        assert r['noise_seed']==80800000+r['seed']+p['source_index']*100003
        assert len(r['features'])==82 and all(math.isfinite(x) for x in r['features'])
        assert r['features'][-4:]==r['noise'] and all(0<=x<1 for x in r['noise'])
        close(float(((np.array(r['features'])-mean)/std)@coef+model['intercept']),r['score'])
        for name in ('fixed4','split22','ar_reference'):
            assert len(r[name])==4 and all(isinstance(t,int) and 0<=t<30522 for t in r[name])
        assert r['gain']==sum(a==t for a,t in zip(r['split22'],r['ar_reference']))-sum(a==t for a,t in zip(r['fixed4'],r['ar_reference']))
        assert r['no_eos_all_branches']==all(102 not in r[k] for k in ('fixed4','split22','ar_reference'))
        assert r['teacher_nll_fixed4']>=0 and r['teacher_nll_split22']>=0
        close(r['nll_gain'],r['teacher_nll_fixed4']-r['teacher_nll_split22'])
        assert s['locked_threshold']==(r['score']>model['dev_threshold'])
    # The two real-context lengths share noise, but are clustered as one source sentence.
    for source in sources[256:]:
        for seed in manifest['seeds']:
            a=selected_map[f"{source['source_id']}-L16",seed]
            b=selected_map[f"{source['source_id']}-L32",seed]
            assert a['noise']==b['noise']
    analysis=read(root/'analysis.json')
    for length in (6,16,32):
        group=[r for r in selected if r['length']==length]
        gains=np.array([r['gain'] for r in group])
        nll=np.array([r['nll_gain'] for r in group])
        base=np.array([sum(a==t for a,t in zip(r['fixed4'],r['ar_reference'])) for r in group])
        summary=analysis['by_length'][str(length)]
        assert summary['windows']==len(group)
        close(summary['fixed4_match'],base.mean()/4)
        close(summary['all_split_match'],np.mean(base+gains)/4)
        for name,s in summary['policies'].items():
            mask=np.array([r[name] for r in group])
            assert s['split_windows']==mask.sum()
            if name.endswith('_half'):
                assert mask.sum()==len(group)//2
            close(s['split_fraction'],mask.mean())
            close(s['match_fraction'],np.mean(base+mask*gains)/4)
            close(s['versus_random_expectation']['mean'],np.mean((mask.astype(float)-mask.mean())*gains)/4)
            close(s['teacher_nll_reduction_vs_random_per_token']['mean'],np.mean((mask.astype(float)-mask.mean())*nll)/4)
        for name,score in (('learned_half',lambda r:r['score']),('confidence_half',lambda r:-sum(r['features'][2:4])/2),('oracle_half',lambda r:r['gain'])):
            order=sorted(group,key=lambda r:(-score(r),r['prompt_id'],r['seed']))
            keys={(r['prompt_id'],r['seed']) for r in order[:len(group)//2]}
            assert {(r['prompt_id'],r['seed']) for r in group if r[name]}==keys
    for name,lengths,n_sources in (('short',(6,),256),('long',(16,32),128)):
        clusters=defaultdict(list)
        for r in selected:
            if r['length'] in lengths:
                clusters[r['source_id']].append((r['learned_half']-.5)*r['gain']/4)
        assert len(clusters)==n_sources
        values=np.array([np.mean(clusters[k]) for k in sorted(clusters)])
        # Independent vectorized reconstruction of primary bootstrap draws.
        draws=np.random.default_rng(8081704).integers(0,n_sources,size=(4000,n_sources))
        ci=np.quantile(values[draws].mean(axis=1),[.0125,.9875])
        reported=analysis['primary'][name]
        close(reported['mean'],values.mean())
        np.testing.assert_allclose(reported['interval'],ci,atol=1e-12,rtol=1e-12)
        assert reported['coverage']==.975 and reported['passes']==bool(ci[0]>0)
    checks=read(root/'collection_checks.json')
    assert checks['windows']==4096 and checks['model_sha256']==manifest['model_sha256']
    assert checks['first_two_disagreements']==sum(sum(a!=b for a,b in zip(r['fixed4'][:2],r['split22'][:2])) for r in raw)
    assert read(root/'nll_crosscheck.json')['passed']
    assert analysis['local_confirmation_passed']==all(v['passes'] for v in analysis['primary'].values())
    report=dict(passed=True,new_source_sentences=384,contexts=512,windows=4096,
        prior_source_row_overlap=0,prior_sentence_hash_overlap=0,prior_six_token_prefix_overlap=0,
        model_and_projection_unchanged=True,all_predictions_recomputed=True,
        primary_bootstrap_independently_reconstructed=True,long_context_noise_pairing_verified=True,
        nll_single_sequence_crosscheck_branches=24,local_confirmation_passed=analysis['local_confirmation_passed'],
        limitation='No independent full model rerun; NLL crosscheck covers selected rows only. Source-document independence and online quality/performance remain untested.')
    (root/'integrity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
