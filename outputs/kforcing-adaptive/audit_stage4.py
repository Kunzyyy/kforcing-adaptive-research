"""Independent standard-library integrity and aggregate audit; no GPU required."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main():
    root = Path(sys.argv[1])
    prompts = read(root/'prefixes.json')
    pmap = {p['id']: p for p in prompts}
    assert len(prompts) == len(pmap) == 128
    assert len({tuple(p['ids']) for p in prompts}) == 128
    assert len({p['source_row'] for p in prompts}) == 128
    assert all(len(p['ids']) == 6 and p['ids'][0] == 101 for p in prompts)
    old = set()
    for folder in ('pilot-001', 'benefit-002'):
        for group in read(root.parent/folder/'prefixes.json').values():
            old.update(tuple(p['ids']) for p in group)
    old.update(tuple(p['ids']) for p in read(root.parent/'current-003/fresh_prefixes.json'))
    assert not old.intersection(tuple(p['ids']) for p in prompts)
    main_rows = lines(root/'samples.jsonl')
    recommended = lines(root/'recommended_samples.jsonl')
    references = read(root/'references.json')
    methods = ('ar', 'fixed2', 'fixed3', 'fixed4')
    for rows, variants in ((main_rows, methods), (recommended, methods[1:])):
        expected = set(itertools.product(pmap, (617, 619), variants))
        assert len(rows) == len(expected)
        assert {(r['prompt_id'], r['seed'], r['method']) for r in rows} == expected
        for r in rows:
            ids = r['full_ids']
            assert len(ids) == 128 and ids[:6] == pmap[r['prompt_id']]['ids']
            end = next((i+1 for i in range(6, len(ids)) if ids[i] == 102), len(ids))
            assert r['ids'] == ids[:end]
            assert r['prefix_length'] == 6 and r['new_tokens'] == end-6
            assert r['stopped_eos'] == (r['ids'][-1] == 102)
            assert r['teacher_token_count'] == r['new_tokens']
            assert math.isfinite(r['teacher_nll_sum'])
    assert len(references) == 128 and {r['prompt_id'] for r in references} == set(pmap)
    for r in references:
        assert r['ids'] == pmap[r['prompt_id']]['reference_ids']
    for r in recommended:
        assert r['frequency_penalty'] == {'fixed2': .2, 'fixed3': .4, 'fixed4': .5}[r['method']]

    timings = lines(root/'batch_timings.jsonl')
    expected = {(bs, seed, rep, bi, method)
                for bs, seed, rep in [(b, 617, r) for b in (4, 16) for r in range(3)] + [(4, 619, 0)]
                for bi in range(128//bs) for method in methods}
    assert len(timings) == len(expected) == 608
    assert {(r['batch_size'], r['seed'], r['repeat'], r['batch_index'], r['method']) for r in timings} == expected
    fingerprints = defaultdict(set)
    for r in timings:
        assert math.isfinite(r['seconds']) and r['seconds'] > 0
        assert r['fixed_generated_positions'] == 122*r['batch_size']
        assert r['sampler_seed'] == r['seed']+101*r['batch_index']
        fingerprints[r['batch_size'], r['seed'], r['batch_index'], r['method']].add(r['output_sha256'])
    assert all(len(v) == 1 for v in fingerprints.values())
    # Match quality sequences to the recorded hashes of their timed batches.
    quality = {(r['prompt_id'], r['seed'], r['method']): r for r in main_rows}
    for r in timings:
        if r['batch_size'] == 4 and r['repeat'] == 0:
            selected = prompts[r['batch_index']*4:(r['batch_index']+1)*4]
            ids = [quality[p['id'], r['seed'], r['method']]['full_ids'] for p in selected]
            assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == r['output_sha256']
    throughput = read(root/'throughput.json')
    for bs, method in itertools.product((4, 16), methods):
        group = [r for r in timings if r['batch_size'] == bs and r['method'] == method and r['seed'] == 617]
        per_repeat = [sum(r['fixed_generated_positions'] for r in group if r['repeat'] == rep)/
                      sum(r['seconds'] for r in group if r['repeat'] == rep) for rep in range(3)]
        stored = throughput[f'{method}_bs{bs}']
        close(statistics.mean(per_repeat), stored['mean_fixed_positions_per_second'])
        for a, b in zip(per_repeat, stored['repeat_throughputs']):
            close(a, b)
        close(sum(r['seconds'] for r in group), stored['total_decoder_seconds'])
    checks = read(root/'run_checks.json')
    assert checks == dict(quality_samples=1024, batch_timing_records=608,
                          repeated_batch_fingerprint_mismatches=0, all_full_sequences_length_128=True)

    sources = {}
    for dataset, rows in (('lm1b', main_rows), ('lm1b_recommended', recommended),
                          ('lm1b_reference', references), ('wikitext_stage3', lines(root.parent/'current-003/online_samples.jsonl'))):
        for r in rows:
            method = r.get('method', r.get('policy'))
            key = dataset, method, r['prompt_id'], r.get('seed')
            assert key not in sources
            sources[key] = r
    external = lines(root/'external_scores.jsonl')
    assert len(external) == len(sources) == 2240
    seen = set()
    for r in external:
        key = r['dataset'], r['method'], r['prompt_id'], r['seed']
        assert key in sources and key not in seen
        seen.add(key)
        source = sources[key]
        close(r['teacher_nll_sum'], source['teacher_nll_sum'])
        assert r['teacher_token_count'] == source['teacher_token_count']
        assert r['bert_new_tokens'] == source['new_tokens']
        if 'completion' in source:
            assert r['text'] == source['completion']
        assert r['gpt2_scored_tokens'] == max(0, r['gpt2_input_tokens']-1)
        assert r['excluded_short'] == (r['gpt2_input_tokens'] < 2)
        assert math.isfinite(r['gpt2_nll_sum']) and r['gpt2_nll_sum'] >= 0
        if r['excluded_short']:
            assert r['gpt2_nll_sum'] == 0
    summary = read(root/'external_summary.json')
    for key, stored in summary.items():
        dataset, method = key.split('/')
        group = [r for r in external if (r['dataset'], r['method']) == (dataset, method)]
        n = sum(r['gpt2_scored_tokens'] for r in group)
        nll = sum(r['gpt2_nll_sum'] for r in group)/n
        assert stored['samples'] == len(group) and stored['scored_tokens'] == n
        assert stored['excluded_short'] == sum(r['excluded_short'] for r in group)
        close(stored['gpt2_nll'], nll)
        close(stored['gpt2_gen_ppl'], math.exp(nll))
        close(stored['teacher_nll'], sum(r['teacher_nll_sum'] for r in group)/sum(r['teacher_token_count'] for r in group))
        close(stored['mean_bert_new_tokens'], statistics.mean(r['bert_new_tokens'] for r in group))
    scoring_checks = read(root/'external_scoring_checks.json')
    assert scoring_checks['builtin_loss_matches'] and scoring_checks['right_padding_invariance']
    official = read(root/'official_semantic_audit.json')
    assert len(official['sampler_cases']) == 18
    assert all(r['all_13_tokens_match'] for r in official['sampler_cases'])
    model_manifest = read(root/'gpt2_model_manifest.json')
    assert model_manifest['revision'] == '32b71b12589c2f8d625668d2335a01cac3249519'
    assert len(model_manifest['files']) == 6
    weights = next(f for f in model_manifest['files'] if f['name'] == 'model.safetensors')
    assert weights['sha256'] == '5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce'
    assert read(root/'lm1b_manifest.json')['sha256'] == 'd3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
    result = dict(passed=True, independent='Python standard library; recalculates records and aggregates',
                  prefixes=128, prior_prefix_overlap=0, main_samples=1024, recommended_samples=768,
                  references=128, old_samples_rescored=320, external_rows=2240,
                  external_short_exclusions=sum(r['excluded_short'] for r in external),
                  timing_records=608, repeat_fingerprint_mismatches=0,
                  quality_sequences_match_timed_hashes=True, official_same_noise_cases=18,
                  evaluator_builtin_loss_and_padding_checks=True,
                  limits='Integrity is not evidence of method quality or exact paper reproduction.')
    (root/'integrity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
