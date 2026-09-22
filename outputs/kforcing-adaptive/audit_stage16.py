"""Independent source/token/score/statistics audit with first/last batch replay."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a, b): np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-9)


def audit(root):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    m = read(root/'manifest.json')
    for f, h in m['code_hashes'].items(): assert sha(ROOT/f) == h, f
    for f, h in m['input_hashes'].items(): assert sha(root.parent/f) == h, f
    for f, h in m['artifact_hashes'].items(): assert sha(root/f) == h, f
    assert sha(ROOT/'checkpoint_manifest.json') == m['checkpoint_manifest_sha256']
    assert sha(Path('work/pilot-data/vocab.txt')) == m['tokenizer_sha256']
    repo = Path('work/K-Forcing').resolve()
    cmd = ['git', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    commit = subprocess.check_output(cmd+['rev-parse', 'HEAD'], text=True).strip()
    assert commit == '706caa332a69d509b7fc2fa53ac7819f381a8c78'
    released = subprocess.check_output(cmd+['show', commit+':models/transformer.py']).decode('utf-8')
    assert released.replace('\r\n', '\n') == (root/'released_transformer.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    assert 'F.layer_norm(x.float(), [self.dim])' in released
    arch = read(root/'architecture_audit.json')
    state = torch.load('work/checkpoints/pflm_lm1b_k4.ckpt', map_location='cpu', weights_only=True)['state_dict']
    state = {k.removeprefix('backbone.'): v for k, v in state.items()}
    groups = defaultdict(int)
    for key, value in state.items():
        if key == 'rotary_emb.inv_freq': continue
        groups[key.split('.')[0]] += value.numel()
    assert dict(groups) == arch['parameter_groups']
    assert sum(groups.values()) == arch['total_parameters']
    assert not torch.equal(state['vocab_embed.embedding'], state['output_layer.linear.weight'])
    close(arch['embedding_output_max_abs_difference'], float((state['vocab_embed.embedding']-state['output_layer.linear.weight']).abs().max()))
    del state
    probe = read(root/'reference_probe.json')
    assert probe['passed'] and probe['manifest_sha256'] == sha(root/'manifest.json')
    assert probe['reference_sha256'] == sha(ROOT/'reference_pflm.py')
    assert len(probe['checks']) == probe['cases'] == 160
    assert sum(r['k'] for r in probe['checks']) == probe['candidate_positions'] == 400
    assert max(r['max_abs_logit_difference'] for r in probe['checks']) == probe['max_abs_logit_difference'] < .005
    assert sum(r['argmax_disagreements'] for r in probe['checks']) == probe['argmax_disagreements'] == 0
    for r in probe['checks']:
        assert r['actual_argmax'] == r['reference_argmax']
        assert len(r['context_ids']) == r['context_length']
        assert len(r['noise']) == len(r['actual_argmax']) == r['k']
    old = []
    for path in m['input_hashes']:
        if not path.endswith(('prefixes.json', 'sources.json')): continue
        raw = read(root.parent/path)
        if isinstance(raw, dict):
            for group in raw.values(): old.extend(group)
        else: old.extend(raw)
    old_ids = {tuple(p.get('ids', p.get('reference_ids'))[:6]) for p in old}
    old_rows = {p['source_row'] for p in old if 'source_row' in p}
    old_hashes = {p['source_sha256'] for p in old if 'source_sha256' in p}
    prefixes = read(root/'prefixes.json')
    assert len(prefixes) == len({p['prompt_id'] for p in prefixes}) == 1024
    assert len({tuple(p['ids']) for p in prefixes}) == 1024
    assert not old_ids.intersection(tuple(p['ids']) for p in prefixes)
    assert not old_rows.intersection(p['source_row'] for p in prefixes)
    assert not old_hashes.intersection(p['source_sha256'] for p in prefixes)
    corpus = Path('work/lm1b-data/test.parquet'); assert sha(corpus) == m['dataset_sha256']
    texts = pq.read_table(corpus, columns=['text'])['text'].to_pylist()
    tok = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    order = list(range(len(texts))); random.Random(16112001).shuffle(order)
    n = 0
    for index in order:
        digest = hashlib.sha256(texts[index].encode()).hexdigest()
        if index in old_rows or digest in old_hashes: continue
        ids = tok.encode(texts[index], add_special_tokens=True, truncation=True, max_length=128)
        if len(ids) < 12 or tuple(ids[:6]) in old_ids: continue
        p = prefixes[n]
        assert p == dict(prompt_id=f's16-{n:04}', index=n, ids=ids[:6], source_row=index, source_sha256=digest)
        n += 1; old_ids.add(tuple(ids[:6])); old_hashes.add(digest)
        if n == 1024: break
    assert n == 1024
    samples = rows(root/'samples.jsonl')
    assert len(samples) == 2048
    table = {(s['prompt_id'], s['method']): s for s in samples}
    assert set(table) == {(p['prompt_id'], method) for p in prefixes for method in ['ar', 'fixed4']}
    for p in prefixes:
        for method in ['ar', 'fixed4']:
            s = table[p['prompt_id'], method]
            assert len(s['full_ids']) == 128 and s['full_ids'][:6] == p['ids']
            assert s['batch_index'] == p['index']//4 and s['sampler_seed'] == 16120000+s['batch_index']*101
            tail = s['full_ids'][6:]
            stop = tail.index(102)+1 if 102 in tail else len(tail)
            assert s['ids'] == s['full_ids'][:6+stop] and s['new_tokens'] == stop
            assert s['stopped_eos'] == (s['ids'][-1] == 102)
            assert s['completion'] == tok.decode(s['ids'][6:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            assert s['leading_wordpiece_fragment'] == s['completion'].startswith('##')
    generation = read(root/'generation_checks.json')
    assert generation['passed'] and generation['samples_sha256'] == sha(root/'samples.jsonl')
    scores = rows(root/'scores.jsonl')
    new = [r for r in scores if r['cohort'] == 'new']; control = [r for r in scores if r['cohort'] == 'old_control']
    assert len(new) == 2048 and len(control) == 512
    scoring = read(root/'scoring_checks.json')
    assert scoring['passed'] and scoring['scores_sha256'] == sha(root/'scores.jsonl')
    assert scoring['samples_sha256'] == sha(root/'samples.jsonl')
    previous = {(s['prompt_id'], s['seed'], s['method']): s for s in rows(root.parent/'lm1b-004/external_scores.jsonl') if s['dataset'] == 'lm1b'}
    diffs = []
    for r in scores:
        assert r['gpt2_scored_tokens'] == max(0, r['gpt2_input_tokens']-1)
        assert r['excluded_short'] == (r['gpt2_scored_tokens'] == 0)
        if r['excluded_short']: assert r['gpt2_nll_sum'] == 0
        if r['cohort'] == 'old_control':
            s = previous[r['prompt_id'], r['seed'], r['method']]
            assert r['gpt2_scored_tokens'] == s['gpt2_scored_tokens']
            assert math.isclose(r['gpt2_nll_sum'], s['gpt2_nll_sum'], abs_tol=.005, rel_tol=1e-5)
            diffs.append(abs(r['gpt2_nll_sum']-s['gpt2_nll_sum']))
    close(max(diffs), scoring['max_old_control_nll_abs_difference'])
    result = read(root/'analysis.json')
    assert result['scores_sha256'] == sha(root/'scores.jsonl')
    assert result['manifest_sha256'] == sha(root/'manifest.json')
    indices = np.random.default_rng(16112004).integers(0, 1024, (4000, 1024))
    bootstrap = {}
    for method in ['ar', 'fixed4']:
        subset = {r['prompt_id']: r for r in new if r['method'] == method}
        assert len(subset) == 1024
        losses = np.array([subset[p['prompt_id']]['gpt2_nll_sum'] for p in prefixes])
        counts = np.array([subset[p['prompt_id']]['gpt2_scored_tokens'] for p in prefixes])
        mean = losses.sum()/counts.sum()
        boot = np.array([losses[ix].sum()/counts[ix].sum() for ix in indices]); bootstrap[method] = boot
        item = result['methods'][method]
        close(item['nll'], mean); close(item['gen_ppl'], math.exp(mean))
        close(item['gen_ppl_interval95'], np.exp(np.quantile(boot, [.025, .975])))
        assert item['scored_tokens'] == int(counts.sum()) and item['excluded_short'] == int((counts == 0).sum())
        group = [s for s in samples if s['method'] == method]
        close(item['mean_bert_new_tokens'], sum(s['new_tokens'] for s in group)/1024)
        assert item['eos_stopped'] == sum(s['stopped_eos'] for s in group)
        assert item['leading_wordpiece_fragments'] == sum(s['leading_wordpiece_fragment'] for s in group)
    close(result['fixed4_over_ar_ppl_ratio']['estimate'], result['methods']['fixed4']['gen_ppl']/result['methods']['ar']['gen_ppl'])
    close(result['fixed4_over_ar_ppl_ratio']['interval95'], np.exp(np.quantile(bootstrap['fixed4']-bootstrap['ar'], [.025, .975])))
    replay_batches(root, prefixes, table)
    audit_result = dict(passed=True, sources=1024, samples=2048, old_scoring_controls=512,
        original_git_blob_verified=True, released_layernorm_confirmed=True, checkpoint_component_counts_verified=True,
        source_isolation_and_sampling_replayed=True, all_eos_text_and_score_counts_checked=True,
        paired_bootstrap_recomputed=True, first_last_batches_replayed_samples=16,
        reference_probe_records_verified=True, reference_probe_rerun=False,
        analysis_sha256=sha(root/'analysis.json'), audit_code_sha256=sha(Path(__file__)))
    (root/'integrity_audit.json').write_text(json.dumps(audit_result, indent=2), encoding='utf-8')
    print(json.dumps(audit_result, indent=2), flush=True)


@torch.inference_mode()
def replay_batches(root, prefixes, table):
    from core import load_model
    from audit_official import official_cli, quiet_sampler
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    cli = official_cli(); quiet_sampler()
    for method, checkpoint, kind in [('ar', 'ar_best_lm1b.ckpt', 'ar'), ('fixed4', 'pflm_lm1b_k4.ckpt', 'pflm')]:
        model = load_model(Path('work/checkpoints')/checkpoint, kind, 'cuda')
        for start in [0, 1020]:
            group = prefixes[start:start+4]
            prompt = torch.tensor([p['ids'] for p in group], device='cuda')
            torch.manual_seed(16120000+(start//4)*101)
            if method == 'ar': actual = cli.generate_ar_cached(model, prompt, 122)
            else: actual = model.sample_next_k_tokens_with_kv_caches(prompt, torch.ones(4, device='cuda'), 122, k=4, frequency_penalty=0.)
            assert actual.tolist() == [table[p['prompt_id'], method]['full_ids'] for p in group]
        del model
    print('First and last batches, both methods: all 16 full sequences replay exactly.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path, nargs='?', default=ROOT/'results/baseline-016')
    audit(ap.parse_args().root)
