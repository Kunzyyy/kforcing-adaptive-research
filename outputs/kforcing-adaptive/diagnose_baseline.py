"""Stage 16: independent weight-only forward and 1024-source AR/K4 baseline diagnostic."""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from core import load_model
from collect_benefit import verify_checkpoints
from confirm_lowdim import old_sources, OLD_FILES

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/baseline-016'
DATA_SHA = 'd3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def now(): return datetime.now(timezone.utc).isoformat()
def append(f, x): f.write(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()


def freeze(args):
    import pyarrow.parquet as pq
    from transformers import BertTokenizerFast
    out = args.out
    assert not (out/'manifest.json').exists()
    source = Path('work/K-Forcing/models/transformer.py')
    shutil.copyfile(source, out/'released_transformer.py')
    all_old = old_sources(out.parent)+read(out.parent/'lowdim-015/prefixes.json')
    seen = {tuple(p.get('ids', p.get('reference_ids'))[:6]) for p in all_old}
    old_rows = {p['source_row'] for p in all_old if 'source_row' in p}
    old_hashes = {p['source_sha256'] for p in all_old if 'source_sha256' in p}
    dataset = Path('work/lm1b-data/test.parquet'); assert sha(dataset) == DATA_SHA
    texts = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    order = list(range(len(texts))); random.Random(16112001).shuffle(order)
    tok = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    selected = []
    for index in order:
        digest = hashlib.sha256(texts[index].encode()).hexdigest()
        if index in old_rows or digest in old_hashes: continue
        ids = tok.encode(texts[index], add_special_tokens=True, truncation=True, max_length=128)
        if len(ids) < 12 or tuple(ids[:6]) in seen: continue
        selected.append(dict(prompt_id=f's16-{len(selected):04}', index=len(selected), ids=ids[:6],
            source_row=index, source_sha256=digest))
        seen.add(tuple(ids[:6])); old_hashes.add(digest)
        if len(selected) == 1024: break
    assert len(selected) == 1024
    dump(out/'prefixes.json', selected)
    inputs = OLD_FILES+['lowdim-015/prefixes.json', 'lm1b-004/samples.jsonl', 'lm1b-004/external_scores.jsonl']
    files = ['STAGE16_PROTOCOL.md', 'reference_pflm.py', 'diagnose_baseline.py', 'core.py',
        'collect_benefit.py', 'confirm_lowdim.py', 'external_gpt2_score.py', 'audit_official.py',
        'upstream/models/pflm.py', 'upstream/models/transformer.py', 'upstream/models/autoregressive.py',
        'upstream/batch_inference_with_prefix.py']
    dump(out/'manifest.json', dict(created_utc=now(), sources=1024, dataset_sha256=DATA_SHA,
        code_hashes={f: sha(ROOT/f) for f in files}, input_hashes={f: sha(out.parent/f) for f in inputs},
        artifact_hashes={f: sha(out/f) for f in ['prefixes.json', 'released_transformer.py']},
        checkpoint_manifest_sha256=sha(ROOT/'checkpoint_manifest.json'),
        gpt2_manifest_sha256=sha(Path('work/gpt2-large/manifest.json')),
        tokenizer_sha256=sha(Path('work/pilot-data/vocab.txt')),
        paper='https://arxiv.org/html/2606.10820v2', paper_lm1b_genppl=dict(ar=104.8, fixed4=127.6),
        paper_table4=dict(backbone_million=108, noise_encoder_million=5.9, normalization='RMSNorm (pre-norm)'),
        bootstrap_seed=16112004, bootstrap_repeats=4000, exploratory_diagnostic=True))
    print('Frozen 1024 new source sentences and all experiment inputs.', flush=True)


def verify(args):
    m = read(args.out/'manifest.json')
    for f, h in m['code_hashes'].items(): assert sha(ROOT/f) == h, f
    for f, h in m['input_hashes'].items(): assert sha(args.out.parent/f) == h, f
    for f, h in m['artifact_hashes'].items(): assert sha(args.out/f) == h, f
    assert sha(ROOT/'checkpoint_manifest.json') == m['checkpoint_manifest_sha256']
    assert sha(Path('work/gpt2-large/manifest.json')) == m['gpt2_manifest_sha256']
    assert sha(Path('work/pilot-data/vocab.txt')) == m['tokenizer_sha256']
    return m


def setup(args):
    verify(args); verify_checkpoints(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


def norm_source(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LayerNorm')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    calls = [n.func.attr for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'layer_norm' in calls and 'rms_norm' not in calls
    return dict(forward_source=ast.unparse(forward), calls=calls, uses_layer_norm=True)


@torch.inference_mode()
def probe(args):
    from reference_pflm import reference_forward
    setup(args); out = args.out
    assert not (out/'reference_probe.json').exists()
    model = load_model(args.checkpoints/'pflm_lm1b_k4.ckpt', 'pflm', 'cuda')
    state = model.state_dict()
    parameter_groups = defaultdict(int)
    for name, p in model.named_parameters(): parameter_groups[name.split('.')[0]] += p.numel()
    embedding = state['vocab_embed.embedding']; output = state['output_layer.linear.weight']
    arch = dict(parameter_groups=dict(parameter_groups), total_parameters=sum(parameter_groups.values()),
        head_linear_parameters=state['output_layer.linear.weight'].numel()+state['output_layer.linear.bias'].numel(),
        input_embedding_and_output_share_storage=embedding.data_ptr() == output.data_ptr(),
        input_embedding_and_output_equal=bool(torch.equal(embedding, output)),
        embedding_output_max_abs_difference=float((embedding-output).abs().max()),
        local_normalization=norm_source(ROOT/'upstream/models/transformer.py'),
        released_normalization=norm_source(out/'released_transformer.py'),
        paper_normalization='RMSNorm (pre-norm)',
        interpretation='Publication/code description mismatch; no inference of which norm trained the unpublished paper checkpoint.')
    dump(out/'architecture_audit.json', arch)
    old = sorted([r for r in rows(out.parent/'lm1b-004/samples.jsonl') if r['method'] == 'fixed4' and r['seed'] == 617], key=lambda r: r['prompt_id'])[:8]
    results = []
    for length in [6, 10, 14, 30, 70]:
        context = torch.tensor([r['full_ids'][:length] for r in old], device='cuda')
        full_noise = torch.rand((8, 4, 1), generator=torch.Generator().manual_seed(16112003+length*101)).cuda()
        tau = torch.ones(8, 1, 1, device='cuda')
        for width in [1, 2, 3, 4]:
            noise = full_noise[:, :width]
            actual = model(context, noise, tau, mode='inference')
            reference = reference_forward(state, context, noise, tau)
            diff = (actual-reference).abs()
            for i, record in enumerate(old):
                margin = actual[i].topk(2).values
                results.append(dict(prompt_id=record['prompt_id'], context_length=length, k=width,
                    context_ids=context[i].tolist(), noise=noise[i, :, 0].tolist(),
                    max_abs_logit_difference=float(diff[i].max()),
                    argmax_disagreements=int((actual[i].argmax(-1) != reference[i].argmax(-1)).sum()),
                    actual_argmax=actual[i].argmax(-1).tolist(), reference_argmax=reference[i].argmax(-1).tolist(),
                    minimum_top_margin=float((margin[:, 0]-margin[:, 1]).min()),
                    context_includes_generated_eos=102 in record['full_ids'][6:length]))
        print('Independent full-context reference checked, context length', length, flush=True)
    largest = max(r['max_abs_logit_difference'] for r in results)
    different = sum(r['argmax_disagreements'] for r in results)
    result = dict(passed=largest < .005 and different == 0, cases=len(results),
        candidate_positions=sum(r['k'] for r in results), max_abs_logit_difference=largest,
        argmax_disagreements=different, checks=results, manifest_sha256=sha(out/'manifest.json'),
        reference_sha256=sha(ROOT/'reference_pflm.py'), dtype='float32', tf32=False,
        limitations='Matches the released LayerNorm model on a finite probe; not an original H100 environment comparison.')
    dump(out/'reference_probe.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'cases', 'candidate_positions', 'max_abs_logit_difference', 'argmax_disagreements']}), flush=True)
    assert result['passed'], 'Reference disagreement: inspect before any new baseline generation'


@torch.inference_mode()
def generate(args):
    from transformers import BertTokenizerFast
    from audit_official import official_cli, quiet_sampler
    setup(args); out = args.out
    assert read(out/'reference_probe.json')['passed']
    assert not (out/'samples.jsonl').exists()
    official = official_cli(); quiet_sampler()
    tok = BertTokenizerFast(vocab_file='work/pilot-data/vocab.txt', do_lower_case=True)
    prefixes = read(out/'prefixes.json')
    ar = load_model(args.checkpoints/'ar_best_lm1b.ckpt', 'ar', 'cuda')
    pflm = load_model(args.checkpoints/'pflm_lm1b_k4.ckpt', 'pflm', 'cuda')
    with (out/'samples.jsonl').open('w', encoding='utf-8') as f:
        for start in range(0, 1024, 4):
            batch = prefixes[start:start+4]
            context = torch.tensor([p['ids'] for p in batch], device='cuda')
            sampler_seed = 16120000+(start//4)*101
            for name in ['ar', 'fixed4']:
                torch.manual_seed(sampler_seed)
                if name == 'ar': full = official.generate_ar_cached(ar, context, 122)
                else: full = pflm.sample_next_k_tokens_with_kv_caches(context, torch.ones(4, device='cuda'), 122, k=4, frequency_penalty=0.)
                for p, sequence in zip(batch, full.tolist()):
                    assert len(sequence) == 128
                    ids = official.truncate_at_eos(sequence, 102, 6)
                    text = tok.decode(ids[6:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    append(f, dict(prompt_id=p['prompt_id'], method=name, sampler_seed=sampler_seed,
                        batch_index=start//4, prefix_length=6, full_ids=sequence, ids=ids,
                        new_tokens=len(ids)-6, stopped_eos=ids[-1] == 102,
                        leading_wordpiece_fragment=text.startswith('##'), completion=text))
            if (start+4) % 64 == 0: print('New baseline sources', start+4, '/1024, both methods', flush=True)
    dump(out/'generation_checks.json', dict(passed=True, samples=2048, sources=1024,
        samples_sha256=sha(out/'samples.jsonl'), manifest_sha256=sha(out/'manifest.json'),
        reference_probe_sha256=sha(out/'reference_probe.json'), completed_utc=now(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        ar_precision='public CLI internal fp16', pflm_precision='fp32, no outer autocast',
        timing_claim=False))


@torch.inference_mode()
def score(args):
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    from external_gpt2_score import masked_token_nll
    setup(args); out = args.out
    assert not (out/'scores.jsonl').exists()
    assert read(out/'generation_checks.json')['samples_sha256'] == sha(out/'samples.jsonl')
    model_path = Path('work/gpt2-large'); manifest = read(model_path/'manifest.json')
    for item in manifest['files']:
        h = hashlib.sha256()
        with (model_path/item['name']).open('rb') as f:
            while chunk := f.read(8*1024*1024): h.update(chunk)
        assert h.hexdigest() == item['sha256']
    tok = GPT2TokenizerFast.from_pretrained(str(model_path), local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(str(model_path), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation='sdpa').cuda().eval()
    data = [dict(r, cohort='new') for r in rows(out/'samples.jsonl')]
    data += [dict(r, cohort='old_control') for r in rows(out.parent/'lm1b-004/samples.jsonl') if r['method'] in ['ar', 'fixed4']]
    old = {(r['prompt_id'], r['seed'], r['method']): r for r in rows(out.parent/'lm1b-004/external_scores.jsonl') if r['dataset'] == 'lm1b'}
    probe = tok.encode('An independent reference can help locate implementation differences.', return_tensors='pt').cuda()
    result = model(probe, labels=probe, use_cache=False)
    sums, counts = masked_token_nll(result.logits, probe, torch.ones_like(probe))
    torch.testing.assert_close(sums/counts, result.loss.reshape(1), atol=1e-6, rtol=1e-6)
    differences = []
    with (out/'scores.jsonl').open('w', encoding='utf-8') as f:
        for start in range(0, len(data), 4):
            batch = data[start:start+4]
            tokens = [tok.encode(r['completion'], add_special_tokens=False) for r in batch]
            assert all(len(t) <= 1024 for t in tokens)
            width = max(2, max(map(len, tokens)))
            x = torch.full((len(batch), width), tok.eos_token_id, dtype=torch.long, device='cuda'); mask = torch.zeros_like(x)
            for j, t in enumerate(tokens):
                if t: x[j, :len(t)] = torch.tensor(t, device='cuda'); mask[j, :len(t)] = 1
                else: mask[j, 0] = 1
            sums, counts = masked_token_nll(model(x, attention_mask=mask, use_cache=False).logits, x, mask)
            for j, r in enumerate(batch):
                item = dict(cohort=r['cohort'], prompt_id=r['prompt_id'], method=r['method'],
                    gpt2_input_tokens=len(tokens[j]), gpt2_scored_tokens=int(counts[j]),
                    gpt2_nll_sum=float(sums[j]) if counts[j] else 0., excluded_short=not bool(counts[j]))
                if r['cohort'] == 'old_control':
                    item['seed'] = r['seed']
                    control = old[r['prompt_id'], r['seed'], r['method']]
                    assert item['gpt2_scored_tokens'] == control['gpt2_scored_tokens']
                    assert math.isclose(item['gpt2_nll_sum'], control['gpt2_nll_sum'], rel_tol=1e-5, abs_tol=.005)
                    differences.append(abs(item['gpt2_nll_sum']-control['gpt2_nll_sum']))
                append(f, item)
            if (start+4) % 256 == 0: print('Baseline GPT-2 scoring', start+4, '/', len(data), flush=True)
    dump(out/'scoring_checks.json', dict(passed=True, records=len(data), control_records=len(differences),
        max_old_control_nll_abs_difference=max(differences), builtin_loss_crosscheck=True,
        scores_sha256=sha(out/'scores.jsonl'), samples_sha256=sha(out/'samples.jsonl'),
        evaluator_revision=manifest['revision'], manifest_sha256=sha(out/'manifest.json')))


def analyze(args):
    verify(args); out = args.out
    assert not (out/'analysis.json').exists()
    scoring = read(out/'scoring_checks.json')
    assert scoring['passed'] and scoring['scores_sha256'] == sha(out/'scores.jsonl')
    data = [r for r in rows(out/'scores.jsonl') if r['cohort'] == 'new']
    samples = rows(out/'samples.jsonl')
    prefixes = read(out/'prefixes.json')
    ids = [p['prompt_id'] for p in prefixes]
    draws = np.random.default_rng(16112004).integers(0, len(ids), (4000, len(ids)))
    result, bootstrap = {}, {}
    for method in ['ar', 'fixed4']:
        by_id = {r['prompt_id']: r for r in data if r['method'] == method}
        nll = np.array([by_id[i]['gpt2_nll_sum'] for i in ids]); count = np.array([by_id[i]['gpt2_scored_tokens'] for i in ids])
        boots = nll[draws].sum(1)/count[draws].sum(1); bootstrap[method] = boots
        group = [s for s in samples if s['method'] == method]
        mean = float(nll.sum()/count.sum())
        result[method] = dict(samples=len(by_id), nll=mean, gen_ppl=math.exp(mean),
            gen_ppl_interval95=np.exp(np.quantile(boots, [.025, .975])).tolist(), scored_tokens=int(count.sum()),
            excluded_short=int((count == 0).sum()), mean_bert_new_tokens=float(np.mean([s['new_tokens'] for s in group])),
            eos_stopped=sum(s['stopped_eos'] for s in group), leading_wordpiece_fragments=sum(s['leading_wordpiece_fragment'] for s in group))
    ratio = dict(estimate=math.exp(result['fixed4']['nll']-result['ar']['nll']),
        interval95=np.exp(np.quantile(bootstrap['fixed4']-bootstrap['ar'], [.025, .975])).tolist())
    analysis = dict(sources=len(ids), methods=result, fixed4_over_ar_ppl_ratio=ratio,
        paper_reference=read(out/'manifest.json')['paper_lm1b_genppl'], paper_reference_not_hypothesis_test=True,
        source_cluster_bootstrap_repeats=4000, bootstrap_seed=16112004, scores_sha256=sha(out/'scores.jsonl'),
        manifest_sha256=sha(out/'manifest.json'), reference_probe_passed=read(out/'reference_probe.json')['passed'],
        limitations=['Original paper prompts and exact checkpoint correspondence remain unknown.',
            'Local CLI precision/RTX/SDPA environment; not H100/bf16 reproduction.',
            'Fresh baseline-only diagnostic, not a new adaptive method confirmation.'])
    dump(out/'analysis.json', analysis)
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['freeze', 'probe', 'generate', 'score', 'analyze'])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--checkpoints', type=Path, default=Path('work/checkpoints'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.action](args)
