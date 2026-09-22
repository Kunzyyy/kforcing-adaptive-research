"""Retrospective reward audit; preserves stage15 outputs and frozen decisions."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/lowdim-015'
OUT = ROOT / 'results/reward-audit'
INPUTS = ['test_pairs.jsonl', 'test_labels.jsonl', 'test_predictions.jsonl',
          'frozen_predictions.json', 'analysis.json', 'test_scoring.json', 'locked_model.json']
SEED = 21092026
REPS = 4000


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    assert not path.exists(), f'Preserve existing output: {path}'
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def lines(path, values):
    assert not path.exists(), f'Preserve existing output: {path}'
    with path.open('w', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    historical = read(ROOT / 'stage16_file_hashes.json')
    inputs = {name: sha(SOURCE / name) for name in INPUTS}
    for name, digest in inputs.items():
        assert historical[f'results/lowdim-015/{name}'] == digest
    dump(OUT / 'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         study='Retrospective diagnostic; no independent test or refitting',
         inputs=inputs, code_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT / 'REWARD_AUDIT_PLAN.md'), bootstrap_seed=SEED, bootstrap_reps=REPS))
    print('Frozen historical inputs, audit plan and code.', flush=True)


def verify():
    manifest = read(OUT / 'manifest.json')
    assert manifest['code_sha256'] == sha(Path(__file__))
    assert manifest['plan_sha256'] == sha(ROOT / 'REWARD_AUDIT_PLAN.md')
    for name, digest in manifest['inputs'].items():
        assert sha(SOURCE / name) == digest, name
    return manifest


def score():
    import torch
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    verify()
    assert not (OUT / 'token_losses.jsonl').exists()
    model_path = Path('work/gpt2-large')
    manifest = read(model_path / 'manifest.json')
    for item in manifest['files']:
        assert sha(model_path / item['name']) == item['sha256']
    pairs = rows(SOURCE / 'test_pairs.jsonl')
    old = {r['pair_id']: r for r in rows(SOURCE / 'test_labels.jsonl')}
    expected = {}
    for pair in pairs:
        for branch in ('keep4', 'split22'):
            text = pair[branch]['completion']
            label = old[pair['pair_id']][branch]
            value = (label['gpt2_nll_sum'], label['gpt2_scored_tokens'])
            assert text not in expected or expected[text] == value
            expected[text] = value
    texts = sorted(expected)
    assert len(texts) == 8810
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = GPT2TokenizerFast.from_pretrained(str(model_path), local_files_only=True)
    encoded = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    assert max(map(len, encoded)) <= 1024
    model = GPT2LMHeadModel.from_pretrained(str(model_path), local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation='sdpa').cuda().eval()

    def loss_vector(x, mask):
        logits = model(x, attention_mask=mask, use_cache=False).logits
        return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1), reduction='none').view_as(x[:, 1:])

    max_old = max_sum = 0.
    output = []
    with torch.inference_mode():
        probe = tokenizer.encode('A careful evaluation should compare the same number of positions.',
                                  return_tensors='pt').cuda()
        full_loss = loss_vector(probe, torch.ones_like(probe))
        builtin = model(probe, labels=probe, use_cache=False).loss
        torch.testing.assert_close(full_loss.mean(), builtin, rtol=1e-6, atol=1e-6)
        truncated = probe[:, :probe.shape[1]//2]
        short_loss = loss_vector(truncated, torch.ones_like(truncated))
        torch.testing.assert_close(full_loss[:, :short_loss.shape[1]], short_loss, rtol=1e-5, atol=1e-5)
        max_truncation = float((full_loss[:, :short_loss.shape[1]] - short_loss).abs().max())
        for start in range(0, len(texts), 4):
            bt, ids = texts[start:start+4], encoded[start:start+4]
            width = max(2, max(map(len, ids)))
            x = torch.full((len(bt), width), tokenizer.eos_token_id, dtype=torch.long, device='cuda')
            mask = torch.zeros_like(x)
            for j, tokens in enumerate(ids):
                if tokens:
                    x[j, :len(tokens)] = torch.tensor(tokens, device='cuda')
                    mask[j, :len(tokens)] = 1
                else:
                    mask[j, 0] = 1
            losses = loss_vector(x, mask)
            sums = (losses * mask[:, 1:]).sum(1).cpu().tolist()
            for j, text in enumerate(bt):
                n = max(0, len(ids[j])-1)
                values = losses[j, :n].cpu().tolist()
                expected_sum, expected_count = expected[text]
                assert n == expected_count
                old_diff = abs(sums[j] - expected_sum)
                sum_diff = abs(math.fsum(values) - expected_sum)
                assert old_diff <= 1e-4 and sum_diff <= 1e-4, (start+j, old_diff, sum_diff)
                max_old, max_sum = max(max_old, old_diff), max(max_sum, sum_diff)
                output.append(dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(), text=text,
                    input_ids=ids[j], token_nll=values, torch_nll_sum=sums[j], scored_tokens=n))
            if (start+4) % 512 == 0:
                print(f'GPT-2 token losses: {start+4}/{len(texts)}; old score max diff {max_old:g}', flush=True)
    lines(OUT / 'token_losses.jsonl', output)
    dump(OUT / 'scoring_checks.json', dict(passed=True, unique_texts=len(output),
        max_old_sum_abs_difference=max_old, max_float64_vector_sum_abs_difference=max_sum,
        builtin_loss_matches=True, causal_truncation_max_difference=max_truncation,
        evaluator_files_verified=True, evaluator_revision=manifest['revision'],
        token_losses_sha256=sha(OUT / 'token_losses.jsonl'), code_sha256=sha(Path(__file__))))
    print('Token-level scores completed and matched to historical scores.', flush=True)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def analyze():
    verify()
    checks = read(OUT / 'scoring_checks.json')
    assert checks['passed'] and checks['token_losses_sha256'] == sha(OUT / 'token_losses.jsonl')
    losses = {r['text']: r for r in rows(OUT / 'token_losses.jsonl')}
    labels = {r['pair_id']: r for r in rows(SOURCE / 'test_labels.jsonl')}
    predictions = {r['state_id']: r for r in rows(SOURCE / 'test_predictions.jsonl')}
    records = []
    groups = defaultdict(list)
    for pair in rows(SOURCE / 'test_pairs.jsonl'):
        if pair['state_id'] not in predictions:
            continue
        label = labels[pair['pair_id']]
        k, s = label['keep4'], label['split22']
        lk, ls = [losses[pair[b]['completion']]['token_nll'] for b in ('keep4', 'split22')]
        m = min(len(lk), len(ls))
        assert label['valid_label'] and m > 0
        common_k, common_s = math.fsum(lk[:m]), math.fsum(ls[:m])
        c = (common_k-common_s)/m
        d = label['gain']
        record = dict(pair_id=pair['pair_id'], state_id=pair['state_id'], prompt_id=pair['prompt_id'],
            replicate=pair['replicate'], offset=pair['offset'], full_gain=d, common_gain=c,
            remainder=d-c, teacher_gain=k['teacher_nll']-s['teacher_nll'],
            bert_length_change=s['visible_new_tokens']-k['visible_new_tokens'],
            gpt2_length_change=len(ls)-len(lk), calls_change=s['forward_calls']-k['forward_calls'],
            candidate_change=s['computed_candidates']-k['computed_candidates'], common_tokens=m,
            common_keep_sum=common_k, common_split_sum=common_s,
            keep_nll_sum=k['gpt2_nll_sum'], split_nll_sum=s['gpt2_nll_sum'],
            keep_tokens=len(lk), split_tokens=len(ls),
            teacher_keep_sum=k['teacher_nll_sum'], teacher_split_sum=s['teacher_nll_sum'],
            teacher_keep_tokens=k['teacher_token_count'], teacher_split_tokens=s['teacher_token_count'],
            identical_text=label['identical_decoded_text'],
            keep_eos=pair['keep4']['stopped_eos'], split_eos=pair['split22']['stopped_eos'])
        records.append(record)
        groups[pair['state_id']].append(record)
    metrics = ['full_gain', 'common_gain', 'remainder', 'teacher_gain', 'bert_length_change',
               'gpt2_length_change', 'calls_change', 'candidate_change']
    totals = ['common_tokens', 'common_keep_sum', 'common_split_sum', 'keep_nll_sum', 'split_nll_sum',
              'keep_tokens', 'split_tokens', 'teacher_keep_sum', 'teacher_split_sum',
              'teacher_keep_tokens', 'teacher_split_tokens']
    states = []
    for state_id in sorted(groups):
        rs = sorted(groups[state_id], key=lambda r: r['replicate'])
        assert [r['replicate'] for r in rs] == list(range(8))
        p = predictions[state_id]
        state = dict(state_id=state_id, prompt_id=rs[0]['prompt_id'], offset=rs[0]['offset'],
                     selected=p['selected'])
        state.update({key: float(np.mean([r[key] for r in rs])) for key in metrics})
        state.update({key: math.fsum(r[key] for r in rs) for key in totals})
        state.update({key+'_half'+str(h): float(np.mean([r[key] for r in rs[h*4:(h+1)*4]]))
                      for key in ('full_gain', 'common_gain') for h in (0, 1)})
        assert abs(state['full_gain'] - p['mean_gain']) < 1e-12
        states.append(state)
    sources = sorted({s['prompt_id'] for s in states})
    assert len(states) == 682 and len(sources) == 381 and len(records) == 5456
    indices = {key: i for i, key in enumerate(sources)}
    clusters = np.array([indices[s['prompt_id']] for s in states])
    rng = np.random.default_rng(SEED)
    # Equivalent to drawing 381 sources with replacement; reused for all metrics.
    draws = rng.integers(0, len(sources), size=(REPS, len(sources)))
    weights = np.zeros((REPS, len(sources)), dtype=np.int64)
    for i, draw in enumerate(draws):
        weights[i] = np.bincount(draw, minlength=len(sources))

    def source_sums(values):
        return np.bincount(clusters, weights=values, minlength=len(sources))

    def ratio(numerator, denominator):
        ns, ds = source_sums(numerator), source_sums(denominator)
        boots = (weights @ ns) / (weights @ ds)
        return dict(estimate=float(np.sum(numerator)/np.sum(denominator)),
                    interval=np.quantile(boots, [.025, .975]).tolist(), coverage=.95)

    def interval(values):
        return ratio(values, np.ones(len(states)))

    values = {key: np.array([s[key] for s in states]) for key in metrics+totals}
    means = {key: interval(values[key]) for key in metrics}
    original = read(SOURCE / 'analysis.json')
    assert abs(means['full_gain']['estimate'] - original['mean_all_split_gain']['estimate']) < 1e-12
    policies = {}
    mh = np.array([s['selected']['heuristic'] for s in states], dtype=float)
    for name in ('linear14', 'rbf14', 'heuristic'):
        mask = np.array([s['selected'][name] for s in states], dtype=float)
        assert mask.sum() == 341
        policies[name] = {key: dict(vs_random=interval((mask-mask.mean())*values[key]),
                                   vs_heuristic=interval((mask-mh)*values[key]))
                          for key in ('full_gain', 'common_gain', 'teacher_gain')}
    corpus = {}
    for name, ksum, ssum, kn, sn in (
        ('full_gpt2', 'keep_nll_sum', 'split_nll_sum', 'keep_tokens', 'split_tokens'),
        ('common_gpt2', 'common_keep_sum', 'common_split_sum', 'common_tokens', 'common_tokens'),
        ('ar_teacher', 'teacher_keep_sum', 'teacher_split_sum', 'teacher_keep_tokens', 'teacher_split_tokens')):
        keep, split = ratio(values[ksum], values[kn]), ratio(values[ssum], values[sn])
        corpus[name] = {}
        for branch, item in (('keep4', keep), ('split22', split)):
            corpus[name][branch] = dict(mean_nll=item,
                ppl=dict(estimate=math.exp(item['estimate']), interval=[math.exp(v) for v in item['interval']]))
        kb = (weights @ source_sums(values[ksum]))/(weights @ source_sums(values[kn]))
        sb = (weights @ source_sums(values[ssum]))/(weights @ source_sums(values[sn]))
        corpus[name]['nll_gain'] = dict(estimate=keep['estimate']-split['estimate'],
            interval=np.quantile(kb-sb, [.025, .975]).tolist(), coverage=.95)
    strata = {}
    for name, sign in (('split_shorter', -1), ('equal', 0), ('split_longer', 1)):
        rs = [r for r in records if np.sign(r['gpt2_length_change']) == sign]
        strata[name] = dict(pairs=len(rs),
            mean_full_gain=float(np.mean([r['full_gain'] for r in rs])),
            mean_common_gain=float(np.mean([r['common_gain'] for r in rs])),
            mean_gpt2_length_change=float(np.mean([r['gpt2_length_change'] for r in rs])))
    correlations = {a+'_vs_'+b: corr(values[a], values[b]) for a, b in (
        ('full_gain', 'common_gain'), ('full_gain', 'teacher_gain'), ('common_gain', 'teacher_gain'),
        ('full_gain', 'gpt2_length_change'), ('common_gain', 'gpt2_length_change'))}
    for key in ('full_gain', 'common_gain'):
        correlations[key+'_split_half'] = corr([s[key+'_half0'] for s in states], [s[key+'_half1'] for s in states])
    lines(OUT / 'pair_diagnostics.jsonl', records)
    lines(OUT / 'state_diagnostics.jsonl', states)
    dump(OUT / 'analysis.json', dict(retrospective=True, refitted=False, states=len(states), sources=len(sources),
        pairs=len(records), excluded_pairs=5464-len(records), bootstrap_reps=REPS, bootstrap_seed=SEED,
        means=means, frozen_policies=policies, corpus=corpus, outcome_defined_length_strata=strata,
        state_correlations=correlations,
        identical_text_pairs=sum(r['identical_text'] for r in records),
        eos_counts={b: sum(r[b+'_eos'] for r in records) for b in ('keep', 'split')},
        offsets={str(off): {key: float(np.mean([s[key] for s in states if s['offset']==off]))
                 for key in metrics} for off in (0, 8)},
        prior_gate_unchanged=original['advance_gate_passed'],
        code_sha256=sha(Path(__file__)), token_losses_sha256=sha(OUT / 'token_losses.jsonl'),
        limitations=['Previously examined test data; descriptive diagnostic only.',
          'Common horizon is outcome-dependent; remainder is not a causal length effect.',
          'GPT-2 is the training proxy; AR teacher is a model diagnostic, not human quality.',
          'Frozen cohort half-selection is not a deployable policy or timing comparison.',
          'Intervals condition on fitted models, cohort ranks and the eight sampled futures.']))
    print(json.dumps(dict(means=means, correlations=correlations, strata=strata), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['freeze', 'score', 'analyze'])
    args = ap.parse_args()
    globals()[args.action]()
