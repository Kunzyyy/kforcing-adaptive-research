"""Descriptive post-generation diagnostics; never feeds back into selection."""
import argparse
import json
from pathlib import Path
import numpy as np


def diagnose(root):
    data = [json.loads(s) for s in (root / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(data) == 640
    learned = [r for r in data if r['method'] == 'learned']
    result = {}
    for method in ('learned', 'random_matched'):
        group = [r for r in data if r['method'] == method]
        eligible = [s for r in group for s in r['steps'] if s['width'] == 4]
        pre_eos = [s for r in group for s in r['steps']
                   if s['width'] == 4 and s['position'] < 6 + r['visible_new_tokens']]
        after_eos = [s for r in group for s in r['steps']
                     if s['width'] == 4 and s['position'] >= 6 + r['visible_new_tokens']]
        result[method] = dict(all_windows=len(eligible), all_split_fraction=float(np.mean([s['split'] for s in eligible])),
            windows_starting_before_eos=len(pre_eos), split_fraction_before_eos=float(np.mean([s['split'] for s in pre_eos])),
            windows_starting_at_or_after_visible_end=len(after_eos),
            split_fraction_after_visible_end=float(np.mean([s['split'] for s in after_eos])) if after_eos else None,
            split_count_min=min(r['split_count'] for r in group), split_count_max=max(r['split_count'] for r in group))
    positional = []
    for j in range(30):
        steps = [r['steps'][j] for r in learned]
        alive = [r['steps'][j] for r in learned if r['steps'][j]['position'] < 6 + r['visible_new_tokens']]
        positional.append(dict(window_index=j, context_length=6 + j * 4,
            all_split_fraction=float(np.mean([s['split'] for s in steps])),
            all_mean_score=float(np.mean([s['score'] for s in steps])),
            visible_histories=len(alive), visible_split_fraction=float(np.mean([s['split'] for s in alive])) if alive else None))
    index = {(r['prompt_id'], r['seed'], r['method']): r for r in data}
    identities = sorted({(r['prompt_id'], r['seed']) for r in data})
    identical = {}
    for other in ('fixed2', 'fixed3', 'fixed4', 'random_matched'):
        identical[other] = dict(full=sum(index[k + ('learned',)]['ids'] == index[k + (other,)]['ids'] for k in identities),
            visible=sum(index[k + ('learned',)]['visible_ids'] == index[k + (other,)]['visible_ids'] for k in identities))
    timing = [json.loads(s) for s in (root / 'timings.jsonl').read_text(encoding='utf-8').splitlines()]
    repetition = []
    for rep in range(3):
        durations = {method: sum(t['seconds'] for t in timing if t['method'] == method and t['repetition'] == rep)
                     for method in ('learned', 'random_matched')}
        repetition.append(dict(repetition=rep, learned_over_random_throughput=durations['random_matched'] / durations['learned']))
    result.update(learned_position_statistics=positional, learned_identical_outputs=identical,
        timing_ratios_by_repetition=repetition,
        note='Exploratory description after generation; no refit. Visible-window budgets need not match even though full-horizon counts match. Post-EOS states are artificial benchmark continuations.')
    (root / 'budget_diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'learned_position_statistics'}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path)
    diagnose(ap.parse_args().root)
