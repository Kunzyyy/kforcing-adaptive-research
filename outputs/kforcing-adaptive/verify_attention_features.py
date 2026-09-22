"""CPU-only audit independent of the feature extractor and ridge fitting code."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/attention-features'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, atol=1e-11):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def six_features(probabilities, history):
    # Deliberately use per-query/per-head loops and float64 rather than Torch reductions.
    means, maxima, entropies = [], [], []
    for tail in range(2):
        masses, normalized = [], []
        for head in probabilities:
            row = head[tail].astype(np.float64)
            masses.append(float(row[history] + row[history + 1]))
            positive = row[row > 0]
            normalized.append(float(-sum(positive * np.log(positive)) / np.log(history + 3 + tail)))
        means.append(float(np.mean(masses)))
        maxima.append(float(max(masses)))
        entropies.append(float(np.mean(normalized)))
    return np.array(means + maxima + entropies)


def main(output=None):
    manifest = read(OUT / 'manifest.json')
    collection = read(OUT / 'collection.json')
    analysis = read(OUT / 'analysis.json')
    expected_inputs = {
        'results/features-014/development_data.jsonl',
        'results/decision-features/features.jsonl',
        'results/decision-features/analysis.json',
        'results/decision-features/manifest.json',
        'results/averaged-013/train_states.jsonl',
        'results/averaged-013/dev_states.jsonl',
        'ATTENTION_FEATURE_PLAN.md', 'attention_features.py',
        'diagnose_attention_features.py', 'core.py', 'current_features.py',
        'upstream/models/pflm.py', 'upstream/models/transformer.py',
    }
    assert set(manifest['inputs']) == expected_inputs
    for name, digest in manifest['inputs'].items():
        assert sha(ROOT / name) == digest, name
    for inventory in [collection['outputs'], analysis['output_hashes']]:
        for name, digest in inventory.items():
            assert sha(OUT / name) == digest, name
    assert manifest['alpha'] == 1000 and manifest['one_candidate']
    assert manifest['retrospective_development'] and not manifest['old_test_inputs']
    seeds = [14112001, 14112002, 14112003]
    assert manifest['fold_seeds'] == seeds and manifest['bootstrap_seed'] == 22232001

    data = rows(ROOT / 'results/features-014/development_data.jsonl')
    prior = rows(ROOT / 'results/decision-features/features.jsonl')
    features = rows(OUT / 'features.jsonl')
    stored_oof = rows(OUT / 'oof_predictions.jsonl')
    fits = read(OUT / 'fits.json')
    ids = [r['state_id'] for r in data]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) == 446
    for records in [prior, features, stored_oof]:
        assert [r['state_id'] for r in records] == ids
        assert [r['prompt_id'] for r in records] == [r['prompt_id'] for r in data]
    assert {r['source_split'] for r in data} == {'train', 'dev'}
    sources = sorted({r['prompt_id'] for r in data})
    assert len(sources) == 253
    current = {}
    for split in ['train', 'dev']:
        for r in rows(ROOT / f'results/averaged-013/{split}_states.jsonl'):
            if r['status'] == 'paired':
                current[r['state_id']] = {k: r[k] for k in [
                    'prompt_id', 'history_ids', 'noise', 'initial_candidates', 'offset']}
    # The frozen development table predates this experiment. One paired state has
    # incomplete valid labels and was already absent; do not redefine eligibility.
    assert len(current) == 447 and set(ids).issubset(current)
    assert set(current) - set(ids) == {'s13-122-1301-0'}
    x = np.array([r['features'] for r in features], dtype=np.float64)
    y = np.array([r['gain'] for r in data], dtype=np.float64)
    assert x.shape == (446, 25) and np.isfinite(x).all() and np.isfinite(y).all()
    close(y, [np.mean(r['gains']) for r in data])
    assert all(len(r['gains']) == 8 for r in data)
    close(x[:, :19], [r['features'] for r in prior], atol=0)
    probabilities = np.load(OUT / 'attention_probabilities.npz', allow_pickle=False)
    assert set(probabilities.files) == set(ids)
    max_feature_error = 0.
    for i, (f, old, d) in enumerate(zip(features, prior, data)):
        s = current[f['state_id']]
        for key in ['prompt_id', 'history_ids', 'noise', 'initial_candidates', 'offset']:
            assert s[key] == old[key]
        assert f['source_split'] == d['source_split'] == old['source_split']
        history = f['history_length']
        assert history == len(s['history_ids']) == 6 + s['offset']
        assert f['offset'] == s['offset'] == d['offset']
        assert f['heads'] == collection['heads'] and f['last_block'] == collection['last_block']
        a = probabilities[f['state_id']]
        assert a.shape == (f['heads'], 2, history + 4)
        assert np.isfinite(a).all() and (a >= 0).all()
        close(a.sum(axis=-1, dtype=np.float64), 1, atol=3e-7)
        assert (a[:, 0, -1] == 0).all()
        recovered = six_features(a, history)
        close(recovered, x[i, 19:], atol=2e-6)
        max_feature_error = max(max_feature_error, float(np.max(abs(recovered - x[i, 19:]))))

    # Independent split-half RoPE: compute rotated halves explicitly in float64.
    probes = np.load(OUT / 'numeric_probes.npz', allow_pickle=False)
    indices = np.linspace(0, 445, 12, dtype=int).tolist()
    assert probes['indices'].tolist() == collection['independent_probe_indices'] == indices
    assert set(probes.files) == {'indices', 'inv_freq'} | {f'{i}_{kind}' for i in indices for kind in ['qkv', 'actual']}
    max_av_error, max_probability_error, max_probe_feature_error = 0., 0., 0.
    for index in indices:
        f = features[index]
        raw = probes[f'{index}_qkv'].astype(np.float64)[0]
        length, width = raw.shape
        heads = f['heads']
        dim = width // (3 * heads)
        assert length == f['history_length'] + 4 and dim % 2 == 0
        packed = raw.reshape(length, 3, heads, dim)
        angles = np.outer(np.arange(length, dtype=np.float64), probes['inv_freq'].astype(np.float64))
        cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
        rotated = []
        for component in [0, 1]:
            left = packed[:, component, :, :dim // 2]
            right = packed[:, component, :, dim // 2:]
            rotated.append(np.concatenate([left * cosine - right * sine, right * cosine + left * sine], axis=-1))
        q, k = rotated
        a = np.zeros((heads, 2, length), dtype=np.float64)
        av = np.zeros((2, heads, dim), dtype=np.float64)
        for head in range(heads):
            for tail in range(2):
                query_position = length - 2 + tail
                scores = np.array([np.dot(q[query_position, head], k[j, head]) / np.sqrt(dim) for j in range(query_position + 1)])
                weights = np.exp(scores - max(scores))
                weights /= sum(weights)
                a[head, tail, :query_position + 1] = weights
                for key, weight in enumerate(weights):
                    av[tail, head] += weight * packed[key, 2, head]
        reference_a = probabilities[f['state_id']]
        close(a, reference_a, atol=2e-5)
        av = av.reshape(2, heads * dim)
        close(av, probes[f'{index}_actual'], atol=2e-4)
        recovered = six_features(a, f['history_length'])
        close(recovered, x[index, 19:], atol=2e-5)
        max_av_error = max(max_av_error, float(np.max(abs(av - probes[f'{index}_actual']))))
        max_probability_error = max(max_probability_error, float(np.max(abs(a - reference_a))))
        max_probe_feature_error = max(max_probe_feature_error, float(np.max(abs(recovered - x[index, 19:]))))
    assert collection['passed'] and collection['states'] == 446
    assert collection['candidate_positions'] == 1784 and collection['unchanged_hook_logit_cases'] == 12
    assert collection['attention_sdpa_comparisons'] == 446 and collection['new_future_generations'] == 0

    dimensions = {'confidence14': 14, 'combined19': 19, 'attention25': 25}
    assert len(fits) == 45
    lookup = {(r['repeat'], r['fold'], r['method']): r for r in fits}
    assert len(lookup) == 45
    pred = {name: np.full((3, 446), np.nan) for name in dimensions}
    selected = {name: np.zeros((3, 446)) for name in dimensions}
    random_fraction = np.full((3, 446), np.nan)
    heuristic = np.zeros((3, 446))
    train_mse = {name: [] for name in dimensions}
    train_weights = {name: [] for name in dimensions}
    max_prediction_error = 0.

    def choose(values, valid_ids):
        order = sorted(range(len(values)), key=lambda j: (-values[j], ids[valid_ids[j]]))
        selection = np.zeros(len(values), dtype=int)
        selection[order[:len(values) // 2]] = 1
        return selection

    for repeat, seed in enumerate(seeds):
        shuffled = np.random.default_rng(seed).permutation(sources).tolist()
        for fold in range(5):
            validation_sources = set(shuffled[fold::5])
            valid = np.array([i for i, r in enumerate(data) if r['prompt_id'] in validation_sources])
            train = np.array([i for i, r in enumerate(data) if r['prompt_id'] not in validation_sources])
            assert not {data[i]['prompt_id'] for i in train} & {data[i]['prompt_id'] for i in valid}
            random_fraction[repeat, valid] = (len(valid) // 2) / len(valid)
            heuristic[repeat, valid] = choose(-(x[valid, 2] + x[valid, 3]) / 2, valid)
            for name, dim in dimensions.items():
                record = lookup[repeat, fold, name]
                assert record['train'] == train.tolist() and record['valid'] == valid.tolist()
                mean = np.mean(x[train, :dim], axis=0)
                std = np.std(x[train, :dim], axis=0)
                std[std < 1e-8] = 1.
                intercept = float(np.mean(y[train]))
                z = (x[train, :dim] - mean) / std
                augmented = np.vstack([z, np.sqrt(1000) * np.eye(dim)])
                target = np.concatenate([y[train] - intercept, np.zeros(dim)])
                coef = np.linalg.lstsq(augmented, target, rcond=None)[0]
                p = (x[valid, :dim] - mean) / std @ coef + intercept
                choice = choose(p, valid)
                for actual, expected in [(mean, record['mean']), (std, record['std']),
                                         (coef, record['coef']), (intercept, record['intercept']),
                                         (p, record['predictions'])]:
                    close(actual, expected)
                close(choice, record['selected'], atol=0)
                error = float(np.mean((z @ coef + intercept - y[train]) ** 2))
                close(error, record['train_mse'])
                train_mse[name].append(error)
                train_weights[name].append(len(train))
                pred[name][repeat, valid] = p
                selected[name][repeat, valid] = choice
                max_prediction_error = max(max_prediction_error, float(np.max(abs(p - record['predictions']))))
    for name in dimensions:
        assert np.isfinite(pred[name]).all()
    for i, row in enumerate(stored_oof):
        close(row['gain'], y[i], atol=0)
        close(row['fraction'], random_fraction[:, i], atol=0)
        close(row['heuristic'], heuristic[:, i], atol=0)
        for name in dimensions:
            close(row['predictions'][name], pred[name][:, i])
            close(row['selected'][name], selected[name][:, i], atol=0)

    # Bootstrap via a source-count matrix, independently of indexed sums in the runner.
    source_index = {s: j for j, s in enumerate(sources)}
    state_source = np.array([source_index[r['prompt_id']] for r in data])
    source_sizes = np.bincount(state_source, minlength=253)
    rng = np.random.default_rng(22232001)
    multiplicities = np.zeros((4000, 253), dtype=np.int64)
    for b in range(4000):
        multiplicities[b] = np.bincount(rng.integers(253, size=253), minlength=253)
    denominators = multiplicities @ source_sizes
    intervals_checked = 0

    def check_interval(values, record, coverage):
        nonlocal intervals_checked
        source_totals = np.bincount(state_source, weights=values, minlength=253)
        draws = multiplicities @ source_totals / denominators
        tail = (1 - coverage) / 2
        bounds = np.quantile(draws, [tail, 1 - tail])
        assert record['coverage'] == coverage
        close(record['estimate'], np.mean(values))
        close(record['interval'], bounds)
        intervals_checked += 1
        return bounds

    for name, dim in dimensions.items():
        record = analysis['methods'][name]
        assert record['dimensions'] == dim
        close(record['oof_mse'], np.mean((pred[name] - y) ** 2))
        close(record['train_mse'], np.average(train_mse[name], weights=train_weights[name]))
        controls = {'random': random_fraction, 'confidence14': selected['confidence14'],
                    'combined19': selected['combined19'], 'heuristic': heuristic}
        for control, choices in controls.items():
            values = np.mean((selected[name] - choices) * y, axis=0)
            check_interval(values, record[f'gain_vs_{control}'], .95)
    primary_bounds = []
    for key, choices in [('vs_combined19', selected['combined19']), ('vs_heuristic', heuristic)]:
        values = np.mean((selected['attention25'] - choices) * y, axis=0)
        primary_bounds.append(check_interval(values, analysis['primary'][key], .975))
    gate = all(bounds[0] > 0 for bounds in primary_bounds) and np.mean((pred['attention25'] - y) ** 2) < np.mean((pred['combined19'] - y) ** 2)
    assert analysis['development_screen_passed'] == bool(gate)
    old = read(ROOT / 'results/decision-features/analysis.json')['methods']
    for name in ['confidence14', 'combined19']:
        for key in ['oof_mse', 'train_mse']:
            close(analysis['methods'][name][key], old[name][key])
        close(analysis['methods'][name]['gain_vs_random']['estimate'], old[name]['gain_vs_random']['estimate'])
    assert analysis['states'] == 446 and analysis['sources'] == 253 and analysis['fits'] == 45
    assert analysis['old_controls_reproduced'] and analysis['retrospective_development']
    for key in ['independent_test', 'old_test_inputs', 'new_human_quality_result', 'new_online_timing']:
        assert analysis[key] is False
    for j, record in enumerate(analysis['feature_ranges'], 19):
        close([record[k] for k in ['min', 'max', 'mean', 'std']],
              [np.min(x[:, j]), np.max(x[:, j]), np.mean(x[:, j]), np.std(x[:, j])])
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)), manifest_sha256=sha(OUT / 'manifest.json'),
        analysis_sha256=sha(OUT / 'analysis.json'), collection_sha256=sha(OUT / 'collection.json'),
        states=446, sources=253, existing_development_eligibility_preserved=True,
        features_reconstructed=2676, numeric_qkv_probes=12,
        max_stored_attention_feature_error=max_feature_error, max_float64_sdpa_output_error=max_av_error,
        max_float64_attention_probability_error=max_probability_error, max_float64_probe_feature_error=max_probe_feature_error,
        augmented_least_squares_fits=45, oof_predictions_recomputed=4014,
        max_prediction_error=max_prediction_error, intervals_recomputed=intervals_checked,
        primary_gate_reproduced=True, development_screen_passed=bool(gate),
        prior_control_points_reproduced=True, train_dev_input_whitelist_verified=True,
        independent_of_runner_imports=True, new_gpu_replay=False, new_human_rating=False,
        limitation='Independent arithmetic audit of saved train/dev evidence, not an independent research replication or new confirmation.')
    if output is not None:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Optional verification report; default only prints, leaving evidence unchanged.')
    main(parser.parse_args().output)
