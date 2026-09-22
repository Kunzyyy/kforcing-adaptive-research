"""Analyze independently returned ratings; no ratings or model-derived substitutes supplied.

Usage: python analyze_blind_quality.py --ratings reviewer1.json reviewer2.json --out analysis.json
No arguments prints the honest empty status. Synthetic fixtures are only accepted by
analyze(..., allow_synthetic=True), never by the command-line entry point.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import unicodedata

ROOT = Path(__file__).resolve().parent / 'results/blind-quality'
CHOICES = {'A', 'B', 'TIE', 'SKIP'}

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_packet(packet, key):
    require(packet.get('schema_version') == 1, 'Unsupported packet version')
    digest = hashlib.sha256(canonical(packet['tasks'])).hexdigest()
    require(digest == packet['tasks_sha256'] == key['tasks_sha256'], 'Task digest mismatch')
    require(packet['packet_id'] == key['packet_id'] == 'bq-' + digest[:16], 'Packet ID mismatch')
    ids = [t['id'] for t in packet['tasks']]
    require(len(ids) == len(set(ids)) == 80 and set(ids) == set(key['items']), 'Expected 80 unique task IDs')
    require(all(set(t) == {'id', 'prefix', 'A', 'B'} for t in packet['tasks']), 'Unexpected public task fields')
    kinds = Counter(v['kind'] for v in key['items'].values())
    require(kinds == {'main': 64, 'swap': 8, 'identical': 8}, 'Unexpected design counts')
    main = {i: k for i, k in key['items'].items() if k['kind'] == 'main'}
    require(len({k['prompt_id'] for k in main.values()}) == 64, 'Main tasks must use distinct sources')
    require(Counter(k['A_method'] for k in main.values()) == {'split22': 32, 'keep4': 32}, 'Unbalanced orientation')
    tasks = {t['id']: t for t in packet['tasks']}
    for i, k in key['items'].items():
        t = tasks[i]
        if k['kind'] in {'main', 'swap'}:
            require({k['A_method'], k['B_method']} == {'keep4', 'split22'}, 'Invalid method mapping')
        if k['kind'] == 'main':
            require(k['natural_identical'] == (t['A'] == t['B']), 'Natural-identical flag mismatch')
        else:
            require(k['parent_id'] in main, 'Invalid control parent')
            parent = tasks[k['parent_id']]
            require(t['prefix'] == parent['prefix'], 'Control prefix mismatch')
            if k['kind'] == 'swap':
                require(t['A'] == parent['B'] and t['B'] == parent['A'] and t['A'] != t['B'], 'Invalid swap control')
                require(k['A_method'] == main[k['parent_id']]['B_method'], 'Invalid swapped method')
            else:
                require(t['A'] == t['B'] == parent['A'], 'Invalid identical control')
    return main

def validate_rating(raw, packet, allow_synthetic=False):
    require(raw.get('schema_version') == 1 and raw.get('packet_id') == packet['packet_id'] and
            raw.get('tasks_sha256') == packet['tasks_sha256'], 'Rating packet/version mismatch')
    synthetic = raw.get('data_origin') == 'synthetic'
    require(raw.get('data_origin') == 'human_self_reported' or (allow_synthetic and synthetic), 'Synthetic/unknown origin rejected')
    reviewer = raw.get('reviewer', {})
    alias = reviewer.get('alias')
    require(isinstance(alias, str) and bool(alias.strip()) and len(alias) <= 80, 'Reviewer alias required')
    require(reviewer.get('human_confirmed') is True or (allow_synthetic and synthetic), 'Human declaration required')
    require(raw.get('complete') is True, 'Partial rating file rejected; use SKIP for explicit uncertainty')
    answers = raw.get('answers')
    require(isinstance(answers, list) and len(answers) == 80, 'Expected all 80 answers')
    by_id = {}
    expected = {t['id'] for t in packet['tasks']}
    for answer in answers:
        require(isinstance(answer, dict), 'Invalid answer')
        ident = answer.get('task_id')
        require(isinstance(ident, str) and ident in expected and ident not in by_id, 'Unknown/duplicate task ID')
        require(isinstance(answer.get('choice'), str) and answer['choice'] in CHOICES, 'Unanswered/invalid choice')
        require(type(answer.get('both_bad')) is bool, 'both_bad must be boolean')
        require(isinstance(answer.get('comment'), str) and len(answer['comment']) <= 2000, 'Invalid comment')
        by_id[ident] = answer
    normalized_alias = unicodedata.normalize('NFKC', alias.strip()).casefold()
    return alias.strip(), normalized_alias, by_id, synthetic

def score(answer, mapping):
    choice = answer['choice']
    if choice == 'SKIP':
        return None
    if choice == 'TIE':
        return 0
    return 1 if mapping[choice + '_method'] == 'split22' else -1

def fraction(n, d):
    return n / d if d else None

def percentile(values, p):
    position = (len(values) - 1) * p
    i = int(position)
    weight = position - i
    return values[i] * (1 - weight) + values[min(i + 1, len(values) - 1)] * weight

def analyze(packet, key, raw_ratings, allow_synthetic=False):
    main = validate_packet(packet, key)
    require(len(raw_ratings) >= 2, 'At least two independent human reviewers required')
    ratings = [validate_rating(r, packet, allow_synthetic) for r in raw_ratings]
    require(len({r[1] for r in ratings}) == len(ratings), 'Duplicate reviewer aliases')
    synthetic = any(r[3] for r in ratings)
    require(not synthetic or all(r[3] for r in ratings), 'Do not mix synthetic and human ratings')
    main_ids = sorted(main, key=lambda i: main[i]['prompt_id'])
    scores = {i: [score(r[2][i], main[i]) for r in ratings] for i in main_ids}
    complete = [i for i in main_ids if all(s is not None for s in scores[i])]
    observed = [s for group in scores.values() for s in group if s is not None]
    complete_votes = [s for i in complete for s in scores[i]]
    total = 64 * len(ratings)
    missing = total - len(observed)
    source_means = [sum(scores[i]) / len(ratings) for i in complete]
    sufficient = len(complete) >= 48
    estimate = interval = None
    if sufficient:
        estimate = sum(source_means) / len(source_means)
        rng = random.Random(22212005)
        boot = sorted(sum(rng.choice(source_means) for _ in source_means) / len(source_means) for _ in range(4000))
        interval = [percentile(boot, .025), percentile(boot, .975)]
    def counts(values):
        c = Counter(values)
        return dict(n=len(values), split_wins=c[1], keep_wins=c[-1], ties=c[0],
                    split_rate=fraction(c[1], len(values)), keep_rate=fraction(c[-1], len(values)), tie_rate=fraction(c[0], len(values)))
    reviewer_quality = []
    for alias, _, answers, _synthetic in ratings:
        swaps = []; identical = []; swap_skip = 0; identical_skip = 0
        for ident, mapping in key['items'].items():
            if mapping['kind'] == 'swap':
                a = score(answers[ident], mapping)
                b = score(answers[mapping['parent_id']], main[mapping['parent_id']])
                if a is None or b is None:
                    swap_skip += 1
                else:
                    swaps.append(a == b)
            elif mapping['kind'] == 'identical':
                choice = answers[ident]['choice']
                if choice == 'SKIP':
                    identical_skip += 1
                else:
                    identical.append(choice == 'TIE')
        reviewer_quality.append(dict(alias=alias, main_skipped=sum(answers[i]['choice'] == 'SKIP' for i in main_ids),
            both_bad_main=sum(answers[i]['both_bad'] for i in main_ids), both_bad_denominator=64,
            swap_consistent=sum(swaps), swap_comparable=len(swaps), swap_missing=swap_skip,
            swap_consistency=fraction(sum(swaps), len(swaps)), identical_ties=sum(identical),
            identical_nonmissing=len(identical), identical_missing=identical_skip,
            identical_tie_rate=fraction(sum(identical), len(identical)), excluded=False))
    agreements = []
    for r1, r2 in itertools.combinations(range(len(ratings)), 2):
        comparable = [i for i in main_ids if scores[i][r1] is not None and scores[i][r2] is not None]
        equal = sum(scores[i][r1] == scores[i][r2] for i in comparable)
        agreements.append(dict(reviewers=[ratings[r1][0], ratings[r2][0]], agreed=equal,
                               comparable=len(comparable), agreement=fraction(equal, len(comparable))))
    return dict(packet_id=packet['packet_id'], tasks_sha256=packet['tasks_sha256'],
        data_origin='synthetic' if synthetic else 'human_self_reported', identity_independence_verified=False,
        human_ratings_received=0 if synthetic else len(ratings), reviewer_files=len(ratings),
        quality_result='synthetic_test_only' if synthetic else ('descriptive_human_preferences' if sufficient else 'insufficient_complete_sources'),
        scope='single keep4/split22 intervention; does not evaluate an adaptive selector or matched runtime',
        main=dict(total_sources=64, complete_sources=len(complete), incomplete_sources=64-len(complete),
            retention_rate=len(complete)/64, minimum_complete_sources=48,
            natural_identical_sources=sum(k['natural_identical'] for k in main.values()),
            natural_identical_rate=sum(k['natural_identical'] for k in main.values())/64,
            mean_net_split_preference=estimate, source_bootstrap_95ci=interval, bootstrap_repeats=4000,
            bootstrap_seed=22212005, interval_scope='source sampling only, conditional on these reviewers',
            complete_case_votes=counts(complete_votes), all_observed_votes=counts(observed),
            missing_votes=missing, total_possible_votes=total,
            all_source_missing_bounds=[(sum(observed)-missing)/total, (sum(observed)+missing)/total]),
        reviewer_checks=reviewer_quality, pairwise_raw_agreement=agreements,
        new_adaptive_effect_established=False, no_posthoc_reviewer_exclusions=True)

def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, nargs='+')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.ratings:
        print(json.dumps(dict(human_ratings_received=0, quality_result='not_evaluated',
                             note='No rating files supplied to this invocation.'), indent=2))
        return
    if args.out is None:
        parser.error('--out is required when ratings are supplied')
    if args.out.exists():
        parser.error('Refusing to overwrite an existing analysis file')
    try:
        report = analyze(read(ROOT/'reviewer/tasks.json'), read(ROOT/'private/key.json'), [read(p) for p in args.ratings])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    report['rating_input_sha256'] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.ratings]
    report['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('quality_result', 'human_ratings_received', 'main')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main_cli()
