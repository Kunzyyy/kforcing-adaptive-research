"""Reconstruct sampling and review text independently; verify the public/private boundary."""
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unittest
import zipfile
from transformers import BertTokenizer
import test_blind_quality

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/blind-quality'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')

def main():
    manifest=read(OUT/'manifest.json'); packet=read(OUT/'reviewer/tasks.json'); key=read(OUT/'private/key.json')
    expected={'results/decision-confirm/test_pairs.jsonl','results/decision-confirm/prefixes.json',
              'BLIND_QUALITY_PLAN.md','build_blind_quality.py','blind_quality_viewer.html','analyze_blind_quality.py'}
    assert set(manifest['input_whitelist'])==expected
    for name,digest in manifest['input_whitelist'].items(): assert sha(ROOT/name)==digest,name
    assert manifest['label_or_prediction_files_read'] is False
    builder=(ROOT/'build_blind_quality.py').read_text(encoding='utf-8')
    assert 'test_labels.jsonl' not in builder and 'frozen_predictions.json' not in builder
    rows=[json.loads(line) for line in (ROOT/'results/decision-confirm/test_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    prefixes={p['prompt_id']:p for p in read(ROOT/'results/decision-confirm/prefixes.json')}
    groups=defaultdict(list)
    for row in rows:
        if row['status']=='paired': groups[row['state_id']].append(row)
    eligible={s:batch for s,batch in groups.items() if len(batch)==8 and sorted(p['replicate'] for p in batch)==list(range(8))}
    sources=sorted({p['prompt_id'] for batch in eligible.values() for p in batch})
    rng=random.Random(22212001); selected=rng.sample(sources,64); selected_pairs=[]
    for source in selected:
        states=sorted(s for s,b in eligible.items() if b[0]['prompt_id']==source)
        state=states[rng.randrange(len(states))]; replicate=rng.randrange(8)
        selected_pairs.append(next(row for row in eligible[state] if row['replicate']==replicate))
    main_keys={v['prompt_id']:(i,v) for i,v in key['items'].items() if v['kind']=='main'}
    assert len(main_keys)==64 and set(main_keys)==set(selected)
    vocab=ROOT.parents[1]/'work/pilot-data/vocab.txt'
    assert sha(vocab)==manifest['vocab_sha256']
    tokenizer=BertTokenizer(vocab_file=str(vocab),do_lower_case=True)
    public={t['id']:t for t in packet['tasks']}
    for row in selected_pairs:
        ident,mapping=main_keys[row['prompt_id']]; task=public[ident]
        assert row['pair_id']==mapping['pair_id'] and row['state_id']==mapping['state_id'] and row['replicate']==mapping['replicate']
        assert task['prefix']==tokenizer.decode(prefixes[row['prompt_id']]['ids'],skip_special_tokens=True,clean_up_tokenization_spaces=True)
        for side in ['A','B']:
            ids=row[mapping[side+'_method']]['ids']
            assert ids[:6]==prefixes[row['prompt_id']]['ids']
            assert 102 not in ids[6:-1]
            assert task[side]==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=True)
    assert all(set(t)=={'id','prefix','A','B'} for t in packet['tasks'])
    assert hashlib.sha256(canonical(packet['tasks'])).hexdigest()==packet['tasks_sha256']==key['tasks_sha256']
    assert set(packet)=={'schema_version','packet_id','tasks_sha256','tasks'}
    assert Counter(v['kind'] for v in key['items'].values())=={'main':64,'swap':8,'identical':8}
    # Independently reconstruct balanced orientation, opaque IDs, controls and shuffled task order.
    orient=random.Random(22212002); methods=['split22']*32+['keep4']*32; orient.shuffle(methods)
    main_ids=[]
    for row,method in zip(selected_pairs,methods):
        ident=f'item-{orient.getrandbits(64):016x}'; main_ids.append(ident)
        assert main_keys[row['prompt_id']][0]==ident and key['items'][ident]['A_method']==method
    controls=random.Random(22212003); nonidentical=[i for i in main_ids if public[i]['A']!=public[i]['B']]
    task_order=list(main_ids)
    for kind,parents in [('swap',controls.sample(nonidentical,8)),('identical',controls.sample(main_ids,8))]:
        for parent in parents:
            ident=f'item-{orient.getrandbits(64):016x}'; task_order.append(ident)
            assert key['items'][ident]['kind']==kind and key['items'][ident]['parent_id']==parent
    random.Random(22212004).shuffle(task_order)
    assert task_order==[t['id'] for t in packet['tasks']]
    template=(ROOT/'blind_quality_viewer.html').read_text(encoding='utf-8')
    html=(OUT/'reviewer/index.html').read_text(encoding='utf-8')
    match=re.search(r'const PACKET = (.*?);\nconst \$',html,re.S)
    assert match and json.loads(match.group(1))==packet
    assert '<' not in match.group(1) and '\u2028' not in match.group(1) and '\u2029' not in match.group(1)
    assert html.count('<script>')==html.count('</script>')==1
    assert 'innerHTML' not in template and 'fetch(' not in template and 'XMLHttpRequest' not in template
    assert 'split22' not in html and 'keep4' not in html and 'GPT-2' not in html
    for name,digest in manifest['public_hashes'].items(): assert sha(OUT/'reviewer'/name)==digest
    assert sha(OUT/'private/key.json')==manifest['private_key_sha256']
    archive=ROOT.parent/'kforcing-blind-review.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())=={'README.md','tasks.json','index.html'}
        for name in z.namelist(): assert z.read(name)==(OUT/'reviewer'/name).read_bytes()
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_blind_quality)
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    status=read(OUT/'status.json')
    assert status['human_ratings_received']==0 and status['quality_result']=='not_evaluated'
    audit=dict(passed=True,packet_id=packet['packet_id'],manifest_sha256=sha(OUT/'manifest.json'),
        verifier_sha256=sha(Path(__file__)),test_code_sha256=sha(ROOT/'test_blind_quality.py'),
        public_archive_sha256=sha(archive),public_archive_files=3,public_archive_bytes=archive.stat().st_size,
        independently_reconstructed_sources=64,independently_decoded_branch_texts=128,
        main_A_split22=32,main_A_keep4=32,main_sources=64,swap_controls=8,identical_controls=8,
        natural_identical_main=manifest['natural_identical_main'],structural_sampling_matches=True,
        label_or_prediction_input_in_builder=False,public_packet_contains_key=False,
        synthetic_test_cases=tests.testsRun,synthetic_fixtures_written_as_ratings=False,
        human_ratings_received=0,quality_result='not_evaluated')
    (OUT/'integrity_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
