"""Build a method/score-blinded offline human review packet from frozen paired texts."""
from collections import defaultdict
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile
from transformers import BertTokenizerFast

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/blind-quality'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def select(pairs):
    bystate=defaultdict(list)
    for pair in pairs:
        if pair.get('status')=='paired': bystate[pair['state_id']].append(pair)
    bysource=defaultdict(list)
    for state,batch in sorted(bystate.items()):
        if len(batch)==8 and {p['replicate'] for p in batch}==set(range(8)):
            assert len({p['prompt_id'] for p in batch})==1
            bysource[batch[0]['prompt_id']].append(state)
    rng=random.Random(22212001)
    source_ids=rng.sample(sorted(bysource),64)
    chosen=[]
    for source in source_ids:
        state=rng.choice(bysource[source]); replicate=rng.randrange(8)
        chosen.append(next(p for p in bystate[state] if p['replicate']==replicate))
    return chosen,dict(eligible_sources=len(bysource),eligible_states=sum(map(len,bysource.values())),
        total_input_pairs=len(pairs),sampled_sources=64)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocab',type=Path,default=ROOT.parents[1]/'work/pilot-data/vocab.txt')
    args=parser.parse_args()
    archive=ROOT.parent/'kforcing-blind-review.zip'
    assert not OUT.exists() and not archive.exists(), 'Refusing to overwrite a frozen review packet'
    inputs=['results/decision-confirm/test_pairs.jsonl','results/decision-confirm/prefixes.json',
            'BLIND_QUALITY_PLAN.md','build_blind_quality.py','blind_quality_viewer.html','analyze_blind_quality.py']
    hashes={name:sha(ROOT/name) for name in inputs}
    pairs=[json.loads(s) for s in (ROOT/inputs[0]).read_text(encoding='utf-8').splitlines()]
    prefixes={p['prompt_id']:p for p in read(ROOT/inputs[1])}
    chosen,counts=select(pairs)
    tokenizer=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=True)
    order_rng=random.Random(22212002); orientations=['split22']*32+['keep4']*32; order_rng.shuffle(orientations)
    tasks=[]; key={}; main_ids=[]
    def append_task(prefix,a,b,metadata):
        ident=f'item-{order_rng.getrandbits(64):016x}'
        assert ident not in key
        tasks.append(dict(id=ident,prefix=prefix,A=a,B=b)); key[ident]=metadata
        return ident
    for pair,a_method in zip(chosen,orientations):
        b_method='keep4' if a_method=='split22' else 'split22'
        for method in ['keep4','split22']:
            ids=pair[method]['ids']; assert ids[:6]==prefixes[pair['prompt_id']]['ids']
            assert 102 not in ids[6:-1]
        prefix=tokenizer.decode(prefixes[pair['prompt_id']]['ids'],skip_special_tokens=True,clean_up_tokenization_spaces=True)
        texts={method:tokenizer.decode(pair[method]['ids'],skip_special_tokens=True,clean_up_tokenization_spaces=True)
               for method in ['keep4','split22']}
        main_ids.append(append_task(prefix,texts[a_method],texts[b_method],dict(kind='main',
            pair_id=pair['pair_id'],state_id=pair['state_id'],prompt_id=pair['prompt_id'],
            replicate=pair['replicate'],A_method=a_method,B_method=b_method,natural_identical=texts[a_method]==texts[b_method])))
    lookup={x['id']:x for x in tasks}; controls=random.Random(22212003)
    candidates=[i for i in main_ids if lookup[i]['A']!=lookup[i]['B']]
    for ident in controls.sample(candidates,8):
        t=lookup[ident]; k=key[ident]
        append_task(t['prefix'],t['B'],t['A'],dict(kind='swap',parent_id=ident,A_method=k['B_method'],B_method=k['A_method']))
    for ident in controls.sample(main_ids,8):
        t=lookup[ident]
        append_task(t['prefix'],t['A'],t['A'],dict(kind='identical',parent_id=ident))
    random.Random(22212004).shuffle(tasks)
    packet_id='bq-'+hashlib.sha256(canonical(tasks)).hexdigest()[:16]
    payload=dict(schema_version=1,packet_id=packet_id,tasks_sha256=hashlib.sha256(canonical(tasks)).hexdigest(),tasks=tasks)
    OUT.mkdir(exist_ok=False); public=OUT/'reviewer'; private=OUT/'private'; public.mkdir(); private.mkdir()
    dump(public/'tasks.json',payload)
    # All model/score/source identifiers remain private; the page embeds public task text only.
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    template=(ROOT/'blind_quality_viewer.html').read_text(encoding='utf-8')
    assert template.count('/*TASK_DATA*/')==1
    (public/'index.html').write_text(template.replace('/*TASK_DATA*/',encoded),encoding='utf-8')
    (public/'README.md').write_text('# 文本比较评审\n\n解压后用浏览器打开 index.html。所有操作在本地完成。使用不同评审代号独立作答，完成后导出JSON并保留一份副本。无需发送真实姓名。\n\n请先阅读页内规则；评价前文衔接、语法、连贯与自然程度，不按文本长短投票。A/B均包含相同前文；可选相当或无法判断。不要查看其他评审答案或方法标签。\n\n刷新会尝试恢复本浏览器进度；无痕模式、换浏览器或清理网站数据可能丢失。建议定期导出，可用“导入进度”继续。这个评审包不自动发送任何答案。\n',encoding='utf-8')
    dump(private/'key.json',dict(packet_id=packet_id,tasks_sha256=payload['tasks_sha256'],items=key))
    dump(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),packet_id=packet_id,
        input_whitelist=hashes,vocab_sha256=sha(args.vocab),sampling=counts,
        source_seed=22212001,orientation_seed=22212002,control_seed=22212003,order_seed=22212004,
        main_tasks=64,swap_controls=8,identical_controls=8,public_tasks=80,
        natural_identical_main=sum(key[i]['natural_identical'] for i in main_ids),
        label_or_prediction_files_read=False,human_ratings_received=0,quality_result='not_evaluated',
        public_hashes={p.name:sha(p) for p in public.iterdir()},private_key_sha256=sha(private/'key.json')))
    dump(OUT/'status.json',dict(packet_ready=True,human_ratings_received=0,quality_result='not_evaluated',
        externally_sent=False,new_adaptive_effect_established=False))
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(public.iterdir()): z.write(p,p.name)
    print(json.dumps(dict(packet_id=packet_id,**counts,tasks=len(tasks),archive=str(archive),bytes=archive.stat().st_size),indent=2))

if __name__=='__main__': main()
