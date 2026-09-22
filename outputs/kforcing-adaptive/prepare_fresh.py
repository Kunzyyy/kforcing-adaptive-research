"""Select new test articles only after the current-block controller is locked."""
import hashlib
import json
from pathlib import Path
import random
import sys
from transformers import BertTokenizerFast
from collect_benefit import articles,dump


def main():
    root=Path(sys.argv[1])
    data=Path(sys.argv[2])
    if (root/'fresh_prefixes.json').exists():
        raise RuntimeError('Fresh prefixes already frozen')
    controller=json.loads((root/'locked_controller.json').read_text())
    assert controller['passed_development_gate']
    previous=root.parent
    old1=json.loads((previous/'pilot-001/prefixes.json').read_text(encoding='utf-8'))
    old2=json.loads((previous/'benefit-002/prefixes.json').read_text(encoding='utf-8'))
    old_ids={tuple(p['ids']) for d in (old1,old2) for rows in d.values() for p in rows}
    old1_ids={tuple(p['ids']) for rows in old1.values() for p in rows}
    excluded={p['article_id'] for rows in old2.values() for p in rows}
    tok=BertTokenizerFast(vocab_file=str(data/'vocab.txt'),do_lower_case=True)
    groups=articles(data/'test.parquet')
    records={}
    first_stage_articles=set()
    for article,texts in groups.items():
        records[article]=[]
        for text in texts:
            ids=[101]+tok.encode(text,add_special_tokens=False)[:5]
            if tuple(ids) in old1_ids:
                first_stage_articles.add(article)
            records[article].append(dict(article_id=article,ids=ids,prefix=tok.decode(ids),
                source_sha256=hashlib.sha256(text.encode()).hexdigest()))
    excluded|=first_stage_articles
    candidates=[]
    for article,rows in sorted(records.items()):
        if article in excluded:
            continue
        random.Random(article+'stage3').shuffle(rows)
        unique=[]
        seen_local=set()
        for p in rows:
            ids=tuple(p['ids'])
            if ids not in old_ids and ids not in seen_local:
                seen_local.add(ids)
                unique.append(p)
        candidates.extend(unique[:4])
    random.Random(338701).shuffle(candidates)
    selected=[]
    seen=set(old_ids)
    for p in candidates:
        if tuple(p['ids']) in seen:
            continue
        seen.add(tuple(p['ids']))
        p.update(id=f'fresh-{len(selected):03}',split='fresh_test')
        selected.append(p)
        if len(selected)==32:
            break
    inventory=dict(total_test_articles=len(groups),prior_stage1_articles=len(first_stage_articles),
        excluded_test_articles=len(set(groups)&excluded),available_article_count=len(set(groups)-excluded),
        available_capped_candidates=len(candidates),selected_prompts=len(selected),
        selected_articles=len({p['article_id'] for p in selected}),excluded_article_ids=sorted(excluded))
    dump(root/'fresh_inventory.json',inventory)
    if len(selected)!=32:
        raise RuntimeError(f'Insufficient untouched test articles: {inventory}')
    dump(root/'fresh_prefixes.json',selected)
    print(json.dumps({k:v for k,v in inventory.items() if k!='excluded_article_ids'},indent=2))


if __name__=='__main__':
    main()
