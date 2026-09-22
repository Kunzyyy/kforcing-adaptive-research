"""Pinned first-party dataset-converter held-out parquet; no remote Python execution."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request
import pyarrow.parquet as pq
from transformers import BertTokenizerFast

REVISION='8d52bfd3cc2819fc1166dbb8144c328e2690de3e'
SHA256='d3c7b4b2a48c24e9ae8353d6316dbac1453b08f2d870da9fa096c941e8f4cc8a'
URL=f'https://huggingface.co/datasets/billion-word-benchmark/lm1b/resolve/{REVISION}/plain_text/test/0000.parquet'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--vocab',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.dest.mkdir(parents=True,exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.dest/'test.parquet'
    if not path.exists():
        data=urllib.request.urlopen(URL,timeout=90).read()
        assert hashlib.sha256(data).hexdigest()==SHA256
        path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA256
    manifest=dict(repo='billion-word-benchmark/lm1b',revision=REVISION,url=URL,sha256=SHA256,
        size=path.stat().st_size,split='test',source_builder_revision='35161838ea9e05371a25a8db037f94fcae4c2064',
        note='Hugging Face conversion of heldout-monolingual.tokenized.shuffled, not training data.')
    (args.out/'lm1b_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    target=args.out/'prefixes.json'
    if target.exists():
        raise RuntimeError('LM1B prefixes already frozen')
    texts=pq.read_table(path,columns=['text'])['text'].to_pylist()
    indices=list(range(len(texts)))
    random.Random(4041601).shuffle(indices)
    tokenizer=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=True)
    seen=set()
    for folder in ('pilot-001','benefit-002'):
        old=args.out.parent/folder/'prefixes.json'
        if old.exists():
            for rows in json.loads(old.read_text(encoding='utf-8')).values():
                seen.update(tuple(r['ids']) for r in rows)
    old=args.out.parent/'current-003/fresh_prefixes.json'
    if old.exists():
        seen.update(tuple(r['ids']) for r in json.loads(old.read_text(encoding='utf-8')))
    result=[]
    for i in indices:
        ids=tokenizer.encode(texts[i],add_special_tokens=True,truncation=True,max_length=128)
        if len(ids)<12 or tuple(ids[:6]) in seen:
            continue
        seen.add(tuple(ids[:6]))
        result.append(dict(id=f'lm1b-{len(result):03}',source_row=i,
            source_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),ids=ids[:6],reference_ids=ids,
            prefix=tokenizer.decode(ids[:6],skip_special_tokens=False)))
        if len(result)==128:
            break
    assert len(result)==128
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(total_heldout_rows=len(texts),selected_prefixes=len(result),manifest=manifest),indent=2))


if __name__=='__main__':
    main()
