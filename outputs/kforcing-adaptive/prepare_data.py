"""Download pinned public WikiText-2 splits and BERT vocabulary; no remote code."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def get(url):
    return urllib.request.urlopen(url, timeout=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', type=Path, required=True)
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    dataset = {'sha': 'b08601e04326c79dfdd32d625aee71d232d685c3'}
    tokenizer = {'sha': '86b5e0934494bd15c9632b12f734a8a67f723594'}
    files = [(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{dataset["sha"]}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet', f'{split}.parquet') for split in ['validation', 'test']]
    files.append((f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{tokenizer["sha"]}/vocab.txt', 'vocab.txt'))
    manifest = {'dataset': 'Salesforce/wikitext/wikitext-2-raw-v1',
        'dataset_revision': dataset['sha'], 'tokenizer_revision': tokenizer['sha'], 'files': []}
    for url, name in files:
        content = get(url).read()
        (args.dest / name).write_bytes(content)
        manifest['files'].append(dict(name=name, url=url, sha256=hashlib.sha256(content).hexdigest(), size=len(content)))
        print(name, len(content), flush=True)
    (args.dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
