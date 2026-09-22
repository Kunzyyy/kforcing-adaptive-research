import concurrent.futures
import argparse
import hashlib
import json
import pathlib
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--dest', type=pathlib.Path, required=True)
ROOT = parser.parse_args().dest
REV = '16b984316bdd4585f17b11d0e61a58d236c2febc'
URL = 'https://huggingface.co'
metadata = json.load(urllib.request.urlopen(f'{URL}/api/models/zwave/K-Forcing/tree/{REV}', timeout=30))
selected = [x for x in metadata if x['path'] in ('ar_best_lm1b.ckpt', 'pflm_lm1b_k4.ckpt')]
ROOT.mkdir(parents=True, exist_ok=True)

def fetch(item):
    name = item['path']
    target = ROOT / name
    temp = target.with_suffix('.part')
    if not target.exists():
        req = urllib.request.Request(f'{URL}/zwave/K-Forcing/resolve/{REV}/{name}?download=true')
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(req, timeout=120) as src, temp.open('wb') as dst:
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if count % (64 * 1024 * 1024) == 0:
                    print(f'{name}: {count/1024**2:.0f}/{item["size"]/1024**2:.0f} MiB', flush=True)
        assert count == item['size'], (name, count)
        assert digest.hexdigest() == item['lfs']['oid'], name
        temp.replace(target)
    else:
        digest = hashlib.sha256()
        with target.open('rb') as src:
            while chunk := src.read(8*1024*1024):
                digest.update(chunk)
        assert digest.hexdigest() == item['lfs']['oid'], name
    print(f'VERIFIED {name} {digest.hexdigest()}', flush=True)
    return {'filename': name, 'sha256': digest.hexdigest(), 'size': item['size'], 'revision': REV}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(fetch, selected))
(ROOT / 'manifest.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
