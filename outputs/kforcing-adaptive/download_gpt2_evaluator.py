"""Pinned GPT-2-Large safetensors and tokenizer, with LFS SHA-256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed

REPO='openai-community/gpt2-large'
REVISION='32b71b12589c2f8d625668d2335a01cac3249519'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dest',type=Path,required=True)
    args=ap.parse_args()
    args.dest.mkdir(parents=True,exist_ok=True)
    meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=60))
    names=['config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors']
    manifest=dict(repo=REPO,revision=REVISION,files=[])
    for name in names:
        entry=next(s for s in meta['siblings'] if s['rfilename']==name)
        path=args.dest/name
        url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true'
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.part')
            total=partial.stat().st_size if partial.exists() else 0
            chunk_size=16*1024*1024
            chunk_dir=args.dest/(name+'.chunks')
            chunk_dir.mkdir(exist_ok=True)
            chunks=[(offset,min(offset+chunk_size,entry['size'])) for offset in range(total,entry['size'],chunk_size)]
            def fetch_chunk(bounds):
                begin,end=bounds
                target=chunk_dir/f'{begin}-{end}'
                if target.exists() and target.stat().st_size==end-begin:
                    return target
                error=None
                for attempt in range(4):
                    req=urllib.request.Request(url+f'&start={begin}&end={end}&attempt={attempt}',
                        headers={'Range':f'bytes={begin}-{end-1}'})
                    try:
                        with urllib.request.urlopen(req,timeout=25) as source:
                            assert source.status==206 and source.headers.get('Content-Range','').startswith(f'bytes {begin}-'),'Server ignored range'
                            content=source.read(end-begin+1)
                        assert len(content)==end-begin,'Incomplete range'
                        target.write_bytes(content)
                        return target
                    except (TimeoutError,OSError,AssertionError) as e:
                        error=e
                raise RuntimeError(f'Chunk {begin} failed: {error}')
            completed=total
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(fetch_chunk,b):b for b in chunks}
                for future in as_completed(futures):
                    target=future.result()
                    completed+=target.stat().st_size
                    print(f'{name}: verified-length chunks {completed/1024**2:.0f}/{entry["size"]/1024**2:.0f} MiB',flush=True)
            with partial.open('ab') as dest:
                for begin,end in chunks:
                    target=chunk_dir/f'{begin}-{end}'
                    dest.write(target.read_bytes())
                    total+=end-begin
            assert total==entry['size'],'Incomplete download'
            partial.replace(path)
        h=hashlib.sha256()
        with path.open('rb') as source:
            while block:=source.read(8*1024*1024):
                h.update(block)
        if 'lfs' in entry:
            assert h.hexdigest()==entry['lfs']['sha256'],'LFS hash mismatch'
        else:
            content=path.read_bytes()
            git_blob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            assert git_blob==entry['blobId'],'Git blob hash mismatch'
        manifest['files'].append(dict(name=name,size=path.stat().st_size,sha256=h.hexdigest(),url=url))
        (args.dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('Verified',name,path.stat().st_size,flush=True)


if __name__=='__main__':
    main()
