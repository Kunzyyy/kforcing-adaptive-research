"""Extract historical EMA storage byte ranges; no historical pickle execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib
import numpy as np
import torch
from core import load_model

ROOT = Path(__file__).resolve().parent


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('model',choices=['pflm','ar']); args=ap.parse_args()
    historical=ROOT/'results/public-history'/('historical_'+args.model)
    meta=read(historical/'metadata_summary.json'); inspection=read(historical/'inspection.json')
    out=Path('work/historical-ema')/args.model; out.mkdir(parents=True,exist_ok=True)
    destination=out/'ema.ckpt'; assert not destination.exists()
    entries={x['name'].split('/data/')[-1]:x for x in inspection['entries'] if '/data/' in x['name']}
    tensors=meta['ema']['shadow_params']
    selected=[entries[t['storage_key']] for t in tensors]
    start=min(x['header_offset'] for x in selected)
    end=max(x['header_offset']+x['bytes']+len(x['name'].encode())+512 for x in selected)
    assert end < inspection['file_bytes']
    span=out/'ema_storage_span.bin'
    chunks=[(i,min(i+8*1024*1024,end)) for i in range(start,end,8*1024*1024)]
    def download(pair):
        a,b=pair; target=out/f'chunk-{a}-{b}.bin'
        if target.exists() and target.stat().st_size==b-a:
            data=target.read_bytes()
            return dict(start=a,end_exclusive=b,sha256=hashlib.sha256(data).hexdigest(),reused=True)
        for attempt in range(3):
            try:
                url=inspection['url']+f'?download=true&start={a}&end={b-1}&attempt={attempt}'
                req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b-1}'})
                with urllib.request.urlopen(req,timeout=25) as response:
                    assert response.status==206
                    assert response.headers.get('Content-Range')==f'bytes {a}-{b-1}/{inspection["file_bytes"]}'
                    data=response.read(b-a+1)
                assert len(data)==b-a
                temporary=target.with_suffix('.part'); temporary.write_bytes(data); temporary.replace(target)
                return dict(start=a,end_exclusive=b,sha256=hashlib.sha256(data).hexdigest(),reused=False)
            except Exception:
                if attempt==2: raise
                time.sleep(1)
    records=[]
    with ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(download,pair) for pair in chunks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%8==0: print(args.model,'EMA range chunks',len(records),'/',len(chunks),flush=True)
    with span.open('wb') as f:
        for a,b in chunks: f.write((out/f'chunk-{a}-{b}.bin').read_bytes())
    current_name='pflm_lm1b_k4.ckpt' if args.model=='pflm' else 'ar_best_lm1b.ckpt'
    model=load_model(Path('work/checkpoints')/current_name,args.model,'cpu')
    names=list(dict(model.named_parameters()))
    expected=[k.removeprefix('backbone.') for k in meta['state_dict'] if k!='backbone.rotary_emb.inv_freq']
    assert names==expected and len(names)==len(tensors)
    state=model.state_dict(); verified=[]
    with span.open('rb') as f:
        for name,t,entry in zip(names,tensors,selected):
            f.seek(entry['header_offset']-start)
            header=f.read(30); fields=struct.unpack('<IHHHHHIIIHH',header)
            assert fields[0]==0x04034b50 and fields[3]==0
            filename=f.read(fields[-2]).decode(); extra=f.read(fields[-1])
            assert filename==entry['name']
            raw=f.read(entry['bytes'])
            assert len(raw)==entry['bytes'] and zlib.crc32(raw)==entry['crc32']
            assert t['dtype']=='torch FloatStorage' and t['offset']==0
            value=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(t['shape'])
            assert list(value.shape)==list(state[name].shape) and list(value.stride())==t['stride']
            assert value.numel()==t['storage_elements'] and torch.isfinite(value).all()
            state[name]=value
            verified.append(dict(name=name,storage_key=t['storage_key'],shape=t['shape'],bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),crc32=entry['crc32']))
    model.load_state_dict(state,strict=True)
    torch.save({'state_dict':{'backbone.'+k:v for k,v in state.items()}},destination)
    manifest=dict(model=args.model,historical_revision=inspection['historical_revision'],
        historical_lfs_sha256=inspection['published_lfs_sha256'],full_historical_file_sha256_verified=False,
        extraction='Exact HTTP ranges; each tensor ZIP CRC32 checked; no historical pickle execution.',
        parameter_mapping='Historical state_dict order without rotary buffer matches published named_parameters and all EMA shapes; training EMA registration order remains unconfirmed.',
        source_metadata_sha256=sha(historical/'data.pkl'),protocol_sha256=sha(ROOT/'HISTORICAL_EMA_PLAN.md'),
        current_checkpoint_sha256=sha(Path('work/checkpoints')/current_name),
        checkpoint=str(destination),checkpoint_sha256=sha(destination),
        span_start=start,span_end_exclusive=end,span_sha256=sha(span),chunks=sorted(records,key=lambda x:x['start']),
        tensors=verified,strict_load=True,all_finite=True,original_buffer='rotary_emb.inv_freq')
    (historical/'ema_extraction.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['checkpoint','checkpoint_sha256','strict_load']},indent=2),flush=True)


if __name__=='__main__': main()
