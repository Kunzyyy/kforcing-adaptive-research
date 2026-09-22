"""Inspect public historical checkpoint ZIP/pickle syntax without unpickling code."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickletools
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REV = '6b357e2893399a3c0219df6ef62af8fc6af747a4'
FILES = {
    'pflm': ('pflm_lm1b_k4.ckpt', 2206111286,
             '91f99042a89226570d3ee02def49c0546175c1e261b3b7922cc4f07dd49f1fdb'),
    'ar': ('ar_best_lm1b.ckpt', 2111620036,
           'c2ec9d523286a7efbf71ff469c0b880b30383561538d5e13889922be1ae27fe9'),
}


class RangedFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.requests = []

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos

    def read(self, length=-1):
        if length < 0: length = self.size-self.pos
        length = min(length, max(0, self.size-self.pos))
        if not length: return b''
        if length > 2*1024*1024: raise ValueError('Metadata-only read exceeds 2 MiB limit')
        start, end = self.pos, self.pos+length-1
        # Unique query avoids a shared cache returning an unrelated byte interval.
        url = self.url+f'?download=true&range_start={start}&range_end={end}'
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}',
                                                   'User-Agent': 'KForcing-history-metadata'})
        with urllib.request.urlopen(req, timeout=25) as response:
            content_range = response.headers.get('Content-Range', '')
            expected = f'bytes {start}-{end}/{self.size}'
            if response.status != 206 or content_range != expected:
                raise ValueError(f'Unverified byte range: {response.status}, {content_range!r}; expected {expected}')
            data = response.read(length+1)
            if len(data) != length: raise ValueError('Range response length mismatch')
        self.requests.append(dict(start=start, end=end, bytes=len(data),
                                  sha256=hashlib.sha256(data).hexdigest()))
        self.pos += len(data)
        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=FILES)
    args = ap.parse_args()
    filename, size, lfs = FILES[args.model]
    out = ROOT/'results/public-history'/('historical_'+args.model)
    out.mkdir(exist_ok=False)
    url = f'https://huggingface.co/zwave/K-Forcing/resolve/{REV}/{filename}'
    remote = RangedFile(url, size)
    with zipfile.ZipFile(remote) as archive:
        entries = [dict(name=x.filename, bytes=x.file_size, compressed=x.compress_size,
                        crc32=x.CRC, header_offset=x.header_offset) for x in archive.infolist()]
        names = [x.filename for x in archive.infolist() if x.filename.endswith('/data.pkl')]
        assert len(names) == 1
        raw = archive.read(names[0])  # ZIP verifies CRC; no pickle.load / torch.load.
        (out/'data.pkl').write_bytes(raw)
    disassembly = io.StringIO()
    pickletools.dis(raw, out=disassembly)
    (out/'data_pickle_disassembly.txt').write_text(disassembly.getvalue(), encoding='utf-8')
    globals_used = sorted({arg for op, arg, _ in pickletools.genops(raw) if op.name == 'GLOBAL'})
    strings = [dict(position=pos, value=arg) for op, arg, pos in pickletools.genops(raw)
               if op.name in {'BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE'}]
    record = dict(observed_utc=datetime.now(timezone.utc).isoformat(), model=args.model,
        historical_revision=REV, url=url, file_bytes=size, published_lfs_sha256=lfs,
        full_file_hash_verified=False, verification='Exact HTTP Content-Range, byte counts, ZIP metadata member CRC32.',
        pickle_executed=False, pickle_sha256=hashlib.sha256(raw).hexdigest(),
        requests=remote.requests, bytes_transferred=sum(x['bytes'] for x in remote.requests),
        entries=entries, globals=globals_used, strings=strings)
    (out/'inspection.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({k: record[k] for k in ['model', 'file_bytes', 'bytes_transferred', 'pickle_sha256', 'globals']}, indent=2))


if __name__ == '__main__': main()
