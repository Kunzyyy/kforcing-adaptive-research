"""Check published files byte-for-byte against PUBLICATION_MANIFEST.json.

Read-only integrity check, runnable from any working directory and with no
third-party dependency. It answers one narrow question: do the files copied
into this repository still have the SHA256 recorded when the publication
snapshot was made? It says nothing about whether any research claim holds.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'PUBLICATION_MANIFEST.json'

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
recorded=manifest['copied_files']

matched=[];missing=[];mismatched=[]
for rel,expected in sorted(recorded.items()):
    path=ROOT/rel
    if not path.is_file():
        missing.append(rel)
    elif sha256(path)==expected:
        matched.append(rel)
    else:
        mismatched.append(rel)

# The manifest is a snapshot taken at publication time. Files added afterwards
# (README, CI, packaging) are intentionally outside it and are not failures.
present={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
ignored_roots=('.git/','.venv/','venv/','__pycache__/')
added=sorted(
    rel for rel in present - set(recorded)
    if not rel.startswith(ignored_roots) and '/__pycache__/' not in rel
)

# An empty or truncated manifest must not be reported as a pass: with nothing
# recorded there is nothing to check, which is a broken manifest, not success.
passed=bool(recorded) and not missing and not mismatched
report={
    'passed':passed,
    'manifest_created_utc':manifest.get('created_utc'),
    'recorded_files':len(recorded),
    'matched':len(matched),
    'missing':missing,
    'mismatched':mismatched,
    'added_after_snapshot':added,
    'scope':'Byte-level integrity of the published snapshot only. Not a numerical audit, '
            'not a research result, and not evidence that any adaptive method works.',
}
print(json.dumps(report,ensure_ascii=False,indent=2))
if not passed:
    sys.exit(1)
