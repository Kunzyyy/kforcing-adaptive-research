"""Run three CPU-only, read-only numerical audits from any working directory."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/'outputs/kforcing-adaptive'
reports={}
for name in ['verify_attention_features.py','verify_local_mechanism.py','verify_local_target_transfer.py']:
    result=subprocess.run([sys.executable,str(RESEARCH/name)],cwd=ROOT,text=True,encoding='utf-8',capture_output=True,check=True)
    if result.stderr.strip():
        print(result.stderr,file=sys.stderr)
    report=json.loads(result.stdout)
    if not report['passed']:
        raise RuntimeError(name+' did not pass')
    reports[name]={'passed':True,'scope':report.get('scope',report.get('limitation','Numerical verification only.'))}
print(json.dumps({'passed':True,'audits':reports,'algorithmic_success_claimed':False},ensure_ascii=False,indent=2))
