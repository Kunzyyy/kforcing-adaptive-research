"""Recompute selected batched branch scores via existing single-sequence teacher scorer."""
import json
import math
from pathlib import Path
import sys
import torch
from core import load_model,teacher_metrics

root=Path(sys.argv[1])
assert not (root/'nll_crosscheck.json').exists(),'Preserve verification record'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
data=[json.loads(x) for x in (root/'windows.jsonl').read_text().splitlines()]
teacher=load_model(Path('work/checkpoints/ar_best_lm1b.ckpt'),'ar','cuda')
checks=[]
for length in (6,16,32):
    candidates=[r for r in data if r['length']==length]
    for index in (0,8,16,24):
        r=candidates[index]
        for method in ('fixed4','split22'):
            expected=r['teacher_nll_'+method]
            actual=teacher_metrics(teacher,r['prefix']+r[method],length)['teacher_nll_sum']
            assert math.isclose(expected,actual,rel_tol=1e-5,abs_tol=.001),(expected,actual)
            checks.append(dict(prompt_id=r['prompt_id'],seed=r['seed'],method=method,
                               batched_nll=expected,single_nll=actual,absolute_difference=abs(actual-expected)))
report=dict(passed=True,branches=len(checks),max_abs_difference=max(r['absolute_difference'] for r in checks),
            tolerance=dict(relative=1e-5,absolute=.001),checks=checks,
            scope='First four distinct source sentences at each context length; existing core.teacher_metrics, batch 1. Not an external quality evaluator.')
(root/'nll_crosscheck.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
