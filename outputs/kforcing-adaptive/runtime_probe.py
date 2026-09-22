"""Small process-isolated reproductions; preserve natural child exit codes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(args):
    import gc
    import torch
    from core import load_model,generate,teacher_metrics
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    models=[]
    for kind in ('pflm','ar'):
        if args.case==kind or args.case in ('dual','alternating'):
            filename='pflm_lm1b_k4.ckpt' if kind=='pflm' else 'ar_best_lm1b.ckpt'
            models.append(load_model(args.checkpoints/filename,kind,'cuda'))
    prefix=torch.tensor([[101,1996,2068,2003,1037,2307]],device='cuda')
    for i in range(args.iterations):
        if args.case=='ar':
            r=teacher_metrics(models[0],prefix[0].tolist()+[2154,102],6)
        else:
            r=generate(models[0],prefix,policy='fixed4',seed=71+i,max_new=32)
            if len(models)>1:
                r.update(teacher_metrics(models[1],r['ids'],6))
        if args.case=='alternating':
            generate(models[0],prefix,policy='fixed2',seed=11+i,max_new=32)
        print(json.dumps({'iteration':i,'metric':r.get('teacher_nll'),'tokens':r.get('new_tokens')}),flush=True)
    del models
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print('returned_from_child_work',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--case',choices=['pflm','ar','dual','alternating'])
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--iterations',type=int,default=8)
    ap.add_argument('--suite',choices=['baseline','nojit','portable'],default='portable')
    args=ap.parse_args()
    if args.case:
        child(args)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'summary.json').exists():
        raise RuntimeError('Use a new probe directory; preserve earlier exit records')
    report=[]
    cases=([('pflm',4),('ar',4),('dual',4),('alternating',4),('dual',1),('alternating',1)]
           if args.suite=='baseline' else [('ar',4),('dual',4),('alternating',4)])
    for case,threads in cases:
        start=time.time()
        cmd=[sys.executable,'-X','faulthandler',__file__,'--checkpoints',str(args.checkpoints),
             '--case',case,'--threads',str(threads),'--iterations',str(args.iterations)]
        jit='0' if args.suite=='nojit' else '1'
        backend='eager' if args.suite=='portable' else 'script'
        env=dict(os.environ,PYTORCH_JIT=jit,KFORCING_HELPERS=backend)
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=90,env=env)
        log=args.out/f'{case}-threads{threads}-jit{jit}.log'
        log.write_text(p.stdout+'\nSTDERR\n'+p.stderr,encoding='utf-8')
        row=dict(case=case,threads=threads,jit=jit,helper_backend=backend,iterations=args.iterations,exit_code=p.returncode,
                 seconds=time.time()-start,completed_marker='returned_from_child_work' in p.stdout)
        report.append(row)
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
