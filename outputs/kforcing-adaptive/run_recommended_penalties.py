"""Released script's k-specific penalty recommendations, frozen before GPT-2 scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from transformers import BertTokenizerFast
from core import load_model,teacher_metrics
from collect_benefit import verify_checkpoints,dump
from audit_official import official_cli,quiet_sampler


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,default=Path('work/checkpoints'))
    ap.add_argument('--vocab',type=Path,default=Path('work/pilot-data/vocab.txt'))
    args=ap.parse_args()
    path=args.out/'recommended_samples.jsonl'
    if path.exists():
        raise RuntimeError('Recommended-penalty results already exist')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    verify_checkpoints(args)
    cli=official_cli()
    quiet_sampler()
    tokenizer=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=True)
    prompts=json.loads((args.out/'prefixes.json').read_text(encoding='utf-8'))
    model=load_model(args.checkpoints/'pflm_lm1b_k4.ckpt','pflm','cuda')
    all_rows=[]
    schedule=[(seed,i,k) for seed in (617,619) for i in range(32) for k in (2,3,4)]
    random.Random(418221).shuffle(schedule)
    for index,(seed,bi,k) in enumerate(schedule):
        batch=prompts[4*bi:4*bi+4]
        torch.manual_seed(seed+bi*101)
        model.max_k=k
        penalty={2:.2,3:.4,4:.5}[k]
        result=model.sample_next_k_tokens_with_kv_caches(torch.tensor([p['ids'] for p in batch],device='cuda'),
            torch.ones(4,device='cuda'),122,k=k,frequency_penalty=penalty).cpu().tolist()
        for p,full in zip(batch,result):
            ids=cli.truncate_at_eos(full,102,6)
            all_rows.append(dict(prompt_id=p['id'],seed=seed,method=f'fixed{k}',frequency_penalty=penalty,
                full_ids=full,ids=ids,prefix_length=6,new_tokens=len(ids)-6,stopped_eos=ids[-1]==102,
                completion=tokenizer.decode(ids[6:],skip_special_tokens=True,clean_up_tokenization_spaces=True)))
        if (index+1)%32==0:
            print('Recommended configuration batches',index+1,'/',len(schedule),flush=True)
    del model
    torch.cuda.empty_cache()
    teacher=load_model(args.checkpoints/'ar_best_lm1b.ckpt','ar','cuda')
    with path.open('w',encoding='utf-8') as dest:
        for r in all_rows:
            r.update(teacher_metrics(teacher,r['ids'],6))
            dest.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    dump(args.out/'recommended_configuration.json',dict(samples=len(all_rows),penalties={'2':.2,'3':.4,'4':.5},
        source='Published scripts/run_pflm_lm1b.sh at 706caa332a69d509b7fc2fa53ac7819f381a8c78',
        selected_without_external_scores=True,scope='One generation run per prefix/seed/method, quality sensitivity only'))
    print('Recommended penalty samples complete',len(all_rows),flush=True)


if __name__=='__main__':
    main()
