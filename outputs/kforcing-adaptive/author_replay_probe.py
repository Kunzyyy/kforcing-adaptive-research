"""Run the frozen K-Forcing probe in a separately supplied official repository."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(8*1024*1024): h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True, help='K-Forcing repository with installed dependencies')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--cases', type=Path, default=Path(__file__).with_name('probe_cases.json'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = ap.parse_args()
    assert not args.output.exists(), 'Preserve previous probe output; choose a new output filename.'
    bundle = json.loads(args.cases.read_text(encoding='utf-8'))
    digest = file_hash(args.checkpoint)
    assert digest == bundle['checkpoint_sha256'], 'This probe requires the exact released checkpoint listed in the input bundle.'
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from omegaconf import OmegaConf
    from models.pflm import MTP
    import models.pflm as module
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = {k.removeprefix('backbone.'): v for k, v in state.get('state_dict', state).items()}
    vocab, width = state['vocab_embed.embedding'].shape
    heads = width//(len(state['rotary_emb.inv_freq'])*2)
    layers = 1+max(int(re.match(r'blocks\.(\d+)\.', k).group(1)) for k in state if k.startswith('blocks.'))
    config = OmegaConf.create(dict(model=dict(hidden_size=width, vocab_size=vocab, n_heads=heads,
        n_blocks=layers, cond_dim=1024, dropout=0., causal=True, scale_by_sigma=False)))
    model = MTP(config, vocab_size=vocab, mask_index=103, max_k=4)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    grouped = defaultdict(list)
    for r in bundle['cases']: grouped[r['context_length'], r['k']].append(r)
    records = []
    with torch.inference_mode():
        for key in sorted(grouped):
            batch = grouped[key]
            assert len(batch) == 8
            context = torch.tensor([r['context_ids'] for r in batch], device=args.device)
            noise = torch.tensor([r['noise'] for r in batch], device=args.device, dtype=torch.float32).unsqueeze(-1)
            tau = torch.ones(len(batch), 1, 1, device=args.device)
            logits = model(context, noise, tau, mode='inference')
            for r, pred in zip(batch, logits):
                predicted = pred.argmax(-1).tolist()
                expected = r['expected_argmax']
                records.append(dict(prompt_id=r['prompt_id'], context_length=r['context_length'], k=r['k'],
                    actual_argmax=predicted, expected_argmax=expected,
                    differing_positions=sum(a != b for a, b in zip(predicted, expected))))
    result = dict(checkpoint_sha256=digest, cases_sha256=file_hash(args.cases),
        repository=str(args.repo.resolve()), imported_model_source=str(Path(module.__file__).resolve()),
        imported_model_source_sha256=file_hash(Path(module.__file__)),
        torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
        precision='fp32 weights, no outer autocast, TF32 disabled',
        cases=len(records), candidate_positions=sum(r['k'] for r in records),
        argmax_disagreements=sum(r['differing_positions'] for r in records), records=records,
        interpretation='A match checks these fixed inputs only. A mismatch identifies an environment/source difference to inspect, not an attribution of fault.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['cases', 'candidate_positions', 'argmax_disagreements', 'device']}, indent=2))


if __name__ == '__main__': main()
