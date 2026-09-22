"""Export the actual CUDA noise values, verifying recorded hashes before sharing."""
import hashlib
import json
from pathlib import Path
import sys
import torch


root=Path(sys.argv[1])
target=root/'replay_noise.json'
assert not target.exists(),'Preserve existing noise export'
records=json.loads((root/'noise_manifest.json').read_text())
export=[]
for r in records:
    torch.manual_seed(r['seed']+101*r['batch_index'])
    tape=torch.stack([torch.rand(4,4,1,device='cuda') for _ in range(31)]).cpu()
    assert hashlib.sha256(tape.numpy().tobytes()).hexdigest()==r['sha256']
    export.append(dict(seed=r['seed'],batch_index=r['batch_index'],sha256=r['sha256'],
                       dtype='float32',shape=[31,4,4,1],values=tape.flatten().tolist()))
target.write_text(json.dumps(export,separators=(',',':')),encoding='utf-8')
print('Exported 64 hash-verified noise batches; consumers can reconstruct tensors without regenerating random values.')
