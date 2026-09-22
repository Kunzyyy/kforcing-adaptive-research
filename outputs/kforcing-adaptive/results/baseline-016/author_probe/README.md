# K-Forcing fixed-input reproduction probe

This package is prepared for comparison and has not been sent to the authors.
It contains no model weights and makes no network requests.

## Purpose

Compare the same published LM1B PFLM checkpoint on the authors' original environment and our portable environment. The 160 fixed cases contain 400 candidate token positions. Contexts, uniform noise, temperature=1, batching and expected argmax outputs are fixed. Some long contexts extend beyond an earlier EOS; these are numerical probes, not text-quality examples.

## Run in the official repository environment

Install the official repository's dependencies as usual, including its native attention/rotary dependencies. Do not replace source modules with our portable versions for the original-environment comparison.

```sh
python author_replay_probe.py --repo /path/to/K-Forcing --checkpoint /path/to/pflm_lm1b_k4.ckpt --output original_environment_result.json
```

Run this command from the extracted probe directory, or pass an absolute --cases path. The script requires the checkpoint SHA-256 in probe_cases.json, safely loads weights_only=True, and records the model source actually imported. It does not modify the supplied repository or checkpoint. Use a new output filename for each run.

## Interpretation

A zero-disagreement result supports consistency on these inputs only. It does not prove full sequence/distribution equivalence or resolve the published Gen-PPL difference. Nonzero disagreements should be inspected with precision/backend/source details before attributing a cause.

Our local independent reference uses raw weights, manual centering LayerNorm, non-interleaved RoPE and explicit QK/softmax/V attention. It matched all 400 candidate positions; maximum logit difference from the portable model was 0.000343323.

## Specific clarification requests (draft, not sent)

1. Is the released checkpoint SHA the model used for LM1B Table 1? If not, what model revision/configuration corresponds to the table?
2. Table 4 says RMSNorm, while the released transformer code calls F.layer_norm. Which normalization was used in the reported checkpoint?
3. Can the exact held-out prefixes, GPT-2-Large scoring script, temperature, penalty and precision settings be provided?
4. Please compare this fixed-input probe on the original environment to help separate portability from checkpoint/evaluation differences.

The publication is https://arxiv.org/html/2606.10820v2 and the pinned repository is https://github.com/alibaba-damo-academy/K-Forcing/tree/706caa332a69d509b7fc2fa53ac7819f381a8c78 . No claim of a paper/code defect is made.
