# Experiment files

This index connects the experiment source directories to their original logs and the final main-model checkpoint.

## Final evaluation checkpoint

[Download gray-scott-l1](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-gray-scott-l1.tar.gz)

Install into `GS/L1/ckpt/epoch_60/` from the repository root:

```bash
python artifacts/download.py --bundle gray-scott-l1
```

[Checkpoint file list and SHA-256 checksums](../../artifacts/manifest.json). The bundle is the main evaluation model; initialization-comparison arms use their own checkpoint paths declared by their launch scripts.

Evaluation entry points: [code/eval_ord.py](code/eval_ord.py).

## Code and original logs

### GS/L1/ablation

[Open experiment code](ablation)

| Original log | Size (bytes) |
|---|---:|
| [gsEvS_46632.err](../../logs/historical/GS/L1/ablation/logs/gsEvS_46632.err) | 385268 |
| [gsEvS_46632.out](../../logs/historical/GS/L1/ablation/logs/gsEvS_46632.out) | 5322 |
| [gsablEv_46030.err](../../logs/historical/GS/L1/ablation/logs/gsablEv_46030.err) | 385538 |
| [gsablEv_46030.out](../../logs/historical/GS/L1/ablation/logs/gsablEv_46030.out) | 5246 |
| [gsabl_46023.err](../../logs/historical/GS/L1/ablation/logs/gsabl_46023.err) | 780556 |
| [gsabl_46023.out](../../logs/historical/GS/L1/ablation/logs/gsabl_46023.out) | 101809 |
| [gsabl_46024.err](../../logs/historical/GS/L1/ablation/logs/gsabl_46024.err) | 400 |
| [gsabl_46024.out](../../logs/historical/GS/L1/ablation/logs/gsabl_46024.out) | 105353 |
| [gsablseed_46594.err](../../logs/historical/GS/L1/ablation/logs/gsablseed_46594.err) | 400 |
| [gsablseed_46594.out](../../logs/historical/GS/L1/ablation/logs/gsablseed_46594.out) | 100150 |
| [gsablseed_46595.err](../../logs/historical/GS/L1/ablation/logs/gsablseed_46595.err) | 773805 |
| [gsablseed_46595.out](../../logs/historical/GS/L1/ablation/logs/gsablseed_46595.out) | 99672 |
| [smoke_46021.err](../../logs/historical/GS/L1/ablation/logs/smoke_46021.err) | 50 |
| [smoke_46021.out](../../logs/historical/GS/L1/ablation/logs/smoke_46021.out) | 821 |

### GS/L1/extension/experiments

[Open experiment code](extension/experiments)

| Original log | Size (bytes) |
|---|---:|
| [seed_continue_L1.log](../../logs/historical/GS/L1/extension/experiments/seed_continue_L1.log) | 101317 |

### GS/L1/code

[Open experiment code](code)

| Original log | Size (bytes) |
|---|---:|
| [ae_train.log](../../logs/historical/GS/L1/logs/ae_train.log) | 4291 |
| [gs_p3loc_44967.err](../../logs/historical/GS/L1/logs/gs_p3loc_44967.err) | 389310 |
| [gs_p3loc_44967.out](../../logs/historical/GS/L1/logs/gs_p3loc_44967.out) | 77749 |
| [ordA_L1_45396.err](../../logs/historical/GS/L1/logs/ordA_L1_45396.err) | 385428 |
| [ordA_L1_45396.out](../../logs/historical/GS/L1/logs/ordA_L1_45396.out) | 39179 |


## Evaluation data and plotting

[Download evaluation data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-gray-scott-l1-data.tar.gz) · [Evaluation-to-figure commands](../../docs/REVIEWER_WORKFLOW.md)

Install with `python artifacts/download.py --bundle gray-scott-l1-data` from the repository root.
