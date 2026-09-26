# Experiment files

This index connects the experiment source directories to their original logs and the final main-model checkpoint.

## Final evaluation checkpoint

[Download elliptic-two-parameter](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-two-parameter.tar.gz)

Install into `a2a4/best_ckpt/best_model/` from the repository root:

```bash
python artifacts/download.py --bundle elliptic-two-parameter
```

[Checkpoint file list and SHA-256 checksums](../artifacts/manifest.json). The bundle is the main evaluation model; initialization-comparison arms use their own checkpoint paths declared by their launch scripts.

Evaluation entry points: [test/generate_test.py](test/generate_test.py), [test/compute_stats.py](test/compute_stats.py).

## Code and original logs

### a2a4/code

[Open experiment code](code)

| Original log | Size (bytes) |
|---|---:|
| [a2a4_2stage_45856.err](../logs/historical/a2a4/code/logs/a2a4_2stage_45856.err) | 1554022 |
| [a2a4_2stage_45856.out](../logs/historical/a2a4/code/logs/a2a4_2stage_45856.out) | 26182 |
| [ex2_scratch_46109.err](../logs/historical/a2a4/code/logs/ex2_scratch_46109.err) | 344 |
| [ex2_scratch_46109.out](../logs/historical/a2a4/code/logs/ex2_scratch_46109.out) | 0 |
| [ex2_scratch_46111.err](../logs/historical/a2a4/code/logs/ex2_scratch_46111.err) | 774937 |
| [ex2_scratch_46111.out](../logs/historical/a2a4/code/logs/ex2_scratch_46111.out) | 26553 |
| [phase1_45841.err](../logs/historical/a2a4/code/logs/phase1_45841.err) | 591 |
| [phase1_45841.out](../logs/historical/a2a4/code/logs/phase1_45841.out) | 12791 |
| [a2a4_2stage_45760.err](../logs/historical/a2a4/train_log/a2a4_2stage_45760.err) | 1555127 |
| [a2a4_2stage_45760.out](../logs/historical/a2a4/train_log/a2a4_2stage_45760.out) | 27526 |

### a2a4/data_gen

[Open experiment code](data_gen)

| Original log | Size (bytes) |
|---|---:|
| [fix_data_45840.err](../logs/historical/a2a4/data_gen/fix_data_45840.err) | 0 |
| [fix_data_45840.out](../logs/historical/a2a4/data_gen/fix_data_45840.out) | 1021 |

### a2a4/test

[Open experiment code](test)

| Original log | Size (bytes) |
|---|---:|
| [eval_scratch_46113.err](../logs/historical/a2a4/test/eval_scratch_46113.err) | 96555 |
| [eval_scratch_46113.out](../logs/historical/a2a4/test/eval_scratch_46113.out) | 1710 |


## Evaluation data and plotting

[Download evaluation data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-elliptic-two-parameter-data.tar.gz) · [Evaluation-to-figure commands](../docs/REVIEWER_WORKFLOW.md)

Install with `python artifacts/download.py --bundle elliptic-two-parameter-data` from the repository root.
