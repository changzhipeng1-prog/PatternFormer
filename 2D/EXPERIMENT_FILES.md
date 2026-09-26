# Experiment files

This index connects the experiment source directories to their original logs and the final main-model checkpoint.

## Final evaluation checkpoint

[Download elliptic-2d](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-2d.tar.gz)

Install into `2D/best_ckpt/best_model/` from the repository root:

```bash
python artifacts/download.py --bundle elliptic-2d
```

[Checkpoint file list and SHA-256 checksums](../artifacts/manifest.json). The bundle is the main evaluation model; initialization-comparison arms use their own checkpoint paths declared by their launch scripts.

Evaluation entry points: [test/generate_test.py](test/generate_test.py), [test/compute_stats.py](test/compute_stats.py).

## Code and original logs

### 2D/code

[Open experiment code](code)

| Original log | Size (bytes) |
|---|---:|
| [ex3_scratch_46110.err](../logs/historical/2D/code/logs/ex3_scratch_46110.err) | 344 |
| [ex3_scratch_46110.out](../logs/historical/2D/code/logs/ex3_scratch_46110.out) | 0 |
| [ex3_scratch_46112.err](../logs/historical/2D/code/logs/ex3_scratch_46112.err) | 771203 |
| [ex3_scratch_46112.out](../logs/historical/2D/code/logs/ex3_scratch_46112.out) | 66919 |
| [2d_2stage_45773.err](../logs/historical/2D/train_log/2d_2stage_45773.err) | 775085 |
| [2d_2stage_45773.out](../logs/historical/2D/train_log/2d_2stage_45773.out) | 71253 |
| [unet2d_v2_38702.err](../logs/historical/2D/train_log/unet2d_v2_38702.err) | 0 |
| [unet2d_v2_38702.out](../logs/historical/2D/train_log/unet2d_v2_38702.out) | 9751 |

### 2D/test

[Open experiment code](test)

| Original log | Size (bytes) |
|---|---:|
| [eval_scratch_46114.err](../logs/historical/2D/test/eval_scratch_46114.err) | 95984 |
| [eval_scratch_46114.out](../logs/historical/2D/test/eval_scratch_46114.out) | 1192 |

### 2D/test/timing

[Open experiment code](test/timing)

| Original log | Size (bytes) |
|---|---:|
| [ours_2d_45855.err](../logs/historical/2D/test/timing/ours_2d_45855.err) | 96291 |
| [ours_2d_45855.out](../logs/historical/2D/test/timing/ours_2d_45855.out) | 544 |


## Evaluation data and plotting

[Download evaluation data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-elliptic-2d-data.tar.gz) · [Evaluation-to-figure commands](../docs/REVIEWER_WORKFLOW.md)

Install with `python artifacts/download.py --bundle elliptic-2d-data` from the repository root.
