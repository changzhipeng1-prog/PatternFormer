# Experiment files

This index connects the experiment source directories to their original logs and the final main-model checkpoint.

## Final evaluation checkpoint

[Download elliptic-1d](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-1d.tar.gz)

Install into `1D_p/best_ckpt/best_model/` from the repository root:

```bash
python artifacts/download.py --bundle elliptic-1d
```

[Checkpoint file list and SHA-256 checksums](../artifacts/manifest.json). The bundle is the main evaluation model; initialization-comparison arms use their own checkpoint paths declared by their launch scripts.

Evaluation entry points: [test/generate_test.py](test/generate_test.py), [test/compute_stats.py](test/compute_stats.py).

## Code and original logs

### 1D_p/ablation_pretrain

[Open experiment code](ablation_pretrain)

| Original log | Size (bytes) |
|---|---:|
| [abl_gen_46012.err](../logs/historical/1D_p/ablation_pretrain/logs/abl_gen_46012.err) | 96542 |
| [abl_gen_46012.out](../logs/historical/1D_p/ablation_pretrain/logs/abl_gen_46012.out) | 1130 |
| [abl_gen_46013.err](../logs/historical/1D_p/ablation_pretrain/logs/abl_gen_46013.err) | 50 |
| [abl_gen_46013.out](../logs/historical/1D_p/ablation_pretrain/logs/abl_gen_46013.out) | 1113 |
| [arm1_pretrained_45989.err](../logs/historical/1D_p/ablation_pretrain/logs/arm1_pretrained_45989.err) | 116 |
| [arm1_pretrained_45989.out](../logs/historical/1D_p/ablation_pretrain/logs/arm1_pretrained_45989.out) | 0 |
| [arm1_pretrained_45991.err](../logs/historical/1D_p/ablation_pretrain/logs/arm1_pretrained_45991.err) | 776549 |
| [arm1_pretrained_45991.out](../logs/historical/1D_p/ablation_pretrain/logs/arm1_pretrained_45991.out) | 47687 |
| [arm2_random_45988.err](../logs/historical/1D_p/ablation_pretrain/logs/arm2_random_45988.err) | 116 |
| [arm2_random_45988.out](../logs/historical/1D_p/ablation_pretrain/logs/arm2_random_45988.out) | 0 |
| [arm2_random_45990.err](../logs/historical/1D_p/ablation_pretrain/logs/arm2_random_45990.err) | 400 |
| [arm2_random_45990.out](../logs/historical/1D_p/ablation_pretrain/logs/arm2_random_45990.out) | 41098 |

### 1D_p/test

[Open experiment code](test)

| Original log | Size (bytes) |
|---|---:|
| [gen_1D_p_46010.err](../logs/historical/1D_p/test/gen_1D_p_46010.err) | 107 |
| [gen_1D_p_46010.out](../logs/historical/1D_p/test/gen_1D_p_46010.out) | 0 |
| [gen_1D_p_46011.err](../logs/historical/1D_p/test/gen_1D_p_46011.err) | 96309 |
| [gen_1D_p_46011.out](../logs/historical/1D_p/test/gen_1D_p_46011.out) | 809 |

### 1D_p/code

[Open experiment code](code)

| Original log | Size (bytes) |
|---|---:|
| [1dp_2stage_45759.err](../logs/historical/1D_p/train_log/1dp_2stage_45759.err) | 1552084 |
| [1dp_2stage_45759.out](../logs/historical/1D_p/train_log/1dp_2stage_45759.out) | 40222 |

