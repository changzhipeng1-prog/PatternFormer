# Experiment files

This index connects the experiment source directories to their original logs and the final main-model checkpoint.

## Final evaluation checkpoint

[Download gray-scott-l2](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-gray-scott-l2.tar.gz)

Install into `GS/L2/ckpt/epoch_60/` from the repository root:

```bash
python artifacts/download.py --bundle gray-scott-l2
```

[Checkpoint file list and SHA-256 checksums](../../artifacts/manifest.json). The bundle is the main evaluation model; initialization-comparison arms use their own checkpoint paths declared by their launch scripts.

Evaluation entry points: [code/eval_ord.py](code/eval_ord.py).

## Code and original logs

### GS/L2/data_gen

[Open experiment code](data_gen)

| Original log | Size (bytes) |
|---|---:|
| [build_dense.log](../../logs/historical/GS/L2/data_gen/L2_gen/build_dense.log) | 426 |
| [shard_0.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_0.log) | 922 |
| [shard_1.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_1.log) | 918 |
| [shard_2.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_2.log) | 753 |
| [shard_3.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_3.log) | 757 |
| [shard_4.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_4.log) | 758 |
| [shard_5.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_5.log) | 759 |
| [shard_6.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_6.log) | 759 |
| [shard_7.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs/shard_7.log) | 758 |
| [shard6_0.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard6_0.log) | 2020 |
| [shard6_1.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard6_1.log) | 1984 |
| [shard6_1_resume.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard6_1_resume.log) | 1984 |
| [shard6_2.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard6_2.log) | 1932 |
| [shard6_3.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard6_3.log) | 1990 |
| [shard6_4.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard6_4.log) | 1948 |
| [shard6_5.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard6_5.log) | 1986 |
| [shard_0.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard_0.log) | 214 |
| [shard_1.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard_1.log) | 274 |
| [shard_2.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard_2.log) | 276 |
| [shard_3.log](../../logs/historical/GS/L2/data_gen/L2_gen/genlogs_dense/shard_3.log) | 143 |

### GS/L2/code

[Open experiment code](code)

| Original log | Size (bytes) |
|---|---:|
| [L2_3a_mix_45459.err](../../logs/historical/GS/L2/logs/L2_3a_mix_45459.err) | 194018 |
| [L2_3a_mix_45459.out](../../logs/historical/GS/L2/logs/L2_3a_mix_45459.out) | 113086 |
| [L2_ae_45146.err](../../logs/historical/GS/L2/logs/L2_ae_45146.err) | 0 |
| [L2_ae_45146.out](../../logs/historical/GS/L2/logs/L2_ae_45146.out) | 5705 |

### GS/L2/try

[Open experiment code](try)

| Original log | Size (bytes) |
|---|---:|
| [gsTB_46026.err](../../logs/historical/GS/L2/try/logs/gsTB_46026.err) | 96337 |
| [gsTB_46026.out](../../logs/historical/GS/L2/try/logs/gsTB_46026.out) | 1019 |

