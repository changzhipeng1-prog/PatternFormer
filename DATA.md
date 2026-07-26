# Data & checkpoints

The datasets, model checkpoints, and rendered figures are **too large for GitHub** (~19 GB
total) and are excluded from this repository. Deposit them in a public archive (e.g. Zenodo)
and place each file back under the path shown below to reproduce the experiments.

Everything listed here is regenerable from the code in this repo: the `data/` and `data_gen/`
outputs from the classical solvers under each `*/data_gen/`, and the checkpoints from the
training scripts under each `*/code/`.

## Datasets (ground‑truth solution sets)

| Path | Size | Contents |
|------|------|----------|
| `GS/L2/data/gs_lookup_L2.pt` | 2.2 GB | Gray–Scott small‑D solution lookup (main‑text regime) |
| `GS/L1/data/gs_lookup_big.pt` | 1.3 GB | Gray–Scott large‑D solution lookup (ablation regime) |
| `a2a4/data/bvp_region_train.pt` | 781 MB | Example 2 training solution sets |
| `a2a4/data/bvp_region_{val,test}.pt` | 98 MB each | Example 2 val/test solution sets |
| `GS/L2/data_gen/L2_gen/gs_dataset_L2_d4.pt` | 418 MB | Gray–Scott small‑D raw generated dataset (D4‑reduced) |
| `1D_p/data/p_solutions_lookup.pt` | 224 MB | Example 1 solution lookup |
| `1D_p/data_gen/cbmfem/branch_*_S_new.npy` | 226–282 MB each | Example 1 raw CBMFEM branches |
| `2D/data/p_solutions_lookup_2d_filtered.pt` | — | Example 3 solution lookup |

Split‑index files (`*/data/test_p_idx*.pt`, `split_info*.pt`, `norm_stats*.pt`) are small and
define the train/val/test partitions (all seed 42, parameter‑level splits):
Ex.1 12600/2700/2701 · Ex.2 21103/2637/2641 · Ex.3 1739 · GS large‑D 758/162/163 ·
GS small‑D 1350/288/292.

## Checkpoints

| Path pattern | Size | Contents |
|--------------|------|----------|
| `<example>/best_ckpt/best_model/` | — | final model: LoRA adapter + input/output projectors + dual head |
| `<example>/best_ckpt/best_model/{unet,autoencoder2d}.pt` | — | frozen autoencoder weights |
| `GS/L{1,2}/ckpt/epoch_60/adapter_model.safetensors` | 39 MB | Gray–Scott fixed‑K LoRA adapter |
| `GS/L{1,2}/ckpt/stop_base/adapter_model.safetensors` | 39 MB | stop‑token warm‑start checkpoint |
| `GS/L1/ablation/ckpt_arm{1,2}_*/...` | 39 MB × many | pretraining/warm‑start ablation checkpoints |
| `1D_p/ablation_pretrain/gen_arm{1,2}_*.pt` | 119 MB each | ablation generation dumps |

The Qwen2.5‑7B‑Instruct base weights are **not** redistributed here — download from the
official source (`Qwen/Qwen2.5-7B-Instruct`); the LoRA adapters above apply on top of it.

## Precomputed evaluation results (kept in‑repo)

The small per‑experiment result JSONs used to assemble the figures **are** included
(`*/results/`, `*/extension/experiments/*.json`, `*/try/results_grid/*.json`), so the figure
scripts run without re‑executing the models.
