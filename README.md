# PatternFormer

Code for **PatternFormer: Learning Multiple Solution Patterns in Reaction–Diffusion Systems**.

PatternFormer adapts a pretrained large language model (Qwen2.5‑7B‑Instruct, frozen, with
LoRA) into a **set‑valued solution operator**: from a PDE parameter `p` it autoregressively
generates the *entire set* of coexisting steady‑state solutions `S(p) = {u_1, …, u_K(p)}` in a
single forward pass. Solution fields live in a 256‑dim latent space produced by a frozen
autoencoder; a lightweight input/output projector and a dual (regression + marker) head sit
around the frozen backbone.

This repository contains the **source code, configs, and figure scripts** to reproduce every
experiment and figure in the paper. Large binary artifacts (datasets, model checkpoints,
rendered figures) are **not** in the repo — see [`DATA.md`](DATA.md) for how to obtain them.
Historical development notes are kept in [`README_dev.md`](README_dev.md).

---

## Repository layout

Each PDE example is a self‑contained sub‑project with the same internal structure
(`code/` model+training, `data_gen/` classical solvers that build the ground truth,
`preprocess/`, `test/` evaluation, `extension/` extrapolation, `train_log/`, `*.py` figure
makers).

| Folder | Paper example | Equation | Parameter |
|--------|---------------|----------|-----------|
| `1D_p/` | **Example 1** | `-u'' + u^2(u^2 - p) = 0` on `(0,1)` | scalar `p ∈ [0,18]` |
| `a2a4/` | **Example 2** | `-u'' + a_4 u^4 + a_2 u^2 = 0` on `(0,1)` | `(a_4, a_2)` |
| `2D/`   | **Example 3** | `-Δu - u^2 = -s·sin(πx)sin(πy)` on `(0,1)^2` | scalar `s` |
| `GS/`   | **Gray–Scott** | steady reaction–diffusion on `[0,1]^2` | feed/kill `(ρ, μ)` |

### Gray–Scott diffusion regimes (note the naming)

`GS/` has two regimes. The `L` refers to **domain size** (`L=1` vs `L=2`), so the diffusion
coefficients run **opposite** to the `L` index:

| Folder | Diffusion | Role | Figure |
|--------|-----------|------|--------|
| `GS/L1/` | **large** `D_A=2.5e-4, D_S=5e-4` (coarser patterns, max 24 solutions) | ablation / SI | `combined_L1.pdf` |
| `GS/L2/` | **small** `D_A=6.25e-5, D_S=1.25e-4` (finer, richer patterns, max 29 solutions) | **main‑text Fig. 3** | `combined_L2.pdf` |

The fixed generation budget `K=24` is the maximum multiplicity of the `L1` (large‑D) training
set; the few `L2` parameters whose true count exceeds 24 are truncated to the budget.

### Top‑level figure scripts

`fig_ablation_full.py`, `fig5_who_solves.py`, `fig_row1_1dp.py`, `fig_gs_combined.py`,
`plot_extension_2x3.py`, `plot_ex2ex3_panel.py` assemble the composite manuscript figures from
each sub‑project's evaluation outputs.

---

## Environment

```bash
conda create -n patternformer python=3.10
conda activate patternformer
pip install torch transformers peft numpy scipy matplotlib   # + as needed
```

Experiments were run with PyTorch 2.x + CUDA, `transformers`, and `peft` (LoRA). The 7B
backbone requires a GPU for training and inference; the classical solvers and Newton
refinement run on CPU (numpy/scipy).

## Reproducing an experiment

1. Obtain the datasets and/or checkpoints (see [`DATA.md`](DATA.md)) and place them under each
   sub‑project's `data/` and `best_ckpt/` (or `ckpt/`) directory.
2. Training config lives in `<example>/code/config.py`. Train with the provided launch
   scripts (`*.slurm` / `*.sh`) under each sub‑project.
3. Evaluate with the scripts under `<example>/test/`; extrapolation under
   `<example>/extension/`.
4. Regenerate figures with the top‑level `*.py` figure scripts.

Ground‑truth data is produced by the classical multiple‑solution solvers under each
`data_gen/` (companion‑based multilevel FEM, homotopy/shooting continuation, and a GPU
quasi‑Newton tensor‑product solver for Gray–Scott).

---

## Before making this repository public

- **Absolute paths:** ~41 scripts contain machine‑specific paths (`/home/…`). Replace with
  paths relative to the repo root or a configurable data directory before release.
- **Data & checkpoints:** deposit the large artifacts listed in [`DATA.md`](DATA.md) to a
  public archive (e.g. Zenodo) and update the download links / paths.
- **Licence:** add a `LICENSE` file (the paper commits to an open‑source MIT licence on
  acceptance).

## Citation

```
@article{patternformer,
  title  = {PatternFormer: Learning Multiple Solution Patterns in Reaction--Diffusion Systems},
  author = {Chang, Zhipeng and Yin, Wenpeng and Hao, Wenrui},
  year   = {2025}
}
```
