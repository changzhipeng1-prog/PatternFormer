# Paper artifacts — Qwen multi-solution PDE solver (3 problems)

Each subfolder is a **self-contained** copy of the full pipeline for one problem —
data generation → preprocessing → model → training → post-processing:

```
paper/
  1D_p/   a2a4/   2D/
    data_gen/      # GROUND-TRUTH generation (the traditional bifurcation/Newton solver)
    preprocess/    # build the training lookup + train/val/test splits from raw gen output
    code/          # config.py, train.py, model/ (incl. the post-processing solver), run_*.sh
    data/          # datasets (.pt): GT solution lookup + train/val/test split indices
    train_log/     # two-stage training stdout/stderr
    best_ckpt/     # final nocontext stage-2 best_model (Qwen adapter + projectors + heads + frozen AE)
    smoke_test.py  # exercises data-gen + preprocess + post-process on tiny/real-GT inputs
```

`code/` keeps only the **final training** scripts (`run_twostage_*.sh`); all eval / viz /
phase / finetune scripts were pruned (the deliverable is the final ckpt + the pipeline code).

## Pipeline stages (per problem)

| stage | 1D_p | a2a4 | 2D |
|---|---|---|---|
| **data-gen** | `data_gen/cbmfem/run_branch.py` (8-branch continuation; `Sexample1.faSh1`+`Newton.py`) | `data_gen/generate_fullspace_gpu.py` (GPU BFS+Newton), `simplify_dataset.py` | `data_gen/generate_dataset_2d.py` (FEM Newton march) + `Companion_Method/*.dat` mesh, `filter_d4_dataset.py` |
| **preprocess** | `preprocess/build_lookup.py` → lookup + splits | `preprocess/split_dataset.py` → 80/10/10 splits | `preprocess/make_splits.py` → 70/15/15 split indices |
| **model+train** | `code/` (`config.py`, `train.py`, `model/`, `run_twostage_pde.sh`) | same | same (+ `train_unet*.py` AE pretrain) |
| **post-proc** | `code/model/traditional_refine.py` → local `data_gen/cbmfem` (damped Newton, **same solver as data-gen**) | `code/model/newton_fdm.py` (FDM damped Newton, **same residual `Lu/h+h·f` as data-gen**) | `code/model/newton_refine_2d.py` (FEM Newton, **same weak form as data-gen**) |

Post-processing == data-generation solver in every problem (this is the paper's "zero post-processing
floor" claim). The post-proc residual norm is exactly the discrete operator the GT satisfies.

## Run / verify

```bash
# smoke-test the whole non-training pipeline (no GPU, no Qwen needed):
cd paper/1D_p && python smoke_test.py      # likewise a2a4/, 2D/

# regenerate splits from shipped data (preprocess defaults read ../data):
cd paper/2D/preprocess && python make_splits.py      # bit-matches the shipped *_p_idx.pt
```

All three `smoke_test.py` pass: data-gen solver converges, preprocess builds correctly-structured
splits, and post-processing Newton drives a noised real-GT solution back to ~1e-12 residual.
The 2D `make_splits.py` **bit-reproduces** the shipped split. For 1D_p / a2a4 the shipped
`*_p_idx.pt` are the canonical splits; the preprocess scripts implement the same documented recipe
(seed/ratios in `p_solutions_meta.pt`), but the original raw `branch_*.npy` / `bvp_region.pt` were
not archived here, so an exact-permutation rematch isn't reproducible from this folder alone.

NOTE: `code/config.py` references datasets as `./data/...` (relative to the run dir). Run training
from the project root (e.g. `paper/1D_p/`) with `data/` alongside, or adjust the paths.

Huge intermediate generator artifacts are intentionally NOT copied (1D_p `branch_*.npy`,
a2a4 `bvp_complete.pt`/`bvp_simplified.pt` ~45 GB, 2D unfiltered lookup) — only the code +
the final training datasets are shipped.

**data/ contents** (configs originally reference `./data/...` or `../data_gen/data/...`; repoint to
this `data/` to re-run):
- 1D_p: `p_solutions_lookup.pt` (GT, 224M) + `p_solutions_meta.pt` + {train,val,test}_p_idx.pt
- a2a4: `bvp_region_{train,val,test}.pt` (1.2G total; train is 971M)
- 2D:  `p_solutions_lookup_2d_filtered.pt` + `split_info.pt` + {train,val,test}_p_idx.pt

| problem | PDE | chain init | train log | best_ckpt source |
|---|---|---|---|---|
| 1D_p | −u″ + u²(u²−p) = 0 | from-HF (chain head) | 1dp_2stage_45759 | checkpoints_pde/best_model |
| a2a4 | −u″ + a4·u⁴ + a2·u² = 0 | from 1D_p best | a2a4_2stage_45760 | checkpoints_pde/best_model |
| 2D | −Δu − u² = −s·sin(πx)sin(πy) | from a2a4 best | 2d_2stage_45773 | checkpoints_chain/best_model |

## A. Unified settings (same across all three — the experiment design)
- **Two-stage**: stage1 withcontext (no_context_prob 0.0) → stage2 nocontext (1.0). Deliverable = stage2.
- **LR**: stage1 `lr_proj 1e-4 / lr_lora 2e-5`, stage2 `2e-5 / 5e-6`; ReduceLROnPlateau on val_mse.
- **Physics**: `lambda_pde 0.05`, GT-referenced excess `relu(R²(pred) − R²(D(E(u_gt))))`, warmup epoch 5→30.
- **Cross-project warm-start chain**: transfer LoRA + output_projector + dual_head + special_tokens; re-init input_projector (param_dim differs); each project keeps its own frozen AE.
- **Model**: latent 256, Qwen2.5-7B hidden 3584, LoRA r16/α32, weight_decay 0.01.
- **DDP-safe early stopping** on val_mse.
- **Eval (item-2)**: M1 = direct Qwen output, k smallest residual; M2 = Qwen output as Newton initial guess, k fewest steps. Both: canonicalize + **Hungarian matching** + zero-safe rel-L2. Baseline = each project's **nocontext** baseline ckpt. **Residual computed by the post-processing solver itself = the data-generation solver** (1D_p cbmfem FEM; a2a4 FDM `Lu/h+h·f`; 2D FEM `‖F_free‖₂`).

## B. Resource/memory-driven differences (not part of the design)
| | 1D_p | a2a4 | 2D |
|---|---|---|---|
| batch×grad_accum×nproc (eff. batch) | 8×2×8 (128) | 4×2×8 (64, memory) | 8×4×4 (128, 4-GPU) |
| lr_plateau patience / factor | 8 / 0.3 | 8 / 0.3 | 25 / 0.5 |
| EARLY_STOP_PATIENCE | 15 | 15 | 50 (must exceed plateau patience) |
| num_epochs cap / stage | 100 | 60 | 200 |
| lambda_mse | 1.0 | 1.0 | 0.5 |

## C. Problem-specific (necessarily different)
solution_dim 1024/1024/145; param_dim 1/2/1; pde_dx 1/1023, 1/1024, mesh; max_solutions 8/6/4;
discretization & BC & post-processing solver (cbmfem FEM / FDM Neumann+ghost / FEM); each own frozen AE.

## Final results (median; baseline = nocontext baseline, trained = our chain+GT-ref)
| problem | val_mse (base→ours) | M1 rel_l2 | M1 resid | M2 rel_l2 | M2 resid | M2 Newton steps |
|---|---|---|---|---|---|---|
| 1D_p | 5.17e-3 → **4.91e-3** | 0.019 → 0.082 | 378 → 359 | 3.2e-8 → 3.2e-8 | ~9e-12 | 3 → 3 |
| a2a4 | 1.76e-2 → **1.60e-2** | 0.091 → **0.071** | 209 → 229 | 4.6e-4 → 5.0e-4 | 3.5e-3 | 2 → 2 |
| 2D | 1.69e-2 → **1.45e-2** | 0.0069 → **0.0061** | 3.14 → **2.19** | 1.8e-3 → **1.4e-3** | ~9e-14 | 4 → 4 |

Notes: residual units differ per problem (different discretizations) — compare only within a problem.
M2 (Newton post-processing) is essentially identical for baseline vs ours in all three (same step count,
same converged accuracy): the chain improves val_mse modestly but does **not** make the direct output a
better Newton initial guess. 1D_p M1 is slightly worse for ours (lower residual but farther from the GT branch).

Dependency note: 1D_p post-processing/residual calls the CBM-FEM solver, now shipped in
`1D_p/data_gen/cbmfem/` (Sexample1.faSh1, Newton.py). `code/model/traditional_refine.py`
resolves it relative to its own location (override via `BBBB_PDE_NEWTON_DIR`).
