# GS L=1 — ordered, no-STOP, fixed-K multi-solution generator

Domain `[0,1]²`, Neumann BC, 128×128, `DA=2.5e-4`, `DS=5e-4`.
Learn `(ρ,μ) → the SET of coexisting Gray-Scott steady states`, deterministically.

**Final result:** deterministic **5.18** distinct solutions / param (163 test params),
vs the STOP baseline **2.53**. Coverage 0.336, mean K_true 9.40.
~1.93 of the 5.18 are *new* (beyond the GT enumeration); 70% of params yield ≥1 new.

## Method

- Solutions sorted by `mean(A)` — a **D4-invariant** scalar → a unique autoregressive order.
- **Fixed K=24 slots, no STOP head** (K ≥ global max K_t = 24).
- **Head** (k<K_t): teacher forcing + ordered MSE (decoded field `λ=0.5` + latent `λ=0.5`)
  + scheduled sampling (`ss_prob=0.25`, fights exposure bias) + hinged FDM-residual physics
  (`λ=0.02`, floor `1e-3` so real & trivial states both incur 0 — see the physics caution below).
- **Tail** (k≥K_t): free self-prediction, physics-only (`λ=0.01`) — best-effort discovery beyond GT.
- Architecture: frozen conv-AE (`2×128×128 ↔ 256` latent) + Qwen2.5-7B + LoRA(r16, q/k/v/o)
  + dual head. Trained **no-context**, finetuned 60 ep from the STOP checkpoint (`ckpt/stop_base`).
- **Eval** = `generate_fixed_k(24)` → FDM quasi-Newton refine → D4-dedup (rel-L2 > 0.15)
  → count distinct converged steady states + GT coverage.

## Layout

```
L1/
├── data_gen/      data-generation pipeline (run on a cluster; produces data/)
│   ├── gs_torch.py          batched torch quasi-Newton FDM solver (matches MATLAB to ~1e-15)
│   ├── gs_continuation.py   BFS parameter-continuation: neighbor warm-start + poly perturbation → many solutions
│   ├── filter_trivial.py    drop spatially-uniform (trivial) states
│   ├── filter_d4.py         D4 symmetry reduction (8 transforms) + canonical pose + mean(A) sort
│   ├── build_big.py / package_dataset.py   assemble the lookup + splits
│   └── op_n128.mat, matlab_ref.mat, seeds_base.mat, validate_seeds.mat   FDM operator + seeds
├── data/          gs_lookup_big.pt (1083 params, ~10.3k sols, mean ~9.5/param) + {train,val,test}_p_idx + norm_stats + split_info  (split 758/162/163)
├── code/          config.py · train.py · eval_ord.py · train_ae.py · gs_torch.py · op_n128.mat · model/ · data/
├── ckpt/
│   ├── epoch_60/      ★ DELIVERABLE (ordered, det 5.18)   — use this for eval
│   └── stop_base/     STOP baseline ckpt (= the finetune start; det 2.53)
├── baseline_stop/ STOP model code (qwen) + same eval metric → reproduces the 2.53 for comparison plots
├── results/       ord_eval_L1.json (the 163-param headline) + smoke_*.json
├── logs/          ordA_L1_45396.{out,err}  (final ordered-training SLURM log)
└── scripts/       run_train_L1.slurm
```

## Reproduce the final number (eval)

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --ckpt ../ckpt/epoch_60 --out ../results/ord_eval_L1.json
#   -> det_distinct ≈ 5.18, coverage ≈ 0.336, Ktrue ≈ 9.40   (163 test params, ~30-60 min)
# fast sanity (one high-Kt param):
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --ckpt ../ckpt/epoch_60 --idx_start 113 --idx_end 114 --out ../results/smoke_ordered.json
```

## STOP baseline (for "ours is better" comparison)

Same metric, same K=24, same FDM refine — only the checkpoint + model code differ
(STOP design = STOP head + CE, trained with `generate()`; ordered = no STOP). The STOP
model has an extra `output_projector` so it must be loaded with its own (`qwen`) model
code, kept in `baseline_stop/`:

```bash
cd baseline_stop
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --ckpt ../ckpt/stop_base --out ../results/stop_eval_L1.json
#   -> det_distinct ≈ 2.53   (vs ordered 5.18)
```

### Noise experiments (ordered model)

The three GS methods compared are: **STOP (2.53)**, our **fixed-K (5.18)**, and
**fixed-K + noise (9.16)**.

- **Single noisy pass**: `code/experiments/noise_variants.py` *swaps* the deterministic pass
  for one noisy pass, σ = 0/0.05/0.1/0.2 → 5.18/5.04/5.01/5.01 — a single noisy draw is not
  better than deterministic (do NOT read this as "noise doesn't help").
- **The paper's noise knob = UNION of several noisy passes**:
  `code/experiments/eval_noise_curve.py` unions a few noisy passes (each an extra forward),
  raising the mean **5.18 → 9.16** per param (≈ the reference solver's 9.40), gain mostly beyond-GT.

=> ordered's **deterministic 5.18** already beats the STOP baseline (2.53) with no sampling;
union-of-noisy-passes is an *optional* knob for extra coverage.

## Full pipeline (data → AE → generator), for reference

The dataset + checkpoints are already in `data/` and `ckpt/`; this is how they were made.

```bash
# 1. DATA  (cluster; quasi-Newton FDM solver + parameter continuation + D4 reduction)
cd data_gen
python gs_continuation.py          # solve all params -> per-param solution sets
python filter_trivial.py           # drop uniform states
python filter_d4.py                # D4-reduce + canonicalize + sort  -> gs_dataset_d4.pt
python build_big.py                # -> ../data/gs_lookup_big.pt + splits + norm_stats
# 2. AUTOENCODER  (single GPU; frozen afterwards)
cd ../code && python train_ae.py   # -> code/checkpoints/autoencoder2d.pt  (val rel_l2 ≈ 0.0019)
# 3. GENERATOR  (4-GPU DDP, 60 ep, finetune from ckpt/stop_base)
sbatch ../scripts/run_train_L1.slurm   # adjust SBATCH resource lines for your cluster
#   -> code/checkpoints_final/epoch_60   (the deliverable, copied here to ckpt/epoch_60)
```

> **Physics caution** (`code/model/pde_residual.py`): the FDM residual's global minimum is
> the *trivial* bleached state (~1e-12), 8 orders below a real solution's AE round-trip
> floor (~1.7e-4). The hinge (`floor=1e-3`) gives both 0 penalty, so physics polishes
> without pulling generations toward the trivial state. `λ_pde` is deliberately tiny.

> **Notes.** `val_mse` is **not** the metric (it measures head reconstruction, already
> converged in the finetune, so it stays flat) — the real signal is eval distinct-count.
> Data-gen scripts in `data_gen/` are reference-grade (cluster + many intermediate files);
> the `--outdir`/seed defaults may need adjusting. The big dataset (`data/`, 1.3 GB) and
> checkpoints are real files here (server-complete); GitHub packaging is a later step.
