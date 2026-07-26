# GS L=2 — ordered, no-STOP, fixed-K multi-solution generator (finer patterns)

Domain `[0,2]²` ⟺ same 128×128 grid with **`DA,DS ÷ 4`** (`DA=6.25e-5`, `DS=1.25e-4`).
The 2× larger domain produces finer, denser patterns (stripes / mazes / lattices),
aligning with the xmorphia phase diagram. Same code/method as L=1.

**Final result (latest / expanded dataset):** deterministic **2.90** distinct / param
(292 test params of the expanded 1930-param set), coverage 0.106, mean K_true 9.73.
(Below L=1's 5.18 — L=2 is intrinsically harder: tiny Newton basins from the ÷4 diffusion.)
On the original-72 subset it scored 2.82; **the project now uses the expanded set throughout.**

## The L=2 story (collapse → diagnosis → fix)

The naive L=2 run **collapsed to det 1.00**. Diagnosis ruled out the AE (reconstructs
L=2 GT at 1.1%) and ruled out missing GT multiplicity (K_true ≈ 7) — the regressor
simply never learned the map on too-little data (only 329 train params). **Fix = data
expansion** (471 → 1930 params), retrained → **2.82**. The L1-ordered LoRA prior in the
mixed init adds only +0.03 over data-only (2.82 vs 2.79) — **data is the lever.**
0-solution params dropped 50% → 29%.

| eval | det distinct | file |
|---|---|---|
| **DELIVERABLE — expanded set, 292 test params (latest)** | **2.90** | `results/ord_eval_L2_expanded292.json` |
| original collapse, original-72 (strict pp) | 1.00 | `results/ord_eval_L2.json` |
| relaxed post-proc baseline, original-72 | 2.14 | `results/ord_eval_L2_relaxB.json` |
| deliverable on original-72 subset (reference) | 2.82 | `results/ord_eval_L2_3a_ep60.json` |
| expanded-data-only ablation, original-72 | 2.79 | `results/ord_eval_L2_3b_ep60.json` |

(The 2.90 is aggregated from the all-1930-param eval in `results/full_chunks_L2/`.)

## Layout

Same as L=1 (`../L1/README.md`), with L=2 specifics:

```
L2/
├── data_gen/      L2_gen pipeline: gen_L2.py / gen_L2_dense.py (synthetic seed bank, NOT spot-continuation;
│                  applies DA,DS/=Lscale²) + build_L2.py / build_L2_dense.py + filter_d4.py + op_n128.mat
├── data/          the (latest) EXPANDED set: gs_lookup_L2.pt (1930 params, ~17.9k sols) + splits (1350/288/292) + norm_stats
├── code/          same code as L1; eval applies op.DA,DS /= 4 via the --L2 flag
├── ckpt/
│   ├── epoch_60/      ★ DELIVERABLE (ordered "3a-mixed", det 2.90 on expanded-292)
│   └── stop_base/     L2 STOP baseline ckpt (det 0.92 on original-72)
├── baseline_stop/ qwen_L2 STOP model + same eval metric  → reproduces 0.92
├── results/       ord_eval_L2_expanded292.json (★ 2.90) + full_chunks_L2/ (all-1930 eval) + orig-72 JSONs + smoke_*.json
├── logs/          L2_3a_mix_45459 (deliverable training) + L2_3b_dat_45460 (data-only ablation)
└── scripts/       run_train_L2.slurm
```

## Reproduce the final number (eval) — patient refiner

L=2 steady states converge **more slowly** under the FDM refiner, so use `--maxiter 30000
--early_iter 0` (the only change vs L=1):

The config default `data/` is the expanded set, so eval runs over its 292 test params.
The full 292×30000 eval is slow — it is already aggregated in `results/full_chunks_L2/`
(→ 2.90). To re-run a slice:

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --L2 --ckpt ../ckpt/epoch_60 \
    --maxiter 30000 --early_iter 0 --out ../results/ord_eval_L2.json
#   -> det_distinct ≈ 2.90  (expanded 292 test)

# STOP baseline (same metric; original-72 number is 0.92, expanded-292 eval not yet re-run):
cd ../baseline_stop
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --L2 --ckpt ../ckpt/stop_base \
    --maxiter 30000 --early_iter 0 --out ../results/stop_eval_L2.json
```

## Full pipeline (data → AE → generator), for reference

```bash
# 1. DATA  (cluster; synthetic seed bank, NOT L1 spot-continuation — L2 maze basins are unreachable from spots)
cd data_gen
python gen_L2.py --jmod r,m          # original 471-param grid (shard r of m across GPUs)
python gen_L2_dense.py --jmod r,m    # densified grid for the expansion
python build_L2_dense.py             # merge -> ../data/ (1930 params, old 72-test split preserved)
# 2. AUTOENCODER  (L=2-specific; val rel_l2 ≈ 0.0084)
cd ../code && python train_ae.py
# 3. GENERATOR  (2-GPU DDP, 60 ep)
sbatch ../scripts/run_train_L2.slurm
```

> **⚠ Reproduction gap — mixed init.** `run_train_L2.slurm` resumes from
> `./init_mixed_L1prior` (= L1-ordered LoRA/special_tokens × L2 AE/dual_head/input_projector).
> That intermediate dir was **手工拼装的, deleted after training** and is not in this snapshot —
> the **deliverable `ckpt/epoch_60` is complete and evaluable as-is**; only a from-scratch
> retrain would need the mixed init rebuilt (or just resume from `../L1/ckpt/epoch_60` LoRA
> + the L2 AE). The data-only ablation `3b` (no mixed init) reaches 2.79, ~the same.

> Same conventions as L=1: `val_mse` is not the metric; data-gen scripts are cluster-grade
> reference; the expanded dataset (`data/`, 2.2 GB) is a real file here (server-complete).
