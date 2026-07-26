# GS — Gray-Scott 2D multi-solution generator (L=1 and L=2)

Self-contained, runnable snapshot of the Gray-Scott project: learn the map
**parameter `(ρ,μ)` → the SET of all coexisting steady-state solutions** of the 2D
Gray-Scott reaction-diffusion system, with a single deterministic autoregressive
generator (frozen conv-autoencoder + Qwen2.5-7B+LoRA + dual head).

```
DA·∇²A − S·A² + (μ+ρ)·A = 0
DS·∇²S + S·A² − ρ·(1−S) = 0          on the unit square, Neumann BC, 128×128
```

`ρ = F` (feed), `μ = k` (kill). The same `(ρ,μ)` admits **many** coexisting patterns
(spots / stripes / mazes); a deterministic single-output regressor would average the
modes (collapse). The method here generates the whole set in a fixed number of slots.

## Headline results (this is the "final" deliverable)

| setup | test params | **ordered (ours)** det distinct | STOP baseline | K_true mean | coverage |
|---|---|---|---|---|---|
| **L=1** (`L1/`) | 163 | **5.18** | 2.53 | 9.40 | 0.336 |
| **L=2** (`L2/`) | 292 (latest/expanded set) | **2.90** | 0.92 † | 9.73 | 0.106 |

† All numbers are on the **latest** dataset for each setup. L=2 uses the **expanded
1930-param** set (its 292 test params); the L=2 STOP baseline (0.92) was measured on the
original 72-param set — a same-set (expanded-292) STOP eval has not been re-run.
The ordered 2.90 comes from the all-1930 eval `L2/results/full_chunks_L2/` (aggregated to
the 292 test params in `L2/results/ord_eval_L2_expanded292.json`).

- *distinct* = of the K=24 deterministic generations, how many converge under an FDM
  quasi-Newton refine to **distinct** (D4-deduped) genuine steady states, averaged per param.
- **L=1 is a clean ~2× win**: the deterministic ordered model (5.18) beats the STOP
  baseline (2.53). ~1.93 of the 5.18 are *new* solutions beyond the GT enumeration;
  70% of params yield ≥1 new.
- **L=2 is intrinsically harder** (domain 2× larger → diffusion ÷4 → finer, denser
  patterns, tiny Newton basins). The naive run collapsed; the fix was **data expansion**
  (471→1930 params). On the latest (expanded) set its 292 test params give **2.90** — far
  above the STOP base (0.92). (On the original 72-param subset it scored 2.82; the project
  now uses the expanded set throughout.)

**Noise experiments** (ordered model). A *single* noisy pass is not better than deterministic
(`L1/code/experiments/noise_variants.py`: σ=0/0.05/0.1/0.2 → 5.18/5.04/5.01/5.01), but the
paper's noise knob is the **union of several noisy passes** (`L1/code/experiments/eval_noise_curve.py`),
which raises the mean to **9.16** per param (≈ the reference solver's 9.40), the gain mostly
beyond-GT. So ordered already beats the STOP baseline deterministically (5.18 > 2.53), and
noise-by-union is an optional knob for extra coverage. The three GS methods compared are:
STOP (2.53), our fixed-K (5.18), and fixed-K + noise (9.16).

## Method (one paragraph)

Solutions are sorted by `mean(A)` (a D4-invariant scalar → a unique autoregressive
order). The model fills a **fixed K=24 slots, no STOP head**. During training: the
**head** (k<K_t) is teacher-forced with an *ordered* MSE (latent + decoded field) +
scheduled sampling (fights exposure bias) + a small hinged FDM-residual physics term;
the **tail** (k≥K_t) is free self-prediction with physics-only loss (best-effort
discovery beyond GT). Eval = `generate_fixed_k(24)` → FDM quasi-Newton refine →
D4-dedup → count distinct converged steady states + GT coverage. See `L1/README.md`.

## Layout

```
GS/
├── L1/                         # domain [0,1]², DA=2.5e-4, DS=5e-4
│   ├── data_gen/   quasi-Newton FDM solver + parameter-continuation multi-solution generator + D4 reduction
│   ├── data/       final dataset: gs_lookup_big.pt (1083 params, ~10.3k sols) + splits + norm stats
│   ├── code/       config / train / eval_ord / model(ordered, no-STOP) / data / train_ae
│   ├── ckpt/       epoch_60  (DELIVERABLE, det 5.18)   +  stop_base (STOP baseline, det 2.53)
│   ├── baseline_stop/  qwen STOP model code + same eval metric  → STOP number for comparison plots
│   ├── results/    ord_eval_L1.json (the 163-param headline)  + smoke_*.json
│   ├── logs/       final training SLURM log
│   └── scripts/    run_train_L1.slurm
└── L2/                         # domain [0,2]² ⟺ DA,DS ÷4 ; same code, --L2 flag
    └── …  (same structure; data/ = expanded 1930-param set; ckpt/epoch_60 = det 2.82)
```

## Environment

- conda env **`torch124`** (torch 2.8, peft, transformers, cuda). One A100/H100 (~16 GB) for eval.
- base model **`Qwen/Qwen2.5-7B-Instruct`** (downloaded from HF on first use).

## Quick start (reproduce the final numbers)

```bash
# ---- L=1: ordered (ours) — full 163-param headline ----
cd GS/L1/code
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --ckpt ../ckpt/epoch_60 --out ../results/ord_eval_L1.json
# -> det_distinct ≈ 5.18, coverage ≈ 0.336

# ---- L=1: STOP baseline (same metric, for comparison) ----
cd GS/L1/baseline_stop
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --ckpt ../ckpt/stop_base --out ../results/stop_eval_L1.json
# -> det_distinct ≈ 2.53

# ---- L=2: ordered (ours) — needs the patient refiner (slower L=2 dynamics) ----
cd GS/L2/code
CUDA_VISIBLE_DEVICES=0 python eval_ord.py --L2 --ckpt ../ckpt/epoch_60 \
    --maxiter 30000 --early_iter 0 --baseline_det 0.92 --out ../results/ord_eval_L2.json
# -> det_distinct ≈ 2.82
```

A 2-param **smoke test** (fast sanity check that the pipeline loads + runs in this
location) is in each `results/smoke_*.json`; regenerate with `--n 2`. See per-folder READMEs
for the full data→train→eval pipeline and notes.
