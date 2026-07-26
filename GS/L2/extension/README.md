# GS L=2 — parameter extrapolation (OOD parameters)

Same experiment as L=1 (`../../L1/extension/README.md`) but on the L=2 regime (`D_A,D_S /4`),
inference-only (no OOD finetune checkpoints were trained for L=2).

## Result (headline)

| regime | L=2 | (L=1 for reference) |
|---|---|---|
| **in-grid rescue** (0-score in-band params) | **73.8%** (321/435), mean 2.2 | 74.8% |
| **off-grid OOD** | **6.25%** (9/144), mean 1.9 | 3.5% |
| **latent-opt along band** | +2.5× (band μ=−0.196ρ+0.0764) | +2.5× |

Same conclusion as L=1: strong in-distribution, weak extrapolation. L=2's OOD rate is
slightly higher (6.25 vs 3.5%) but its solution counts are lower (the harder /4-diffusion regime).

## Layout

```
extension/
├── experiments/ extrapolate.py , latent_opt_extrap.py , gen_extrap_fields.py
│                + result JSONs: extrap_rescue_L2 / extrap_grid_L2 / latent_opt_extrap_L2 / nv_L2
│                + latent_opt_extrap_L2.png , extrap_fields.npz
├── scripts/     run_extrap_L2 / run_extrap_grid_L2_fast / run_extrap_viz_L2 / run_latopt_L2  (.slurm)
└── logs/        SLURM logs
```

Scripts import `config`/`model`/`gs_torch` from `../../code` (adapted). L=2 uses the patient
refiner (`--L2 --maxiter 30000 --early_iter 0` or the fast `18000/5000` variant).

## Reproduce

```bash
cd experiments
python extrapolate.py --L2 --mode rescue --n 0 --sigmas 0.15,0.3 --N 2 --maxiter 30000 --early_iter 0 --out extrap_rescue_L2.json
python extrapolate.py --L2 --mode grid --grid 15 --sigmas 0.2,0.3 --N 2 --maxiter 30000 --early_iter 0 --out extrap_grid_L2.json
python latent_opt_extrap.py --L2 --ckpt ../../code/checkpoints_final/epoch_60 --npts 25 --maxiter 18000 --early_iter 5000 --out latent_opt_extrap_L2.json
```
