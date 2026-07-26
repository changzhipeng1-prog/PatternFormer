# GS L=1 — parameter extrapolation (OOD parameters)

Tests whether the ordered model generates solutions at **(ρ,μ) outside the training band**.
The in-band result is 5.18 distinct/param; this folder asks what happens beyond it.

## Result (headline)

| regime | success | mean solutions | meaning |
|---|---|---|---|
| **in-grid rescue** (params inside the grid that the base eval scored 0) | **74.8%** (92/123) | 2.2 | multi-noise inference recovers most 0-score in-band params |
| **off-grid OOD** (truly outside the training band) | **3.5%** (5/144) | — | the model barely steps outside the training distribution |
| **latent-opt along the solution band** (test-time) | +2.5× | 4 → 14 | test-time latent gradient descent helps a lot, but the ceiling is the model's generalization |
| **physics-only finetune** on OOD points (v1/v2) | ~flat (41→42 with-solution) | ~3.1 | no-GT extrapolation data carries too little signal |

**Conclusion:** strong **in-distribution** multi-solution generation, **weak extrapolation**. To cross
the parameter boundary you need denser OOD-labelled data / stronger physics priors, not just a better optimizer.

## Layout

```
extension/
├── data/        build_extrap_data.py (v1: ±25% box, 180 OOD pts) / build_extrap_data2.py (v2: along solution band, ~88 pts)
│                + data_extrap/ , data_extrap2/  (GT + OOD param lookups, 1.3 GB each)
├── experiments/ extrapolate.py (rescue + grid modes) , latent_opt_extrap.py (test-time latent GD) ,
│                gen_extrap_fields.py , noise_variants.py , plot_corners.py
│                + result JSONs: extrap_rescue_L1 / extrap_grid_L1 / extrap_grid_ft*_ep* / latent_opt_extrap / nv_L1
│                + figures: grid_extrap_heatmap.png , latent_opt_extrap.png , corners_fields.png , extrap_fields.npz
├── ckpt/        checkpoints_extrap/ (v1 finetune) , checkpoints_extrap2/ (v2 finetune)  — best_model + epoch_10..40
├── scripts/     run_extrap_L1 / run_finetune_extrap_L1[_v2] / run_eval_extrap_ckpt / run_latent_opt  (.slurm)
└── logs/        SLURM logs
```

The experiment scripts import `config`/`model`/`gs_torch` from `../../code` (path already adapted).

## Reproduce

```bash
cd experiments                      # uses ../../code for config/model/gs_torch
# rescue 0-score in-band params + OOD grid (no training)
python extrapolate.py --mode rescue --n 0 --sigmas 0.15,0.3 --N 3 --out extrap_rescue_L1.json
python extrapolate.py --mode grid --grid 15 --sigmas 0.2,0.3 --N 2 --out extrap_grid_L1.json
# test-time latent optimization along the solution band
python latent_opt_extrap.py --ckpt ../../code/checkpoints_final/epoch_60 --npts 25 --out latent_opt_extrap.json
# (optional) physics-only finetune on OOD data — see scripts/run_finetune_extrap_L1*.slurm
```
