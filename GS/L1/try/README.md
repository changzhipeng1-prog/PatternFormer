# GS L=1 — result figures (`try/`)

Academic-style figures for showcasing the L=1 result, using the paper's shared
`test_plot_style.py` (Nature-style: sans-serif bold, dpi-300 PNG + PDF). Each script
reads the existing eval JSONs / checkpoints — regenerate with the commands below.

| figure | shows | data / script |
|---|---|---|
| `fig_method_bar` | **headline win**: STOP 2.53 → **ordered (ours) 5.18** → fixed-K+noise 9.16 | `ord_eval_L1.json` · `make_figs.py` |
| `fig_distinct_vs_kt` | recovered distinct vs GT multiplicity K_true (+ y=x); points above y=x = beyond-GT | `ord_eval` + `full_chunks_L1` · `make_figs.py` |
| `fig_rhomu_distinct` | **(ρ,μ) solution distribution**: where the model is strong/weak across the param plane | `full_chunks_L1` (1083) · `make_figs.py` |
| `fig_distribution` | histogram of distinct/param (mean 5.18) + beyond-GT (mean 1.93, 70% of params find ≥1 new) | `ord_eval_L1.json` · `make_figs.py` |
| `fig_ae_recon` | input vs AE round-trip (rel-L2 ≈ 1.9e-3) → the frozen AE is **not** the bottleneck | `epoch_60/autoencoder2d.pt` · `make_ae_recon.py` (GPU) |
| `fig_noise_multipass` | cumulative distinct vs # noise passes (det ∪ noise union keeps finding new basins) | model gen · `make_multipass.py` (GPU) |

The **solution montage** (max-distinct param, cover-data vs new-solution) is the top-level
`../../fig1_maxsol_L1-ti786_L2-ti402_noise.png` (built by `../../make_fig1.py`).

## Regenerate

```bash
python make_figs.py                              # 4 json-based figs (no GPU)
CUDA_VISIBLE_DEVICES=0 python make_ae_recon.py   # AE reconstruction (GPU)
CUDA_VISIBLE_DEVICES=0 python make_multipass.py  # noise multipass curve (GPU)
```

All figures export `.png` (dpi 300) **and** `.pdf`, consistent with the other paper figures.
