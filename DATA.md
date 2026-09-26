# Data, checkpoints and figure inputs

Download the five final main-model checkpoint bundles using [the artifact installer](artifacts/README.md).
Evaluation datasets and figure inputs are available through the same installer; see
[the reviewer workflow](docs/REVIEWER_WORKFLOW.md#download-inputs).
Original experiment logs are indexed in [logs/](logs/README.md).

Place artifact files under the repository root using the paths below. Keep checkpoint
components together: a directory containing only `adapter_config.json` or a model-card
README is not the weight bundle. Large tensors and rendered images are excluded from
Git by `.gitignore`.

## Model and dataset layout

| Problem | Evaluation checkpoint | Evaluation data |
|---|---|---|
| Single-parameter elliptic | `1D_p/best_ckpt/best_model/` | `1D_p/data/p_solutions_lookup.pt`, `test_p_idx.pt` |
| Two-parameter elliptic | `a2a4/best_ckpt/best_model/` | `a2a4/data/bvp_region_test.pt` |
| Two-dimensional elliptic | `2D/best_ckpt/best_model/` | `2D/data/p_solutions_lookup_2d_filtered.pt`, `test_p_idx.pt` |
| GS larger diffusion | `GS/L1/ckpt/epoch_60/` | `GS/L1/data/gs_lookup_big.pt`, `test_p_idx_big.pt`, `norm_stats_big.pt` |
| GS smaller diffusion | `GS/L2/ckpt/epoch_60/` | `GS/L2/data/gs_lookup_L2.pt`, `test_p_idx_L2.pt`, `norm_stats_L2.pt` |

Each data filename after the first entry is relative to the same `data/` directory.
GS evaluation also uses `code/op_n128.mat`. Keep the train and validation indices when
running training-parameter or context-based experiments.

A model bundle contains the LoRA adapter and its configuration, special-token weights,
input projection, dual head, and the AE (`unet.pt` or `autoencoder2d.pt`). Include the
output projector where used by the model implementation. The Qwen base model is loaded
separately using `model_name` in `code/config.py`.

## Figure inputs

| Figure group | Input files |
|---|---|
| Elliptic examples and errors | `test/generated_solutions.pt`, `test/stats.pt`, `test/stats.csv`; the 1D example panel uses `1D_p/test/examples_solutions.pt` |
| Timing panels | Tables under each `test/timing/` directory, including the comparison CSV files read by `plot_efficiency*.py` |
| Elliptic composite panels | Individual PDF **and** PNG panels at the paths declared by `make_panel_*.py` |
| GS composites | Lookup and split files; `results/full_chunks_*/chunk_*.json`; `try/results_grid/*_shard*.json`; `results/fig1_*_noise.pt`; extrapolation arrays under `extension/experiments/` |
| Extrapolation | Arrays listed in `plot_extension_2x3.py`; `grid_direct.pt`, `grid_gt_cont.pt`; continuation seed arrays |
| Generation/refinement comparison | `1D_p/test/who_npz.npz`, `a2a4/test/who_npz.npz`, `2D/test/who_npz.npz` |
| Initialization comparison | Saved generations under `1D_p/ablation_pretrain/`; warm/scratch 2D outputs; GS ablation evaluation JSONs; training-curve logs; `gs_map_data.npz` |
| GS training-parameter gallery | `GS/L2/try/results/train_beyond.pt` containing fields, parameter values and data-membership flags |

For the filenames consumed by an individual figure, run:

```bash
python experiments/run.py main-fig-04
```

The catalogue lists primary inputs. The linked figure scripts specify additional
panel-specific dependencies. Checkpoint-based commands are in [INFERENCE.md](docs/INFERENCE.md).
