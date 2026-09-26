# Paper figures and experiment entry points

Run commands from the repository root. `experiments/run.py` displays the working directory, figure command and output paths for each entry. Input paths refer to the [data layout](../DATA.md).

| Paper figure | Experiment | Figure command | Output |
|---|---|---|---|
| Main Figure 1 | Model architecture and sequence construction | Architecture source links below |  |
| Main Figure 2 | Single-parameter elliptic equation | `python experiments/run.py main-fig-02 --run` | `1D_p/fig_1Dp_panel.pdf` |
| Main Figure 3 | Gray–Scott, smaller diffusion | `python experiments/run.py main-fig-03 --run` | `GS/combined_L2.pdf` |
| Main Figure 4 | Extrapolation across three elliptic problems | `python experiments/run.py main-fig-04 --run` | `fig_extension_2x3.pdf` |
| Main Figure 5 | Direct generation and numerical refinement | `python experiments/run.py main-fig-05 --run` | `fig5_who_solves.pdf` |
| Main Figure 6 | Pretraining and cross-equation initialization | `python experiments/run.py main-fig-06 --run` | `fig_ablation_full.pdf` |
| Supplementary Figure S1 | Two-parameter elliptic equation | `python experiments/run.py supp-fig-01 --run` | `a2a4/fig_a2a4_panel.pdf` |
| Supplementary Figure S2 | Two-dimensional elliptic equation | `python experiments/run.py supp-fig-02 --run` | `2D/fig_2D_panel.pdf` |
| Supplementary Figure S3 | Near-zero two-dimensional solutions | `python experiments/run.py supp-fig-03 --run` | `2D/test/fig_2D_examples_nearzero.pdf` |
| Supplementary Figure S4 | Nearby two-dimensional solution branches | `python experiments/run.py supp-fig-04 --run` | `2D/test/fig_2D_examples_notcollapse.pdf` |
| Supplementary Figure S5 | Two-parameter extrapolation maps | `python experiments/run.py supp-fig-05 --run` | `a2a4/extension/fig_a2a4_appendix.pdf` |
| Supplementary Figure S6 | Far extrapolation with continuation | `python experiments/run.py supp-fig-06 --run` | `1D_p/extension/fig_farp_continuation.pdf` |
| Supplementary Figure S7 | Gray–Scott, larger diffusion | `python experiments/run.py supp-fig-07 --run` | `GS/combined_L1.pdf` |
| Supplementary Figure S8 | Gray–Scott generation at training parameters | [Training-parameter evaluation](INFERENCE.md#grayscott-training-parameter-examples) | `GS/L2/try/results/train_beyond.pt` |

Main Figure 1 is the conceptual architecture illustration; its implementation is in [v2_model.py](../1D_p/code/model/v2_model.py), [v3_model.py](../GS/L1/code/model/v3_model.py) and [sequence_builder.py](../GS/L1/code/data/sequence_builder.py).

Main Figure 6 is exported by the script as `fig_ablation_full.pdf`; the manuscript asset is named `fig_ablation.pdf`. `GS/make_combined.py` builds both diffusion regimes in one invocation. Supplementary Figure S6 runs CPU continuation before plotting. For S8, the evaluation entry writes the solution fields and membership flags used for the gallery.

## Experiment sources

### Main Figure 1: Model architecture and sequence construction

[1D_p/code/model/v2_model.py](../1D_p/code/model/v2_model.py), [GS/L1/code/data/sequence_builder.py](../GS/L1/code/data/sequence_builder.py).

### Main Figure 2: Single-parameter elliptic equation

[1D_p/test/generate_test.py](../1D_p/test/generate_test.py), [1D_p/test/compute_stats.py](../1D_p/test/compute_stats.py).

### Main Figure 3: Gray–Scott, smaller diffusion

[GS/L2/code/eval_ord.py](../GS/L2/code/eval_ord.py), [GS/L2/try/eval_grid.py](../GS/L2/try/eval_grid.py), [GS/_example_gen.py](../GS/_example_gen.py).

### Main Figure 4: Extrapolation across three elliptic problems

[1D_p/extension/compute_1dp_three_methods.py](../1D_p/extension/compute_1dp_three_methods.py), [2D/extension/experiment_three_methods_2d.py](../2D/extension/experiment_three_methods_2d.py), [a2a4/extension/compute_a2a4_cov_M3_bfs.py](../a2a4/extension/compute_a2a4_cov_M3_bfs.py).

### Main Figure 5: Direct generation and numerical refinement

[1D_p/test/_who_npz.py](../1D_p/test/_who_npz.py), [a2a4/test/_who_npz.py](../a2a4/test/_who_npz.py), [2D/test/_who_npz.py](../2D/test/_who_npz.py).

### Main Figure 6: Pretraining and cross-equation initialization

[1D_p/ablation_pretrain/generate_arm.py](../1D_p/ablation_pretrain/generate_arm.py), [2D/test/compare_warm_scratch.py](../2D/test/compare_warm_scratch.py), [GS/L1/ablation/run_eval.sh](../GS/L1/ablation/run_eval.sh).

### Supplementary Figure S1: Two-parameter elliptic equation

[a2a4/test/generate_test.py](../a2a4/test/generate_test.py), [a2a4/test/compute_stats.py](../a2a4/test/compute_stats.py).

### Supplementary Figure S2: Two-dimensional elliptic equation

[2D/test/generate_test.py](../2D/test/generate_test.py), [2D/test/compute_stats.py](../2D/test/compute_stats.py).

### Supplementary Figure S3: Near-zero two-dimensional solutions

[2D/test/generate_test.py](../2D/test/generate_test.py).

### Supplementary Figure S4: Nearby two-dimensional solution branches

[2D/test/generate_test.py](../2D/test/generate_test.py).

### Supplementary Figure S5: Two-parameter extrapolation maps

[a2a4/extension/grid_generate_a2a4.py](../a2a4/extension/grid_generate_a2a4.py), [a2a4/extension/grid_gt_continuation.py](../a2a4/extension/grid_gt_continuation.py).

### Supplementary Figure S6: Far extrapolation with continuation

[1D_p/extension/experiment_p18seed.py](../1D_p/extension/experiment_p18seed.py).

### Supplementary Figure S7: Gray–Scott, larger diffusion

[GS/L1/code/eval_ord.py](../GS/L1/code/eval_ord.py), [GS/L1/try/eval_grid.py](../GS/L1/try/eval_grid.py), [GS/_example_gen.py](../GS/_example_gen.py).

### Supplementary Figure S8: Gray–Scott generation at training parameters

[GS/L2/try/eval_train_beyond.py](../GS/L2/try/eval_train_beyond.py).

## Supplementary tables

| Table | Subject | Source or input |
|---|---|---|
| S1 | Notation | Model and sequence definitions under each `code/` directory |
| S2 | Autoencoder reconstruction | Bundled AE weights; `model/unet1d_v6.py`, `model/autoencoder2d.py`; GS example visualization: `GS/L1/try/make_ae_recon.py` |
| S3 | Datasets and partitions | Lookup datasets and train/validation/test indices under each `data/` directory |
| S4 | Training settings | Per-problem `code/config.py` and training launch scripts |
| S5 | Gray–Scott boundary-seeded extrapolation | `GS/L1/extension/experiments/seed_continue_extrap.py`; JSON outputs under `GS/L1/extension/experiments/` |
| S6 | Small-diffusion extrapolation atlas | `GS/L2/extension/experiments/extrapolate.py`, `gen_extrap_fields.py`; `extrap_grid_L2.json` |
| S7 | Initialization comparison | `GS/L1/ablation/eval/*.json` and `GS/L1/ablation/run_eval.sh` |

Table S1 is notation, S3–S4 describe data/configuration, and the other tables summarize experiment outputs. They are typeset in the manuscript.

## Experiment links

[Code, original logs and final checkpoints by experiment](ARTIFACTS.md).
