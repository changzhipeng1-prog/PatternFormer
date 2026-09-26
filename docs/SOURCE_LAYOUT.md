# Source layout and execution flow

The problem folders keep their original import paths. The `experiments/` directory
provides the common paper-oriented entry point.

```text
checkpoint + test parameters
          |
          v
    generation script ----> generated solution fields
                                      |
                                      v
                            refinement / evaluation
                                      |
                                      v
                             result arrays / CSV / JSON
                                      |
                                      v
                              individual panels
                                      |
                                      v
                              composite figure
```

| Task | Elliptic problems (`1D_p`, `a2a4`, `2D`) | Gray–Scott (`GS/L1`, `GS/L2`) |
|---|---|---|
| Model and input configuration | `code/config.py`, `code/model/` | `code/config.py`, `code/model/` |
| Checkpoint location | `best_ckpt/best_model/` | `ckpt/epoch_60/` |
| Test inference | `test/generate_test.py` | `code/eval_ord.py` |
| Refinement and statistics | `test/compute_stats.py` | `code/eval_ord.py`, `try/eval_grid.py` |
| Main result plots | `test/plot_*.py`, `make_panel_*.py` | `GS/make_combined.py` |
| Extrapolation | `extension/` | `extension/experiments/` |
| Initialization comparisons | `1D_p/ablation_pretrain/`, `2D/test/compare_warm_scratch.py` | `GS/L1/ablation/` |
| Dataset construction | `data_gen/`, `preprocess/` | `data_gen/` |

Start from a [paper figure](EXPERIMENTS.md), then follow the listed experiment sources.
Training entry points remain under `code/` and the ablation directories. Loading a
checkpoint for evaluation does not invoke training.
