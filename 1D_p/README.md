# Single-parameter elliptic equation

Paper locations: **Main Figures 2, 4–6; Supplementary Figure S6**.

| Directory | Purpose |
|---|---|
| `code/` | Model, configuration and training entry points |
| `best_ckpt/best_model/` | Evaluation checkpoint and AE components |
| `data/` | Evaluation data and split indices |
| `test/` | Checkpoint inference, refinement, statistics and example plots |
| `test/timing/` | Timing measurements and timing-panel plots |
| `extension/` | Extrapolation experiments |
| `data_gen/` | Classical data-generation solvers and branch plots |

From the repository root:

```bash
python 1D_p/test/generate_test.py
python 1D_p/test/compute_stats.py
```

For figure assembly and its saved inputs, follow the [paper figure guide](../docs/EXPERIMENTS.md)
and [figure recipes](../docs/FIGURE_RECIPES.md). Model/data placement is in [DATA.md](../DATA.md).
