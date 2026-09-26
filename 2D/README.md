# Two-dimensional elliptic equation

Paper locations: **Main Figures 4–6; Supplementary Figures S2–S4**.

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
python 2D/test/generate_test.py
python 2D/test/compute_stats.py
```

For figure assembly and its saved inputs, follow the [paper figure guide](../docs/EXPERIMENTS.md)
and [figure recipes](../docs/FIGURE_RECIPES.md). Model/data placement is in [DATA.md](../DATA.md).

## Code, logs and checkpoint

[Experiment-by-experiment log index](EXPERIMENT_FILES.md) · [Download final evaluation checkpoint](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-2d.tar.gz)

Install with `python artifacts/download.py --bundle elliptic-2d` from the repository root. Files are placed in `2D/best_ckpt/best_model/`.
