# Two-parameter elliptic equation

Paper locations: **Main Figures 4–5; Supplementary Figures S1 and S5**.

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
python a2a4/test/generate_test.py
python a2a4/test/compute_stats.py
```

For figure assembly and its saved inputs, follow the [paper figure guide](../docs/EXPERIMENTS.md)
and [figure recipes](../docs/FIGURE_RECIPES.md). Model/data placement is in [DATA.md](../DATA.md).

## Code, logs and checkpoint

[Experiment-by-experiment log index](EXPERIMENT_FILES.md) · [Download final evaluation checkpoint](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-two-parameter.tar.gz)

Install with `python artifacts/download.py --bundle elliptic-two-parameter` from the repository root. Files are placed in `a2a4/best_ckpt/best_model/`.

## Evaluation data and plotting

[Download evaluation data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-elliptic-two-parameter-data.tar.gz) · [Evaluation-to-figure commands](../docs/REVIEWER_WORKFLOW.md)

Install with `python artifacts/download.py --bundle elliptic-two-parameter-data` from the repository root.
