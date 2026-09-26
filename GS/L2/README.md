# Gray–Scott L2: smaller diffusion coefficients

Paper locations: **Main Figure 3 and Supplementary Figure S8**.

- `code/eval_ord.py`: fixed-budget checkpoint evaluation.
- `try/eval_grid.py`: noise, random-start and STOP-model comparisons.
- `ckpt/epoch_60/`: ordered model; `ckpt/stop_base/`: STOP model.
- `results/` and `try/results_grid/`: per-parameter evaluation outputs.
- `extension/experiments/`: extrapolation entry points and outputs.

Use the [inference guide](../../docs/INFERENCE.md) for commands and the
[figure recipes](../../docs/FIGURE_RECIPES.md) for assembly. Input paths are listed
in [DATA.md](../../DATA.md). Run `python experiments/run.py --list` from the repository
root to locate the corresponding figure.
