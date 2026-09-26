# Gray–Scott L1: larger diffusion coefficients

Paper locations: **Supplementary Figure S7 and Main Figure 6**.

- `code/eval_ord.py`: fixed-budget checkpoint evaluation.
- `try/eval_grid.py`: noise, random-start and STOP-model comparisons.
- `ckpt/epoch_60/`: ordered model; `ckpt/stop_base/`: STOP model.
- `results/` and `try/results_grid/`: per-parameter evaluation outputs.
- `extension/experiments/`: extrapolation entry points and outputs.

Use the [inference guide](../../docs/INFERENCE.md) for commands and the
[figure recipes](../../docs/FIGURE_RECIPES.md) for assembly. Input paths are listed
in [DATA.md](../../DATA.md). Run `python experiments/run.py --list` from the repository
root to locate the corresponding figure.

## Code, logs and checkpoint

[Experiment-by-experiment log index](EXPERIMENT_FILES.md) · [Download final evaluation checkpoint](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-gray-scott-l1.tar.gz)

Install with `python artifacts/download.py --bundle gray-scott-l1` from the repository root. Files are placed in `GS/L1/ckpt/epoch_60/`.

## Evaluation data and plotting

[Download evaluation data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-gray-scott-l1-data.tar.gz) · [Evaluation-to-figure commands](../../docs/REVIEWER_WORKFLOW.md)

Install with `python artifacts/download.py --bundle gray-scott-l1-data` from the repository root.
