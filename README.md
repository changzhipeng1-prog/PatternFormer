# PatternFormer

Code for **PatternFormer: Learning Multiple Solution Patterns in Reaction–Diffusion Systems**.

Start with the [paper figure guide](docs/EXPERIMENTS.md) to find an experiment by its
main-text or supplementary figure number. Use saved experiment outputs to assemble
figures, or load trained checkpoints to run inference and numerical refinement.

Follow the [download → evaluation → plotting workflow](docs/REVIEWER_WORKFLOW.md)
for executable commands and the input dependencies of each figure.

## Getting started

1. Follow [environment setup](docs/SETUP.md).
2. [Download final main checkpoints](artifacts/README.md); see [DATA.md](DATA.md) for dataset and figure-input paths.
3. Choose a figure from the [experiment guide](docs/EXPERIMENTS.md).
4. To train models, follow [training and checkpoint construction](docs/TRAINING.md).
5. For model evaluation, follow [checkpoint inference](docs/INFERENCE.md).

```bash
python experiments/run.py --list
python experiments/run.py main-fig-02
python experiments/run.py main-fig-02 --run
```

Without `--run`, the command displays the source scripts, working directory, inputs
and outputs. `--run` invokes the figure script. The figure commands use saved outputs;
training is a separate workflow.

## Repository layout

| Directory | Contents | Paper location |
|---|---|---|
| `experiments/` | Figure-number catalogue and command launcher | Main and supplementary figures |
| `docs/` | Setup, figure recipes and checkpoint inference | Reader guide |
| `1D_p/` | Single-parameter elliptic problem | Figures 2, 4–6; Figure S6 |
| `a2a4/` | Two-parameter elliptic problem | Figures 4–5; Figures S1, S5 |
| `2D/` | Two-dimensional elliptic problem | Figures 4–6; Figures S2–S4 |
| `GS/L2/` | Gray–Scott, smaller diffusion coefficients | Figures 3, S8 |
| `GS/L1/` | Gray–Scott, larger diffusion coefficients | Figure 6, Figure S7 |
| Root `fig_*.py`, `plot_*.py` | Composite figure scripts | See the figure catalogue |

Within each problem, `code/` contains the model and configuration, `test/` the evaluation
and plotting scripts, `extension/` extrapolation, and `data_gen/` the classical solvers.
The GS problems use `results/` and `try/` for evaluation outputs and comparison scripts.
See [source layout](docs/SOURCE_LAYOUT.md) for the execution flow.

Original experiment logs: [logs/](logs/README.md).

## Citation

```bibtex
@article{patternformer,
  title  = {PatternFormer: Learning Multiple Solution Patterns in Reaction--Diffusion Systems},
  author = {Chang, Zhipeng and Yin, Wenpeng and Hao, Wenrui},
  year   = {2025}
}
```
