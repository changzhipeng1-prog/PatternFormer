# Download, evaluate and plot

The final checkpoints, evaluation data and historical experiment logs are indexed
by problem in [ARTIFACTS.md](ARTIFACTS.md). No model training is needed for the
main checkpoint evaluations below. Use Python 3.10 and [SETUP.md](SETUP.md).

## Download inputs

From the repository root, download the model and data for each desired problem:

| Problem | Final model bundle | Evaluation data bundle |
|---|---|---|
| 1D single parameter | `elliptic-1d` | `elliptic-1d-data` |
| 1D two parameters | `elliptic-two-parameter` | `elliptic-two-parameter-data` |
| 2D elliptic | `elliptic-2d` | `elliptic-2d-data` |
| Gray–Scott L1 | `gray-scott-l1` | `gray-scott-l1-data` |
| Gray–Scott L2 | `gray-scott-l2` | `gray-scott-l2-data` |

```bash
python artifacts/download.py --bundle elliptic-1d
python artifacts/download.py --bundle elliptic-1d-data
python artifacts/download.py --bundle paper-figure-inputs
```

Repeat the first two commands with the relevant bundle IDs in the table. For all
paper figures, install all five data bundles and `paper-figure-inputs`. The Qwen
base model is obtained separately through Hugging Face, as described in setup.

Data are installed at the original code paths. Lookups retain their full parameter
ordering so the supplied train/validation/test indices remain valid. The two-parameter
data bundle contains its test partition; the other bundles contain the indexed
lookups, split files and, for Gray–Scott, normalization and numerical operators.

## Evaluate elliptic checkpoints

Choose a new result directory for each run. `--limit 0` evaluates the complete test
partition; the default limit of 2 is a short execution check. A short subset does
not contain all parameter examples used in the paper figures.

```bash
python experiments/evaluate.py --problem 1D_p --limit 0 --cpus 4 --output ../evaluation/1D_p
python experiments/evaluate.py --problem a2a4 --limit 0 --cpus 4 --output ../evaluation/a2a4
python experiments/evaluate.py --problem 2D --limit 0 --cpus 4 --output ../evaluation/2D
```

Run each GPU command inside a GPU allocation. Each output directory contains
`generated_solutions.pt`, `stats.pt`, `stats.csv`, and its execution log. The statistics
are computed from that run's generated fields, including numerical refinement.

## Plot the elliptic evaluation outputs

The plotting wrapper creates a separate workspace under a new output directory
outside the repository. It copies in the selected evaluation results before running
the original panel scripts; original inputs are not overwritten.

```bash
python experiments/plot.py main-fig-02 --evaluation 1D_p=../evaluation/1D_p --output ../figures/main02
python experiments/plot.py supp-fig-01 --evaluation a2a4=../evaluation/a2a4 --output ../figures/supp01
python experiments/plot.py supp-fig-02 --evaluation 2D=../evaluation/2D --output ../figures/supp02
python experiments/plot.py supp-fig-03 --evaluation 2D=../evaluation/2D --output ../figures/supp03
python experiments/plot.py supp-fig-04 --evaluation 2D=../evaluation/2D --output ../figures/supp04
python experiments/plot.py main-fig-05 \
  --evaluation 1D_p=../evaluation/1D_p \
  --evaluation a2a4=../evaluation/a2a4 \
  --evaluation 2D=../evaluation/2D --output ../figures/main05
```

The example and error panels read the selected evaluation outputs. Figure 2's five
example parameters are selected from those outputs by `1D_p/test/select_examples.py`.
Timing panels read the original timing CSVs; ordinary inference does not measure the
separate classical-solver timing experiment. Dataset branch panels use the lookup
or the supplied original branch-panel files. Each plotting run writes `inputs.json`
and `plot.log` and prints its PDF/PNG paths.

## Evaluate and plot Gray–Scott

```bash
python experiments/evaluate_gs.py --setup L1 --limit 0 --output ../evaluation/GS-L1
python experiments/evaluate_gs.py --setup L2 --limit 0 --output ../evaluation/GS-L2
python experiments/plot.py supp-fig-07 --gs-evaluation L1=../evaluation/GS-L1/results.json --output ../figures/supp07
python experiments/plot.py main-fig-03 --gs-evaluation L2=../evaluation/GS-L2/results.json --output ../figures/main03
```

The default evaluation limit is one test parameter; `--limit 0` selects all test
parameters. The figure wrapper requires the complete test-index set when replacing
the fixed-K test records. Fixed-K test counts are read from the supplied new JSON.
The composite also uses distinct experiment inputs: noise sweeps, random-start and
STOP baseline records, representative noise-generated fields, and extrapolation
fields. These are supplied in the repository and figure-input bundle. Training and
validation atlas entries retain their saved records; they are not test evaluations.
See [INFERENCE.md](INFERENCE.md#grayscott-noise-and-baseline-comparisons) for the
separate noise and baseline evaluation commands and example-field generation.

## Other figure experiments

The full list of figure commands is `python experiments/run.py --list`.
Extrapolation and initialization comparisons use their own experiment results,
rather than the main test inference. Render the supplied historical inputs with:

```bash
python experiments/plot.py main-fig-04 --saved-inputs --output ../figures/main04
python experiments/plot.py main-fig-06 --saved-inputs --output ../figures/main06
python experiments/plot.py supp-fig-05 --saved-inputs --output ../figures/supp05
python experiments/plot.py supp-fig-06 --saved-inputs --output ../figures/supp06
python experiments/plot.py supp-fig-08 --saved-inputs --output ../figures/supp08
```

Use `--saved-inputs` with the other figure IDs to render the supplied historical
outputs too. Figure S6 plots saved continuation fields. Figure S8 plots the saved
training-parameter fields at indices 638 and 482; to create new fields first, follow
[training-parameter evaluation](INFERENCE.md#grayscott-training-parameter-examples).
Source commands for extrapolation, initialization comparisons and timing experiments
are linked in [EXPERIMENTS.md](EXPERIMENTS.md) and [FIGURE_RECIPES.md](FIGURE_RECIPES.md).

## Training records and code

[TRAINING.md](TRAINING.md) links data preparation, autoencoder training, model stages,
initialization components and evaluation. [ARTIFACTS.md](ARTIFACTS.md) connects each
problem's code to its original training logs and final evaluation checkpoint. Only
final evaluation checkpoints are released; no intermediate training epochs are in
the model bundles. Comparison-arm outputs in the figure-input bundle remain associated
with their own source code and logs; they are not replaced by the five main models.
