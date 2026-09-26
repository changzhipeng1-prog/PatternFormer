# Figure preparation recipes

Commands below start in the repository root and use the saved inputs described in
[DATA.md](../DATA.md). They do not run model training. The composite panel assemblers
consume both PDF and PNG versions of their individual panels.

## Main Figure 2: single-parameter elliptic problem

```bash
python 1D_p/test/plot_examples.py
python 1D_p/test/plot_box.py
python 1D_p/data_gen/make_bifurcation_1Dp.py
python 1D_p/test/timing/plot_efficiency.py
python experiments/run.py main-fig-02 --run
```

Inputs include `test/examples_solutions.pt`, `test/stats.csv`, the lookup dataset and
timing tables. For the model-overlay version of the branch panel, use
`1D_p/data_gen/make_bifurcation_overlay.py` with its additional inputs. The composite
output is `1D_p/fig_1Dp_panel.pdf`.

## Main Figure 3 and Supplementary Figure S7: Gray–Scott

```bash
python GS/make_combined.py
```

This writes `GS/combined_L2.pdf` (main Figure 3) and `GS/combined_L1.pdf` (Figure S7).
Both regimes' inputs must be present: lookup/split files, population-result shards,
example fields, noise-curve records and extrapolation inputs consumed by the script.
`GS/_gridload.py` loads population records, `GS/make_fig1.py` draws the solution gallery,
and `GS/make_combined.py` assembles the atlas, gallery and summary panels.

## Main Figure 4: extrapolation

```bash
python experiments/run.py main-fig-04 --run
```

Inputs: `three_methods_1dp.pt`, `three_methods_2d.pt`, `a2a4_cov_M3_bfs.pt`,
`grid_gt_cont.pt`, `gt_extension.pt`, and the two `sols_row2.pt` files. Their paths
are printed by the catalogue. The output is `fig_extension_2x3.pdf`.

## Main Figure 5: direct outputs and refinement

Prepare the per-problem field arrays from saved predictions and statistics:

```bash
python 1D_p/test/_who_npz.py
python a2a4/test/_who_npz.py
python 2D/test/_who_npz.py
python experiments/run.py main-fig-05 --run
```

The first three scripts perform CPU refinement and write `test/who_npz.npz` in each
problem directory. If these arrays are already present, run only the last command.
Output: `fig5_who_solves.pdf`.

## Main Figure 6: initialization comparisons

```bash
python experiments/run.py main-fig-06 --run
```

This reads saved generations, evaluation arrays/JSONs, training-curve logs and
`gs_map_data.npz`. It writes `fig_ablation_full.pdf`, corresponding to the manuscript
asset `fig_ablation.pdf`. The script itself contains the per-panel input paths.

## Supplementary Figures S1–S2: elliptic composite panels

```bash
python a2a4/test/plot_examples.py
python a2a4/test/plot_box.py
python a2a4/test/timing/plot_efficiency_a2a4.py
python experiments/run.py supp-fig-01 --run

python 2D/test/plot_examples.py
python 2D/test/plot_box.py
python 2D/test/timing/plot_efficiency_2d.py
python experiments/run.py supp-fig-02 --run
```

Place the branch-panel assets `a2a4/data_gen/fig_a2a4_bifurcation.{pdf,png}` and
`2D/data_gen/fig_2D_bifurcation.{pdf,png}` before assembly. The composite outputs are
`a2a4/fig_a2a4_panel.pdf` and `2D/fig_2D_panel.pdf`.

## Supplementary Figures S3–S4: two-dimensional examples

```bash
python 2D/test/plot_examples.py
```

Reads `2D/test/generated_solutions.pt`. It writes the main example panel plus
`fig_2D_examples_nearzero.pdf` and `fig_2D_examples_notcollapse.pdf` under `2D/test/`.
Example selection and rendering are implemented in this script.

## Supplementary Figure S5: two-parameter extrapolation

```bash
python experiments/run.py supp-fig-05 --run
```

Reads `a2a4/extension/grid_direct.pt` and `grid_gt_cont.pt` and writes
`a2a4/extension/fig_a2a4_appendix.pdf`.

## Supplementary Figure S6: far continuation

```bash
python experiments/run.py supp-fig-06 --run
```

The launcher selects `1D_p/extension/` as the working directory. The script reads
`p18seed_result.pt` and `../data_gen/cbmfem/initial_S2.npy`, performs CPU continuation,
and writes `farp_fields_p100.pt` and `fig_farp_continuation.pdf` there.

## Supplementary Figure S8: training-parameter fields

Use the [training-parameter evaluation command](INFERENCE.md#gray-scott-training-parameter-examples)
to produce `GS/L2/try/results/train_beyond.pt`. The records contain `sols` (solution
fields), `is_gt` (membership flags), `param` (parameter values), and `gt_count`.
The gallery displays the activator component `sols[:, 0]`, with frame colours
indicating the saved membership flag. The shared tile renderer is
[`GS/make_fig1.py`](../GS/make_fig1.py), function `panel`.
