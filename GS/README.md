# Gray–Scott experiments

| Folder | Diffusion coefficients | Paper figures |
|---|---|---|
| `L1/` | `D_A=2.5e-4`, `D_S=5e-4` | Main Figure 6; Supplementary Figure S7 |
| `L2/` | `D_A=6.25e-5`, `D_S=1.25e-4` | Main Figure 3; Supplementary Figure S8 |

Each problem contains `code/` (model and evaluator), `data/` (lookup, splits and
normalization), `ckpt/` (saved models), `results/` (evaluation records), `try/`
(comparison scripts) and `extension/` (parameter extrapolation).

From the repository root, `python GS/make_combined.py` assembles both diffusion
regimes' composite figures from saved inputs. See the [figure recipes](../docs/FIGURE_RECIPES.md),
[checkpoint inference commands](../docs/INFERENCE.md) and [data layout](../DATA.md).

Shared modules: `_example_gen.py` generates representative solution fields;
`_gridload.py` loads per-parameter records; `make_fig1.py` provides gallery rendering;
`make_combined.py` assembles the composite panels.
