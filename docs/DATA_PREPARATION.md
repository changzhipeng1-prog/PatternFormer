# Numerical data and partition preparation

Model training consumes the datasets, parameter partitions and normalization files
listed in [DATA.md](../DATA.md). Numerical data construction precedes AE training.
Keep the lookup and its split indices together: an index refers to a particular
ordering of parameters in its lookup file.

## Single-parameter elliptic equation

The continuation scripts use `1D_p/data_gen/cbmfem/initial_S2.npy` as the seed solution
array at the starting parameter. `MTG.py`, `Newton.py` and `Filtering.py` implement the
classical multilevel solver; `run_branch.py` continues one column of the seed array.
Run each branch from that directory, with its column index as the argument:

```bash
(cd 1D_p/data_gen/cbmfem && python run_branch.py 0)
```

Repeat for each seed column. The outputs are `branch_<b>_p_new.npy` and
`branch_<b>_S_new.npy`. Assemble the lookup and parameter splits:

```bash
python 1D_p/preprocess/build_lookup.py \
  --branch_dir 1D_p/data_gen/cbmfem --out_dir 1D_p/data
```

This writes the lookup, metadata and train/validation/test index files consumed by
`1D_p/code/config.py`.

## Two-parameter elliptic equation

`a2a4/data_gen/generate_fullspace_gpu.py` performs the parameter-grid continuation
using seed datasets. Its command-line options select the parameter grid, seed files,
output path and allocated device indices. `simplify_dataset.py` converts the generated
solution records. The selected region dataset is the input to the split command:

```bash
python a2a4/preprocess/split_dataset.py \
  --region a2a4/data/bvp_region.pt --out_dir a2a4/data
```

The model reads `bvp_region_train.pt`, `bvp_region_val.pt` and `bvp_region_test.pt`.
The FDM refinement entry is:

```bash
NCPU=4 python a2a4/data_gen/correct_dataset_fdm.py
```

That script backs up the split files in `data/orig_backup/` and writes refined records
to `data/`. Inspect its input/output paths before applying it to an existing dataset.
The original launch recipes under `data_gen/` specify seed-dataset and grid arguments;
use the devices provided by your scheduler allocation.

## Two-dimensional elliptic equation

`2D/data_gen/generate_dataset_2d.py` builds the numerical lookup using the FEM mesh
and boundary-condition input files named in the script. Its output directory is
`2D/data_gen/2d_eq61_dataset/`. Place the resulting unfiltered
`p_solutions_lookup_2d.pt` under `2D/data/`, then run:

```bash
python 2D/data_gen/filter_d4_dataset.py
python 2D/preprocess/make_splits.py \
  --lookup 2D/data/p_solutions_lookup_2d_filtered.pt --out_dir 2D/data
```

The 2D AE trainer reads the unfiltered lookup. The main model reads the filtered
lookup and the split indices created for it. Keep the mesh coordinates, elements,
free-node indices and boundary metadata in the lookup.

## Gray–Scott

The numerical operators and seed inputs include `op_n128.mat` and, for the seed-bank
scripts, `matlab_ref.mat`. Their paths are resolved by the data-generation scripts.
The cell archives contain activator/substrate fields and `(rho, mu)` parameter values.

| Regime | Raw cell inputs | Lookup builder | Model inputs |
|---|---|---|---|
| L1 | `GS/L1/data_gen/data_merged/`, `data_dense/` | `GS/L1/data_gen/build_big.py` | `GS/L1/data/*_big.pt` |
| L2 | `GS/L2/data_gen/L2_gen/data_L2/`, `data_L2_dense/` | `GS/L2/data_gen/build_L2_dense.py` | `GS/L2/data/*_L2.pt` |

L1 continuation is implemented in `GS/L1/data_gen/gs_continuation.py`. L2 seed-bank
entry points are `GS/L2/data_gen/L2_gen/gen_L2.py` and `gen_L2_dense.py`. Their CLI
arguments select grid ranges, refinement settings, output directories and shards.
After collecting the cell archives:

```bash
python GS/L1/data_gen/build_big.py
python GS/L2/data_gen/build_L2_dense.py
```

The builders remove homogeneous fields, group solutions by parameter, reduce square
symmetries and create lookup, partition and normalization files. L2's expanded builder
uses a prior L2 lookup/split assignment to retain existing parameter memberships;
keep that input dataset available before writing the expanded outputs. `OLDQ` and
`QDATA` in that script select the input and output directories.

Several GS construction steps use CUDA. Allocate their resources explicitly; run
classical CPU-only solvers in CPU jobs. Do not infer available devices from a low
utilization reading.

## Continue to model training

Once the model inputs are in place, follow [TRAINING.md](TRAINING.md): train the AE,
then the model stages, then evaluate the selected checkpoint and prepare the figures.
For evaluation using existing datasets, retain the associated supplied partition
files instead of rerunning a split command over a different parameter ordering.
