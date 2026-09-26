# Training and checkpoint construction

The workflow has four kinds of inputs: numerical datasets and split indices, the Qwen
base model, the problem-specific autoencoder, and a parent checkpoint for transfer
stages. The stage catalogue is [training.json](../experiments/training.json).

```text
numerical datasets + parameter splits
        |
        +--> problem-specific AE
        |           |
        |           +--> 1D context --> 1D no-context
        |                                      |
        |                              two-parameter context
        |                                      |
        |                              two-parameter no-context
        |                                      |
        |                                 2D initialization
        |                                      |
        |                              2D context --> 2D no-context
        |
        +--> GS AE + STOP checkpoint --> L1 ordered training
        |                                      |
        |                 L2 STOP components + L1 adapter
        |                                      |
        |                              L2 mixed initialization
        |                                      |
        |                              L2 ordered training
        |
        +--> independent initialization comparison arms
```

## Environment and stage launcher

Follow [SETUP.md](SETUP.md) and [DATA_PREPARATION.md](DATA_PREPARATION.md). Use the
Python interpreter with PyTorch, Transformers, PEFT and the other model dependencies.
All commands below start at the repository root. `RUN` is a new output directory:

```bash
export RUN="$PWD/runs/training"
python experiments/train.py --list
python experiments/train.py ae-1d --run-root "$RUN"
```

Without `--run`, a command only prints the exact launch plan. Add `--run` to execute
it inside your GPU allocation. Stages retain the GPU count, batch size, accumulation,
epoch budget and learning rates of the source recipe listed in the catalogue. The
launcher uses `torch.distributed.run` for multi-GPU stages and never chooses devices
outside the allocation. Use a new `--run-root` for a new experiment; an existing stage
output directory is not overwritten.

Every executed stage writes `run.json` (command and timestamps), `environment.txt`
and `train.log` beside its checkpoints. Monitor a job with `squeue` and its `train.log`.
For example, after activating the environment, a single-node SLURM launch is:

```bash
mkdir -p logs
sbatch --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=32 --mem=256G \
  --time=72:00:00 --output=logs/ae_1d_%j.out \
  --wrap="python experiments/train.py ae-1d --run-root '$RUN' --run"
```

Set resources to the stage's documented GPU count and your cluster's partition limits.
Finish a parent stage before launching a dependent one. The launcher's `--parent`
means weight initialization for a new training stage; it is not an optimizer-state
resume after interruption.

## Autoencoders

```bash
python experiments/train.py ae-1d --run-root "$RUN" --run
python experiments/train.py ae-two-param --run-root "$RUN" --run
python experiments/train.py ae-2d --run-root "$RUN" --run
python experiments/train.py ae-gs-l1 --run-root "$RUN" --run
python experiments/train.py ae-gs-l2 --run-root "$RUN" --run
```

Each command is a separate job. The two 1D AE trainers use eight GPUs; the 2D and GS
AE trainers use one GPU each. Outputs:

| Stage | AE file relative to `$RUN` |
|---|---|
| `ae-1d` | `ae-1d/unet.pt` |
| `ae-two-param` | `ae-two-param/unet.pt` |
| `ae-2d` | `ae-2d/checkpoints/best_model/autoencoder2d.pt` |
| `ae-gs-l1` | `ae-gs-l1/checkpoints/autoencoder2d.pt` |
| `ae-gs-l2` | `ae-gs-l2/checkpoints/autoencoder2d.pt` |

The AE selection metric is the validation reconstruction loss in each trainer: MSE
for elliptic AEs, and the mean of validation-batch relative-L2 values for GS AEs.
`2D/code/train_unet.py` uses the 256-dimensional AE and the unfiltered 2D lookup;
`train_unet_v2.py` is a separate, 512-dimensional architecture. The 2D AE trainer writes
its split indices under the stage's `data/`; model training reads the parameter splits
under `2D/data/` associated with the filtered lookup.

## Single-parameter model

```bash
python experiments/train.py 1d-context --run-root "$RUN" \
  --ae "$RUN/ae-1d/unet.pt" --run
python experiments/train.py 1d-nocontext --run-root "$RUN" \
  --ae "$RUN/ae-1d/unet.pt" \
  --parent "$RUN/1d-context/checkpoints/best_model" --run
```

The final evaluation bundle is `$RUN/1d-nocontext/checkpoints/best_model/`.
Both stages use eight GPUs; the second initializes from the first stage's selected
validation checkpoint and trains without context examples.

## Two-parameter model

```bash
python experiments/train.py two-param-context --run-root "$RUN" \
  --ae "$RUN/ae-two-param/unet.pt" \
  --parent "$RUN/1d-nocontext/checkpoints/best_model" --run
python experiments/train.py two-param-nocontext --run-root "$RUN" \
  --ae "$RUN/ae-two-param/unet.pt" \
  --parent "$RUN/two-param-context/checkpoints/best_model" --run
```

The model transfers the Qwen-side components from the 1D model and uses its own AE
and parameter input projection. Final bundle:
`$RUN/two-param-nocontext/checkpoints/best_model/`.

## Two-dimensional model

Construct the input checkpoint explicitly, retaining the 2D AE and transferring the
adapter, output projector, head and special tokens from the chosen two-parameter model:

```bash
python experiments/prepare_init.py elliptic-2d \
  --parent "$RUN/two-param-nocontext/checkpoints/best_model" \
  --ae "$RUN/ae-2d/checkpoints/best_model/autoencoder2d.pt" \
  --out "$RUN/init-2d"
python experiments/train.py 2d-context --run-root "$RUN" \
  --ae "$RUN/ae-2d/checkpoints/best_model/autoencoder2d.pt" \
  --parent "$RUN/init-2d" --run
python experiments/train.py 2d-nocontext --run-root "$RUN" \
  --ae "$RUN/ae-2d/checkpoints/best_model/autoencoder2d.pt" \
  --parent "$RUN/2d-context/checkpoints/best_model" --run
```

The input projector is initialized for the new parameter dimension. Both stages use
four GPUs. Final bundle: `$RUN/2d-nocontext/checkpoints/best_model/`. To use an existing
transfer parent, pass its directory to `prepare_init.py` instead. The original launch
recipe `2D/code/run_twostage_chain.sh` names that parent explicitly.

## Gray–Scott ordered models

The ordered training stage consumes an existing STOP-model bundle. The bundle must
contain the AE, adapter, input projector, dual head and special-token weights; the
associated STOP architectures are under `GS/L1/baseline_stop/` and
`GS/L2/baseline_stop/`.

```bash
python experiments/train.py gs-l1-ordered --run-root "$RUN" \
  --ae "$PWD/GS/L1/ckpt/stop_base/autoencoder2d.pt" \
  --parent "$PWD/GS/L1/ckpt/stop_base" --run
```

For L2, assemble the mixed initialization before training. Adapter and special tokens
come from the L1 ordered model; the input projector, head and AE come from the L2 STOP
bundle. Keep the AE and its latent-space head together.

```bash
python experiments/prepare_init.py gs-l2-mixed \
  --parent "$RUN/gs-l1-ordered/checkpoints/epoch_60" \
  --target "$PWD/GS/L2/ckpt/stop_base" --out "$RUN/init-gs-l2"
python experiments/train.py gs-l2-ordered --run-root "$RUN" \
  --ae "$RUN/init-gs-l2/autoencoder2d.pt" \
  --parent "$RUN/init-gs-l2" --run
```

L1 uses four GPUs and L2 two. Both write `checkpoints/epoch_60/` and
`checkpoints/best_model/`. The figure-evaluation routes use the `epoch_60` checkpoint;
`best_model` is the separately saved minimum-validation-MSE checkpoint. Do not
interchange these names when selecting an evaluation input.

## Initialization comparisons

The catalogue contains separate stages for each comparison arm:

| Figure 6 group | Context stage | No-context stage | GPUs |
|---|---|---|---|
| 1D pretrained | `1d-pretrained-context` | `1d-pretrained-nocontext` | 4 |
| 1D random frozen backbone | `1d-random-context` | `1d-random-nocontext` | 4 |
| 2D fresh Qwen initialization | `2d-fresh-context` | `2d-fresh-nocontext` | 4 |
| GS pretrained | `gs-l1-pretrained-context` | `gs-l1-pretrained-nocontext` | 4 |
| GS random frozen backbone | `gs-l1-random-context` | `gs-l1-random-nocontext` | 4 |

Use `--ae` to pass the same problem-specific AE to both arms. The no-context stage
receives its **own arm's** `checkpoints/best_model` through `--parent`. For example:

```bash
python experiments/train.py 1d-random-context --run-root "$RUN" \
  --ae "$RUN/ae-1d/unet.pt" --run
python experiments/train.py 1d-random-nocontext --run-root "$RUN" \
  --ae "$RUN/ae-1d/unet.pt" \
  --parent "$RUN/1d-random-context/checkpoints/best_model" --run
```

The two 1D arms use the same four-GPU recipe. `2d-fresh-context` starts from Qwen
without an a2a4 parent; compare its final checkpoint with `2d-nocontext`. The GS arms
start directly from the selected backbone rather than the STOP parent. Random-backbone
weights are reconstructed using the model's configured initialization seed; preserve
that configuration for inference.

## Checkpoint selection and evaluation

The elliptic training scripts save `best_model/` whenever validation MSE improves.
Training summaries and logs retain the selected loss and stage settings. Training and
validation examples use their corresponding partition files. Test inference is a
separate command, described in [INFERENCE.md](INFERENCE.md).

Keep the whole checkpoint directory together; copying only the adapter omits the AE
and prediction heads. To evaluate a new bundle, point the evaluation entry at the
selected directory, then run refinement and the [figure recipes](FIGURE_RECIPES.md).
Figure 6 also reads training curves: retain `train.log` from both stages alongside
inference outputs. Its plot readers select filenames in the plotting script; point
those inputs at the corresponding arm logs when plotting a new run.
