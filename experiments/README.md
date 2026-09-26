# Paper-oriented commands

The figure catalogue is [figures.json](figures.json). Each entry contains its paper
number, descriptive name, existing experiment sources, figure command, primary input
paths and output paths.

```bash
python experiments/run.py --list
python experiments/run.py main-fig-04
python experiments/run.py main-fig-04 --run
```

The launcher runs each script in its specified working directory and propagates a
nonzero exit code. It does not train a model or change any experiment configuration.
Some figure scripts perform CPU refinement or continuation using saved predictions.
See [the figure recipes](../docs/FIGURE_RECIPES.md) for preparing individual panels and
[the full paper map](../docs/EXPERIMENTS.md) for tables and model-source entries.

## Training stages

`python experiments/train.py --list` lists the training stages. Their exact arguments
are stored in [training.json](training.json). Follow [TRAINING.md](../docs/TRAINING.md)
for input checkpoints, execution order and stage outputs. `prepare_init.py` assembles
transfer inputs in a new directory without replacing existing weights.
