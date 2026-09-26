# Development entry points

- [Source layout](docs/SOURCE_LAYOUT.md)
- [Paper figures](docs/EXPERIMENTS.md)
- [Figure preparation](docs/FIGURE_RECIPES.md)
- [Checkpoint inference](docs/INFERENCE.md)

Model implementation and configuration are under each problem's `code/` directory.
Classical solvers are under `data_gen/`; dataset conversion and splitting are under
`preprocess/` where present. Training scripts are separate from the checkpoint
inference commands in the reader guide.
