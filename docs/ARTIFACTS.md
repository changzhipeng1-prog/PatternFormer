# Experiment code, logs and final checkpoints

| Problem | Experiment file index | Final checkpoint |
|---|---|---|
| [1D_p](../1D_p/README.md) | [Code and original logs](../1D_p/EXPERIMENT_FILES.md) | [Download](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-1d.tar.gz) |
| [a2a4](../a2a4/README.md) | [Code and original logs](../a2a4/EXPERIMENT_FILES.md) | [Download](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-two-parameter.tar.gz) |
| [2D](../2D/README.md) | [Code and original logs](../2D/EXPERIMENT_FILES.md) | [Download](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-elliptic-2d.tar.gz) |
| [GS/L1](../GS/L1/README.md) | [Code and original logs](../GS/L1/EXPERIMENT_FILES.md) | [Download](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-gray-scott-l1.tar.gz) |
| [GS/L2](../GS/L2/README.md) | [Code and original logs](../GS/L2/EXPERIMENT_FILES.md) | [Download](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/research-artifacts-v1/patternformer-gray-scott-l2.tar.gz) |

The checkpoint links select one final main-model checkpoint per problem. Each problem index groups the original logs by experiment source directory; log filenames preserve stage, arm, and scheduler identifiers.

Shared Gray–Scott data construction: [code](../GS) · [original build log](../logs/historical/GS/build.log).

See [paper figure guide](EXPERIMENTS.md) for the figure-to-code mapping.

## Evaluation datasets

- [elliptic-1d-data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-elliptic-1d-data.tar.gz)
- [elliptic-two-parameter-data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-elliptic-two-parameter-data.tar.gz)
- [elliptic-2d-data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-elliptic-2d-data.tar.gz)
- [gray-scott-l1-data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-gray-scott-l1-data.tar.gz)
- [gray-scott-l2-data](https://github.com/changzhipeng1-prog/PatternFormer/releases/download/evaluation-inputs-v1/patternformer-gray-scott-l2-data.tar.gz)

[Download, evaluate and plot](REVIEWER_WORKFLOW.md) connects these inputs to the figure commands.
