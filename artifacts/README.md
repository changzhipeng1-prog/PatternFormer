# Final evaluation checkpoints

Download bundles from the [research artifacts release](https://github.com/changzhipeng1-prog/PatternFormer/releases/tag/research-artifacts-v1). The five bundles contain only the final main-model checkpoints used for evaluation, one per problem. Intermediate training checkpoints are excluded. The Qwen base model is downloaded separately as described in [setup](../docs/SETUP.md).

From the repository root, with Python 3.10 or newer:

```bash
python artifacts/download.py --bundle elliptic-1d
python artifacts/download.py --bundle elliptic-two-parameter
python artifacts/download.py --bundle elliptic-2d
python artifacts/download.py --bundle gray-scott-l1
python artifacts/download.py --bundle gray-scott-l2
```

Each command downloads its bundle, checks SHA-256 hashes, and installs files at the paths used by the experiment scripts. Existing identical files are retained; different existing files are never overwritten. Use `--dest /path/to/another/directory` to select another destination.

[manifest.json](manifest.json) lists every packaged file and download asset with its size and checksum. Main checkpoints are installed under `best_ckpt/best_model/` for elliptic problems and `ckpt/epoch_60/` for Gray–Scott. Datasets, separate ablation checkpoints and saved figure inputs are not included. See [DATA.md](../DATA.md) for their expected paths.

`build_bundles.py --source /path/to/experiment/root --output /path/to/new/bundles` builds the same bundle structure from experiment files.

## Experiment links

[Code, original logs and final checkpoints by experiment](../docs/ARTIFACTS.md).
