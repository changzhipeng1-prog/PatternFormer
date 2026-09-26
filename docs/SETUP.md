# Environment and execution

Use Python 3.10. Install a CUDA-enabled PyTorch build appropriate for your GPU when
running model inference. Figure assembly from saved arrays and evaluation tables
uses CPU; some figure-preparation scripts also perform CPU numerical refinement.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency versions are pinned in `requirements.txt`. GPU evaluation uses
CUDA PyTorch; install a wheel compatible with your driver before the other packages.
The 7B backbone is loaded in bfloat16. The original environment uses CUDA 12.8
on A100 80 GB GPUs; memory needs also depend on batch and solution counts.

For the model, obtain `Qwen/Qwen2.5-7B-Instruct` through Hugging Face or place its files
in a local directory and set `model_name` in the relevant `code/config.py` to that
path. The task-specific LoRA adapter is loaded with its bundled AE and projection/head
weights; see [DATA.md](../DATA.md).

Run documentation commands from the repository root unless a `cd` is shown. Scripts
use paths relative to their own file or to the working directory specified in the
figure catalogue. Keep the data and model component filenames as listed.

On a compute cluster, request a GPU allocation before running inference. The existing
`.slurm` and `.sh` files are examples of cluster launches: set the Python environment,
project path, output path and scheduler resources for your system before submitting.
For headless figure rendering, set `MPLBACKEND=Agg`.

Figure outputs use the filenames in [EXPERIMENTS.md](EXPERIMENTS.md). Running a script
again can overwrite its corresponding generated files. Use a separate working copy
when comparing configurations.

Training commands and stage dependencies are in [TRAINING.md](TRAINING.md).

For a fixed Qwen snapshot, the source checkpoint revision is
`a09a35458c702b33eeacc393d103063234e8bc28`:

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='a09a35458c702b33eeacc393d103063234e8bc28'))"
```

Use the printed directory as `model_name` in the problem configuration.
