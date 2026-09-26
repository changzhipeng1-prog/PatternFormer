# Ablation: does the PRETRAINED backbone matter? (1D_p)

Reviewer-facing control for the title claim *"fine-tuning a **pretrained** large language
model"*. The setup is identical to the FPT test (Lu et al., *Pretrained Transformers as
Universal Computation Engines*, AAAI 2022): hold everything fixed and swap only the frozen
backbone.

All arms reuse the **delivered 1D_p pipeline** (`../code`, `../data`, frozen AE at
`../best_ckpt/best_model/unet.pt`) and the **same 2-stage recipe** (withcontext → nocontext),
on 4 GPUs with eff-batch `8*4*4 = 128` (matches the delivered 8-GPU run `8*2*8 = 128`).

| arm | script | backbone | trainable | what it tests |
|---|---|---|---|---|
| **1** pretrained (control) | `run_arm1_pretrained.sh` | Qwen2.5-7B **pretrained**, frozen | LoRA + projectors + heads | the delivered method, re-run at 4-GPU for a matched control |
| **2** random-init | `run_arm2_random.sh` | Qwen2.5-7B-arch **random**, frozen | LoRA + projectors + heads | **does pretraining help?** only diff vs Arm 1 is `--random_init_backbone 1` |
| **3** from-scratch | `run_arm3_scratch.sh` | small (hidden 1024, 12 layers) **random**, full-FT | **all** backbone params | **do you need 7B?** purpose-built small model trained properly |

If Arm 2 ≈ Arm 1, the pretraining is doing little (retitle to "a large transformer").
If Arm 1 ≫ Arm 2, the pretrained prior is real and the title is earned.

## Code hooks (in `../code`, all default-off, guarded)
- `config.py`: `random_init_backbone`, `full_finetune_backbone`, `scratch_hidden`,
  `scratch_layers`, `backbone_init_seed`.
- `model/v2_model.py`: `__init__` and `from_pretrained` build the backbone from
  `AutoConfig`→`from_config` (random) instead of `from_pretrained` when a flag is set;
  the random base is **seeded** so every DDP rank (and the stage1→stage2 resume) gets an
  identical frozen base. Shrinking `hidden_size` also fixes the head geometry
  (`num_attention_heads = hidden/head_dim`).
- `train.py`: CLI `--random_init_backbone / --full_finetune_backbone /
  --scratch_hidden / --scratch_layers`. Full-FT needs no optimizer change — every
  `requires_grad` backbone param is collected into the existing LoRA optimizer group.

Validated by `scratchpad/smoke_ablation.py` (tiny random model: Arm 2 = LoRA-only
trainable on a frozen random base; Arm 3 = full backbone trainable; both forward cleanly).

## Run / evaluate
```bash
cd paper/1D_p/ablation_pretrain
sbatch run_arm2_random.sh        # Arm 2 (the decisive control)
sbatch run_arm1_pretrained.sh    # Arm 1 (matched control)
sbatch run_arm3_scratch.sh       # Arm 3 (size control; lr may need tuning)
# deliverable of each = ckpt_<arm>/best_model ; evaluate with the standard
# paper/1D_p/test/compute_stats.py pointed at that checkpoint.
```
Compare arms on the SAME metrics as the paper: val_mse, M1 direct rel-L2, exact
count-matching, M2 Newton steps. Report all numbers from the final on-disk checkpoints.

## Related experiment files

[Original logs for this experiment](../EXPERIMENT_FILES.md#1d_pablation_pretrain) · [Problem overview](../README.md).
