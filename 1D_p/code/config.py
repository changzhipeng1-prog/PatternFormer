"""
Configuration: V2 — Hybrid Vocabulary Dual-Head P-Conditioned Model

Architecture overview:
  - UNet1d v6 (frozen): 1024-dim solution → 256-dim latent
  - SpecialTokenEmbeddings: 5 learnable token embeddings (P_START/SOL/STOP/VECTOR/PAD)
  - InputProjector: 2-layer GELU MLP, [z; p] (latent_dim+1) → qwen_hidden_dim (Soft Prompt)
  - OutputProjector: 2-layer GELU MLP, qwen_hidden_dim → latent_dim
  - DualHead: cls_head (D→5) + reg_head (D→latent_dim)
  - Qwen2.5-7B-Instruct with LoRA
  - Physical canonicalization: solutions sorted by L1 integral (ascending)
  - Full LossMask + 2D causal+padding attention mask
  - 8-GPU DDP via torchrun

Special token IDs:
  P_START = 0, SOL = 1, STOP = 2, VECTOR = 3, PAD = 4
"""
import os
from dataclasses import dataclass, field
from typing import List


# ----- self-contained paths (resolved relative to paper/1D_p/code/) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
_CKPT = os.path.join(_HERE, "..", "best_ckpt", "best_model")


# ----- special token constants (used everywhere) -----
P_START_ID = 0
SOL_ID     = 1
STOP_ID    = 2
VECTOR_ID  = 3
PAD_ID     = 4


@dataclass
class Config:
    # ==================== Data paths (resolved relative to paper/1D_p/) ====================
    branch_data_dir: str = os.path.join(_HERE, "..", "data_gen", "cbmfem")
    data_lookup_path: str = os.path.join(_DATA, "p_solutions_lookup.pt")
    train_p_idx_path: str = os.path.join(_DATA, "train_p_idx.pt")
    val_p_idx_path: str = os.path.join(_DATA, "val_p_idx.pt")
    test_p_idx_path: str = os.path.join(_DATA, "test_p_idx.pt")
    data_meta_path: str = os.path.join(_DATA, "p_solutions_meta.pt")

    # ==================== Data params ====================
    solution_dim: int = 1024
    num_branches: int = 8
    p_round_decimals: int = 3
    drop_zero_solution: bool = True
    zero_sol_threshold: float = 1e-4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ==================== Sequence / vocab ====================
    vocab_size: int = 5              # P_START, SOL, STOP, VECTOR, PAD
    max_solutions_per_p: int = 8     # max K solutions per p block
    max_context_blocks: int = 10     # max ICL context blocks
    max_seq_len: int = 256           # hard padding length
    # Rough upper bound per sample:
    # context: max_context_blocks * (2 + max_solutions_per_p + 1) = 10*(2+8+1)=110
    # target:  (2 + max_solutions_per_p + 1) = 11
    # total  ≤ 121  → 256 gives ample headroom

    # ==================== UNet architecture ====================
    unet_channels: List[int] = field(default_factory=lambda: [1, 16, 32, 64, 128, 256])
    latent_dim: int = 256            # v2: 256 (v1 was 512)

    # ==================== UNet checkpoint ====================
    unet_checkpoint_path: str = os.path.join(_CKPT, "unet.pt")

    # ==================== Qwen model ====================
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    qwen_hidden_dim: int = 3584      # actual Qwen2.5-7B hidden size

    # ============ Ablation: does the PRETRAINED backbone matter? ============
    # Arm 1 (default): pretrained Qwen weights, frozen + LoRA.
    # Arm 2: random_init_backbone=True -> same architecture, RANDOM frozen weights + LoRA.
    # Arm 3: full_finetune_backbone=True (+ scratch_hidden/layers) -> small random
    #        backbone, ALL params trained from scratch (no LoRA).
    random_init_backbone: bool = False
    full_finetune_backbone: bool = False
    scratch_hidden: int = 0          # >0 overrides backbone hidden_size (also set qwen_hidden_dim)
    scratch_layers: int = 0          # >0 overrides backbone num_hidden_layers
    backbone_init_seed: int = 1234   # seed for random-init backbone -> identical base on every
                                     # DDP rank and across the stage1->stage2 resume

    # ==================== LoRA params ====================
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ==================== Training hyperparameters ====================
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 50
    lr_projector: float = 1e-4      # InputProjector + OutputProjector + SpecialTokens + DualHead
    lr_lora: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"

    # ==================== Loss weights ====================
    lambda_mse: float = 1.0
    lambda_ce: float = 1.0
    lambda_pde: float = 0.05        # target value (after warmup)
    # GT-referenced physics (qwen_pde): penalize only the EXCESS of the prediction's
    # FDM residual over the AE-decoded GT's residual (the per-sample "AE floor").
    # The raw residual here is dominated by the AE floor (measured: AE-floor/true-GT
    # residual ≈ 2200x; weighted floor λ_pde·floor ≈ 4.5x the MSE term), so the raw
    # loss taxes correct solutions and over-smooths.  GT-ref removes that floor.
    pde_use_gt_ref: bool = True
    pde_warmup_start: int = 5       # epoch at which PDE loss begins
    pde_warmup_end: int = 30        # epoch at which PDE loss reaches lambda_pde (matches 2D)
    pde_dx: float = 1.0 / 1023

    # CE class weights: STOP is ~3x underrepresented vs VECTOR (VECTOR/STOP ≈ 3)
    ce_stop_weight: float = 3.0     # weight for STOP_ID in cross-entropy loss
    ce_vector_weight: float = 1.0   # weight for VECTOR_ID in cross-entropy loss

    # ==================== Dynamic context sampling ====================
    max_context_p: int = 10         # k ~ Uniform[1, max_context_p]
    sampler_seed: int = 42
    target_repeat_per_epoch: int = 1   # repeat each target p with new contexts
    multisol_weight_power: float = 1.5  # k_gt^power for per-p weighted sampling

    # ==================== LR plateau decay ====================
    lr_plateau_patience: int = 8    # epochs without val_mse improvement → decay LR
    lr_plateau_factor: float = 0.3  # multiplicative LR decay factor on plateau

    # ==================== Training control ====================
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    save_total_limit: int = 3
    num_gpus: int = 8

    # ==================== Output paths ====================
    output_dir: str = "./checkpoints"
    log_dir: str = "./outputs/logs"
    test_output_dir: str = "./outputs/test_results"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data_lookup_path) or ".", exist_ok=True)


default_config = Config()
