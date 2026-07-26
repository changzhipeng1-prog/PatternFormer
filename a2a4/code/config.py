"""
Configuration: V2 — Hybrid Vocabulary Dual-Head (a4, a2)-Conditioned Model
BVP: -u'' + a4*u^4 + a2*u^2 = 0,  u'(0)=0, u(1)=0
"""
import os
from dataclasses import dataclass, field
from typing import List

P_START_ID = 0
SOL_ID     = 1
STOP_ID    = 2
VECTOR_ID  = 3
PAD_ID     = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# shipped datasets live in paper/a2a4/data/ ; final ckpt in paper/a2a4/best_ckpt/best_model/
DATA_GEN = os.path.join(BASE_DIR, "..", "data")
_CKPT = os.path.join(BASE_DIR, "..", "best_ckpt", "best_model")


@dataclass
class Config:
    # ==================== Data paths ====================
    train_data_path: str = os.path.join(DATA_GEN, "bvp_region_train.pt")
    val_data_path:   str = os.path.join(DATA_GEN, "bvp_region_val.pt")
    test_data_path:  str = os.path.join(DATA_GEN, "bvp_region_test.pt")

    # ==================== Data params ====================
    solution_dim: int  = 1024
    param_dim:    int  = 2       # (a4, a2)
    max_k:        int  = 6       # max solutions per sample

    # ==================== Sequence / vocab ====================
    vocab_size:           int = 5
    max_solutions_per_p:  int = 6
    max_context_blocks:   int = 10
    max_seq_len:          int = 256

    # ==================== UNet architecture ====================
    unet_channels: List[int] = field(default_factory=lambda: [1, 16, 32, 64, 128, 256])
    latent_dim:    int = 256

    # ==================== UNet checkpoint ====================
    unet_checkpoint_path: str = os.path.join(_CKPT, "unet.pt")

    # ==================== 1D_p pretrained ckpt (Phase 2 warm-start) ====================
    pretrained_1dp_ckpt: str = os.path.join(
        BASE_DIR, "..", "..", "1D_p", "best_ckpt", "best_model"
    )

    # ==================== Qwen model ====================
    model_name:      str = "Qwen/Qwen2.5-7B-Instruct"
    qwen_hidden_dim: int = 3584

    # ==================== LoRA params ====================
    lora_r:              int       = 16
    lora_alpha:          int       = 32
    lora_dropout:        float     = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ==================== Training hyperparameters ====================
    batch_size:                  int   = 4
    gradient_accumulation_steps: int   = 4
    num_epochs:                  int   = 60
    lr_projector:                float = 1e-4
    lr_lora:                     float = 2e-5
    weight_decay:                float = 0.01
    max_grad_norm:                float = 1.0
    mixed_precision:             str   = "bf16"

    # ==================== Loss weights ====================
    lambda_mse:       float = 1.0
    lambda_ce:        float = 1.0
    lambda_pde:       float = 0.05
    # GT-referenced physics (qwen_pde): penalize only relu(R^2(pred) - R^2(AE(gt))).
    # The raw residual is dominated by the AE floor (Lu/h amplifies AE recon error by
    # ~1/h; measured AE-floor ~1e4, ~1e7x the MSE term), so the raw loss over-smooths.
    pde_use_gt_ref:   bool  = True
    pde_warmup_start: int   = 5
    pde_warmup_end:   int   = 30
    pde_dx:           float = 1.0 / 1024   # h = 1/N

    ce_stop_weight:   float = 3.0
    ce_vector_weight: float = 1.0

    # ==================== Dynamic context sampling ====================
    max_context_p:             int   = 8
    sampler_seed:              int   = 42
    target_repeat_per_epoch:   int   = 2
    # Inverse-frequency class weighting to handle k imbalance
    use_weighted_sampler:      bool  = True

    # ==================== LR plateau decay ====================
    lr_plateau_patience: int   = 8
    lr_plateau_factor:   float = 0.3

    # ==================== Output paths ====================
    output_dir:      str = "./checkpoints"
    log_dir:         str = "./outputs/logs"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


default_config = Config()
