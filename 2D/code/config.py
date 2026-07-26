"""
Configuration: V3 — 2D PDE, Hybrid Vocabulary Dual-Head P-Conditioned Model

Differences from V2:
  - solution_dim: 1024 → 145  (2D FEM mesh, ell=3)
  - UNet1d_v6   → SolutionAutoencoder2D  (MLP-based)
  - branch_data_dir: points to 2D dataset
  - num_branches: 8 → 10  (max solutions per s: 2 sym + 4+4 asym)
  - max_solutions_per_p: 8 → 10
  - pde_dx: removed (2D PDE loss uses mesh, not uniform grid)
"""
import os
from dataclasses import dataclass, field
from typing import List

# ----- self-contained paths (resolved relative to paper/2D/code/) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
_CKPT = os.path.join(_HERE, "..", "best_ckpt", "best_model")

# Special token IDs (same as v2)
P_START_ID = 0
SOL_ID     = 1
STOP_ID    = 2
VECTOR_ID  = 3
PAD_ID     = 4


@dataclass
class Config:
    # ==================== Data paths (resolved relative to paper/2D/) ====================
    data_lookup_path: str = os.path.join(_DATA, "p_solutions_lookup_2d_filtered.pt")
    train_p_idx_path: str = os.path.join(_DATA, "train_p_idx.pt")
    val_p_idx_path:   str = os.path.join(_DATA, "val_p_idx.pt")
    test_p_idx_path:  str = os.path.join(_DATA, "test_p_idx.pt")
    data_meta_path:   str = os.path.join(_DATA, "split_info.pt")

    # ==================== Data params ====================
    solution_dim: int = 145           # 2D FEM: 145 nodes (ell=3 mesh)
    num_branches: int = 4             # max solutions per s after D4 filtering (2 sym + 2 orbit reps)
    p_round_decimals: int = 0         # s is integer
    drop_zero_solution: bool = False  # all solutions are non-trivial
    zero_sol_threshold: float = 1e-4
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15

    # ==================== Sequence / vocab ====================
    vocab_size: int = 5
    max_solutions_per_p: int = 4     # up to 4 solutions per s after D4 filtering
    max_context_blocks:  int = 10
    max_seq_len:         int = 128
    # max tokens per sample (after D4 filtering, max 4 solutions per s):
    # context: 10*(2+4+1) = 70
    # target:     (2+4+1) =  7
    # total <= 77 → 128 gives headroom

    # ==================== Autoencoder ====================
    # (replaces UNet1d channels list)
    autoencoder_hidden_dim: int = 512   # MLP hidden width
    latent_dim: int = 256               # bottleneck

    # ==================== Autoencoder checkpoint ====================
    unet_checkpoint_path: str = os.path.join(_CKPT, "autoencoder2d.pt")

    # ==================== Qwen model ====================
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    qwen_hidden_dim: int = 3584

    # ==================== LoRA params ====================
    lora_r: int = 16              # same as v2
    lora_alpha: int = 32          # same as v2 (alpha/r = 2)
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ==================== Training hyperparameters ====================
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1000
    lr_projector: float = 1e-4    # same as v2
    lr_lora: float = 2e-5         # same as v2
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"

    # ==================== Normalization scales ====================
    # p values range from -138 to 1600 in v3 dataset.
    # Dividing by p_input_scale maps p to roughly [-0.09, 1.0],
    # preventing the large p values from dominating [z; p] concatenation.
    p_input_scale:   float = 1600.0

    # ==================== Loss weights ====================
    # MSE is now computed in latent space (z_MSE ~ 0.1).
    # lambda_mse = 1.0 gives effective latent MSE contribution ~0.1, dominant over CE ~0.001.
    lambda_mse: float = 0.5
    lambda_ce:  float = 1.0
    # PDE physics loss is ON in this variant (qwen_pde), as a GT-referenced residual
    # EXCESS — penalize only how much a prediction's FEM residual exceeds the residual
    # the AE-decoded ground truth reaches for the same (s, branch).  A correct branch
    # then incurs zero physics penalty (no conflict with the multi-branch MSE); only
    # slots less physical than the AE can represent are pulled toward the Newton basin.
    lambda_pde: float = 0.05    # physics weight (warmup ep5->30)
    pde_use_gt_ref: bool = True # use AE-decoded GT residual as the per-sample reference
    pde_warmup_start: int = 5
    pde_warmup_end:   int = 30

    ce_stop_weight:   float = 3.0
    ce_vector_weight: float = 1.0

    # ==================== Dynamic context sampling ====================
    max_context_p: int = 10
    sampler_seed: int = 42
    target_repeat_per_epoch: int = 10
    multisol_weight_power: float = 0.0   # uniform: no k=4 oversampling

    # ==================== LR plateau decay ====================
    lr_plateau_patience: int = 25
    lr_plateau_factor:   float = 0.5
    lr_min_ratio:        float = 0.10   # min_lr = lr * lr_min_ratio (prevents LR decaying to zero)

    # ==================== Training control ====================
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    save_total_limit: int = 3
    num_gpus: int = 6

    # ==================== Output paths ====================
    output_dir:      str = "./checkpoints"
    log_dir:         str = "./outputs/logs"
    test_output_dir: str = "./outputs/test_results"

    def __post_init__(self):
        os.makedirs(self.output_dir,      exist_ok=True)
        os.makedirs(self.log_dir,         exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data_lookup_path) or ".", exist_ok=True)


default_config = Config()
