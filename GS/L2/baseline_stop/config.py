"""Config — Gray-Scott 2D, conv-AE + Qwen multi-solution generator.

Adapted from 2D_multisolution/qwen/config.py. Key differences:
  - parameter is 2D: (rho, mu)  (reference used a scalar p)
  - solution is a 2x128x128 image (reference: 145-d FEM vector) -> conv AE
  - no D4 reduction yet, so cap solutions/param and lengthen max_seq_len
"""
import os
from dataclasses import dataclass, field
from typing import List

# Special token IDs (same as reference)
P_START_ID = 0
SOL_ID     = 1
STOP_ID    = 2
VECTOR_ID  = 3
PAD_ID     = 4

# Package-relative paths (this file lives in GS/L2/code/; data in GS/L2/data/).
# Using __file__ keeps the package self-contained regardless of the cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))
_CKPT = os.path.normpath(os.path.join(_HERE, "..", "ckpt"))


@dataclass
class Config:
    # ==================== Data paths ====================
    data_lookup_path: str = f"{_DATA}/gs_lookup_L2.pt"
    train_p_idx_path: str = f"{_DATA}/train_p_idx_L2.pt"
    val_p_idx_path:   str = f"{_DATA}/val_p_idx_L2.pt"
    test_p_idx_path:  str = f"{_DATA}/test_p_idx_L2.pt"
    norm_stats_path:  str = f"{_DATA}/norm_stats_L2.pt"

    # ==================== Data params ====================
    param_dim:    int = 2                 # (rho, mu)
    solution_dim: int = 32768             # 2*128*128 (informational; conv AE ignores)
    img_size:     int = 128
    in_ch:        int = 2                 # A, S

    # ==================== Sequence / vocab ====================
    vocab_size: int = 5
    max_solutions_per_p: int = 24         # D4-reduced data: max 24, mean 9.8 (no truncation)
    max_context_p:       int = 10         # # of in-context example params (reference parity)
    max_seq_len:         int = 768
    # block = 3 + 2K tokens; worst case (11 blocks * (3+2*24)=51) = 561 < 768
    # ordered fixed-K (qwen_ord): every TARGET block has EXACTLY fixed_k SOL slots.
    # head k<K_t = GT teacher-forced (ordered MSE); tail k>=K_t = self-pred free-run
    # (physics-only). fixed_k >= global max K_t (L1=24, L2=19) so K_t<=K always holds.
    fixed_k:             int = 24

    # ==================== Autoencoder ====================
    latent_dim: int = 256
    unet_checkpoint_path: str = f"{_CKPT}/epoch_60/autoencoder2d.pt"

    # ==================== Qwen model ====================
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    qwen_hidden_dim: int = 3584

    # ==================== LoRA ====================
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ==================== Training ====================
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 60          # FINE-TUNE from STOP best_model -> short
    lr_projector: float = 1e-4
    lr_lora: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"

    # ==================== Loss weights ====================
    lambda_mse: float = 0.5     # decoded-image (field-space) MSE, normalized — HEAD only
    lambda_ce:  float = 0.0     # qwen_ord: NO STOP head -> CE off (was 1.0 in STOP design)
    lambda_latent: float = 0.5  # (3) direct latent-space MSE on reg head output — HEAD only
    lambda_pde: float = 0.02    # (2) FDM residual on HEAD — SMALL: physics min is the trivial
                                #     bleached state (~1e-12), 8 orders below the AE floor
    lambda_pde_tail: float = 0.01  # tail (k>=K_t) physics-only, SMALL so it can't bias the
                                #   head; tail = best-effort discovery of solutions beyond GT
    pde_floor: float = 1e-3     # hinge: only penalize residual ABOVE ~2x the AE round-trip
                                #     ceiling (4.8e-4) -> real & trivial states incur 0 penalty
    pde_warmup_start: int = 0   # FINE-TUNE from converged base: patterns already formed ->
    pde_warmup_end:   int = 10  #   physics on from the start (gentle ramp), polishes only
    ce_stop_weight:   float = 1.0   # (unused: lambda_ce=0)
    ce_vector_weight: float = 1.0
    # ---- (4) scheduled sampling (fights exposure bias) — ON for ordered finetune ----
    # with prob ss_prob, replace a target sol_proj input by the model's own (detached)
    # prediction, so training conditions on inference-like (self-generated) history.
    # tail slots (k>=K_t) are ALWAYS self-pred (prob 1) — handled separately in forward.
    ss_prob:         float = 0.25
    ss_warmup_start: int = 0    # ramp ss_prob over [start,end] epochs (on from start: finetune)
    ss_warmup_end:   int = 15
    # physics-loss operator (FDM, matches the data generator)
    op_path: str = f"{_HERE}/op_n128.mat"
    DA: float = 6.25e-5    # L=2: 2.5e-4 / 4
    DS: float = 1.25e-4    # L=2: 5e-4   / 4

    # ==================== Context sampling ====================
    sampler_seed: int = 42
    target_repeat_per_epoch: int = 10
    # "random" = any candidate param as context; "local" = nearest in (rho,mu).
    # Local context is ~4x more relevant (nearby params share solution structure)
    # and is a test-time-valid prior; the val/eval context library is the TRAIN set.
    context_mode: str = "random"
    # with-context and no-context are trained as SEPARATE phases (never mixed):
    #   Phase 2 (run_phase2): no_context_prob = 0.0  (pure with-context)
    #   Phase 3 (run_phase3): no_context_prob = 1.0  (pure no-context, resume from Phase 2)
    no_context_prob: float = 0.0

    # ==================== LR plateau ====================
    lr_plateau_patience: int = 25
    lr_plateau_factor:   float = 0.5
    lr_min_ratio:        float = 0.10

    # ==================== Control ====================
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    save_total_limit: int = 3
    num_gpus: int = 6

    # ==================== Output ====================
    output_dir:      str = "./checkpoints_final"
    log_dir:         str = "./outputs/logs"
    test_output_dir: str = "./outputs/test_results"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)


default_config = Config()
