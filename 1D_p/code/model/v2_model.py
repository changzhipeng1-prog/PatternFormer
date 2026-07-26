"""
V2PDEModel — Full V2 Architecture

Components (in order of data flow):
    1. UNet1d v6          (frozen encoder + frozen decoder)
    2. SpecialTokenEmbeddings  (4 learnable + 1 fixed-zero)
    3. InputProjector     ([z; p] → Qwen hidden space)
    4. OutputProjector    (Qwen hidden → latent)
    5. DualHead           (cls_head + reg_head)
    6. Qwen2.5-7B-Instruct + LoRA

Training (Teacher Forcing):
    - Receive pre-built batch from collate_fn:
        inputs_embeds  [B, L, D]
        attention_mask [B, L]    (1D, for padding; causal handled internally)
        loss_mask      [B, L]    bool, True at target-block solution positions
        cls_targets    [B, L]    long, -100 at ignored positions
        reg_targets    [B, L, latent_dim]  float32
        p_targets      [B, L]   float32   (p value at each solution position)
    - One full Transformer forward pass.
    - DualHead applied to all positions, losses gated by masks.

Inference (generate):
    - Accepts context blocks (precomputed embeddings) + target_p.
    - Autoregressively routes through DualHead until STOP or max_solutions.
    - Returns list of 1024-dim solutions (already UNet-decoded).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import P_START_ID, SOL_ID, STOP_ID, VECTOR_ID, PAD_ID
from model.unet1d_v6 import UNet1d
from model.special_tokens import SpecialTokenEmbeddings
from model.input_projector import InputProjector
from model.output_projector import OutputProjector
from model.dual_head import DualHead
from model.pde_loss import compute_pde_loss, get_pde_lambda
from data.dataset import build_2d_causal_mask


class V2PDEModel(nn.Module):
    """
    End-to-end V2 model.

    Args:
        config: Config instance
        unet_checkpoint: path to pre-trained UNet weights (required)
    """

    def __init__(self, config, unet_checkpoint: str = None, local_rank: int = None):
        super().__init__()
        self.config = config

        # Determine which GPU this rank owns.
        # In DDP (torchrun), LOCAL_RANK is set in the environment.
        # If not in DDP (single-GPU or testing), fall back to cuda:0.
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.component_device = torch.device(f"cuda:{local_rank}")

        # ---- 1. UNet (frozen) ----
        self.unet = UNet1d(
            layers=config.unet_channels,
            latent_dim=config.latent_dim,
            solution_dim=config.solution_dim,
        )
        if unet_checkpoint and os.path.exists(unet_checkpoint):
            print(f"[rank{local_rank}] Loading UNet weights from {unet_checkpoint}")
            state = torch.load(unet_checkpoint, map_location="cpu", weights_only=True)
            self.unet.load_state_dict(state)
        elif unet_checkpoint:
            print(f"[rank{local_rank}] Warning: UNet checkpoint not found at {unet_checkpoint}")

        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False
        print(f"[rank{local_rank}] UNet frozen (encoder + decoder)")

        # ---- 2. Special token embeddings ----
        self.special_tok = SpecialTokenEmbeddings(config.qwen_hidden_dim)

        # ---- 3. Input projector ----
        self.input_proj = InputProjector(config.latent_dim, config.qwen_hidden_dim)

        # ---- 4. Output projector ----
        self.output_proj = OutputProjector(config.qwen_hidden_dim, config.latent_dim)

        # ---- 5. Dual head ----
        self.dual_head = DualHead(
            hidden_dim=config.qwen_hidden_dim,
            latent_dim=config.latent_dim,
            vocab_size=config.vocab_size,
        )

        # ---- 6. Qwen + LoRA (or pretrained-vs-random ablation backbones) ----
        # IMPORTANT: pin this rank's Qwen to its own GPU only.
        # device_map="auto" would spread across ALL visible GPUs, causing
        # conflicts when 8 processes run simultaneously under torchrun.
        rand_init = bool(getattr(config, "random_init_backbone", False))
        full_ft   = bool(getattr(config, "full_finetune_backbone", False))
        sh        = int(getattr(config, "scratch_hidden", 0) or 0)
        sl        = int(getattr(config, "scratch_layers", 0) or 0)
        if rand_init or full_ft or sh or sl:
            # ABLATION: build the SAME architecture (optionally shrunk) with RANDOM weights.
            from transformers import AutoConfig
            qcfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            if sh:
                head_dim = qcfg.hidden_size // qcfg.num_attention_heads   # keep original head_dim (128)
                qcfg.hidden_size = sh
                qcfg.num_attention_heads = max(1, sh // head_dim)
                qcfg.num_key_value_heads = max(1, qcfg.num_attention_heads // 4)
                qcfg.intermediate_size = sh * 4
            if sl:
                qcfg.num_hidden_layers = sl
            tag = "FULL-FT-from-scratch" if full_ft else "random-init (frozen)"
            print(f"[rank{local_rank}] ABLATION {tag}: build backbone from config "
                  f"(hidden={qcfg.hidden_size}, layers={qcfg.num_hidden_layers}) → cuda:{local_rank}")
            # seed so every DDP rank builds the IDENTICAL random base (a frozen base
            # that differs across ranks would make the averaged LoRA gradients incoherent)
            torch.manual_seed(int(getattr(config, "backbone_init_seed", 1234)))
            self.qwen_model = AutoModelForCausalLM.from_config(
                qcfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(f"cuda:{local_rank}")
        else:
            print(f"[rank{local_rank}] Loading Qwen: {config.model_name} → cuda:{local_rank}")
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{local_rank}"},   # pin to this rank's GPU only
                trust_remote_code=True,
            )

        if full_ft:
            # train ALL backbone params (no LoRA); the optimizer collects every
            # requires_grad param of qwen_model, so this needs no optimizer change.
            for p in self.qwen_model.parameters():
                p.requires_grad = True
            n_train = sum(p.numel() for p in self.qwen_model.parameters() if p.requires_grad)
            print(f"[rank{local_rank}] full fine-tune from scratch: {n_train:,} backbone params trainable")
        else:
            self._setup_lora()

        # Move all small trainable modules to this rank's GPU
        for m in [self.special_tok, self.input_proj, self.output_proj,
                  self.dual_head, self.unet]:
            m.to(device=self.component_device, dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_lora(self):
        cfg = self.config
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.qwen_model = get_peft_model(self.qwen_model, lora_cfg)
        self.qwen_model.print_trainable_parameters()

    def _get_transformer(self):
        """Unwrap PEFT (when present) to reach the raw Transformer stack."""
        if isinstance(self.qwen_model, PeftModel):
            return self.qwen_model.base_model.model.model
        return self.qwen_model.model

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, batch: dict, current_epoch: int = 1) -> dict:
        """
        Teacher-Forcing forward pass with LossMask.

        Args:
            batch: output of make_collate_fn (see dataset.py for full field list).
                   The key addition vs the pre-built inputs_embeds is the raw
                   target block data (target_sols_padded, target_k_counts,
                   target_p_vals, target_block_offsets) which lets us rebuild
                   target block embeddings with gradient tracking so that
                   InputProjector and SpecialTokenEmbeddings actually get trained.
            current_epoch: for PDE warmup scheduler

        Returns dict with scalar losses:
            total_loss, mse_loss, pde_loss, ce_loss, lambda_pde
        """
        dev   = self.component_device
        dtype = torch.bfloat16

        # Pre-built (context + placeholder target, detached CPU→GPU)
        inputs_embeds  = batch["inputs_embeds"].to(dev, dtype)    # [B, L, D]
        attention_mask = batch["attention_mask"].to(dev)           # [B, L]
        loss_mask      = batch["loss_mask"].to(dev)                # [B, L]
        cls_targets    = batch["cls_targets"].to(dev)              # [B, L]
        reg_targets    = batch["reg_targets"].to(dev, torch.float32)  # [B, L, lat]
        p_targets      = batch["p_targets"].to(dev, torch.float32)    # [B, L]
        actual_lens    = batch["actual_lens"].to(dev)              # [B]

        # Rebuild target block embeddings with gradient tracking
        # so that InputProjector and SpecialTokenEmbeddings get trained.
        inputs_embeds = self._rebuild_target_block_embeds(
            base_embeds          = inputs_embeds,
            target_block_offsets = batch["target_block_offsets"].to(dev),
            target_sols_padded   = batch["target_sols_padded"].to(dev, dtype),
            target_k_counts      = batch["target_k_counts"].to(dev),
            target_p_vals        = batch["target_p_vals"].to(dev, dtype),
        )   # [B, L, D]  — target block now has gradient; context is detached
        # Guard against any implicit fp32 promotion from mixed module outputs.
        inputs_embeds = inputs_embeds.to(dev, dtype)

        B, L, D = inputs_embeds.shape

        # Build 2D causal + padding attention mask [B, 1, L, L]
        # HuggingFace expects bool mask: True=attend, converted to float additive
        bool_mask_2d = build_2d_causal_mask(actual_lens, L, dev)   # [B, 1, L, L]
        # Convert to float additive mask for HF (0 / -inf)
        float_mask_2d = torch.zeros(B, 1, L, L, dtype=dtype, device=dev)
        float_mask_2d.masked_fill_(~bool_mask_2d, float("-inf"))

        # Transformer forward (no generation, teacher-forced full sequence)
        transformer = self._get_transformer()
        out = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=float_mask_2d,
            return_dict=True,
        )
        hidden = out.last_hidden_state   # [B, L, D]

        # DualHead on all positions
        cls_logits, reg_latent = self.dual_head(hidden)   # [B, L, 5], [B, L, lat]

        # ---- CE loss (classification head) ----
        # Valid where cls_targets != -100
        cls_valid = cls_targets != -100                    # [B, L]
        ce_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        if cls_valid.any():
            # Class weights: STOP is ~3× underrepresented relative to VECTOR
            # (VECTOR appears k_gt times per sample; STOP appears exactly once).
            cls_w = torch.ones(self.config.vocab_size, device=dev, dtype=torch.float32)
            cls_w[STOP_ID]   = self.config.ce_stop_weight
            cls_w[VECTOR_ID] = self.config.ce_vector_weight
            ce_loss = F.cross_entropy(
                cls_logits[cls_valid].float(),            # [N_valid, 5]
                cls_targets[cls_valid],
                weight=cls_w,
                reduction="mean",
            )

        # ---- MSE + PDE loss (regression head, masked to solution positions) ----
        mse_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        pde_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)

        if loss_mask.any():
            # Extract positions: [N_sol, lat]
            reg_latent_sel = reg_latent[loss_mask].float()    # [N_sol, lat]
            reg_targets_sel = reg_targets[loss_mask]           # [N_sol, lat]
            p_targets_sel   = p_targets[loss_mask]             # [N_sol]

            # Decode latent → 1024-dim physical solution
            pred_u = self.unet(reg_latent_sel.to(dtype=dtype), "decode").float()   # [N_sol, 1024]
            gt_latent = reg_targets_sel.to(dtype=dtype)
            gt_u      = self.unet(gt_latent, "decode").float()                     # [N_sol, 1024]

            mse_loss = F.mse_loss(pred_u, gt_u)

            # PDE loss
            lambda_pde = get_pde_lambda(
                current_epoch,
                warmup_start=self.config.pde_warmup_start,
                warmup_end=self.config.pde_warmup_end,
                target_lambda=self.config.lambda_pde,
            )
            if lambda_pde > 0:
                # GT-referenced excess: subtract the AE-decoded GT's residual (the
                # "AE floor") per-sample so a correct branch incurs ~0 penalty.
                ref_u = gt_u.detach() if getattr(self.config, "pde_use_gt_ref", True) else None
                pde_loss = compute_pde_loss(
                    pred_u,                               # [N_sol, 1024]
                    p_targets_sel,                        # [N_sol]
                    self.config.pde_dx,
                    ref_u=ref_u,
                )
        else:
            lambda_pde = 0.0

        total_loss = (
            self.config.lambda_mse * mse_loss
            + lambda_pde * pde_loss
            + self.config.lambda_ce * ce_loss
        )

        return {
            "total_loss": total_loss,
            "mse_loss":   mse_loss,
            "pde_loss":   pde_loss,
            "ce_loss":    ce_loss,
            "lambda_pde": lambda_pde,
        }

    # ------------------------------------------------------------------
    # Target block embedding rebuild (with gradient tracking)
    # ------------------------------------------------------------------

    def _rebuild_target_block_embeds(
        self,
        base_embeds:          torch.Tensor,   # [B, L, D]  detached
        target_block_offsets: torch.Tensor,   # [B]        long
        target_sols_padded:   torch.Tensor,   # [B, K_max, 1024]  bfloat16
        target_k_counts:      torch.Tensor,   # [B]        long
        target_p_vals:        torch.Tensor,   # [B]        bfloat16
    ) -> torch.Tensor:
        """
        Re-compute target block embeddings inside forward() so that gradients
        flow back through InputProjector and SpecialTokenEmbeddings.

        Context positions use the pre-built detached embeddings (no loss there,
        so no gradient is needed).  Only target positions are rebuilt live.

        Returns [B, L, D] where context region is detached and target region
        is gradient-tracked.
        """
        dev, dtype = self.component_device, torch.bfloat16
        B, L, D = base_embeds.shape

        rebuilt_rows = []
        for b in range(B):
            offset = int(target_block_offsets[b].item())
            K_t    = int(target_k_counts[b].item())
            p_val  = float(target_p_vals[b].item())

            z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)

            target_embs = []
            # [P_START]
            target_embs.append(self._special_emb(P_START_ID))
            # [p_proj]
            target_embs.append(self._input_proj(z_zero, p_val))

            for k in range(K_t):
                sol_k = target_sols_padded[b, k]              # [1024]
                z_k   = self.unet(sol_k.unsqueeze(0), "encode").squeeze(0)  # [lat]
                # [SOL] — loss position; gradient flows here to SpecialTokenEmbeddings
                target_embs.append(self._special_emb(SOL_ID))
                # [sol_proj_k] — gradient flows here to InputProjector
                target_embs.append(self._input_proj(z_k, p_val))

            # [STOP]
            target_embs.append(self._special_emb(STOP_ID))

            target_len  = len(target_embs)
            target_block = torch.stack(target_embs, dim=0)   # [target_len, D]

            # Concatenate: detached context | gradient target | detached padding
            context_part = base_embeds[b, :offset]                         # detached
            remaining    = L - offset - target_len
            if remaining > 0:
                pad_part  = base_embeds[b, offset + target_len:]           # detached zeros
                row = torch.cat([context_part, target_block, pad_part], dim=0)
            else:
                row = torch.cat([context_part, target_block[:L - offset]], dim=0)

            rebuilt_rows.append(row)

        # Keep dtype consistent with Qwen weights (bf16) to avoid q_proj dtype mismatch.
        return torch.stack(rebuilt_rows, dim=0).to(device=dev, dtype=dtype)   # [B, L, D]

    # ------------------------------------------------------------------
    # Inference (generate)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_p_vals:  list,            # list[float]
        context_sols:    list,            # list[Tensor [K_i, 1024]]
        target_p_val:    float,
        max_solutions:   int = 8,
    ) -> list:
        """
        Autoregressive generation for a single sample (B=1).

        Builds context embedding, then iteratively routes through DualHead
        until STOP is predicted or max_solutions reached.

        Returns:
            list of Tensor [1024]  — generated PDE solutions (canonicalized order)
        """
        self.eval()
        dev   = self.component_device
        dtype = torch.bfloat16
        D     = self.config.qwen_hidden_dim

        # ---- Build context prefix (no target solutions yet) ----
        from model.canonicalize import canonicalize_solutions

        prefix_embeds = []   # list[Tensor [D]]

        for p_v, sols in zip(context_p_vals, context_sols):
            sols = canonicalize_solutions(
                sols.to(dev, dtype)
            )                               # [K, 1024]
            K = sols.shape[0]

            prefix_embeds.append(self._special_emb(P_START_ID))
            z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
            prefix_embeds.append(self._input_proj(z_zero, p_v))

            for k in range(K):
                prefix_embeds.append(self._special_emb(SOL_ID))
                z_k = self.unet(sols[k:k+1], "encode").squeeze(0)  # [lat]
                prefix_embeds.append(self._input_proj(z_k, p_v))

            prefix_embeds.append(self._special_emb(STOP_ID))

        # ---- Append target P_START + p_embed ----
        prefix_embeds.append(self._special_emb(P_START_ID))
        z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
        prefix_embeds.append(self._input_proj(z_zero, target_p_val))

        # Initial sequence: [1, L_prefix, D]
        seq = torch.stack(prefix_embeds, dim=0).unsqueeze(0).to(dev, dtype)   # [1, L, D]

        transformer = self._get_transformer()
        past_kv = None
        solutions = []

        def _cached_seq_len(cache_obj) -> int:
            # transformers>=4.4x may return DynamicCache; older versions return tuples.
            if cache_obj is None:
                return 0
            if hasattr(cache_obj, "get_seq_length"):
                try:
                    return int(cache_obj.get_seq_length())
                except TypeError:
                    return int(cache_obj.get_seq_length(0))
            return int(cache_obj[0][0].shape[2])

        for step in range(max_solutions):
            # Append [SOL] marker (signals "a vector follows")
            sol_tok = self._special_emb(SOL_ID).unsqueeze(0).unsqueeze(0).to(dev, dtype)  # [1,1,D]

            if past_kv is None:
                # First step: full sequence forward
                full_seq = torch.cat([seq, sol_tok], dim=1).to(dev, dtype)    # [1, L+1, D]
                L_full = full_seq.shape[1]
                attn_1d = torch.ones(1, L_full, dtype=torch.long, device=dev)
                out = transformer(
                    inputs_embeds=full_seq,
                    attention_mask=attn_1d,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                cur_len = _cached_seq_len(past_kv) + 1
                attn_1d = torch.ones(1, cur_len, dtype=torch.long, device=dev)
                out = transformer(
                    inputs_embeds=sol_tok,
                    attention_mask=attn_1d,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
            past_kv = out.past_key_values
            last_h  = out.last_hidden_state[:, -1, :]          # [1, D]

            token_id, reg_lat = self.dual_head.route(last_h)   # [1], [1, lat]
            tid = token_id.item()

            if tid == STOP_ID:
                break

            if tid == VECTOR_ID:
                # reg_lat is already [1, latent_dim] (from reg_head: hidden_dim → lat).
                # Do NOT apply output_proj here — output_proj expects hidden_dim input
                # but reg_lat is already in latent space.  Decode directly via UNet.
                sol_pred = self.unet(reg_lat.to(dtype), "decode")    # [1, 1024]
                solutions.append(sol_pred.squeeze(0).cpu())          # [1024]

                # Teacher-force next input: InputProjector of predicted solution
                z_pred = self.unet(sol_pred, "encode").squeeze(0)   # [lat]
                next_emb = self._input_proj(z_pred, target_p_val)   # [D]
                next_tok = next_emb.unsqueeze(0).unsqueeze(0).to(dev, dtype)  # [1,1,D]

                cur_len = _cached_seq_len(past_kv) + 1
                attn_1d = torch.ones(1, cur_len, dtype=torch.long, device=dev)
                out2 = transformer(
                    inputs_embeds=next_tok,
                    attention_mask=attn_1d,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = out2.past_key_values
                # last hidden at sol_proj position is consumed next iteration
                # (next [SOL] marker will be appended at loop top)

        return solutions   # list of [1024] tensors

    # ------------------------------------------------------------------
    # Internal embedding helpers (single-vector, no batch dim)
    # ------------------------------------------------------------------

    def _special_emb(self, token_id: int) -> torch.Tensor:
        """[D] bfloat16"""
        tid = torch.tensor([token_id], dtype=torch.long, device=self.component_device)
        return self.special_tok(tid).squeeze(0)

    def _input_proj(self, z: torch.Tensor, p_val: float) -> torch.Tensor:
        """z: [lat], returns [D]"""
        dev   = self.component_device
        dtype = torch.bfloat16
        z2d   = z.unsqueeze(0).to(dev, dtype)                 # [1, lat]
        p_t   = torch.tensor([[p_val]], dtype=dtype, device=dev)  # [1, 1]
        return self.input_proj(z2d, p_t).squeeze(0)            # [D]

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        if hasattr(self.qwen_model, "save_pretrained"):
            self.qwen_model.save_pretrained(save_dir)

        torch.save(self.special_tok.state_dict(),
                   os.path.join(save_dir, "special_tokens.pt"))
        torch.save(self.input_proj.state_dict(),
                   os.path.join(save_dir, "input_projector.pt"))
        torch.save(self.output_proj.state_dict(),
                   os.path.join(save_dir, "output_projector.pt"))
        torch.save(self.dual_head.state_dict(),
                   os.path.join(save_dir, "dual_head.pt"))
        torch.save(self.unet.state_dict(),
                   os.path.join(save_dir, "unet.pt"))
        print(f"V2PDEModel saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, config):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = config

        # UNet (frozen)
        model.unet = UNet1d(
            layers=config.unet_channels,
            latent_dim=config.latent_dim,
            solution_dim=config.solution_dim,
        )
        unet_path = os.path.join(checkpoint_dir, "unet.pt")
        if os.path.exists(unet_path):
            model.unet.load_state_dict(
                torch.load(unet_path, map_location="cpu", weights_only=True)
            )
        model.unet.eval()
        for p in model.unet.parameters():
            p.requires_grad = False

        # Learnable modules
        model.special_tok  = SpecialTokenEmbeddings(config.qwen_hidden_dim)
        model.input_proj   = InputProjector(config.latent_dim, config.qwen_hidden_dim)
        model.output_proj  = OutputProjector(config.qwen_hidden_dim, config.latent_dim)
        model.dual_head    = DualHead(config.qwen_hidden_dim, config.latent_dim,
                                      config.vocab_size)

        def _load(module, fname):
            p = os.path.join(checkpoint_dir, fname)
            if os.path.exists(p):
                module.load_state_dict(torch.load(p, map_location="cpu",
                                                   weights_only=True))

        _load(model.special_tok,  "special_tokens.pt")
        _load(model.input_proj,   "input_projector.pt")
        _load(model.output_proj,  "output_projector.pt")
        _load(model.dual_head,    "dual_head.pt")

        # Qwen backbone — pin to this rank's GPU; honor the ablation flags so a
        # resumed run rebuilds the SAME backbone it was trained with.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rand_init = bool(getattr(config, "random_init_backbone", False))
        full_ft   = bool(getattr(config, "full_finetune_backbone", False))
        sh        = int(getattr(config, "scratch_hidden", 0) or 0)
        sl        = int(getattr(config, "scratch_layers", 0) or 0)
        if full_ft:
            # full-FT backbone was saved in full (config + weights) -> load it back
            model.qwen_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir, torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{local_rank}"}, trust_remote_code=True)
            for p in model.qwen_model.parameters():
                p.requires_grad = True
        else:
            if rand_init or sh or sl:
                from transformers import AutoConfig
                qcfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
                if sh:
                    head_dim = qcfg.hidden_size // qcfg.num_attention_heads
                    qcfg.hidden_size = sh
                    qcfg.num_attention_heads = max(1, sh // head_dim)
                    qcfg.num_key_value_heads = max(1, qcfg.num_attention_heads // 4)
                    qcfg.intermediate_size = sh * 4
                if sl: qcfg.num_hidden_layers = sl
                torch.manual_seed(int(getattr(config, "backbone_init_seed", 1234)))
                base = AutoModelForCausalLM.from_config(
                    qcfg, torch_dtype=torch.bfloat16, trust_remote_code=True).to(f"cuda:{local_rank}")
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    config.model_name, torch_dtype=torch.bfloat16,
                    device_map={"": f"cuda:{local_rank}"}, trust_remote_code=True)
            # Resume training from LoRA checkpoint in trainable mode.
            model.qwen_model = PeftModel.from_pretrained(
                base, checkpoint_dir, is_trainable=True)

        model.local_rank       = local_rank
        model.component_device = torch.device(f"cuda:{local_rank}")
        for m in [model.special_tok, model.input_proj, model.output_proj,
                  model.dual_head, model.unet]:
            m.to(device=model.component_device, dtype=torch.bfloat16)

        return model
