"""
V2PDEModel for 1D_a2a4: -u'' + a4*u^4 + a2*u^2 = 0

Differences from 1D_p:
  - InputProjector: [z; a4; a2] (latent_dim+2) instead of [z; p] (latent_dim+1)
  - params in batch are [B, L, 2] (a4, a2) instead of scalar p
  - PDE loss uses a4, a2 tensors
  - from_pretrained_1dp: loads 1D_p weights, reinitialises input_proj
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

    def __init__(self, config, unet_checkpoint: str = None, local_rank: int = None):
        super().__init__()
        self.config = config

        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank       = local_rank
        self.component_device = torch.device(f"cuda:{local_rank}")

        # ---- UNet (frozen) ----
        self.unet = UNet1d(
            layers=config.unet_channels,
            latent_dim=config.latent_dim,
            solution_dim=config.solution_dim,
        )
        if unet_checkpoint and os.path.exists(unet_checkpoint):
            print(f"[rank{local_rank}] Loading UNet from {unet_checkpoint}")
            self.unet.load_state_dict(
                torch.load(unet_checkpoint, map_location="cpu", weights_only=True)
            )
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        # ---- Small learnable modules ----
        self.special_tok = SpecialTokenEmbeddings(config.qwen_hidden_dim)
        self.input_proj  = InputProjector(config.latent_dim, config.qwen_hidden_dim,
                                          param_dim=config.param_dim)
        self.output_proj = OutputProjector(config.qwen_hidden_dim, config.latent_dim)
        self.dual_head   = DualHead(config.qwen_hidden_dim, config.latent_dim,
                                    config.vocab_size)

        # ---- Qwen + LoRA ----
        print(f"[rank{local_rank}] Loading Qwen → cuda:{local_rank}")
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{local_rank}"},
            trust_remote_code=True,
        )
        self._setup_lora()

        dev, dtype = self.component_device, torch.bfloat16
        for m in [self.special_tok, self.input_proj, self.output_proj,
                  self.dual_head, self.unet]:
            m.to(device=dev, dtype=dtype)

    def _setup_lora(self):
        cfg = self.config
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none", task_type="CAUSAL_LM",
        )
        self.qwen_model = get_peft_model(self.qwen_model, lora_cfg)
        self.qwen_model.print_trainable_parameters()

    def _get_transformer(self):
        if hasattr(self.qwen_model, "base_model"):
            return self.qwen_model.base_model.model.model
        return self.qwen_model.model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict, current_epoch: int = 1) -> dict:
        dev, dtype = self.component_device, torch.bfloat16

        inputs_embeds  = batch["inputs_embeds"].to(dev, dtype)
        attention_mask = batch["attention_mask"].to(dev)
        loss_mask      = batch["loss_mask"].to(dev)
        cls_targets    = batch["cls_targets"].to(dev)
        reg_targets    = batch["reg_targets"].to(dev, torch.float32)
        params_targets = batch["params_targets"].to(dev, torch.float32)   # [B, L, 2]
        actual_lens    = batch["actual_lens"].to(dev)

        inputs_embeds = self._rebuild_target_block_embeds(
            base_embeds          = inputs_embeds,
            target_block_offsets = batch["target_block_offsets"].to(dev),
            target_sols_padded   = batch["target_sols_padded"].to(dev, dtype),
            target_k_counts      = batch["target_k_counts"].to(dev),
            target_params        = batch["target_params"].to(dev, dtype),   # [B, 2]
        ).to(dev, dtype)

        B, L, D = inputs_embeds.shape

        bool_mask_2d  = build_2d_causal_mask(actual_lens, L, dev)
        float_mask_2d = torch.zeros(B, 1, L, L, dtype=dtype, device=dev)
        float_mask_2d.masked_fill_(~bool_mask_2d, float("-inf"))

        transformer = self._get_transformer()
        out = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=float_mask_2d,
            return_dict=True,
        )
        hidden = out.last_hidden_state

        cls_logits, reg_latent = self.dual_head(hidden)

        # CE loss
        cls_valid = cls_targets != -100
        ce_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        if cls_valid.any():
            cls_w = torch.ones(self.config.vocab_size, device=dev, dtype=torch.float32)
            cls_w[STOP_ID]   = self.config.ce_stop_weight
            cls_w[VECTOR_ID] = self.config.ce_vector_weight
            ce_loss = F.cross_entropy(
                cls_logits[cls_valid].float(), cls_targets[cls_valid],
                weight=cls_w, reduction="mean",
            )

        # MSE + PDE loss
        mse_loss  = torch.tensor(0.0, device=dev, dtype=torch.float32)
        pde_loss  = torch.tensor(0.0, device=dev, dtype=torch.float32)
        lambda_pde = 0.0

        if loss_mask.any():
            reg_lat_sel    = reg_latent[loss_mask].float()
            reg_tgt_sel    = reg_targets[loss_mask]
            params_sel     = params_targets[loss_mask]        # [N_sol, 2]
            a4_sel         = params_sel[:, 0]
            a2_sel         = params_sel[:, 1]

            pred_u = self.unet(reg_lat_sel.to(dtype), "decode").float()
            gt_u   = self.unet(reg_tgt_sel.to(dtype), "decode").float()
            mse_loss = F.mse_loss(pred_u, gt_u)

            lambda_pde = get_pde_lambda(
                current_epoch,
                warmup_start=self.config.pde_warmup_start,
                warmup_end=self.config.pde_warmup_end,
                target_lambda=self.config.lambda_pde,
            )
            if lambda_pde > 0:
                # GT-referenced excess: subtract the AE-decoded GT's residual (the
                # "AE floor", ~1e4 here) per-sample so a correct branch incurs ~0 penalty.
                ref_u = gt_u.detach() if getattr(self.config, "pde_use_gt_ref", True) else None
                pde_loss = compute_pde_loss(pred_u, a4_sel, a2_sel, self.config.pde_dx,
                                            ref_u=ref_u)

        total_loss = (self.config.lambda_mse * mse_loss
                      + lambda_pde * pde_loss
                      + self.config.lambda_ce * ce_loss)

        return {"total_loss": total_loss, "mse_loss": mse_loss,
                "pde_loss": pde_loss, "ce_loss": ce_loss, "lambda_pde": lambda_pde}

    # ------------------------------------------------------------------
    # Rebuild target block with gradient tracking
    # ------------------------------------------------------------------

    def _rebuild_target_block_embeds(self, base_embeds, target_block_offsets,
                                     target_sols_padded, target_k_counts,
                                     target_params) -> torch.Tensor:
        dev, dtype = self.component_device, torch.bfloat16
        B, L, D   = base_embeds.shape

        rows = []
        for b in range(B):
            offset  = int(target_block_offsets[b].item())
            K_t     = int(target_k_counts[b].item())
            params_b = target_params[b]            # [2]  (a4, a2)

            z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)

            tgt_embs = [self._special_emb(P_START_ID),
                        self._input_proj(z_zero, params_b)]

            for k in range(K_t):
                sol_k = target_sols_padded[b, k]
                z_k   = self.unet(sol_k.unsqueeze(0), "encode").squeeze(0)
                tgt_embs.append(self._special_emb(SOL_ID))
                tgt_embs.append(self._input_proj(z_k, params_b))

            tgt_embs.append(self._special_emb(STOP_ID))

            tgt_len   = len(tgt_embs)
            tgt_block = torch.stack(tgt_embs, dim=0)   # [tgt_len, D]

            ctx  = base_embeds[b, :offset]
            rest = L - offset - tgt_len
            if rest > 0:
                pad = base_embeds[b, offset + tgt_len:]
                row = torch.cat([ctx, tgt_block, pad], dim=0)
            else:
                row = torch.cat([ctx, tgt_block[:L - offset]], dim=0)

            rows.append(row)

        return torch.stack(rows, dim=0).to(dev, dtype)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, context_params: list, context_sols: list,
                 target_params: tuple, max_solutions: int = 6) -> list:
        self.eval()
        dev, dtype = self.component_device, torch.bfloat16
        from model.canonicalize import canonicalize_solutions

        prefix = []
        for p_i, sols_i in zip(context_params, context_sols):
            sols_i = canonicalize_solutions(sols_i.to(dev, dtype))
            K = sols_i.shape[0]
            z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
            p_t    = torch.tensor([list(p_i)], dtype=dtype, device=dev)   # [1, 2]
            prefix.append(self._special_emb(P_START_ID))
            prefix.append(self.input_proj(z_zero.unsqueeze(0), p_t).squeeze(0))
            for k in range(K):
                prefix.append(self._special_emb(SOL_ID))
                z_k = self.unet(sols_i[k:k+1], "encode").squeeze(0)
                prefix.append(self.input_proj(z_k.unsqueeze(0), p_t).squeeze(0))
            prefix.append(self._special_emb(STOP_ID))

        prefix.append(self._special_emb(P_START_ID))
        z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
        tp_t   = torch.tensor([list(target_params)], dtype=dtype, device=dev)
        prefix.append(self.input_proj(z_zero.unsqueeze(0), tp_t).squeeze(0))

        seq = torch.stack(prefix, dim=0).unsqueeze(0).to(dev, dtype)

        transformer = self._get_transformer()
        past_kv  = None
        solutions = []

        def _cache_len(c):
            if c is None: return 0
            if hasattr(c, "get_seq_length"):
                try: return int(c.get_seq_length())
                except TypeError: return int(c.get_seq_length(0))
            return int(c[0][0].shape[2])

        for _ in range(max_solutions):
            sol_tok = self._special_emb(SOL_ID).unsqueeze(0).unsqueeze(0).to(dev, dtype)

            if past_kv is None:
                full = torch.cat([seq, sol_tok], dim=1)
                attn = torch.ones(1, full.shape[1], dtype=torch.long, device=dev)
                out  = transformer(inputs_embeds=full, attention_mask=attn,
                                   use_cache=True, return_dict=True)
            else:
                attn = torch.ones(1, _cache_len(past_kv) + 1,
                                  dtype=torch.long, device=dev)
                out  = transformer(inputs_embeds=sol_tok, attention_mask=attn,
                                   past_key_values=past_kv, use_cache=True, return_dict=True)

            past_kv = out.past_key_values
            last_h  = out.last_hidden_state[:, -1, :]
            tid, reg_lat = self.dual_head.route(last_h)

            if tid.item() == STOP_ID:
                break
            if tid.item() == VECTOR_ID:
                sol_pred = self.unet(reg_lat.to(dtype), "decode")
                solutions.append(sol_pred.squeeze(0).cpu())

                z_pred   = self.unet(sol_pred, "encode").squeeze(0)
                next_emb = self.input_proj(z_pred.unsqueeze(0), tp_t).squeeze(0)
                next_tok = next_emb.unsqueeze(0).unsqueeze(0).to(dev, dtype)
                attn2    = torch.ones(1, _cache_len(past_kv) + 1,
                                      dtype=torch.long, device=dev)
                out2 = transformer(inputs_embeds=next_tok, attention_mask=attn2,
                                   past_key_values=past_kv, use_cache=True, return_dict=True)
                past_kv = out2.past_key_values

        return solutions

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _special_emb(self, token_id: int) -> torch.Tensor:
        tid = torch.tensor([token_id], dtype=torch.long, device=self.component_device)
        return self.special_tok(tid).squeeze(0)

    def _input_proj(self, z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """z: [lat], params: [2] → [D]"""
        dev, dtype = self.component_device, torch.bfloat16
        z2d = z.unsqueeze(0).to(dev, dtype)
        p2d = params.unsqueeze(0).to(dev, dtype) if params.dim() == 1 else params.to(dev, dtype)
        return self.input_proj(z2d, p2d).squeeze(0)

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
        """Resume from a same-architecture checkpoint (Phase 2 → Phase 3)."""
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = config

        model.unet = UNet1d(layers=config.unet_channels,
                            latent_dim=config.latent_dim,
                            solution_dim=config.solution_dim)
        unet_p = os.path.join(checkpoint_dir, "unet.pt")
        if os.path.exists(unet_p):
            model.unet.load_state_dict(
                torch.load(unet_p, map_location="cpu", weights_only=True))
        model.unet.eval()
        for p in model.unet.parameters():
            p.requires_grad = False

        model.special_tok = SpecialTokenEmbeddings(config.qwen_hidden_dim)
        model.input_proj  = InputProjector(config.latent_dim, config.qwen_hidden_dim,
                                           param_dim=config.param_dim)
        model.output_proj = OutputProjector(config.qwen_hidden_dim, config.latent_dim)
        model.dual_head   = DualHead(config.qwen_hidden_dim, config.latent_dim,
                                     config.vocab_size)

        def _load(m, fname):
            p = os.path.join(checkpoint_dir, fname)
            if os.path.exists(p):
                m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))

        _load(model.special_tok, "special_tokens.pt")
        _load(model.input_proj,  "input_projector.pt")
        _load(model.output_proj, "output_projector.pt")
        _load(model.dual_head,   "dual_head.pt")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{local_rank}"}, trust_remote_code=True,
        )
        model.qwen_model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=True)
        model.local_rank       = local_rank
        model.component_device = torch.device(f"cuda:{local_rank}")
        dev, dtype = model.component_device, torch.bfloat16
        for m in [model.special_tok, model.input_proj, model.output_proj,
                  model.dual_head, model.unet]:
            m.to(device=dev, dtype=dtype)
        return model

    @classmethod
    def from_pretrained_1dp(cls, checkpoint_1dp: str, config):
        """
        Warm-start from 1D_p Phase-2 checkpoint.

        Loaded (same shape): UNet, LoRA, special_tokens, output_proj, dual_head.
        Re-initialized (different shape): input_proj (latent+1 → latent+2).
        """
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = config

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model.local_rank       = local_rank
        model.component_device = torch.device(f"cuda:{local_rank}")
        dev, dtype = model.component_device, torch.bfloat16

        # UNet — load from OUR Phase 1 checkpoint, not from 1D_p
        model.unet = UNet1d(layers=config.unet_channels,
                            latent_dim=config.latent_dim,
                            solution_dim=config.solution_dim)
        unet_p = config.unet_checkpoint_path
        if os.path.exists(unet_p):
            model.unet.load_state_dict(
                torch.load(unet_p, map_location="cpu", weights_only=True))
            print(f"[rank{local_rank}] Loaded UNet from Phase 1 ckpt: {unet_p}")
        else:
            print(f"[rank{local_rank}] WARNING: Phase 1 UNet ckpt not found at {unet_p}")
        model.unet.eval()
        for p in model.unet.parameters():
            p.requires_grad = False

        # Small modules
        model.special_tok = SpecialTokenEmbeddings(config.qwen_hidden_dim)
        model.input_proj  = InputProjector(config.latent_dim, config.qwen_hidden_dim,
                                           param_dim=config.param_dim)  # fresh init
        model.output_proj = OutputProjector(config.qwen_hidden_dim, config.latent_dim)
        model.dual_head   = DualHead(config.qwen_hidden_dim, config.latent_dim,
                                     config.vocab_size)

        def _load(m, fname):
            p = os.path.join(checkpoint_1dp, fname)
            if os.path.exists(p):
                m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
                print(f"[rank{local_rank}] Loaded {fname} from 1D_p ckpt")

        _load(model.special_tok, "special_tokens.pt")
        # input_proj: NOT loaded (different param_dim)
        _load(model.output_proj, "output_projector.pt")
        _load(model.dual_head,   "dual_head.pt")
        print(f"[rank{local_rank}] input_proj re-initialized (param_dim: 1→2)")

        # Qwen + LoRA from 1D_p
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{local_rank}"}, trust_remote_code=True,
        )
        model.qwen_model = PeftModel.from_pretrained(base, checkpoint_1dp, is_trainable=True)
        print(f"[rank{local_rank}] Loaded LoRA from 1D_p ckpt")

        for m in [model.special_tok, model.input_proj, model.output_proj,
                  model.dual_head, model.unet]:
            m.to(device=dev, dtype=dtype)
        return model
