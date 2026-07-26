"""GSPDEModel — conv-AE + Qwen+LoRA multi-solution generator for Gray-Scott 2D.

Adapted from 2D_multisolution/qwen/model/v3_model.py:
  - SolutionAutoencoder2D is now a CONV autoencoder over 2x128x128 images
  - parameter is 2D (rho,mu), normalized by (p-mean)/std before InputProjector
  - FEM PDE-loss replaced by an optional hinged FDM residual term, active when
    op_path is set (lambda_pde=0.02 by default; see model/pde_residual.py)

Data flow: frozen conv AE -> special tokens + InputProjector -> Qwen+LoRA ->
DualHead (cls + reg) -> decode. Teacher-forced training.
NOTE on generation: generate() below honors the STOP head, but it under-generates
(spurious early STOP -> count collapse). The PRODUCTION path is fixed-K generation
that ignores STOP (see diversity/*.py det_pool and compare_init.qwen_pool).
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import P_START_ID, SOL_ID, STOP_ID, VECTOR_ID, PAD_ID
from model.autoencoder2d import SolutionAutoencoder2D
from model.special_tokens import SpecialTokenEmbeddings
from model.input_projector import InputProjector
from model.dual_head import DualHead
from data.dataset import build_2d_causal_mask


def get_ss_prob(epoch, start, end, target):
    """Linear warmup of the scheduled-sampling probability over [start,end]."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return target
    return target * (epoch - start) / max(1, end - start)


class GSPDEModel(nn.Module):
    def __init__(self, config, unet_checkpoint: str = None, local_rank: int = None,
                 p_mean: torch.Tensor = None, p_std: torch.Tensor = None):
        super().__init__()
        self.config = config
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.component_device = torch.device(f"cuda:{local_rank}")

        # ---- 1. conv autoencoder (frozen) ----
        self.autoencoder = SolutionAutoencoder2D(latent_dim=config.latent_dim, in_ch=config.in_ch,
                                                 img=config.img_size)
        if unet_checkpoint and os.path.exists(unet_checkpoint):
            print(f"[rank{local_rank}] loading AE from {unet_checkpoint}")
            self.autoencoder.load_state_dict(
                torch.load(unet_checkpoint, map_location="cpu", weights_only=True))
        elif unet_checkpoint:
            print(f"[rank{local_rank}] WARNING: AE checkpoint missing at {unet_checkpoint}")
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False

        # ---- param normalization buffers ----
        pd = config.param_dim
        self.register_buffer("p_mean", (p_mean if p_mean is not None else torch.zeros(pd)).clone())
        self.register_buffer("p_std", (p_std if p_std is not None else torch.ones(pd)).clone())

        # ---- (2) FDM physics-residual module ----
        self.pde_res = None
        if getattr(config, "lambda_pde", 0.0) > 0 and os.path.exists(config.op_path):
            from model.pde_residual import GSResidual
            self.pde_res = GSResidual(config.op_path, config.DA, config.DS,
                                      device=self.component_device,
                                      floor=getattr(config, "pde_floor", 1e-3))
            print(f"[rank{local_rank}] FDM physics residual loss enabled (lambda_pde={config.lambda_pde})")

        # ---- 2-5 small trainable modules ----
        self.special_tok = SpecialTokenEmbeddings(config.qwen_hidden_dim)
        self.input_proj = InputProjector(config.latent_dim, config.qwen_hidden_dim, param_dim=pd)
        # B2: OutputProjector removed (was dead code -- instantiated & trained but never
        # called in forward/generate; regression output comes from dual_head.reg_head).
        self.dual_head = DualHead(config.qwen_hidden_dim, config.latent_dim, config.vocab_size)

        # ---- 6. Qwen + LoRA ----
        if bool(getattr(config, "random_init_backbone", False)):
            from transformers import AutoConfig
            qcfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            torch.manual_seed(int(getattr(config, "backbone_init_seed", 1234)))
            print(f"[rank{local_rank}] ABLATION random-init backbone (frozen) {config.model_name} -> cuda:{local_rank}")
            self.qwen_model = AutoModelForCausalLM.from_config(
                qcfg, torch_dtype=torch.bfloat16, trust_remote_code=True).to(f"cuda:{local_rank}")
        else:
            print(f"[rank{local_rank}] loading Qwen {config.model_name} -> cuda:{local_rank}")
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                config.model_name, torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{local_rank}"}, trust_remote_code=True)
        self._setup_lora()

        for m in [self.special_tok, self.input_proj, self.dual_head,
                  self.autoencoder]:
            m.to(device=self.component_device, dtype=torch.bfloat16)
        self.p_mean = self.p_mean.to(self.component_device)
        self.p_std = self.p_std.to(self.component_device)

    def _setup_lora(self):
        c = self.config
        lora = LoraConfig(r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout,
                          target_modules=c.lora_target_modules, bias="none", task_type="CAUSAL_LM")
        self.qwen_model = get_peft_model(self.qwen_model, lora)
        self.qwen_model.print_trainable_parameters()

    def _get_transformer(self):
        if hasattr(self.qwen_model, "base_model"):
            return self.qwen_model.base_model.model.model
        return self.qwen_model.model

    # ------------------------------------------------------------------
    def forward(self, batch: dict, current_epoch: int = 1) -> dict:
        dev, dtype = self.component_device, torch.bfloat16
        inputs_embeds = batch["inputs_embeds"].to(dev, dtype)
        loss_mask = batch["loss_mask"].to(dev)              # head SOL (k<K_t): ordered MSE
        tail_mask = batch["tail_mask"].to(dev)              # tail SOL (k>=K_t): physics-only
        cls_targets = batch["cls_targets"].to(dev)          # VECTOR_ID marks head SOL (for SS)
        reg_targets = batch["reg_targets"].to(dev, torch.float32)
        actual_lens = batch["actual_lens"].to(dev)
        p_targets = batch["p_targets"].to(dev, torch.float32)   # [B,L,2] (rho,mu) at SOL pos

        inputs_embeds = self._rebuild_target_block_embeds(
            base_embeds=inputs_embeds,
            target_block_offsets=batch["target_block_offsets"].to(dev),
            target_sols_padded=batch["target_sols_padded"].to(dev, dtype),
            target_k_counts=batch["target_k_counts"].to(dev),
            target_p_vals=batch["target_p_vals"].to(dev, dtype),
        ).to(dev, dtype)
        B, L, D = inputs_embeds.shape

        bool_mask = build_2d_causal_mask(actual_lens, L, dev)
        float_mask = torch.zeros(B, 1, L, L, dtype=dtype, device=dev)
        float_mask.masked_fill_(~bool_mask, float("-inf"))

        out = self._get_transformer()(inputs_embeds=inputs_embeds,
                                      attention_mask=float_mask, return_dict=True)
        _, reg_latent = self.dual_head(out.last_hidden_state)   # no cls head (STOP removed)

        # ---- self-prediction substitution (pass 2): tail ALWAYS + head SS (prob ss_prob) ----
        # Tail slots (k>=K_t) have no GT, so their sol_proj input is filled with the model's
        # OWN pass-1 prediction (free-run); the head additionally uses scheduled sampling to
        # fight exposure bias. Both overwrite the sol_proj input at pi+1 with
        # input_proj(detached pass-1 prediction at SOL position pi), then re-run once.
        ss_prob = 0.0
        if self.training and getattr(self.config, "ss_prob", 0.0) > 0.0:
            ss_prob = get_ss_prob(current_epoch, self.config.ss_warmup_start,
                                  self.config.ss_warmup_end, self.config.ss_prob)
        sel = tail_mask.clone()
        if ss_prob > 0.0:
            head_vec = (cls_targets == VECTOR_ID)
            sel = sel | (head_vec & (torch.rand(B, L, device=dev) < ss_prob))
        if L > 1:
            sel[:, L - 1] = False                              # no sol_proj after last pos
        if sel.any():
            bi, pi = sel.nonzero(as_tuple=True)
            z_pred = reg_latent[bi, pi].detach().to(dtype)            # [M,latent]
            p_raw = p_targets[bi, pi]                                  # [M,2] real p
            p_norm = ((p_raw - self.p_mean.float()) / self.p_std.float()).to(dtype)
            new_emb = self.input_proj(z_pred, p_norm).to(dtype)       # [M,hidden]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[bi, pi + 1] = new_emb                       # overwrite sol_proj
            out = self._get_transformer()(inputs_embeds=inputs_embeds,
                                          attention_mask=float_mask, return_dict=True)
            _, reg_latent = self.dual_head(out.last_hidden_state)

        from model.pde_residual import get_pde_lambda
        std = self.autoencoder.std.float()

        # ---- HEAD (k<K_t): ordered latent + decoded-field MSE + small hinge physics ----
        mse_loss = torch.tensor(0.0, device=dev)
        latent_loss = torch.tensor(0.0, device=dev)
        pde_loss = torch.tensor(0.0, device=dev)
        lam_pde = 0.0
        if loss_mask.any():
            tgt_sel = reg_targets[loss_mask]
            reg_sel = reg_latent[loss_mask].float()
            latent_loss = F.mse_loss(reg_sel, tgt_sel)
            pred = self.autoencoder(reg_sel.to(dtype), "decode").float()
            gt = self.autoencoder(tgt_sel.to(dtype), "decode").float()
            mse_loss = (((pred - gt) / std) ** 2).mean()
            if self.pde_res is not None:
                lam_pde = get_pde_lambda(current_epoch, self.config.pde_warmup_start,
                                         self.config.pde_warmup_end, self.config.lambda_pde)
                if lam_pde > 0:
                    p_sel = p_targets[loss_mask]
                    pde_loss, _ = self.pde_res(pred, p_sel[:, 0], p_sel[:, 1])

        # ---- TAIL (k>=K_t): physics-only -> best-effort discovery of solutions beyond GT ----
        # (+ optional non-triviality hinge for the extrapolation finetune: keeps physics-only
        #  slots from collapsing to the trivial flat state, which the hinged residual permits.)
        tail_pde = torch.tensor(0.0, device=dev)
        nontrivial_loss = torch.tensor(0.0, device=dev)
        lam_pde_tail = 0.0
        lam_nt = float(getattr(self.config, "lambda_nontrivial", 0.0))
        if self.pde_res is not None and tail_mask.any() and (
                getattr(self.config, "lambda_pde_tail", 0.0) > 0 or lam_nt > 0):
            lam_pde_tail = get_pde_lambda(current_epoch, self.config.pde_warmup_start,
                                          self.config.pde_warmup_end,
                                          getattr(self.config, "lambda_pde_tail", 0.0))
            reg_tail = reg_latent[tail_mask].float()
            pred_tail = self.autoencoder(reg_tail.to(dtype), "decode").float()
            p_tl = p_targets[tail_mask]
            if lam_pde_tail > 0:
                tf = float(getattr(self.config, "pde_tail_floor", -1.0))
                tail_pde, _ = self.pde_res(pred_tail, p_tl[:, 0], p_tl[:, 1],
                                           floor=(tf if tf >= 0 else None))
            if lam_nt > 0:
                A_tail = pred_tail[:, 0]                                   # [M,128,128]
                rng = A_tail.flatten(1).amax(1) - A_tail.flatten(1).amin(1)  # spatial range per slot
                floor = float(getattr(self.config, "nontrivial_range_floor", 0.30))
                nontrivial_loss = torch.relu(floor - rng).mean()          # penalize too-flat slots

        total = (self.config.lambda_mse * mse_loss
                 + self.config.lambda_latent * latent_loss
                 + lam_pde * pde_loss
                 + lam_pde_tail * tail_pde
                 + lam_nt * nontrivial_loss)

        return {"total_loss": total, "mse_loss": mse_loss, "latent_loss": latent_loss,
                "pde_loss": pde_loss, "tail_pde": tail_pde, "lambda_pde": lam_pde,
                "lambda_pde_tail": lam_pde_tail, "ss_prob": ss_prob,
                "nontrivial_loss": nontrivial_loss, "lambda_nontrivial": lam_nt,
                "ce_loss": torch.tensor(0.0, device=dev)}

    # ------------------------------------------------------------------
    def _rebuild_target_block_embeds(self, base_embeds, target_block_offsets,
                                     target_sols_padded, target_k_counts, target_p_vals):
        dev, dtype = self.component_device, torch.bfloat16
        B, L, D = base_embeds.shape
        rows = []
        for b in range(B):
            offset = int(target_block_offsets[b].item())
            K_t = int(target_k_counts[b].item())
            p_vec = target_p_vals[b]                     # [2]
            z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)

            # FIXED-K block (no STOP): head k<K_t = GT teacher-forced; tail k>=K_t = zero
            # placeholder (forward's pass-2 overwrites it with the model's self-prediction).
            K = self.config.fixed_k
            embs = [self._special_emb(P_START_ID), self._input_proj(z_zero, p_vec)]
            for k in range(K):
                embs.append(self._special_emb(SOL_ID))
                if k < K_t:
                    z_k = self.autoencoder(target_sols_padded[b, k].unsqueeze(0), "encode").squeeze(0)
                    embs.append(self._input_proj(z_k, p_vec))
                else:
                    embs.append(self._input_proj(z_zero, p_vec))   # tail placeholder

            tlen = len(embs)
            block = torch.stack(embs, 0)
            ctx = base_embeds[b, :offset]
            remaining = L - offset - tlen
            if remaining > 0:
                row = torch.cat([ctx, block, base_embeds[b, offset + tlen:]], 0)
            else:
                row = torch.cat([ctx, block[:L - offset]], 0)
            rows.append(row)
        return torch.stack(rows, 0).to(dev, dtype)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, context_p_vals: list, context_sols: list, target_p_val,
                 max_solutions: int = 16) -> list:
        self.eval()
        dev, dtype = self.component_device, torch.bfloat16
        from model.canonicalize import canonicalize_solutions

        prefix = []
        for p_v, sols in zip(context_p_vals, context_sols):
            sols = canonicalize_solutions(sols.to(dev, dtype))
            prefix.append(self._special_emb(P_START_ID))
            z0 = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
            prefix.append(self._input_proj(z0, p_v))
            for k in range(sols.shape[0]):
                prefix.append(self._special_emb(SOL_ID))
                z_k = self.autoencoder(sols[k:k+1], "encode").squeeze(0)
                prefix.append(self._input_proj(z_k, p_v))
            prefix.append(self._special_emb(STOP_ID))
        prefix.append(self._special_emb(P_START_ID))
        z0 = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
        prefix.append(self._input_proj(z0, target_p_val))
        seq = torch.stack(prefix, 0).unsqueeze(0).to(dev, dtype)

        transformer = self._get_transformer()
        past_kv = None
        solutions = []

        def _clen(c):
            if c is None: return 0
            if hasattr(c, "get_seq_length"):
                try: return int(c.get_seq_length())
                except TypeError: return int(c.get_seq_length(0))
            return int(c[0][0].shape[2])

        for step in range(max_solutions):
            sol_tok = self._special_emb(SOL_ID).unsqueeze(0).unsqueeze(0).to(dev, dtype)
            if past_kv is None:
                full = torch.cat([seq, sol_tok], 1)
                out = transformer(inputs_embeds=full,
                                  attention_mask=torch.ones(1, full.shape[1], dtype=torch.long, device=dev),
                                  use_cache=True, return_dict=True)
            else:
                out = transformer(inputs_embeds=sol_tok,
                                  attention_mask=torch.ones(1, _clen(past_kv) + 1, dtype=torch.long, device=dev),
                                  past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            h_last = out.last_hidden_state[:, -1, :]
            tid, reg_lat = self.dual_head.route(h_last)
            if tid.item() == STOP_ID:
                break
            if tid.item() == VECTOR_ID:
                sol_pred = self.autoencoder(reg_lat.to(dtype), "decode")
                z_pred = self.autoencoder(sol_pred, "encode").squeeze(0)
                solutions.append(sol_pred.squeeze(0).cpu())
                next_tok = self._input_proj(z_pred, target_p_val).unsqueeze(0).unsqueeze(0).to(dev, dtype)
                out2 = transformer(inputs_embeds=next_tok,
                                   attention_mask=torch.ones(1, _clen(past_kv) + 1, dtype=torch.long, device=dev),
                                   past_key_values=past_kv, use_cache=True, return_dict=True)
                past_kv = out2.past_key_values
                tid2, _ = self.dual_head.route(out2.last_hidden_state[:, -1, :])
                if tid2.item() == STOP_ID:
                    break
        return solutions

    @torch.no_grad()
    def generate_fixed_k(self, p_value, K, noise_std: float = 0.0, dtype=torch.bfloat16):
        """Fixed-K generation: force K solutions, IGNORING the STOP head.

        This is the PRODUCTION path. generate() above honors STOP and under-
        generates (spurious early STOP -> count collapse). This consolidates the
        det_pool() loop that was duplicated across diversity/*.py. No context.
        Returns np.ndarray [K, 2, img, img] in physical units.
        """
        def _clen(c):
            if c is None:
                return 0
            if hasattr(c, "get_seq_length"):
                try:
                    return int(c.get_seq_length())
                except TypeError:
                    return int(c.get_seq_length(0))
            return int(c[0][0].shape[2])

        dev = self.component_device
        T = self._get_transformer()
        tp = torch.as_tensor(p_value, dtype=torch.float32)
        seq = torch.stack([self._special_emb(P_START_ID),
                           self._input_proj(torch.zeros(self.config.latent_dim), tp)], 0).unsqueeze(0).to(dev, dtype)
        sol_emb = self._special_emb(SOL_ID).unsqueeze(0).unsqueeze(0).to(dev, dtype)
        past = None
        fields = []
        for _ in range(K):
            if past is None:
                o = T(inputs_embeds=torch.cat([seq, sol_emb], 1),
                      attention_mask=torch.ones(1, seq.shape[1] + 1, dtype=torch.long, device=dev),
                      use_cache=True, return_dict=True)
            else:
                o = T(inputs_embeds=sol_emb,
                      attention_mask=torch.ones(1, _clen(past) + 1, dtype=torch.long, device=dev),
                      past_key_values=past, use_cache=True, return_dict=True)
            past = o.past_key_values
            _, reg = self.dual_head(o.last_hidden_state[:, -1, :])
            if noise_std > 0:
                reg = reg + noise_std * torch.randn_like(reg)   # latent noise for extra diversity
            sol = self.autoencoder(reg.to(dtype), "decode")
            fields.append(sol.float().squeeze(0).cpu().numpy())
            z = self.autoencoder(sol, "encode").squeeze(0).float()
            nxt = self._input_proj(z, tp).unsqueeze(0).unsqueeze(0).to(dev, dtype)
            o2 = T(inputs_embeds=nxt,
                   attention_mask=torch.ones(1, _clen(past) + 1, dtype=torch.long, device=dev),
                   past_key_values=past, use_cache=True, return_dict=True)
            past = o2.past_key_values
        return np.stack(fields)

    # ------------------------------------------------------------------
    def _special_emb(self, token_id: int) -> torch.Tensor:
        tid = torch.tensor([token_id], dtype=torch.long, device=self.component_device)
        return self.special_tok(tid).squeeze(0)

    def _input_proj(self, z: torch.Tensor, p_vec) -> torch.Tensor:
        dev, dtype = self.component_device, torch.bfloat16
        z2d = z.unsqueeze(0).to(dev, dtype)
        p_vec = torch.as_tensor(p_vec, dtype=dtype, device=dev).reshape(-1)
        p_norm = ((p_vec - self.p_mean.to(dtype)) / self.p_std.to(dtype)).unsqueeze(0)  # [1,2]
        return self.input_proj(z2d, p_norm).squeeze(0)

    # ------------------------------------------------------------------
    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self.qwen_model, "save_pretrained"):
            self.qwen_model.save_pretrained(save_dir)
        mods = [("special_tokens", self.special_tok), ("input_projector", self.input_proj),
                ("dual_head", self.dual_head),
                ("autoencoder2d", self.autoencoder)]
        for name, mod in mods:
            torch.save(mod.state_dict(), os.path.join(save_dir, f"{name}.pt"))
        print(f"GSPDEModel saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, config, local_rank: int = None,
                        p_mean: torch.Tensor = None, p_std: torch.Tensor = None):
        """Resume a saved GSPDEModel (used for Phase 3: no-context finetune)."""
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.config = config
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        m.local_rank = local_rank
        m.component_device = torch.device(f"cuda:{local_rank}")
        pd = config.param_dim
        m.register_buffer("p_mean", (p_mean if p_mean is not None else torch.zeros(pd)).clone())
        m.register_buffer("p_std", (p_std if p_std is not None else torch.ones(pd)).clone())

        m.autoencoder = SolutionAutoencoder2D(latent_dim=config.latent_dim, in_ch=config.in_ch,
                                              img=config.img_size)
        m.special_tok = SpecialTokenEmbeddings(config.qwen_hidden_dim)
        m.input_proj = InputProjector(config.latent_dim, config.qwen_hidden_dim, param_dim=pd)
        m.dual_head = DualHead(config.qwen_hidden_dim, config.latent_dim, config.vocab_size)

        # ---- (2) FDM physics-residual module (mirror __init__; forward references it) ----
        m.pde_res = None
        if getattr(config, "lambda_pde", 0.0) > 0 and os.path.exists(config.op_path):
            from model.pde_residual import GSResidual
            m.pde_res = GSResidual(config.op_path, config.DA, config.DS,
                                   device=m.component_device,
                                   floor=getattr(config, "pde_floor", 1e-3))
            print(f"[rank{local_rank}] FDM physics residual loss enabled (lambda_pde={config.lambda_pde})")

        def _load(mod, fn):
            p = os.path.join(checkpoint_dir, fn)
            if os.path.exists(p):
                mod.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        # B2: output_projector.pt (if present in older checkpoints) is intentionally
        # NOT loaded -- the module was dead code and has been removed.
        load_list = [("autoencoder2d.pt", m.autoencoder), ("special_tokens.pt", m.special_tok),
                     ("input_projector.pt", m.input_proj),
                     ("dual_head.pt", m.dual_head)]
        for fn, mod in load_list:
            _load(mod, fn)
        m.autoencoder.eval()
        for p in m.autoencoder.parameters():
            p.requires_grad = False

        if bool(getattr(config, "random_init_backbone", False)):
            from transformers import AutoConfig
            qcfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            torch.manual_seed(int(getattr(config, "backbone_init_seed", 1234)))
            print(f"[rank{local_rank}] ABLATION random-init backbone (frozen) for resume -> cuda:{local_rank}")
            base = AutoModelForCausalLM.from_config(
                qcfg, torch_dtype=torch.bfloat16, trust_remote_code=True).to(f"cuda:{local_rank}")
        else:
            base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16,
                                                        device_map={"": f"cuda:{local_rank}"},
                                                        trust_remote_code=True)
        m.qwen_model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=True)
        for mod in [m.special_tok, m.input_proj, m.dual_head, m.autoencoder]:
            mod.to(device=m.component_device, dtype=torch.bfloat16)
        m.p_mean = m.p_mean.to(m.component_device); m.p_std = m.p_std.to(m.component_device)
        return m
