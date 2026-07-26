"""
V3PDEModel — Full V3 Architecture (2D PDE, 145-node FEM mesh)

Differences from V2:
    - SolutionAutoencoder2D replaces UNet1d_v6
    - solution_dim: 1024 → 145
    - lambda_pde: 0.05 with linear warmup (epochs 10→25)
    - Checkpoint key: unet.pt → autoencoder2d.pt

Components (in order of data flow):
    1. SolutionAutoencoder2D  (frozen encoder + frozen decoder)
    2. SpecialTokenEmbeddings (4 learnable + 1 fixed-zero)
    3. InputProjector         ([z; p] → Qwen hidden space)
    4. OutputProjector        (Qwen hidden → latent)
    5. DualHead               (cls_head + reg_head)
    6. Qwen2.5-7B-Instruct + LoRA

Training (Teacher Forcing): same as V2.
Inference (generate):       same as V2, returns list of 145-dim solutions.
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
from model.autoencoder2d import SolutionAutoencoder2D
from model.special_tokens import SpecialTokenEmbeddings
from model.input_projector import InputProjector
from model.output_projector import OutputProjector
from model.dual_head import DualHead
from model.pde_loss_2d import build_pde_data, compute_pde_loss_2d, get_pde_lambda
from data.dataset import build_2d_causal_mask


class V3PDEModel(nn.Module):
    """
    End-to-end V3 model for the 2D PDE bifurcation problem.

    Args:
        config:               Config instance (v3/config.py)
        unet_checkpoint:      path to pre-trained SolutionAutoencoder2D weights
        local_rank:           GPU rank (set automatically in DDP)
    """

    def __init__(self, config, unet_checkpoint: str = None, local_rank: int = None,
                 mesh_coord: torch.Tensor = None, mesh_elem: torch.Tensor = None,
                 mesh_free_nodes: torch.Tensor = None):
        """
        Args:
            mesh_coord:       [145, 2]  node coordinates  (from dataset 'coord' key)
            mesh_elem:        [256, 3]  triangle indices   (from dataset 'elem' key)
            mesh_free_nodes:  [n_free]  free DOF indices   (from dataset 'free_nodes' key)
        """
        super().__init__()
        self.config = config

        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.component_device = torch.device(f"cuda:{local_rank}")

        # ---- 0. (canonicalize uses mean(u) sort; no mesh-coord init needed) ----

        # ---- 1. SolutionAutoencoder2D (frozen) ----
        self.autoencoder = SolutionAutoencoder2D(
            solution_dim=config.solution_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.autoencoder_hidden_dim,
        )
        if unet_checkpoint and os.path.exists(unet_checkpoint):
            print(f"[rank{local_rank}] Loading autoencoder weights from {unet_checkpoint}")
            state = torch.load(unet_checkpoint, map_location="cpu", weights_only=True)
            self.autoencoder.load_state_dict(state)
        elif unet_checkpoint:
            print(f"[rank{local_rank}] Warning: autoencoder checkpoint not found at {unet_checkpoint}")

        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        print(f"[rank{local_rank}] SolutionAutoencoder2D frozen (encoder + decoder)")

        # ---- 1b. Precompute 2D FEM data for PDE loss ----
        self.pde_data = None
        if mesh_coord is not None and mesh_elem is not None and mesh_free_nodes is not None:
            self.pde_data = build_pde_data(mesh_coord, mesh_elem, mesh_free_nodes)
            print(f"[rank{local_rank}] PDE2DData built: "
                  f"{mesh_coord.shape[0]} nodes, {mesh_elem.shape[0]} elements, "
                  f"{mesh_free_nodes.shape[0]} free DOFs")

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

        # ---- 6. Qwen + LoRA ----
        print(f"[rank{local_rank}] Loading Qwen: {config.model_name} → cuda:{local_rank}")
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{local_rank}"},
            trust_remote_code=True,
        )
        self._setup_lora()

        # Move all small trainable modules to this rank's GPU
        for m in [self.special_tok, self.input_proj, self.output_proj,
                  self.dual_head, self.autoencoder]:
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
        """Unwrap PEFT to reach the raw Transformer stack."""
        if hasattr(self.qwen_model, "base_model"):
            return self.qwen_model.base_model.model.model
        return self.qwen_model.model

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, batch: dict, current_epoch: int = 1) -> dict:
        """
        Teacher-Forcing forward pass with LossMask.

        Args:
            batch:         output of make_collate_fn (dataset.py)
            current_epoch: for PDE warmup scheduler (linear 0→lambda_pde over epochs warmup_start→warmup_end)

        Returns dict with scalar losses:
            total_loss, mse_loss, pde_loss, ce_loss, lambda_pde
        """
        dev   = self.component_device
        dtype = torch.bfloat16

        inputs_embeds  = batch["inputs_embeds"].to(dev, dtype)
        attention_mask = batch["attention_mask"].to(dev)
        loss_mask      = batch["loss_mask"].to(dev)
        cls_targets    = batch["cls_targets"].to(dev)
        reg_targets    = batch["reg_targets"].to(dev, torch.float32)
        p_targets      = batch["p_targets"].to(dev, torch.float32)
        actual_lens    = batch["actual_lens"].to(dev)

        # Rebuild target block embeddings with gradient tracking
        inputs_embeds = self._rebuild_target_block_embeds(
            base_embeds          = inputs_embeds,
            target_block_offsets = batch["target_block_offsets"].to(dev),
            target_sols_padded   = batch["target_sols_padded"].to(dev, dtype),
            target_k_counts      = batch["target_k_counts"].to(dev),
            target_p_vals        = batch["target_p_vals"].to(dev, dtype),
        )
        inputs_embeds = inputs_embeds.to(dev, dtype)

        B, L, D = inputs_embeds.shape

        # Build 2D causal + padding attention mask [B, 1, L, L]
        bool_mask_2d  = build_2d_causal_mask(actual_lens, L, dev)
        float_mask_2d = torch.zeros(B, 1, L, L, dtype=dtype, device=dev)
        float_mask_2d.masked_fill_(~bool_mask_2d, float("-inf"))

        # Transformer forward (teacher-forced full sequence)
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
        cls_valid = cls_targets != -100
        ce_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        if cls_valid.any():
            cls_w = torch.ones(self.config.vocab_size, device=dev, dtype=torch.float32)
            cls_w[STOP_ID]   = self.config.ce_stop_weight
            cls_w[VECTOR_ID] = self.config.ce_vector_weight
            ce_loss = F.cross_entropy(
                cls_logits[cls_valid].float(),
                cls_targets[cls_valid],
                weight=cls_w,
                reduction="mean",
            )

        # ---- MSE + PDE loss (regression head, masked to solution positions) ----
        mse_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        pde_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)

        if loss_mask.any():
            reg_latent_sel  = reg_latent[loss_mask].float()    # [N_sol, lat]
            reg_targets_sel = reg_targets[loss_mask]           # [N_sol, lat]
            p_targets_sel   = p_targets[loss_mask]             # [N_sol]

            # Decode latent → 145-dim physical solution
            pred_u = self.autoencoder(
                reg_latent_sel.to(dtype=dtype), "decode"
            ).float()                                           # [N_sol, 145]
            gt_u = self.autoencoder(
                reg_targets_sel.to(dtype=dtype), "decode"
            ).float()                                           # [N_sol, 145]

            mse_loss = F.mse_loss(pred_u, gt_u)

            # PDE loss: GT-referenced FEM residual excess of the predicted solution
            lambda_pde = get_pde_lambda(
                current_epoch,
                warmup_start   = self.config.pde_warmup_start,
                warmup_end     = self.config.pde_warmup_end,
                target_lambda  = self.config.lambda_pde,
            )
            if lambda_pde > 0 and self.pde_data is not None:
                # gt_u = D(E(u_gt)) is the AE-decoded ground truth (already computed
                # above for the MSE term).  Using its residual as the per-sample
                # reference means a correct prediction incurs ZERO physics penalty
                # (no conflict with the multi-branch MSE); only slots LESS physical
                # than the AE can represent get pulled down toward the Newton basin.
                ref_u = gt_u.detach() if getattr(self.config, "pde_use_gt_ref", True) else None
                pde_loss = compute_pde_loss_2d(
                    pred_u,
                    p_targets_sel,
                    self.pde_data,
                    ref_u=ref_u,
                )
        else:
            lambda_pde = 0.0

        total_loss = (
            self.config.lambda_mse * mse_loss
            + lambda_pde            * pde_loss
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
        target_sols_padded:   torch.Tensor,   # [B, K_max, 145]  bfloat16
        target_k_counts:      torch.Tensor,   # [B]        long
        target_p_vals:        torch.Tensor,   # [B]        bfloat16
    ) -> torch.Tensor:
        """
        Re-compute target block embeddings inside forward() so that gradients
        flow back through InputProjector and SpecialTokenEmbeddings.
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
            target_embs.append(self._special_emb(P_START_ID))
            target_embs.append(self._input_proj(z_zero, p_val))

            for k in range(K_t):
                sol_k = target_sols_padded[b, k]              # [145]
                z_k   = self.autoencoder(sol_k.unsqueeze(0), "encode").squeeze(0)  # [lat]
                target_embs.append(self._special_emb(SOL_ID))
                target_embs.append(self._input_proj(z_k, p_val))

            target_embs.append(self._special_emb(STOP_ID))

            target_len   = len(target_embs)
            target_block = torch.stack(target_embs, dim=0)   # [target_len, D]

            context_part = base_embeds[b, :offset]
            remaining    = L - offset - target_len
            if remaining > 0:
                pad_part = base_embeds[b, offset + target_len:]
                row = torch.cat([context_part, target_block, pad_part], dim=0)
            else:
                row = torch.cat([context_part, target_block[:L - offset]], dim=0)

            rebuilt_rows.append(row)

        return torch.stack(rebuilt_rows, dim=0).to(device=dev, dtype=dtype)

    # ------------------------------------------------------------------
    # Inference (generate)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_p_vals:  list,            # list[float]
        context_sols:    list,            # list[Tensor [K_i, 145]]
        target_p_val:    float,
        max_solutions:   int = 10,
    ) -> list:
        """
        Autoregressive generation for a single sample (B=1).

        Returns:
            list of Tensor [145]  — generated PDE solutions (canonicalized order)
        """
        self.eval()
        dev   = self.component_device
        dtype = torch.bfloat16

        from model.canonicalize import canonicalize_solutions

        prefix_embeds = []

        for p_v, sols in zip(context_p_vals, context_sols):
            sols = canonicalize_solutions(sols.to(dev, dtype))  # [K, 145]
            K = sols.shape[0]

            prefix_embeds.append(self._special_emb(P_START_ID))
            z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
            prefix_embeds.append(self._input_proj(z_zero, p_v))

            for k in range(K):
                prefix_embeds.append(self._special_emb(SOL_ID))
                z_k = self.autoencoder(sols[k:k+1], "encode").squeeze(0)
                prefix_embeds.append(self._input_proj(z_k, p_v))

            prefix_embeds.append(self._special_emb(STOP_ID))

        prefix_embeds.append(self._special_emb(P_START_ID))
        z_zero = torch.zeros(self.config.latent_dim, dtype=dtype, device=dev)
        prefix_embeds.append(self._input_proj(z_zero, target_p_val))

        seq = torch.stack(prefix_embeds, dim=0).unsqueeze(0).to(dev, dtype)

        transformer = self._get_transformer()
        past_kv = None
        solutions = []

        def _cached_seq_len(cache_obj) -> int:
            if cache_obj is None:
                return 0
            if hasattr(cache_obj, "get_seq_length"):
                try:
                    return int(cache_obj.get_seq_length())
                except TypeError:
                    return int(cache_obj.get_seq_length(0))
            return int(cache_obj[0][0].shape[2])

        for step in range(max_solutions):
            sol_tok = self._special_emb(SOL_ID).unsqueeze(0).unsqueeze(0).to(dev, dtype)

            if past_kv is None:
                full_seq = torch.cat([seq, sol_tok], dim=1).to(dev, dtype)
                L_full   = full_seq.shape[1]
                attn_1d  = torch.ones(1, L_full, dtype=torch.long, device=dev)
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
            last_h  = out.last_hidden_state[:, -1, :]

            token_id, reg_lat = self.dual_head.route(last_h)
            tid = token_id.item()

            if tid == STOP_ID:
                break

            if tid == VECTOR_ID:
                sol_pred = self.autoencoder(reg_lat.to(dtype), "decode")  # [1, 145]
                solutions.append(sol_pred.squeeze(0).cpu())               # [145]

                z_pred   = self.autoencoder(sol_pred, "encode").squeeze(0)
                next_emb = self._input_proj(z_pred, target_p_val)
                next_tok = next_emb.unsqueeze(0).unsqueeze(0).to(dev, dtype)

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

                # Route on sol_proj hidden: training teaches STOP here after last solution.
                last_h2 = out2.last_hidden_state[:, -1, :]
                tid2, _ = self.dual_head.route(last_h2)
                if tid2.item() == STOP_ID:
                    break

        return solutions

    # ------------------------------------------------------------------
    # Internal embedding helpers
    # ------------------------------------------------------------------

    def _special_emb(self, token_id: int) -> torch.Tensor:
        """[D] bfloat16"""
        tid = torch.tensor([token_id], dtype=torch.long, device=self.component_device)
        return self.special_tok(tid).squeeze(0)

    def _input_proj(self, z: torch.Tensor, p_val: float) -> torch.Tensor:
        """z: [lat], returns [D]. p_val is divided by p_input_scale before projection."""
        dev   = self.component_device
        dtype = torch.bfloat16
        z2d   = z.unsqueeze(0).to(dev, dtype)
        p_normalized = p_val / self.config.p_input_scale
        p_t   = torch.tensor([[p_normalized]], dtype=dtype, device=dev)
        return self.input_proj(z2d, p_t).squeeze(0)

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
        torch.save(self.autoencoder.state_dict(),
                   os.path.join(save_dir, "autoencoder2d.pt"))
        print(f"V3PDEModel saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, config,
                        mesh_coord: torch.Tensor = None,
                        mesh_elem: torch.Tensor = None,
                        mesh_free_nodes: torch.Tensor = None):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = config

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model.local_rank       = local_rank
        model.component_device = torch.device(f"cuda:{local_rank}")

        # ---- 0. PDE data ----
        model.pde_data = None
        if mesh_coord is not None and mesh_elem is not None and mesh_free_nodes is not None:
            model.pde_data = build_pde_data(mesh_coord, mesh_elem, mesh_free_nodes)
            print(f"[rank{local_rank}] PDE2DData built: "
                  f"{mesh_coord.shape[0]} nodes, {mesh_elem.shape[0]} elements, "
                  f"{mesh_free_nodes.shape[0]} free DOFs")

        # ---- 1. Autoencoder ----
        model.autoencoder = SolutionAutoencoder2D(
            solution_dim=config.solution_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.autoencoder_hidden_dim,
        )
        ae_path = os.path.join(checkpoint_dir, "autoencoder2d.pt")
        if os.path.exists(ae_path):
            model.autoencoder.load_state_dict(
                torch.load(ae_path, map_location="cpu", weights_only=True)
            )
        model.autoencoder.eval()
        for p in model.autoencoder.parameters():
            p.requires_grad = False

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

        base = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{local_rank}"},
            trust_remote_code=True,
        )
        model.qwen_model = PeftModel.from_pretrained(
            base, checkpoint_dir, is_trainable=True
        )

        for m in [model.special_tok, model.input_proj, model.output_proj,
                  model.dual_head, model.autoencoder]:
            m.to(device=model.component_device, dtype=torch.bfloat16)

        return model
