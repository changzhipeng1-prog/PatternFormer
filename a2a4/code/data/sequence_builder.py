"""
Sequence Builder for 1D_a2a4 V2 training.

Block structure per sample (params = (a4, a2)):
    [P_START] [params_proj]  [SOL] [sol_proj_1]  [SOL] [sol_proj_2] ... [STOP]

params_proj = InputProjector(z=zeros, params=(a4, a2))
sol_proj_k  = InputProjector(z=UNet.encode(sol_k), params=(a4, a2))
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import P_START_ID, SOL_ID, STOP_ID, VECTOR_ID, PAD_ID
from model.canonicalize import canonicalize_solutions

_IGNORE = -100


class SequenceBuilder:
    def __init__(self, unet, input_proj, special_tok,
                 latent_dim: int, hidden_dim: int, param_dim: int = 2,
                 max_seq_len: int = 256,
                 device: torch.device = torch.device("cpu")):
        self.unet        = unet
        self.input_proj  = input_proj
        self.special_tok = special_tok
        self.latent_dim  = latent_dim
        self.hidden_dim  = hidden_dim
        self.param_dim   = param_dim
        self.max_seq_len = max_seq_len
        self.device      = device

    @torch.no_grad()
    def _encode_solution(self, sol: torch.Tensor) -> torch.Tensor:
        sol2d = sol.to(device=self.device, dtype=torch.bfloat16).unsqueeze(0)
        return self.unet(sol2d, "encode").squeeze(0)

    def _proj(self, z: torch.Tensor, params: tuple) -> torch.Tensor:
        """params: (a4, a2) floats → [hidden_dim]"""
        dev   = self.device
        dtype = torch.bfloat16
        z2d   = z.unsqueeze(0).to(dev, dtype)                           # [1, lat]
        p_t   = torch.tensor([list(params)], dtype=dtype, device=dev)   # [1, 2]
        return self.input_proj(z2d, p_t).squeeze(0)                     # [hidden]

    def _special(self, token_id: int) -> torch.Tensor:
        tid = torch.tensor([token_id], dtype=torch.long, device=self.device)
        return self.special_tok(tid).squeeze(0)

    def _zero_latent(self):
        return torch.zeros(self.latent_dim, dtype=torch.bfloat16, device=self.device)

    def _build_block(self, params: tuple, solutions: torch.Tensor, is_target: bool):
        K = solutions.shape[0]
        embeds, loss_mask, cls_targets, reg_targets, params_targets = [], [], [], [], []

        def _app(emb, lm, ct, rt, pt):
            embeds.append(emb)
            loss_mask.append(lm)
            cls_targets.append(ct)
            reg_targets.append(rt)
            params_targets.append(pt)

        # [P_START]
        _app(self._special(P_START_ID), False, _IGNORE, None, (0.0, 0.0))

        # [params_proj]
        p_embed = self._proj(self._zero_latent(), params)
        next_after_p = SOL_ID if K > 0 else STOP_ID
        _app(p_embed, False,
             next_after_p if is_target else _IGNORE,
             None, (0.0, 0.0))

        for k in range(K):
            sol_k     = solutions[k]
            z_k       = self._encode_solution(sol_k)
            sol_embed = self._proj(z_k, params)

            _app(self._special(SOL_ID), is_target,
                 VECTOR_ID if is_target else _IGNORE,
                 z_k if is_target else None,
                 params if is_target else (0.0, 0.0))

            next_after_sol = SOL_ID if k < K - 1 else STOP_ID
            _app(sol_embed, False,
                 next_after_sol if is_target else _IGNORE,
                 None, (0.0, 0.0))

        # [STOP]
        _app(self._special(STOP_ID), is_target,
             STOP_ID if is_target else _IGNORE,
             None, (0.0, 0.0))

        return embeds, loss_mask, cls_targets, reg_targets, params_targets

    def build(self,
              context_params: list,    # list of (a4, a2) tuples
              context_sols:   list,    # list of Tensor [K_i, 1024]
              target_params:  tuple,   # (a4, a2)
              target_sols:    torch.Tensor,  # [K_t, 1024]
              ) -> dict:
        all_emb = []; all_lm = []; all_ct = []; all_rt = []; all_pt = []

        target_sols = canonicalize_solutions(
            target_sols.to(device=self.device, dtype=torch.bfloat16)
        )

        for params_i, sols_i in zip(context_params, context_sols):
            sols_i = canonicalize_solutions(
                sols_i.to(device=self.device, dtype=torch.bfloat16)
            )
            e, lm, ct, rt, pt = self._build_block(params_i, sols_i, is_target=False)
            all_emb.extend(e); all_lm.extend(lm); all_ct.extend(ct)
            all_rt.extend(rt); all_pt.extend(pt)

        target_block_offset = len(all_emb)

        e, lm, ct, rt, pt = self._build_block(target_params, target_sols, is_target=True)
        all_emb.extend(e); all_lm.extend(lm); all_ct.extend(ct)
        all_rt.extend(rt); all_pt.extend(pt)

        actual_len = len(all_emb)

        # Truncate (preserve target block)
        if actual_len > self.max_seq_len:
            excess = actual_len - self.max_seq_len
            all_emb = all_emb[excess:]; all_lm  = all_lm[excess:]
            all_ct  = all_ct[excess:];  all_rt  = all_rt[excess:]
            all_pt  = all_pt[excess:]
            actual_len          = self.max_seq_len
            target_block_offset = max(0, target_block_offset - excess)

        inputs_embeds = torch.stack(all_emb, dim=0).cpu()   # [L, D]
        loss_mask     = torch.tensor(all_lm, dtype=torch.bool)
        cls_targets   = torch.tensor(all_ct, dtype=torch.long)

        reg_list = []
        for rt in all_rt:
            reg_list.append(rt.float().cpu() if rt is not None
                            else torch.zeros(self.latent_dim))
        reg_targets = torch.stack(reg_list, dim=0)           # [L, lat]

        # params_targets: [L, 2]
        pt_list = [torch.tensor(list(p), dtype=torch.float32) for p in all_pt]
        params_targets = torch.stack(pt_list, dim=0)         # [L, 2]

        return {
            "inputs_embeds":         inputs_embeds,
            "loss_mask":             loss_mask,
            "cls_targets":           cls_targets,
            "reg_targets":           reg_targets,
            "params_targets":        params_targets,
            "actual_len":            actual_len,
            "target_block_offset":   target_block_offset,
            "canonical_target_sols": target_sols.float().cpu(),
            "target_params":         target_params,
        }
