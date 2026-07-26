"""Sequence Builder — Gray-Scott 2D (image solutions, 2D parameter).

Block for parameter p=(rho,mu) with K canonicalized solutions:
    [P_START] [p_proj] [SOL] [sol_proj_1] ... [SOL] [sol_proj_K] [STOP]
    p_proj   = InputProjector(z=zeros, p)
    sol_proj = InputProjector(z=AE.encode(sol_k), p)

AR: hidden_state[i] predicts position i+1. Regression target z_k is placed at
the [SOL] position preceding sol_proj_k. Same scheme as the reference; only the
solution shape ([2,128,128]) and parameter dim (2) differ.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import P_START_ID, SOL_ID, STOP_ID, VECTOR_ID, PAD_ID
from model.canonicalize import canonicalize_solutions

_IGNORE = -100


class SequenceBuilder:
    def __init__(self, unet, input_proj, special_tok,
                 latent_dim: int, hidden_dim: int, param_dim: int = 2,
                 max_seq_len: int = 512, max_solutions_per_p: int = 16,
                 device: torch.device = torch.device("cpu"),
                 p_mean: torch.Tensor = None, p_std: torch.Tensor = None,
                 fixed_k: int = 24):
        self.unet = unet
        self.input_proj = input_proj
        self.special_tok = special_tok
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim
        self.max_seq_len = max_seq_len
        self.max_k = max_solutions_per_p
        self.fixed_k = int(fixed_k)   # ordered fixed-K: every TARGET block has K SOL slots
        self.device = device
        self.p_mean = (p_mean if p_mean is not None else torch.zeros(param_dim)).to(device)
        self.p_std = (p_std if p_std is not None else torch.ones(param_dim)).to(device)

    # ---- helpers ----
    @torch.no_grad()
    def _encode_solution(self, sol: torch.Tensor) -> torch.Tensor:
        sol = sol.to(self.device, torch.bfloat16).unsqueeze(0)   # [1,2,128,128]
        return self.unet(sol, "encode").squeeze(0)               # [latent]

    def _norm_p(self, p_vec: torch.Tensor) -> torch.Tensor:
        p_vec = p_vec.to(self.device, torch.bfloat16)
        return (p_vec - self.p_mean.to(torch.bfloat16)) / self.p_std.to(torch.bfloat16)

    def _proj(self, z: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device, torch.bfloat16).unsqueeze(0)       # [1,latent]
        p_t = self._norm_p(p_vec).unsqueeze(0)                   # [1,param_dim]
        return self.input_proj(z, p_t).squeeze(0)               # [hidden]

    def _special(self, token_id: int) -> torch.Tensor:
        tid = torch.tensor([token_id], dtype=torch.long, device=self.device)
        return self.special_tok(tid).squeeze(0)

    def _zero_latent(self):
        return torch.zeros(self.latent_dim, dtype=torch.bfloat16, device=self.device)

    # ---- block ----
    def _build_block(self, p_vec: torch.Tensor, solutions: torch.Tensor, is_target: bool):
        """TARGET block (qwen_ord): FIXED fixed_k SOL slots, NO STOP.
          head k<K_t : [SOL]+GT_sol  -> ordered MSE (loss_mask=True, reg=z_k, cls=VECTOR_ID)
          tail k>=K_t: [SOL]+placeholder -> physics-only (tail_mask=True; self-pred filled in
                       forward's pass-2). reg=None, loss_mask=False.
        CONTEXT block (is_target=False): K_t SOL slots + STOP, no loss (unchanged).
        """
        K_t = min(solutions.shape[0], self.max_k)
        zero_p = torch.zeros(self.param_dim)
        embeds, loss_mask, cls_t, reg_t, p_t, tail_m = [], [], [], [], [], []

        def _app(e, lm, c, r, p, tm=False):
            embeds.append(e); loss_mask.append(lm); cls_t.append(c)
            reg_t.append(r); p_t.append(p); tail_m.append(tm)

        _app(self._special(P_START_ID), False, _IGNORE, None, zero_p)
        p_embed = self._proj(self._zero_latent(), p_vec)
        _app(p_embed, False, _IGNORE, None, zero_p)

        if is_target:
            for k in range(self.fixed_k):
                is_head = k < K_t
                if is_head:
                    z_k = self._encode_solution(solutions[k])
                    sol_embed = self._proj(z_k, p_vec)
                else:
                    z_k = None
                    sol_embed = self._proj(self._zero_latent(), p_vec)   # placeholder (overwritten)
                # [SOL] position
                _app(self._special(SOL_ID),
                     is_head,                                # MSE loss_mask = head only
                     VECTOR_ID if is_head else _IGNORE,      # cls=VECTOR_ID marks head SOL for SS
                     z_k if is_head else None,               # reg target = GT latent (head only)
                     p_vec,                                  # p at SOL: needed for SS + physics
                     tm=(not is_head))                       # tail_mask = tail SOL (physics-only)
                # [sol_proj / placeholder] input position
                _app(sol_embed, False, _IGNORE, None, zero_p)
        else:
            for k in range(K_t):
                z_k = self._encode_solution(solutions[k])
                sol_embed = self._proj(z_k, p_vec)
                _app(self._special(SOL_ID), False, _IGNORE, None, zero_p)
                _app(sol_embed, False, _IGNORE, None, zero_p)
            _app(self._special(STOP_ID), False, _IGNORE, None, zero_p)

        return embeds, loss_mask, cls_t, reg_t, p_t, tail_m

    # ---- public ----
    def build(self, context_p_vals, context_sols, target_p_val, target_sols) -> dict:
        all_e, all_lm, all_ct, all_rt, all_pt, all_tm = [], [], [], [], [], []

        target_sols = canonicalize_solutions(target_sols.to(self.device, torch.bfloat16))

        for p_vec, sols in zip(context_p_vals, context_sols):
            sols = canonicalize_solutions(sols.to(self.device, torch.bfloat16))
            e, lm, ct, rt, pt, tm = self._build_block(p_vec, sols, is_target=False)
            all_e += e; all_lm += lm; all_ct += ct; all_rt += rt; all_pt += pt; all_tm += tm

        target_block_offset = len(all_e)
        e, lm, ct, rt, pt, tm = self._build_block(target_p_val, target_sols, is_target=True)
        all_e += e; all_lm += lm; all_ct += ct; all_rt += rt; all_pt += pt; all_tm += tm
        actual_len = len(all_e)

        if actual_len > self.max_seq_len:
            excess = actual_len - self.max_seq_len
            all_e = all_e[excess:]; all_lm = all_lm[excess:]; all_ct = all_ct[excess:]
            all_rt = all_rt[excess:]; all_pt = all_pt[excess:]; all_tm = all_tm[excess:]
            actual_len = self.max_seq_len
            target_block_offset = max(0, target_block_offset - excess)

        inputs_embeds = torch.stack(all_e, 0).cpu()
        loss_mask = torch.tensor(all_lm, dtype=torch.bool)
        tail_mask = torch.tensor(all_tm, dtype=torch.bool)
        cls_targets = torch.tensor(all_ct, dtype=torch.long)
        reg_targets = torch.stack(
            [r.float().cpu() if r is not None else torch.zeros(self.latent_dim) for r in all_rt], 0)
        p_targets = torch.stack([p.float().cpu() for p in all_pt], 0)   # [L, param_dim]

        # cap target K to max_k for the padded raw solutions
        K_t = min(target_sols.shape[0], self.max_k)
        return {
            "inputs_embeds": inputs_embeds,
            "loss_mask": loss_mask,
            "tail_mask": tail_mask,
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
            "p_targets": p_targets,
            "actual_len": actual_len,
            "target_block_offset": target_block_offset,
            "canonical_target_sols": target_sols[:K_t].float().cpu(),   # [K_t,2,128,128]
            "target_p_val": target_p_val.float().cpu(),                 # [param_dim]
        }
