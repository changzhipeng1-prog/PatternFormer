"""
Sequence Builder — the data-flow entry point for V3 training (2D PDE, 145-node FEM mesh).

Builds the full hybrid-vocabulary embedding sequence for one training sample.

Sequence structure
------------------
A sample consists of:
    (a) 0..k context blocks  (loss_mask = 0)
    (b) 1 target block       (loss_mask = 1 at [SOL] positions)

Each block for parameter p_i with K_i canonicalized solutions:

    Token stream (positions in embedding sequence):
    ┌──────────────────────────────────────────────────────────────────────┐
    │ [P_START] [p_proj]  [SOL] [sol_proj_1]  [SOL] [sol_proj_2]  [STOP] │
    │    ↑          ↑       ↑        ↑           ↑        ↑          ↑    │
    │  special   InputProj special InputProj  special InputProj   special │
    └──────────────────────────────────────────────────────────────────────┘

    p_proj    = InputProjector(z=zeros, p=p_i)
    sol_proj  = InputProjector(z=Autoencoder.encode(sol_k), p=p_i)

AR formulation: hidden_state[i] predicts what is at position i+1.

Classification targets (cls_targets, for CE loss in target block only):
    [p_proj]     → cls_target = SOL_ID   (next is [SOL])  or STOP_ID (if K==0)
    [SOL]        → cls_target = VECTOR_ID (next is a continuous solution vector)
    [sol_proj_k] → cls_target = SOL_ID   (k < K-1) or STOP_ID (k == K-1)
    all others   → cls_target = -100     (ignored)

Loss masks (regression):
    loss_mask[i] = True  ↔  position i is a [SOL] token in the target block.
    At [SOL] position i: hidden_state[i] predicts sol_{k+1}, so:
        reg_target[i] = z_k = UNet.encode(sol_k)   (the latent being introduced next)
        cls_target[i] = VECTOR_ID

    Context blocks and PAD tokens: loss_mask = 0.

Outputs (all CPU tensors, L = actual sequence length before padding):
    inputs_embeds:          [L, hidden_dim]  bfloat16
    loss_mask:              [L]              bool
    cls_targets:            [L]              long (-100 = ignore)
    reg_targets:            [L, latent_dim]  float32
    p_targets:              [L]              float32
    actual_len:             int
    target_block_offset:    int   — index where target block begins in the sequence
    canonical_target_sols:  [K_t, 1024]  float32  — canonicalized target solutions
    target_p_val:           float
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import P_START_ID, SOL_ID, STOP_ID, VECTOR_ID, PAD_ID
from model.canonicalize import canonicalize_solutions


# Sentinel for "no loss at this position"
_IGNORE = -100


class SequenceBuilder:
    """
    Stateful builder that constructs embedding sequences given the model's
    live modules (UNet, InputProjector, SpecialTokenEmbeddings).

    This class operates on **pre-computed latents** when possible; it only
    calls into the model's UNet encoder at build time, so it must be used
    inside a torch.no_grad() context (or the caller must handle grad tracking).

    Args:
        autoencoder:    SolutionAutoencoder2D (frozen)
        input_proj:     InputProjector
        special_tok:    SpecialTokenEmbeddings
        latent_dim:     int  (e.g. 256)
        hidden_dim:     int  (e.g. 3584)
        max_seq_len:    int  (hard cap; sequences exceeding this are truncated)
        device:         torch.device
    """

    def __init__(self, unet, input_proj, special_tok,
                 latent_dim: int, hidden_dim: int,
                 max_seq_len: int = 256,
                 device: torch.device = torch.device("cpu"),
                 p_scale: float = 1.0):
        self.unet        = unet
        self.input_proj  = input_proj
        self.special_tok = special_tok
        self.latent_dim  = latent_dim
        self.hidden_dim  = hidden_dim
        self.max_seq_len = max_seq_len
        self.device      = device
        self.p_scale     = float(p_scale)   # divide raw p by this before InputProjector
        self._debug_printed_target_stop_once = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_solution(self, sol: torch.Tensor) -> torch.Tensor:
        """Encode a single solution [145] → [latent_dim]."""
        sol2d = sol.to(device=self.device, dtype=torch.bfloat16).unsqueeze(0)  # [1, 145]
        z = self.unet(sol2d, "encode")                                           # [1, latent]
        return z.squeeze(0)                                                      # [latent]

    def _proj(self, z: torch.Tensor, p_scalar: float) -> torch.Tensor:
        """InputProjector forward for a single (z, p) pair → [hidden_dim]."""
        z = z.to(device=self.device, dtype=torch.bfloat16)
        p_normalized = p_scalar / self.p_scale                                      # normalize p
        p_t = torch.tensor([[p_normalized]], dtype=torch.bfloat16, device=self.device)  # [1,1]
        z2d = z.unsqueeze(0)                                                        # [1, latent]
        out = self.input_proj(z2d, p_t)                                             # [1, hidden]
        return out.squeeze(0)                                                        # [hidden]

    def _special(self, token_id: int) -> torch.Tensor:
        """Return special token embedding [hidden_dim]."""
        tid = torch.tensor([token_id], dtype=torch.long, device=self.device)
        return self.special_tok(tid).squeeze(0)                                     # [hidden]

    def _zero_latent(self) -> torch.Tensor:
        return torch.zeros(self.latent_dim, dtype=torch.bfloat16, device=self.device)

    def _zero_hidden(self) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, dtype=torch.bfloat16, device=self.device)

    # ------------------------------------------------------------------
    # Block builder
    # ------------------------------------------------------------------

    def _build_block(self, p_val: float, solutions: torch.Tensor, is_target: bool):
        """
        Build embedding tokens + metadata for one p-block.

        Args:
            p_val:     scalar float
            solutions: [K, 1024]  (already canonicalized)
            is_target: bool — whether this is the target block

        Returns:
            embeds:      list[Tensor[D]]
            loss_mask:   list[bool]
            cls_targets: list[int]   (-100 = ignore)
            reg_targets: list[Tensor[latent] or None]
            p_targets:   list[float]
        """
        K = solutions.shape[0]
        embeds       = []
        loss_mask    = []
        cls_targets  = []
        reg_targets  = []
        p_targets    = []

        def _append(embed, lm, cls_t, reg_t, p_t):
            embeds.append(embed)
            loss_mask.append(lm)
            cls_targets.append(cls_t)
            reg_targets.append(reg_t)
            p_targets.append(p_t)

        # ---- Position 0: [P_START] special token ----
        _append(
            self._special(P_START_ID),
            False,
            _IGNORE,
            None,
            0.0,
        )

        # ---- Position 1: [p_proj] — p-value soft prompt ----
        # z = zero (no prior solution latent at the p marker)
        z_zero = self._zero_latent()
        p_embed = self._proj(z_zero, p_val)

        # cls_target at p_proj: what comes NEXT?
        # If K > 0 → next is [SOL] marker → VECTOR will follow → classify SOL
        # If K = 0 → next is [STOP]
        next_after_p = SOL_ID if K > 0 else STOP_ID
        _append(
            p_embed,
            False,                 # p position never has regression loss
            next_after_p if is_target else _IGNORE,
            None,
            0.0,
        )

        # ---- Positions 2 .. 2+2K-1: interleaved [SOL] + [sol_proj] ----
        for k in range(K):
            # Encode solution first so we can store z_k at the [SOL] position.
            sol_k     = solutions[k]              # [1024]
            z_k       = self._encode_solution(sol_k)   # [latent]
            sol_embed = self._proj(z_k, p_val)    # [hidden]

            # [SOL] marker — AR: model reads this token → hidden_state predicts
            # the next position (sol_proj_k), so regression target z_k goes HERE.
            # cls_target = VECTOR_ID (next position is a continuous vector).
            _append(
                self._special(SOL_ID),
                is_target,                               # loss_mask: regression loss here
                VECTOR_ID if is_target else _IGNORE,     # cls: next is VECTOR
                z_k if is_target else None,              # reg_target = z_k  ← correct AR
                p_val if is_target else 0.0,
            )

            # [sol_proj_k] — teacher-forced input (actual solution embedding).
            # cls_target: what comes NEXT after this?
            #   k < K-1  → next is [SOL]  → SOL_ID
            #   k == K-1 → next is [STOP] → STOP_ID
            # No regression loss here (model reads sol_k → predicts next type, not sol_k).
            next_after_sol = SOL_ID if k < K - 1 else STOP_ID
            _append(
                sol_embed,
                False,                                   # no regression loss at sol_proj
                next_after_sol if is_target else _IGNORE,
                None,
                0.0,
            )

        # ---- Last position: [STOP] token ----
        # Keep STOP supervised in target block so CE sees an explicit STOP label
        # at the STOP token position as requested by training diagnostics.
        _append(
            self._special(STOP_ID),
            is_target,
            STOP_ID if is_target else _IGNORE,
            None,
            0.0,
        )

        return embeds, loss_mask, cls_targets, reg_targets, p_targets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        context_p_vals:  list,       # list of float
        context_sols:    list,       # list of Tensor [K_i, 1024]
        target_p_val:    float,
        target_sols:     torch.Tensor,   # [K_t, 1024]
    ) -> dict:
        """
        Build one complete training sequence.

        Args:
            context_p_vals:  k float values
            context_sols:    k tensors [K_i, 1024]
            target_p_val:    float
            target_sols:     [K_t, 1024]

        Returns dict with CPU tensors:
            inputs_embeds:         [L, hidden_dim]       bfloat16
            loss_mask:             [L]                   bool
            cls_targets:           [L]                   long (-100 = ignore)
            reg_targets:           [L, latent_dim]       float32
            p_targets:             [L]                   float32
            actual_len:            int
            target_block_offset:   int  (index of first token in target block)
            canonical_target_sols: [K_t, 145]            float32  (canonicalized)
            target_p_val:          float
        """
        all_embeds      = []
        all_loss_mask   = []
        all_cls_targets = []
        all_reg_targets = []
        all_p_targets   = []

        # Canonicalize target
        target_sols = canonicalize_solutions(
            target_sols.to(device=self.device, dtype=torch.bfloat16)
        )

        # ---- Context blocks ----
        for p_val, sols in zip(context_p_vals, context_sols):
            sols = canonicalize_solutions(
                sols.to(device=self.device, dtype=torch.bfloat16)
            )
            e, lm, ct, rt, pt = self._build_block(p_val, sols, is_target=False)
            all_embeds.extend(e)
            all_loss_mask.extend(lm)
            all_cls_targets.extend(ct)
            all_reg_targets.extend(rt)
            all_p_targets.extend(pt)

        # Record where target block starts (before appending target)
        target_block_offset = len(all_embeds)

        # ---- Target block ----
        e, lm, ct, rt, pt = self._build_block(target_p_val, target_sols, is_target=True)
        target_block_len = len(e)
        all_embeds.extend(e)
        all_loss_mask.extend(lm)
        all_cls_targets.extend(ct)
        all_reg_targets.extend(rt)
        all_p_targets.extend(pt)

        actual_len = len(all_embeds)

        # Truncate if over limit (preserves target block by truncating context prefix)
        if actual_len > self.max_seq_len:
            excess = actual_len - self.max_seq_len
            all_embeds      = all_embeds[excess:]
            all_loss_mask   = all_loss_mask[excess:]
            all_cls_targets = all_cls_targets[excess:]
            all_reg_targets = all_reg_targets[excess:]
            all_p_targets   = all_p_targets[excess:]
            actual_len      = self.max_seq_len
            target_block_offset = max(0, target_block_offset - excess)

        # Pack into tensors
        inputs_embeds = torch.stack(all_embeds, dim=0).cpu()       # [L, D]

        loss_mask = torch.tensor(all_loss_mask, dtype=torch.bool)  # [L]

        cls_targets = torch.tensor(all_cls_targets, dtype=torch.long)  # [L]

        reg_targets_list = []
        for rt in all_reg_targets:
            if rt is not None:
                reg_targets_list.append(rt.float().cpu())
            else:
                reg_targets_list.append(torch.zeros(self.latent_dim))
        reg_targets = torch.stack(reg_targets_list, dim=0)         # [L, latent]

        p_targets = torch.tensor(all_p_targets, dtype=torch.float32)  # [L]

        # One-time debug check for STOP alignment/mask in target block.
        # 默认关闭；长训多 rank 刷屏。需要时: export BBBB_SEQUENCE_BUILDER_DEBUG=1
        if (
            os.environ.get("BBBB_SEQUENCE_BUILDER_DEBUG", "0") == "1"
            and not self._debug_printed_target_stop_once
        ):
            stop_pos = target_block_offset + target_block_len - 1
            if 0 <= stop_pos < actual_len:
                left = max(0, stop_pos - 3)
                right = min(actual_len, stop_pos + 4)
                print("[SequenceBuilder DEBUG] target STOP alignment check")
                print(f"  inputs_embeds.shape={tuple(inputs_embeds.shape)}")
                print(f"  target_stop_pos={stop_pos}, window=[{left}:{right})")
                print(f"  cls_targets_window={cls_targets[left:right].tolist()}")
                print(f"  loss_mask_window={loss_mask[left:right].tolist()}")
                print(
                    f"  STOP cell -> cls_targets[{stop_pos}]={int(cls_targets[stop_pos].item())}, "
                    f"loss_mask[{stop_pos}]={bool(loss_mask[stop_pos].item())}"
                )
            else:
                print("[SequenceBuilder DEBUG] target STOP not in range after truncation.")
            self._debug_printed_target_stop_once = True

        return {
            "inputs_embeds":         inputs_embeds,                 # [L, hidden_dim]
            "loss_mask":             loss_mask,                     # [L]  bool
            "cls_targets":           cls_targets,                   # [L]  long
            "reg_targets":           reg_targets,                   # [L, latent_dim]
            "p_targets":             p_targets,                     # [L]  float32
            "actual_len":            actual_len,
            "target_block_offset":   target_block_offset,          # int
            "canonical_target_sols": target_sols.float().cpu(),    # [K_t, 1024]
            "target_p_val":          target_p_val,                 # float
        }
