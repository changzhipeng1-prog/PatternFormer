"""
Dual Head: parallel classification + regression heads on Qwen's last hidden state.

Classification head (cls_head):
    Predicts the *type* of the next sequence position:
        P_START=0, SOL=1, STOP=2, VECTOR=3, PAD=4
    In practice, during solution generation the relevant outputs are VECTOR and STOP.

Regression head (reg_head):
    Predicts the UNet latent vector for the next solution.
    Its output is only consumed when cls_head predicts VECTOR.

Training (Teacher Forcing with Loss Mask):
    - Positions where ground-truth next token is a discrete special token:
          CE loss on cls_head; reg_head output ignored.
    - Positions where ground-truth next token is a continuous solution vector:
          MSE + PDE loss on reg_head output (decoded back to 1024-dim);
          cls_head is forced to predict VECTOR_ID via CE loss.
    - Context and PAD positions: both heads' losses are masked to 0.

Inference:
    - cls_head routes: VECTOR → consume reg_head output; STOP → terminate.
"""
import torch
import torch.nn as nn

from config import VECTOR_ID


class DualHead(nn.Module):
    """
    Parallel classification and regression heads.

    Args:
        hidden_dim: Qwen hidden size (e.g. 3584)
        latent_dim: UNet bottleneck size (e.g. 256)
        vocab_size: number of special tokens (5)
    """

    def __init__(self, hidden_dim: int, latent_dim: int, vocab_size: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # Classification head: D → vocab_size logits
        self.cls_head = nn.Linear(hidden_dim, vocab_size, bias=True)

        # Regression head: D → latent_dim continuous vector
        self.reg_head = nn.Linear(hidden_dim, latent_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.xavier_uniform_(self.reg_head.weight)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [..., hidden_dim] — Qwen last_hidden_state at selected positions

        Returns:
            cls_logits: [..., vocab_size]   — raw logits for CE loss / argmax routing
            reg_latent: [..., latent_dim]   — predicted latent (used when cls==VECTOR)
        """
        cls_logits = self.cls_head(h)
        reg_latent = self.reg_head(h)
        return cls_logits, reg_latent

    def route(self, h: torch.Tensor):
        """
        Inference-time routing (batch dim=1 expected, but works for any batch).

        Returns:
            token_ids: [B]  predicted token type
            reg_latent: [B, latent_dim]
        """
        cls_logits, reg_latent = self.forward(h)
        token_ids = cls_logits.argmax(dim=-1)    # [B]
        return token_ids, reg_latent
