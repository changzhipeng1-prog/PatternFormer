"""
Special Token Embeddings for V2 Hybrid Vocabulary.

Token ID mapping (matches config.py constants):
    P_START = 0   — start of a parameter block
    SOL     = 1   — solution-slot marker (precedes each solution embedding)
    STOP    = 2   — end of solution sequence for a given p
    VECTOR  = 3   — classification head output meaning "I am outputting a vector here"
    PAD     = 4   — padding (always zero, not learnable)

Only tokens 0-3 have learnable embeddings; token 4 (PAD) is always the zero vector.
"""
import torch
import torch.nn as nn

from config import P_START_ID, SOL_ID, STOP_ID, VECTOR_ID, PAD_ID

NUM_SPECIAL = 5   # total number of special tokens


class SpecialTokenEmbeddings(nn.Module):
    """
    Lookup table for the 5 special token embeddings.

    Args:
        hidden_dim: Qwen hidden dimension (e.g. 3584)

    Usage:
        embed = SpecialTokenEmbeddings(3584)
        v = embed(torch.tensor([P_START_ID, SOL_ID, STOP_ID]))  # [3, 3584]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable embeddings for tokens 0-3 (P_START, SOL, STOP, VECTOR)
        self.embed = nn.Embedding(NUM_SPECIAL - 1, hidden_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # PAD is a fixed zero vector — registered as buffer so it moves with .to(device)
        self.register_buffer("pad_vec", torch.zeros(1, hidden_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: integer tensor of arbitrary shape [...], values in 0-4

        Returns:
            embeddings: [..., hidden_dim]
        """
        shape = token_ids.shape
        flat = token_ids.reshape(-1)          # [N]

        is_pad = flat == PAD_ID               # [N] bool

        # Replace PAD IDs with 0 to avoid out-of-range in nn.Embedding
        safe_ids = flat.clone()
        safe_ids[is_pad] = 0

        embs = self.embed(safe_ids)           # [N, D]

        # Zero out PAD positions (pure tensor op, no branch on GPU)
        mask = (~is_pad).float().unsqueeze(-1)   # [N, 1]
        embs = embs * mask

        return embs.reshape(*shape, self.hidden_dim)

    def get(self, token_id: int) -> torch.Tensor:
        """
        Convenience: return single token embedding as [1, D].
        """
        tid = torch.tensor([token_id], dtype=torch.long, device=self.pad_vec.device)
        return self.forward(tid)              # [1, D]
