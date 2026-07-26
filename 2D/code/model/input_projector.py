"""
Input Projector: maps UNet latent z and scalar p to Qwen's hidden space.

Architecture:
    concat([z, p]) → Linear(latent_dim+1, mid_dim) → GELU → Linear(mid_dim, hidden_dim)

This replaces V1's MatchLayer(encode) + PEmbedding with a unified 2-layer GELU MLP
that jointly encodes the spatial latent and the PDE parameter as a single Soft Prompt.

mid_dim = (latent_dim + 1 + hidden_dim) // 2  (harmonic interpolation)
"""
import torch
import torch.nn as nn


class InputProjector(nn.Module):
    """
    Jointly project (z, p) → Qwen hidden space.

    Args:
        latent_dim: UNet bottleneck dimension (e.g. 256)
        hidden_dim: Qwen hidden size (e.g. 3584)

    Input:
        z: [..., latent_dim]   UNet-encoded latent
        p: [..., 1]            scalar PDE parameter (must match z's batch dims)

    Output:
        [..., hidden_dim]      Soft Prompt embedding
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        in_dim  = latent_dim + 1
        mid_dim = (in_dim + hidden_dim) // 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [..., latent_dim]
            p: [...] or [..., 1]  — scalar p values

        Returns:
            [..., hidden_dim]
        """
        if p.dim() < z.dim():
            p = p.unsqueeze(-1)              # [..., 1]
        elif p.shape[-1] != 1:
            p = p.unsqueeze(-1)

        x = torch.cat([z, p], dim=-1)       # [..., latent_dim+1]
        return self.net(x)
