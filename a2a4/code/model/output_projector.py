"""
Output Projector: maps Qwen hidden states back to UNet latent space.

Architecture:
    h → Linear(hidden_dim, mid_dim) → GELU → Linear(mid_dim, latent_dim)

This replaces V1's MatchLayer(decode) with a 2-layer GELU MLP.
mid_dim = (hidden_dim + latent_dim) // 2
"""
import torch
import torch.nn as nn


class OutputProjector(nn.Module):
    """
    Project Qwen hidden state → UNet latent space.

    Args:
        hidden_dim: Qwen hidden size (e.g. 3584)
        latent_dim: UNet bottleneck dimension (e.g. 256)

    Input:
        h: [..., hidden_dim]

    Output:
        [..., latent_dim]
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        mid_dim = (hidden_dim + latent_dim) // 2

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, latent_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [..., hidden_dim]

        Returns:
            [..., latent_dim]
        """
        return self.net(h)
