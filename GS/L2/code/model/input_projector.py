"""Input Projector: maps AE latent z and the 2D parameter (rho,mu) into Qwen's
hidden space as a single soft-prompt embedding.

    concat([z, p]) -> Linear(latent_dim+param_dim, mid) -> GELU -> Linear(mid, hidden)

Adapted from the reference (scalar p) to param_dim=2.
"""
import torch
import torch.nn as nn


class InputProjector(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, param_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim
        in_dim = latent_dim + param_dim
        mid_dim = (in_dim + hidden_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, hidden_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """z: [..., latent_dim]   p: [..., param_dim]  ->  [..., hidden_dim]"""
        if p.dim() < z.dim():
            p = p.unsqueeze(-2).expand(*z.shape[:-1], self.param_dim)
        x = torch.cat([z, p], dim=-1)
        return self.net(x)
