"""
Input Projector: maps UNet latent z and params (a4, a2) to Qwen hidden space.

concat([z, a4, a2]) → Linear(latent_dim+2, mid_dim) → GELU → Linear(mid_dim, hidden_dim)
"""
import torch
import torch.nn as nn


class InputProjector(nn.Module):
    """
    Jointly project (z, params) → Qwen hidden space.

    Args:
        latent_dim: UNet bottleneck dimension (e.g. 256)
        hidden_dim: Qwen hidden size (e.g. 3584)
        param_dim:  number of PDE parameters (2 for a4, a2)
    """

    def __init__(self, latent_dim: int, hidden_dim: int, param_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.param_dim  = param_dim
        in_dim  = latent_dim + param_dim
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

    def forward(self, z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      [..., latent_dim]
            params: [..., param_dim]  — (a4, a2) values

        Returns:
            [..., hidden_dim]
        """
        if params.shape[-1] != self.param_dim:
            raise ValueError(f"Expected params dim {self.param_dim}, got {params.shape[-1]}")
        x = torch.cat([z, params], dim=-1)   # [..., latent_dim+param_dim]
        return self.net(x)
