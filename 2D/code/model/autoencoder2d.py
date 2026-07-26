"""
autoencoder2d.py — MLP Autoencoder for 2D FEM solutions (145 nodes)

Replaces UNet1d_v6 for the 2D problem.
Same interface: forward(x, mode) where mode = 'encode' | 'decode'

Architecture
------------
solution_dim = 145  (ell=3 FEM mesh, free + boundary nodes)
latent_dim   = 256

Encoder:  [B, 145] → Linear(145→512) → LN+GELU → Linear(512→512) → LN+GELU
                   → Linear(512→256) → [B, 256]

Decoder:  [B, 256] → Linear(256→512) → LN+GELU → Linear(512→512) → LN+GELU
                   → Linear(512→145) → [B, 145]

Supports batched sequences: input [B, T, 145] is flattened to [B*T, 145]
before encoding and restored to [B, T, latent_dim] after.
"""

import torch
import torch.nn as nn


class _MLPBlock(nn.Module):
    """Linear → LayerNorm → GELU with residual projection."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.act    = nn.GELU()
        # Residual: project input if dims differ
        self.residual = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.norm(self.linear(x))) + self.residual(x)


class SolutionAutoencoder2D(nn.Module):
    """
    MLP Autoencoder for 2D FEM solutions on the ell=3 mesh (145 nodes).

    Args:
        solution_dim : number of FEM nodes (default 145)
        latent_dim   : bottleneck dimension (default 256)
        hidden_dim   : width of intermediate layers (default 512)

    Forward modes:
        mode='encode': [B, solution_dim]  or [B, T, solution_dim]
                       → [B, latent_dim]  or [B, T, latent_dim]
        mode='decode': [B, latent_dim]    or [B, T, latent_dim]
                       → [B, solution_dim] or [B, T, solution_dim]
    """

    def __init__(self, solution_dim: int = 145,
                 latent_dim: int = 256,
                 hidden_dim: int = 512):
        super().__init__()
        self.solution_dim = solution_dim
        self.latent_dim   = latent_dim
        self.hidden_dim   = hidden_dim

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            _MLPBlock(solution_dim, hidden_dim),
            _MLPBlock(hidden_dim,   hidden_dim),
            _MLPBlock(hidden_dim,   hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            _MLPBlock(latent_dim,  hidden_dim),
            _MLPBlock(hidden_dim,  hidden_dim),
            _MLPBlock(hidden_dim,  hidden_dim),
            nn.Linear(hidden_dim, solution_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Public API ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'encode':
            return self._encode(x)
        elif mode == 'decode':
            return self._decode(x)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'encode' or 'decode'.")

    # ── Internal ──────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, D] or [B, T, D] → [B, latent] or [B, T, latent]"""
        seq = x.dim() == 3
        if seq:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
        z = self.encoder(x)          # [B(*T), latent]
        if seq:
            z = z.view(B, T, self.latent_dim)
        return z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent] or [B, T, latent] → [B, D] or [B, T, D]"""
        seq = z.dim() == 3
        if seq:
            B, T, L = z.shape
            z = z.reshape(B * T, L)
        x = self.decoder(z)          # [B(*T), solution_dim]
        if seq:
            x = x.view(B, T, self.solution_dim)
        return x

    # ── Param groups for selective freezing ───────────────────────────────

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_decoder_params(self):
        return self.decoder.parameters()


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = SolutionAutoencoder2D(solution_dim=145, latent_dim=256, hidden_dim=512)

    total = sum(p.numel() for p in model.parameters())
    enc   = sum(p.numel() for p in model.get_encoder_params())
    dec   = sum(p.numel() for p in model.get_decoder_params())
    print(f"SolutionAutoencoder2D")
    print(f"  solution_dim={model.solution_dim}, latent_dim={model.latent_dim}, "
          f"hidden={model.hidden_dim}")
    print(f"  Total params : {total:,}")
    print(f"  Encoder      : {enc:,}")
    print(f"  Decoder      : {dec:,}")

    # Single-batch test
    x = torch.randn(8, 145)
    z = model(x, 'encode')
    r = model(z, 'decode')
    print(f"\n  Single: {x.shape} → {z.shape} → {r.shape}")

    # Sequence test
    x_seq = torch.randn(4, 10, 145)
    z_seq = model(x_seq, 'encode')
    r_seq = model(z_seq, 'decode')
    print(f"  Seq   : {x_seq.shape} → {z_seq.shape} → {r_seq.shape}")
    print("All checks passed.")
