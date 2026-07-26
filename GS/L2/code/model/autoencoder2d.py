"""autoencoder2d.py — Convolutional autoencoder for 2D Gray-Scott fields.

Unlike the reference (MLP on a 145-node FEM mesh), our solutions live on a
REGULAR 128x128 grid with 2 channels (A activator, S substrate), so a conv
autoencoder is the natural encoder/decoder. No skip connections: the 256-d
bottleneck latent is what the Qwen generator consumes.

Same interface as the reference SolutionAutoencoder2D:
    forward(x, mode) with mode = 'encode' | 'decode'
    encode: [B,2,128,128] or [B,T,2,128,128] -> [B,256] or [B,T,256]
    decode: [B,256]       or [B,T,256]       -> [B,2,128,128] or [B,T,2,128,128]

Per-channel normalization (mean/std from train set) is baked in as buffers, so
encode() normalizes inputs and decode() returns physical-unit fields.
"""
import torch
import torch.nn as nn


def _gn(c):
    return nn.GroupNorm(min(32, c), c)


class _Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1), _gn(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, padding=1), _gn(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SolutionAutoencoder2D(nn.Module):
    """Conv AE: 2x128x128 <-> 256-d latent.

    Args mirror the reference for drop-in config compatibility; solution_dim is
    accepted but unused (kept so config.solution_dim doesn't break construction).
    """

    def __init__(self, solution_dim: int = None, latent_dim: int = 256,
                 hidden_dim: int = None, in_ch: int = 2,
                 base: int = 32, img: int = 128,
                 mean: torch.Tensor = None, std: torch.Tensor = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_ch = in_ch
        self.img = img
        chs = [base, base * 2, base * 4, base * 8, base * 8]   # 32,64,128,256,256
        self.bott = img // (2 ** len(chs))                     # 128/32 = 4
        self.cend = chs[-1]

        # encoder: 5 stride-2 blocks  128->64->32->16->8->4
        enc = []
        c = in_ch
        for co in chs:
            enc.append(_Down(c, co)); c = co
        self.enc = nn.Sequential(*enc)
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(self.cend * self.bott * self.bott, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # decoder
        self.from_latent = nn.Linear(latent_dim, self.cend * self.bott * self.bott)
        dec = []
        rev = chs[::-1]                                        # 256,256,128,64,32
        c = self.cend
        for co in rev[1:] + [base]:
            dec.append(_Up(c, co)); c = co
        self.dec = nn.Sequential(*dec)
        self.head = nn.Conv2d(base, in_ch, 3, padding=1)

        # normalization buffers (set from train stats)
        if mean is None:
            mean = torch.zeros(in_ch)
        if std is None:
            std = torch.ones(in_ch)
        self.register_buffer("mean", mean.view(1, in_ch, 1, 1).clone())
        self.register_buffer("std", std.view(1, in_ch, 1, 1).clone())

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- public API ----
    def forward(self, x, mode):
        if mode == "encode":
            return self._encode(x)
        if mode == "decode":
            return self._decode(x)
        raise ValueError(f"mode must be 'encode'|'decode', got {mode}")

    def _encode(self, x):
        seq = x.dim() == 5
        if seq:
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
        x = (x - self.mean) / self.std
        z = self.to_latent(self.enc(x))
        if seq:
            z = z.view(B, T, self.latent_dim)
        return z

    def _decode(self, z):
        seq = z.dim() == 3
        if seq:
            B, T = z.shape[:2]
            z = z.reshape(B * T, self.latent_dim)
        h = self.from_latent(z).view(-1, self.cend, self.bott, self.bott)
        x = self.head(self.dec(h))
        x = x * self.std + self.mean
        if seq:
            x = x.view(B, T, self.in_ch, self.img, self.img)
        return x

    def get_encoder_params(self):
        return list(self.enc.parameters()) + list(self.to_latent.parameters())

    def get_decoder_params(self):
        return list(self.from_latent.parameters()) + list(self.dec.parameters()) + \
            list(self.head.parameters())


if __name__ == "__main__":
    m = SolutionAutoencoder2D(latent_dim=256)
    n = sum(p.numel() for p in m.parameters())
    print(f"params {n/1e6:.2f}M, bottleneck {m.cend}x{m.bott}x{m.bott}")
    x = torch.randn(4, 2, 128, 128)
    z = m(x, "encode"); r = m(z, "decode")
    print("single", tuple(x.shape), "->", tuple(z.shape), "->", tuple(r.shape))
    xs = torch.randn(2, 3, 2, 128, 128)
    zs = m(xs, "encode"); rs = m(zs, "decode")
    print("seq   ", tuple(xs.shape), "->", tuple(zs.shape), "->", tuple(rs.shape))
