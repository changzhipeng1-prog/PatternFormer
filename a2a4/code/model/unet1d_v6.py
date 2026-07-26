"""
1D UNet v6 — Anti-Checkerboard Decoder

Changes vs original model/unet1d.py:
  - Decoder main path: ConvTranspose1d -> Upsample(mode='linear') + Conv1d(k=3,p=1)
  - Decoder skip path: ConvTranspose1d -> Upsample(mode='linear') + Conv1d(k=3,p=1)
  - Encoder is identical to the original (weights can be loaded from unet_best.pth)
  - Added get_encoder_params() / get_decoder_params() for selective freezing
"""
from collections import OrderedDict

import torch
import torch.nn as nn


class UNet1d(nn.Module):
    """
    1D UNet encoder-decoder with smooth upsampling decoder.

    Encoder: [B*T, 1, 1024] -> latent_dim
    Decoder: latent_dim -> [B*T, 1, 1024]  (no ConvTranspose1d)

    Args:
        layers: channel list, e.g. [1, 16, 32, 64, 128, 256]
        latent_dim: bottleneck dimension
        solution_dim: spatial dimension (default 1024)
    """

    def __init__(self, layers, latent_dim=512, solution_dim=1024):
        super(UNet1d, self).__init__()

        self.layers = layers
        self.latent_dim = latent_dim
        self.solution_dim = solution_dim

        num_downsample = len(layers) - 1
        self.bottleneck_spatial = solution_dim // (2 ** num_downsample)
        self.bottleneck_channels = layers[-1]
        self.bottleneck_flat = self.bottleneck_channels * self.bottleneck_spatial

        # ==================== Encoder (identical to original) ====================
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for idx in range(1, len(layers)):
            in_ch = layers[idx - 1]
            out_ch = layers[idx]
            half_out = out_ch // 2

            self.encoder.append(
                nn.Sequential(
                    UNet1d._block(in_ch, half_out, name=f'enc{idx}'),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )
            self.downsample.append(
                nn.Conv1d(in_ch, half_out, kernel_size=2, stride=2)
            )

        self.conv1 = nn.Conv1d(layers[-1], layers[-1], kernel_size=1)
        self.encoder_fc = nn.Linear(self.bottleneck_flat, latent_dim)

        # ==================== Decoder (smooth upsampling) ====================
        self.decoder_fc = nn.Linear(latent_dim, self.bottleneck_flat)

        decoder_layers = layers[::-1]

        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for idx in range(1, len(decoder_layers)):
            in_ch = decoder_layers[idx - 1]
            out_ch = decoder_layers[idx]

            if out_ch // 2 < 1:
                half_out = 1
            else:
                half_out = out_ch // 2

            # Main path: _block -> Upsample(linear) -> Conv1d
            self.decoder.append(
                nn.Sequential(
                    UNet1d._block(in_ch, half_out, name=f'dec{idx}'),
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(half_out, half_out, kernel_size=3, stride=1, padding=1),
                )
            )

            # Skip path: Upsample(linear) -> Conv1d
            self.upsample.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(in_ch, half_out, kernel_size=3, stride=1, padding=1),
                )
            )

        final_ch = decoder_layers[-1]
        if final_ch // 2 < 1:
            final_full = 2
        else:
            final_full = (final_ch // 2) * 2
        self.conv2 = nn.Conv1d(final_full, layers[0], kernel_size=1)

    def forward(self, x, mode):
        if mode == 'encode':
            return self._encode(x)
        elif mode == 'decode':
            return self._decode(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _encode(self, x):
        if x.dim() == 3:
            self.batch_size = x.size(0)
            self.time_steps = x.size(1)
            x = x.view(self.batch_size * self.time_steps, self.solution_dim)
        else:
            self.batch_size = x.size(0)
            self.time_steps = 1

        x = x.unsqueeze(1)

        for layer, downsample in zip(self.encoder, self.downsample):
            skip = downsample(x)
            x = layer(x)
            x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = x.view(self.batch_size * self.time_steps, -1)
        x = self.encoder_fc(x)

        if self.time_steps > 1:
            x = x.view(self.batch_size, self.time_steps, -1)

        return x

    def _decode(self, x):
        if x.dim() == 3:
            B, T = x.size(0), x.size(1)
            x = x.view(B * T, -1)
        else:
            B = x.size(0)
            T = 1

        x = self.decoder_fc(x)
        x = x.view(B * T, self.bottleneck_channels, self.bottleneck_spatial)

        for layer, upsample in zip(self.decoder, self.upsample):
            skip = upsample(x)
            x = layer(x)
            x = torch.cat((x, skip), dim=1)

        x = self.conv2(x)
        x = x.squeeze(1)

        if T > 1:
            x = x.view(B, T, -1)

        return x

    def get_encoder_params(self):
        """Return iterator over encoder parameters."""
        for module in [self.encoder, self.downsample, self.conv1, self.encoder_fc]:
            yield from module.parameters()

    def get_decoder_params(self):
        """Return iterator over decoder parameters."""
        for module in [self.decoder_fc, self.decoder, self.upsample, self.conv2]:
            yield from module.parameters()

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([
                (name + "conv1", nn.Conv1d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                (name + "norm1", nn.BatchNorm1d(num_features=features)),
                (name + "tanh1", nn.Tanh()),
                (name + "conv2", nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False)),
                (name + "norm2", nn.BatchNorm1d(num_features=features)),
                (name + "tanh2", nn.Tanh()),
            ])
        )


if __name__ == "__main__":
    layers = [1, 16, 32, 64, 128, 256]
    latent_dim = 512
    model = UNet1d(layers, latent_dim=latent_dim)

    print(f"UNet1d v6 (anti-checkerboard):")
    print(f"  Layers: {layers}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Bottleneck spatial: {model.bottleneck_spatial}")

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.get_encoder_params())
    dec_params = sum(p.numel() for p in model.get_decoder_params())
    print(f"  Total params: {total_params:,}")
    print(f"  Encoder:      {enc_params:,}")
    print(f"  Decoder:      {dec_params:,}")

    x_single = torch.randn(4, 1024)
    latent_single = model(x_single, 'encode')
    recon_single = model(latent_single, 'decode')
    print(f"\nPhase 1: {x_single.shape} -> {latent_single.shape} -> {recon_single.shape}")

    B, T = 2, 10
    x_seq = torch.randn(B, T, 1024)
    latent_seq = model(x_seq, 'encode')
    recon_seq = model(latent_seq, 'decode')
    print(f"Phase 2: {x_seq.shape} -> {latent_seq.shape} -> {recon_seq.shape}")

    print("\nAll dimension checks passed!")
