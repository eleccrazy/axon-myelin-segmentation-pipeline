"""
File: unet_deep.py
Description: Deeper 4-level U-Net backbone for stain-specific TB and IHC models.
    This mirrors the encoder–decoder depth of UNetStain, but without the
    stain embedding. It uses 4 downsampling steps and a 512-channel bottleneck
    (for bilinear upsampling), with the usual skip connections.

Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive 3×3 convolutions with ReLU activations.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """
    Downscaling with max-pooling followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    """
    Upscaling, concatenation with encoder features, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(
                in_ch // 2, in_ch // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Pad if needed to match skip connection spatial size
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1×1 convolution to map features to output logits.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetDeep(nn.Module):
    """
    4-level U-Net backbone without stain embedding.

    - 4 downsampling steps (feature sizes 64, 128, 256, 512, 512 for bilinear=True)
    - 512-channel bottleneck (base_c*16//2 for base_c=64)
    - symmetric decoder with skip connections

    This matches the depth of UNetStain used in the mixed-stain experiment,
    but is purely stain-agnostic.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_c: int = 64,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, base_c)  # 64
        self.down1 = Down(base_c, base_c * 2)  # 128
        self.down2 = Down(base_c * 2, base_c * 4)  # 256
        self.down3 = Down(base_c * 4, base_c * 8)  # 512

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)  # 512 for bilinear

        # Decoder
        # up1: input = upsampled bottleneck (512) + skip from x4 (512)
        self.up1 = Up(base_c * 16 // factor + base_c * 8, base_c * 8 // factor)

        # up2: input = 256 + skip from x3 (256)
        self.up2 = Up(
            (base_c * 8 // factor) + base_c * 4,
            base_c * 4 // factor,
        )

        # up3: input = 128 + skip from x2 (128)
        self.up3 = Up(
            (base_c * 4 // factor) + base_c * 2,
            base_c * 2 // factor,
        )

        # up4: input = 64 + skip from x1 (64)
        self.up4 = Up(
            (base_c * 2 // factor) + base_c,
            base_c,
        )

        self.outc = OutConv(base_c, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 512

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)  # [B, out_channels, H, W]
