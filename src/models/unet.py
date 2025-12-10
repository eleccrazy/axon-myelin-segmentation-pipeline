"""
File: unet.py
Description: Baseline U-Net architecture for stain-specific TB and IHC models.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3×3 convolutions with ReLU activations.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Lightweight 3-level U-Net used for the stain-specific experiments.

    This matches the architecture used in tb.py and ihc.py:
    - encoder with feature sizes: 64 → 128 → 256
    - two pooling operations
    - symmetric decoder with skip connections
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)  # skip from enc2

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)  # skip from enc1

        self.final = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)  # [B, c1, H,   W]
        e2 = self.enc2(self.pool(e1))  # [B, c2, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, c3, H/4, W/4]

        # Decoder
        d2 = self.up2(e3)  # [B, c2, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # [B, c1, H, W]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)  # logits [B, out_channels, H, W]
