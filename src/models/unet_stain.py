"""
File: unet_stain.py
Description: U-Net variant with stain embedding for the mixed-stain experiment.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from typing import Tuple

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


class StainEmbedding(nn.Module):
    """
    Learnable embedding for stain identity (TB vs IHC).

    Expects integer stain IDs of shape [B], with:
      0 → toluidine blue (TB)
      1 → IHC (DAB)
    """

    def __init__(self, embed_dim: int = 1) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)

    def forward(
        self,
        stain_ids: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        stain_ids
            Tensor of shape [B] with integer stain IDs {0, 1}.
        spatial_size
            Target spatial size (H, W) for the expanded embedding.

        Returns
        -------
        Tensor of shape [B, embed_dim, H, W].
        """
        # [B, embed_dim]
        e = self.emb(stain_ids.long())
        # [B, embed_dim, 1, 1]
        e = e[:, :, None, None]
        # Tile to [B, embed_dim, H, W]
        H, W = spatial_size
        return e.expand(-1, -1, H, W)


class UNetStain(nn.Module):
    """
    U-Net with stain embedding at the bottleneck, used for the mixed-stain model.

    The architecture mirrors the deeper U-Net from mixed_experiment2.py:
    - 4 downsampling steps (feature sizes 64, 128, 256, 512, 512)
    - stain embedding concatenated at the bottleneck (8 channels by default)
    - symmetric decoder with skip connections
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_c: int = 64,
        bilinear: bool = True,
        embed_dim: int = 1,
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

        # Stain embedding
        self.stain_emb = StainEmbedding(embed_dim=embed_dim)

        # Bottleneck concatenation: (base_c*16//factor) + embed_dim
        bottleneck_ch = base_c * 16 // factor + embed_dim  # 512 + 8 = 520

        # Decoder (channels must match concatenations)
        # up1: input = upsampled bottleneck (520) + skip from x4 (512)
        self.up1 = Up(bottleneck_ch + base_c * 8, base_c * 8 // factor)  # 1032 → 256

        # up2: input = 256 + skip from x3 (256)
        self.up2 = Up(
            (base_c * 8 // factor) + base_c * 4,
            base_c * 4 // factor,
        )  # 512 → 128

        # up3: input = 128 + skip from x2 (128)
        self.up3 = Up(
            (base_c * 4 // factor) + base_c * 2,
            base_c * 2 // factor,
        )  # 256 → 64

        # up4: input = 64 + skip from x1 (64)
        self.up4 = Up(
            (base_c * 2 // factor) + base_c,
            base_c,
        )  # 128 → 64

        self.outc = OutConv(base_c, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        stain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with stain-aware bottleneck.

        Parameters
        ----------
        x
            Input image tensor of shape [B, in_channels, H, W].
        stain_ids
            Tensor of shape [B] with integer stain IDs {0, 1}.
        """
        # Encoder
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 512

        # Stain embedding at bottleneck resolution
        B, Cb, H, W = x5.shape
        emb = self.stain_emb(stain_ids, (H, W))  # [B, embed_dim, H, W]
        x5_cat = torch.cat([x5, emb], dim=1)  # [B, 520, H, W]

        # Decoder
        x = self.up1(x5_cat, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)  # logits [B, out_channels, H, W]
