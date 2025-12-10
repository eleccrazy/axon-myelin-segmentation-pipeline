"""
File: losses.py
Description: Common loss functions for TB, IHC and mixed-stain segmentation models.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss operating on logits.

    Applies a sigmoid to logits, computes the soft Dice coefficient per sample,
    and returns 1 - mean(Dice) over the batch.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Composite loss combining BCEWithLogits and Dice.

    The parameter alpha controls the weighting:
        loss = alpha * BCE + (1 - alpha) * Dice.
    An optional positive-class weight can be passed for imbalanced data.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets.float())
        d = self.dice(logits, targets)
        return self.alpha * bce + (1.0 - self.alpha) * d


def gaussian_blur_mask(mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    Apply a Gaussian blur to a binary mask.

    Used by the boundary-aware Dice variant to emphasise border regions.
    """
    from scipy.ndimage import gaussian_filter  # local import to avoid hard dependency

    m = mask.detach().cpu().numpy().astype("float32")
    m_blur = gaussian_filter(m, sigma=sigma)
    return torch.from_numpy(m_blur).to(mask.device).float()


class BoundaryAwareDice(nn.Module):
    """
    Dice loss variant that emphasises boundary regions in the target mask.

    The binary target is smoothed with a Gaussian kernel and combined with a
    boundary_weight factor to up-weight border pixels.
    """

    def __init__(
        self,
        eps: float = 1e-7,
        boundary_weight: float = 3.0,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.boundary_weight = boundary_weight
        self.sigma = sigma

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs, targets: (B, 1, H, W)
        smooth = gaussian_blur_mask(targets, sigma=self.sigma)

        # weight is higher near boundaries
        weight = 1.0 + self.boundary_weight * smooth
        weight = weight.detach()

        p = probs * weight
        t = targets * weight

        inter = (p * t).sum(dim=(1, 2, 3))
        denom = (p * p).sum(dim=(1, 2, 3)) + (t * t).sum(dim=(1, 2, 3))

        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BCEDiceBoundary(nn.Module):
    """
    Composite BCE + boundary-aware Dice loss for the mixed-stain model.

    The parameter alpha controls the weighting:
        loss = alpha * BCE + (1 - alpha) * BoundaryAwareDice.
    An optional positive-class weight can be passed to the BCE term.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        pos_weight: Optional[torch.Tensor] = None,
        boundary_weight: float = 3.0,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.boundary_dice = BoundaryAwareDice(
            boundary_weight=boundary_weight,
            sigma=sigma,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets.float(), pos_weight=self.pos_weight
            )

        probs = torch.sigmoid(logits)
        bd = self.boundary_dice(probs, targets)
        return self.alpha * bce + (1.0 - self.alpha) * bd
