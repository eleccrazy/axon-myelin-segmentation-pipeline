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

    Used by the boundary-aware Dice variant to obtain a smooth version of the
    target mask that emphasises boundary regions.
    """
    from skimage.filters import gaussian

    m = mask.detach().cpu().numpy().astype("float32")
    m_blur = gaussian(m, sigma=sigma)
    return torch.from_numpy(m_blur).to(mask.device).float()


class BoundaryAwareDice(nn.Module):
    """
    Dice loss variant that emphasises boundary regions in the target mask.

    The target mask is smoothed with a Gaussian kernel, and the absolute
    difference |smooth - targets| is used to build a per-pixel weight map.
    This matches the formulation used in the mixed-stain experiment.
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
        targets = targets.float()
        smooth = gaussian_blur_mask(targets, sigma=self.sigma)

        # Weight emphasises boundaries via |smooth - targets|
        weight = 1.0 + self.boundary_weight * torch.abs(smooth - targets)
        weight = weight.detach()

        # Flatten for per-sample Dice computation
        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)
        w = weight.view(weight.size(0), -1)

        # Weighted Dice: inter = (p * t * w), denom = (p * w) + (t * w)
        inter = (p * t * w).sum(dim=1)
        denom = (p * w).sum(dim=1) + (t * w).sum(dim=1)

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
        targets = targets.float()

        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )

        probs = torch.sigmoid(logits)
        bd = self.boundary_dice(probs, targets)
        return self.alpha * bce + (1.0 - self.alpha) * bd
