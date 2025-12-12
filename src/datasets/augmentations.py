"""
File: augmentations.py
Description: Simple geometric augmentations for TB, IHC and mixed-stain datasets.
             All transforms operate on preprocessed single-channel tensors [1, H, W] and
             are applied to image and mask together to keep them aligned.
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def basic_geo_augment(image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Apply simple geometric augmentations:

      1) Random horizontal flip (p=0.5)
      2) Random vertical flip   (p=0.5)
      3) Random rotation by k * 90 degrees, k ∈ {0,1,2,3} (uniform)

    Parameters
    ----------
    image : Tensor
        Preprocessed image tensor of shape [1, H, W].
    mask : Tensor
        Binary mask tensor of shape [1, H, W].

    Returns
    -------
    (image_aug, mask_aug) : Tuple[Tensor, Tensor]
        Augmented image–mask pair.
    """
    assert image.ndim == 3 and mask.ndim == 3, "Expected [1,H,W] tensors"
    assert image.shape == mask.shape, "Image/mask shapes must match"

    # Horizontal flip (left–right)
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # Vertical flip (top–bottom)
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])

    # Random rotation by k * 90 degrees
    k = int(torch.randint(0, 4, ()).item())  # 0,1,2,3
    if k != 0:
        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])

    return image, mask
