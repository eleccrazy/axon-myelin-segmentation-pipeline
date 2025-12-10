"""
File: tb_preprocessing.py
Description: Preprocessing pipeline for toluidine blue LM images.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .common import percentile_clip, zscore


class TBPreprocess:
    """
    Toluidine blue preprocessing: RGB → LAB(L) → shade correction → CLAHE → denoise → optional percentile clip → z-score.

    This class covers both the stain-specific TB preprocessing and the TB branch
    of the mixed-stain experiment. Small behavioural differences are controlled
    through constructor arguments.
    """

    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        illum_sigma: float = 31.0,
        denoise_sigma: float = 0.5,
        use_percentile_clip: bool = False,
    ) -> None:
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.illum_sigma = illum_sigma
        self.denoise_sigma = denoise_sigma
        self.use_percentile_clip = use_percentile_clip

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Apply the toluidine blue preprocessing pipeline to a PIL RGB image.

        Returns a single-channel tensor with shape [1, H, W].
        """
        # PIL RGB -> numpy BGR for OpenCV
        img_rgb = np.asarray(img_pil).astype(np.uint8)
        img_bgr = img_rgb[..., ::-1]

        # RGB -> LAB, take the L channel
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[..., 0].astype(np.float32)

        # Illumination correction using a large Gaussian blur
        if self.illum_sigma > 0:
            bg = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=self.illum_sigma)
            # Equivalent to L - bg + 128 using addWeighted, then clipping to valid range
            L = cv2.addWeighted(L, 1.0, bg, -1.0, 128.0)
            L = np.clip(L, 0.0, 255.0)

        # CLAHE on the L channel
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)
        L_eq = clahe.apply(L.astype(np.uint8)).astype(np.float32)

        # Convert to [0, 1]
        L_eq /= 255.0

        # Optional gentle denoise
        if self.denoise_sigma > 0:
            L_eq = cv2.GaussianBlur(L_eq, ksize=(0, 0), sigmaX=self.denoise_sigma)

        # Optional percentile clipping (used in mixed-stain TB branch)
        if self.use_percentile_clip:
            L_eq = percentile_clip(L_eq)

        # Per-image z-score normalisation
        L_z = zscore(L_eq)

        # Return [1, H, W] tensor
        return torch.from_numpy(L_z).float().unsqueeze(0)


"""
Example instances with typical parameters for TB preprocessing in tb and mixed-stain experiments.

tb_pre = TBPreprocess(
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    illum_sigma=31.0,
    denoise_sigma=0.5,
    use_percentile_clip=False,
)

tb_pre_mixed = TBPreprocess(
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    illum_sigma=30.0,
    denoise_sigma=0.5,
    use_percentile_clip=True,
)
"""
