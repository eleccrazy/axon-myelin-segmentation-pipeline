"""
File: ihc_preprocessing.py
Description: Preprocessing pipeline for IHC (DAB) LM images.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import exposure
from skimage.color import rgb2hed

from .common import percentile_clip, zscore


class DABPreprocess:
    """
    IHC DAB preprocessing: RGB → HED → DAB → invert → CLAHE → denoise →
    optional percentile clip → z-score.

    This class covers both the stain-specific IHC preprocessing and the IHC
    branch of the mixed-stain experiment. Behavioural differences such as
    percentile clipping are controlled through constructor arguments.
    """

    def __init__(
        self,
        clahe_clip: float = 0.02,
        denoise_sigma: float = 0.5,
        use_percentile_clip: bool = False,
    ) -> None:
        self.clahe_clip = clahe_clip
        self.denoise_sigma = denoise_sigma
        self.use_percentile_clip = use_percentile_clip

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Apply the IHC DAB preprocessing pipeline to a PIL RGB image.

        Returns a single-channel tensor with shape [1, H, W].
        """
        # PIL RGB -> float RGB in [0, 1]
        rgb = np.asarray(img_pil).astype(np.float32) / 255.0

        # Colour deconvolution: RGB -> HED, take the DAB channel
        hed = rgb2hed(rgb)
        dab = hed[..., 2]

        # Normalise to [0, 1] and invert so myelin appears bright
        dab = (dab - dab.min()) / (dab.max() - dab.min() + 1e-8)
        dab = 1.0 - dab

        # CLAHE for local contrast enhancement
        dab = exposure.equalize_adapthist(dab, clip_limit=self.clahe_clip)

        # Optional gentle denoise
        if self.denoise_sigma > 0:
            dab = cv2.GaussianBlur(dab, ksize=(0, 0), sigmaX=self.denoise_sigma)

        # Optional percentile clipping (used in mixed-stain IHC branch)
        if self.use_percentile_clip:
            dab = percentile_clip(dab)

        # Per-image z-score normalisation
        dab_z = zscore(dab)

        # Return [1, H, W] tensor
        return torch.from_numpy(dab_z).float().unsqueeze(0)


"""
Example instances with typical parameters for IHC preprocessing in ihc and mixed-stain experiments.

ihc_pre = DABPreprocess(
    clahe_clip=0.02,
    denoise_sigma=0.5,
    use_percentile_clip=False,
)

ihc_pre_mixed = DABPreprocess(
    clahe_clip=0.02,
    denoise_sigma=0.5,
    use_percentile_clip=True,
)

"""
