"""
File: ihc_dataset.py
Description: Dataset wrapper for IHC (DAB) LM images.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from src.configs.preprocessing_configs import IHC_TRAIN_PREPROCESS_CONFIG
from src.datasets.base import PairedImageMaskDataset
from src.preprocessing.ihc_preprocessing import DABPreprocess
from src.utils.paths import IHC_SPLIT_DIR


class IHCLMDataset(PairedImageMaskDataset):
    """
    IHC (DAB) dataset using the split data under data/splitted/ihc/.
    """

    def __init__(
        self,
        split: str,
        augment: Optional[callable] = None,
    ) -> None:
        """
        Construct the IHC dataset for a given split.

        Parameters
        ----------
        split
            One of {"train", "val", "test"}.
        augment
            Optional augmentation callable applied to (image, mask).
        """
        split_root = Path(IHC_SPLIT_DIR) / split
        ihc_pre = DABPreprocess(
            clahe_clip=IHC_TRAIN_PREPROCESS_CONFIG.clahe_clip,
            denoise_sigma=IHC_TRAIN_PREPROCESS_CONFIG.denoise_sigma,
            use_percentile_clip=IHC_TRAIN_PREPROCESS_CONFIG.use_percentile_clip,
        )
        super().__init__(root=split_root, preprocess=ihc_pre, augment=augment)
