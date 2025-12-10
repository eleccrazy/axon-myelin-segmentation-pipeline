"""
File: tb_dataset.py
Description: Dataset wrapper for toluidine blue (TB) LM images.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import Dataset

from src.configs.preprocessing_configs import TB_TRAIN_PREPROCESS_CONFIG
from src.datasets.base import PairedImageMaskDataset
from src.preprocessing.tb_preprocessing import TBPreprocess
from src.utils.paths import TB_SPLIT_DIR


class TBLMDataset(PairedImageMaskDataset):
    """
    Toluidine blue dataset using the split data under data/splitted/tb/.
    """

    def __init__(
        self,
        split: str,
        augment: Optional[
            callable
        ] = None,  # augmentation callable applied in training only
    ) -> None:
        """
        Construct the TB dataset for a given split.

        Parameters
        ----------
        split
            One of {"train", "val", "test"}.
        augment
            Optional augmentation callable applied to (image, mask).
        """
        split_root = Path(TB_SPLIT_DIR) / split
        tb_pre = TBPreprocess(
            clahe_clip=TB_TRAIN_PREPROCESS_CONFIG.clahe_clip,
            clahe_tile=TB_TRAIN_PREPROCESS_CONFIG.clahe_tile,
            illum_sigma=TB_TRAIN_PREPROCESS_CONFIG.illum_sigma,
            denoise_sigma=TB_TRAIN_PREPROCESS_CONFIG.denoise_sigma,
            use_percentile_clip=TB_TRAIN_PREPROCESS_CONFIG.use_percentile_clip,
        )
        super().__init__(root=split_root, preprocess=tb_pre, augment=augment)
