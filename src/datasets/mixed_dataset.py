"""
File: mixed_dataset.py
Description: Mixed-stain dataset combining TB and IHC splits with stain labels.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.configs.preprocessing_configs import (
    MIXED_IHC_PREPROCESS_CONFIG,
    MIXED_TB_PREPROCESS_CONFIG,
)
from src.datasets.base import find_image_mask_pairs
from src.preprocessing.ihc_preprocessing import DABPreprocess
from src.preprocessing.tb_preprocessing import TBPreprocess
from src.utils.paths import IHC_SPLIT_DIR, TB_SPLIT_DIR


class MixedLMDataset(Dataset):
    """
    Mixed-stain dataset for TB + IHC with stain identifiers.

    Each sample consists of:
        - preprocessed image tensor [1, H, W]
        - binary mask tensor [1, H, W]
        - stain_id (0 for TB, 1 for IHC)
    """

    def __init__(
        self,
        split: str,
        augment: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        """
        Construct the mixed-stain dataset for a given split.

        Parameters
        ----------
        split
            One of {"train", "val", "test"}.
        augment
            Optional augmentation callable applied to (image, mask).
        """
        self.split = split
        self.augment = augment

        tb_root = Path(TB_SPLIT_DIR) / split
        ihc_root = Path(IHC_SPLIT_DIR) / split

        self.tb_pre = TBPreprocess(
            clahe_clip=MIXED_TB_PREPROCESS_CONFIG.clahe_clip,
            clahe_tile=MIXED_TB_PREPROCESS_CONFIG.clahe_tile,
            illum_sigma=MIXED_TB_PREPROCESS_CONFIG.illum_sigma,
            denoise_sigma=MIXED_TB_PREPROCESS_CONFIG.denoise_sigma,
            use_percentile_clip=MIXED_TB_PREPROCESS_CONFIG.use_percentile_clip,
        )
        self.ihc_pre = DABPreprocess(
            clahe_clip=MIXED_IHC_PREPROCESS_CONFIG.clahe_clip,
            denoise_sigma=MIXED_IHC_PREPROCESS_CONFIG.denoise_sigma,
            use_percentile_clip=MIXED_IHC_PREPROCESS_CONFIG.use_percentile_clip,
        )

        # Collect TB and IHC samples with stain IDs
        tb_pairs = [(img, mask, 0) for img, mask in find_image_mask_pairs(tb_root)]
        ihc_pairs = [(img, mask, 1) for img, mask in find_image_mask_pairs(ihc_root)]

        self.samples: List[Tuple[Path, Path, int]] = tb_pairs + ihc_pairs
        if not self.samples:
            raise RuntimeError(f"No mixed samples found for split='{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_mask(mask_path: Path) -> torch.Tensor:
        """
        Load a binary mask image and return a tensor of shape [1, H, W].
        """
        mask = Image.open(mask_path).convert("L")
        mask_array = (np.array(mask) > 0).astype("float32")  # [H, W]
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # [1, H, W]
        return mask_tensor

    def __getitem__(self, idx: int):
        img_path, mask_path, stain_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        # Choose stain-specific preprocess
        if stain_id == 0:
            img_tensor = self.tb_pre(img)
        else:
            img_tensor = self.ihc_pre(img)

        # Binary mask [1, H, W]
        mask_array = torch.from_numpy(
            (torch.tensor(mask_img, dtype=torch.uint8).numpy() > 0).astype("float32")
        )
        mask_tensor = mask_array.unsqueeze(0)

        if self.augment is not None:
            img_tensor, mask_tensor = self.augment(img_tensor, mask_tensor)

        stain_tensor = torch.tensor(stain_id, dtype=torch.long)
        return img_tensor, mask_tensor, stain_tensor
