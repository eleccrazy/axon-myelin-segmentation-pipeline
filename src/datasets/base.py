"""
File: base.py
Description: Base dataset for paired microscopy images and binary masks.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def find_image_mask_pairs(
    root: Path,
    image_suffixes: Sequence[str] = (".tiff", ".tif"),
    mask_suffix: str = "_mask",
) -> List[Tuple[Path, Path]]:
    """
    Find image–mask pairs in a directory.

    Images are expected to have extensions in image_suffixes and not contain
    the mask_suffix in their stem. Masks are expected to share the same stem
    with mask_suffix appended before the extension.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    pairs: List[Tuple[Path, Path]] = []

    for img_path in sorted(root.iterdir()):
        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in image_suffixes:
            continue

        if mask_suffix in img_path.stem:
            # skip files that are already mask images
            continue

        mask_stem = img_path.stem + mask_suffix
        mask_path = img_path.with_name(mask_stem + img_path.suffix)

        if mask_path.is_file():
            pairs.append((img_path, mask_path))
        else:
            # silently skip missing masks; split_dataset.py should have ensured pairs
            continue

    if not pairs:
        raise RuntimeError(f"No image–mask pairs found in {root}")

    return pairs


class PairedImageMaskDataset(Dataset):
    """
    Generic dataset for paired microscopy images and binary masks.

    The dataset assumes that images and masks share a common stem, with mask
    filenames using an additional suffix (e.g. *_mask.tiff).
    """

    def __init__(
        self,
        root: Path,
        preprocess: Callable[[Image.Image], torch.Tensor],
        augment: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        """
        Construct a paired image–mask dataset.

        Parameters
        ----------
        root
            Directory containing the image–mask pairs.
        preprocess
            Callable that converts a PIL image to a preprocessed tensor of
            shape [1, H, W].
        augment
            Optional callable that takes (image_tensor, mask_tensor) and
            returns an augmented pair. Typically used only for training.
        """
        self.root = root.expanduser().resolve()
        self.preprocess = preprocess
        self.augment = augment

        self.pairs = find_image_mask_pairs(self.root)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_mask(mask_path: Path) -> torch.Tensor:
        """
        Load a binary mask image and return a tensor of shape [1, H, W].
        """
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)
        # binarise: any non-zero value is treated as foreground
        mask_bin = (mask_array > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # [1, H, W]
        return mask_tensor

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img_pil)  # [1, H, W]

        mask_tensor = self._load_mask(mask_path)  # [1, H, W]

        if self.augment is not None:
            img_tensor, mask_tensor = self.augment(img_tensor, mask_tensor)

        return img_tensor, mask_tensor
