"""
File: split_dataset.py
Description: Split toluidine blue and IHC datasets into train/val/test sets.
Author: Gizachew Kassa
Date Created: 06/11/2025
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.paths import DATA_ROOT, IHC_DIR, TB_DIR

random.seed(42)  # for reproducibility


def collect_image_mask_pairs(
    input_dir: Path, img_ext: str = ".tiff", mask_suffix: str = "_mask.tiff"
) -> List[Tuple[Path, Path]]:
    """
    Find all (image, mask) pairs in a stain directory.
    """
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.name.endswith(img_ext) and not f.name.endswith(mask_suffix)
    ]

    pairs: List[Tuple[Path, Path]] = []
    for img_path in images:
        mask_name = img_path.name.replace(img_ext, mask_suffix)
        mask_path = input_dir / mask_name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        # if the mask is missing, we silently skip that image

    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Shuffle and split image–mask pairs into train/val/test subsets.
    """
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]

    return {"train": train, "val": val, "test": test}


def copy_split(
    splits: Dict[str, List[Tuple[Path, Path]]], output_base: Path
) -> Dict[str, int]:
    """
    Copy image–mask pairs into train/val/test folders under output_base.

    Returns a dict with the number of pairs per split.
    """
    counts: Dict[str, int] = {}

    for split_name, pairs in splits.items():
        split_dir = output_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in pairs:
            shutil.copy2(img_path, split_dir / img_path.name)
            shutil.copy2(mask_path, split_dir / mask_path.name)

        counts[split_name] = len(pairs)

    return counts


def split_stain(input_dir: Path, stain_name: str) -> Dict[str, int]:
    """
    Split a single stain dataset and copy files to data/splitted/<stain_name>.
    """
    pairs = collect_image_mask_pairs(input_dir)
    output_base = DATA_ROOT / "splitted" / stain_name
    output_base.mkdir(parents=True, exist_ok=True)

    splits = split_pairs(pairs)
    counts = copy_split(splits, output_base)
    return counts


def main() -> None:
    """
    Split both toluidine blue and IHC datasets into train/val/test sets.
    """
    print("Starting dataset split for toluidine blue and IHC...")

    tb_counts = split_stain(TB_DIR, "tb")
    ihc_counts = split_stain(IHC_DIR, "ihc")

    print("Toluidine blue split (data/splitted/tb):")
    print(f" - Train: {tb_counts.get('train', 0)} pairs")
    print(f" - Val:   {tb_counts.get('val', 0)} pairs")
    print(f" - Test:  {tb_counts.get('test', 0)} pairs")

    print("IHC split (data/splitted/ihc):")
    print(f" - Train: {ihc_counts.get('train', 0)} pairs")
    print(f" - Val:   {ihc_counts.get('val', 0)} pairs")
    print(f" - Test:  {ihc_counts.get('test', 0)} pairs")


if __name__ == "__main__":
    main()
