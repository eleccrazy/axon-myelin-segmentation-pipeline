"""
File: pre_visualization.py
Description: Create 2×2 preview grids of image–mask pairs for each stain.

Author: Gizachew Kassa
Date Created: 06/11/2025
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils.paths import IHC_DIR, OUTPUTS_ROOT, TB_DIR


def find_mask_pairs(
    input_dir: Path,
    mask_suffix: str = "_mask.tiff",
    image_ext: str = ".tiff",
    max_pairs: int = 2,
) -> List[Tuple[Path, Path]]:
    """
    Find up to `max_pairs` (image, mask) pairs in a stain directory.

    A mask is identified by the given suffix; the corresponding image is
    assumed to share the same base name without the mask suffix.
    """
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    mask_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(mask_suffix.lower())
    )
    mask_files = mask_files[:max_pairs]

    pairs: List[Tuple[Path, Path]] = []
    for mask_name in mask_files:
        mask_path = input_dir / mask_name
        image_name = mask_name.replace(mask_suffix, image_ext)
        image_path = input_dir / image_name
        if image_path.exists():
            pairs.append((image_path, mask_path))
        else:
            # silently skip if the image is missing
            continue

    return pairs


def load_image(path: Path) -> np.ndarray:
    """
    Load an RGB image as a NumPy array.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_visual_mask(path: Path) -> np.ndarray:
    """
    Load a mask and convert it to a binary 0/255 visualisation.
    """
    mask = Image.open(path).convert("L")
    mask_np = np.array(mask)
    mask_vis = (mask_np > 0).astype(np.uint8) * 255
    return mask_vis


def create_preview_grid(
    stain_name: str,
    pairs: List[Tuple[Path, Path]],
    output_path: Path,
) -> None:
    """
    Create a 2×2 grid (2 samples × image/mask) and save it to `output_path`.
    """
    if not pairs:
        return

    # Ensure we have at most 2 pairs
    pairs = pairs[:2]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.tight_layout(pad=3.0)

    for row_idx, (img_path, mask_path) in enumerate(pairs):
        image = load_image(img_path)
        mask = load_visual_mask(mask_path)

        # Left: original image
        ax_img = axes[row_idx, 0]
        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title(f"{stain_name} Image\n{img_path.name}", fontsize=10)

        # Right: binary mask
        ax_mask = axes[row_idx, 1]
        ax_mask.imshow(mask, cmap="gray")
        ax_mask.axis("off")
        ax_mask.set_title(f"{stain_name} Mask\n{mask_path.name}", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Generate preview grids for toluidine blue and IHC stains.

    Uses TB_DIR and IHC_DIR as inputs and writes:
        OUTPUTS_ROOT / 'pre_visualization' / 'tb_preview.png'
        OUTPUTS_ROOT / 'pre_visualization' / 'ihc_preview.png'
    """
    print("Starting pre-visualization grid generation...")

    output_root = OUTPUTS_ROOT / "pre_visualization"
    tb_pairs = find_mask_pairs(TB_DIR)
    ihc_pairs = find_mask_pairs(IHC_DIR)

    tb_out = output_root / "tb_preview.png"
    ihc_out = output_root / "ihc_preview.png"

    create_preview_grid("Toluidine blue", tb_pairs, tb_out)
    create_preview_grid("IHC (DAB)", ihc_pairs, ihc_out)

    print(f"Saved toluidine blue preview to: {tb_out}")
    print(f"Saved IHC preview to:          {ihc_out}")


if __name__ == "__main__":
    main()
