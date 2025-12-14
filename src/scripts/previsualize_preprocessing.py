"""
File: previsualize_preprocessing.py
Description: Script to pre-visualize the preprocessing pipelines for TB and IHC.
Author: Gizachew Kassa
Date Created: 14/12/2025
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.configs.preprocessing_configs import (
    IHC_TRAIN_PREPROCESS_CONFIG,
    TB_TRAIN_PREPROCESS_CONFIG,
)
from src.datasets.base import find_image_mask_pairs
from src.preprocessing.ihc_preprocessing import DABPreprocess
from src.preprocessing.tb_preprocessing import TBPreprocess
from src.utils.paths import IHC_SPLIT_DIR, OUTPUTS_ROOT, TB_SPLIT_DIR


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_one_image_from_split(split_root: Path) -> Path:
    """Pick one image path (from first image-mask pair) from a split folder."""
    pairs = find_image_mask_pairs(split_root)
    img_path, _ = pairs[0]
    return img_path


def _random_crop_pil(
    img: Image.Image, crop_size: int, rng: random.Random
) -> Image.Image:
    """Random 256x256 crop from a PIL image."""
    w, h = img.size
    if w < crop_size or h < crop_size:
        raise ValueError(
            f"Image too small for crop {crop_size}x{crop_size}: got {w}x{h}"
        )
    x0 = rng.randint(0, w - crop_size)
    y0 = rng.randint(0, h - crop_size)
    return img.crop((x0, y0, x0 + crop_size, y0 + crop_size))


def _pre_to_uint8_minmax(pre_crop: np.ndarray) -> np.ndarray:
    """
    Convert preprocessed crop (float) to uint8 grayscale using min-max normalization.
    Matches the logic you provided.
    """
    pre_min, pre_max = float(pre_crop.min()), float(pre_crop.max())
    if pre_max > pre_min:
        pre_norm = (pre_crop - pre_min) / (pre_max - pre_min)
    else:
        pre_norm = np.zeros_like(pre_crop, dtype=np.float32)

    pre_uint8 = (pre_norm * 255.0).astype(np.uint8)
    return pre_uint8


def _save_org_and_pre(
    orig_crop_rgb: Image.Image,
    pre_uint8: np.ndarray,
    out_dir: Path,
    stain: str,
    crop_idx: int,
    upscale: int = 4,
) -> None:
    """
    Save:
      <stain>_crop<idx>_org.png  (RGB)
      <stain>_crop<idx>_pre.png  (grayscale)
    Upscales (e.g., 256->1024) and writes 300 DPI metadata.
    """
    crop_size_eff = orig_crop_rgb.size[0]
    new_size = (crop_size_eff * upscale, crop_size_eff * upscale)

    # Original RGB
    orig_up = orig_crop_rgb.resize(new_size, Image.BICUBIC)

    # Preprocessed grayscale
    pre_img = Image.fromarray(pre_uint8, mode="L")
    pre_up = pre_img.resize(new_size, Image.BICUBIC)

    org_path = out_dir / f"{stain}_crop{crop_idx}_org.png"
    pre_path = out_dir / f"{stain}_crop{crop_idx}_pre.png"

    orig_up.save(org_path, dpi=(300, 300))
    pre_up.save(pre_path, dpi=(300, 300))

    print(f"Saved {org_path}")
    print(f"Saved {pre_path}")


def main() -> None:
    crop_size = 256
    num_crops = 2
    seed = 1337
    upscale = 4  # 256x256 -> 1024x1024

    out_dir = OUTPUTS_ROOT / "pre_visualization" / "orginal_vs_preprocessed"
    _ensure_dir(out_dir)

    rng = random.Random(seed)

    # Build preprocessors from your configs
    tb_pre = TBPreprocess(
        clahe_clip=TB_TRAIN_PREPROCESS_CONFIG.clahe_clip,
        clahe_tile=TB_TRAIN_PREPROCESS_CONFIG.clahe_tile,
        illum_sigma=TB_TRAIN_PREPROCESS_CONFIG.illum_sigma,
        denoise_sigma=TB_TRAIN_PREPROCESS_CONFIG.denoise_sigma,
        use_percentile_clip=TB_TRAIN_PREPROCESS_CONFIG.use_percentile_clip,
    )

    ihc_pre = DABPreprocess(
        clahe_clip=IHC_TRAIN_PREPROCESS_CONFIG.clahe_clip,
        denoise_sigma=IHC_TRAIN_PREPROCESS_CONFIG.denoise_sigma,
        use_percentile_clip=IHC_TRAIN_PREPROCESS_CONFIG.use_percentile_clip,
    )

    # Pick one sample image from each stain (train split)
    tb_img_path = _pick_one_image_from_split(Path(TB_SPLIT_DIR) / "train")
    ihc_img_path = _pick_one_image_from_split(Path(IHC_SPLIT_DIR) / "train")

    tb_img = Image.open(tb_img_path).convert("RGB")
    ihc_img = Image.open(ihc_img_path).convert("RGB")

    # --- TB: 2 crops ---
    for i in range(1, num_crops + 1):
        crop_rgb = _random_crop_pil(tb_img, crop_size=crop_size, rng=rng)
        pre_tensor = tb_pre(crop_rgb)  # [1,H,W], z-scored float
        pre_crop = pre_tensor.squeeze(0).cpu().numpy()
        pre_uint8 = _pre_to_uint8_minmax(pre_crop)

        _save_org_and_pre(
            orig_crop_rgb=crop_rgb,
            pre_uint8=pre_uint8,
            out_dir=out_dir,
            stain="tb",
            crop_idx=i,
            upscale=upscale,
        )

    # --- IHC: 2 crops ---
    for i in range(1, num_crops + 1):
        crop_rgb = _random_crop_pil(ihc_img, crop_size=crop_size, rng=rng)
        pre_tensor = ihc_pre(crop_rgb)  # [1,H,W], z-scored float
        pre_crop = pre_tensor.squeeze(0).cpu().numpy()
        pre_uint8 = _pre_to_uint8_minmax(pre_crop)

        _save_org_and_pre(
            orig_crop_rgb=crop_rgb,
            pre_uint8=pre_uint8,
            out_dir=out_dir,
            stain="ihc",
            crop_idx=i,
            upscale=upscale,
        )

    print(f"\nDone. Outputs saved to:\n{out_dir}")


if __name__ == "__main__":
    main()
