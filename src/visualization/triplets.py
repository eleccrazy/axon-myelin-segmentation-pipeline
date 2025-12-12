"""
File: triplets.py
Description: Generate input/GT/prediction triplet crops for visualization.
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.datasets.ihc_dataset import IHCLMDataset
from src.datasets.mixed_dataset import MixedLMDataset
from src.datasets.tb_dataset import TBLMDataset
from src.utils.paths import OUTPUTS_ROOT
from src.visualization.utils import (
    get_device,
    load_calibrated_threshold,
    load_model_for_exp,
)


def _get_dataset(exp: str, split: str):
    exp = exp.lower()
    if exp == "tb":
        return TBLMDataset(split=split, augment=None)
    if exp == "ihc":
        return IHCLMDataset(split=split, augment=None)
    if exp == "mixed":
        return MixedLMDataset(split=split, augment=None)
    raise ValueError(f"Unknown experiment: {exp}")


def _get_image_path(exp: str, dataset, idx: int) -> Path:
    """Return the original image path for a given dataset index."""
    if exp in {"tb", "ihc"}:
        img_path, _ = dataset.pairs[idx]
    elif exp == "mixed":
        img_path, _, _ = dataset.samples[idx]
    else:  # pragma: no cover
        raise ValueError(f"Unknown experiment: {exp}")
    return Path(img_path)


def _crop_box_from_center(
    H: int, W: int, crop_size: int, cy: int, cx: int
) -> Tuple[int, int, int, int, int]:
    """Square crop around (cy, cx), clipped to image. Returns (y1,y2,x1,x2,s)."""
    s = min(crop_size, H, W)
    half = s // 2

    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    y2 = y1 + s
    x2 = x1 + s

    # if we ran off the edge, shift back
    if y2 > H:
        y2 = H
        y1 = H - s
    if x2 > W:
        x2 = W
        x1 = W - s

    return y1, y2, x1, x2, s


def _get_crop_centres(
    H: int, W: int, n: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """
    Choose n crop centres.
    - Always include the central crop when n>=1
    - Add remaining centres randomly (uniform) within the image bounds
    """
    centres: List[Tuple[int, int]] = []
    if n <= 0:
        return centres

    centres.append((H // 2, W // 2))
    if n == 1:
        return centres

    for _ in range(n - 1):
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        centres.append((cy, cx))
    return centres


def generate_triplets(
    exp: str,
    crop_size: int,
    crops_per_image: int = 3,
    split: str = "test",
    threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = 42,
    save_dir: Optional[os.PathLike] = None,
) -> None:
    """Generate input/GT/prediction triplet crops for a given experiment.

    For each image in the split, this generates `crops_per_image` fragments.
    Each fragment produces three PNG files:
      - *_input.png  (RGB crop from the original image)
      - *_gt.png     (binary GT mask crop as 3-channel 0/255)
      - *_pred.png   (binary prediction crop as 3-channel 0/255)

    Default output layout:
      OUTPUTS_ROOT/figures/<exp>/triplets/<crop_size>/
    """
    exp = exp.lower()
    if device is None:
        device = get_device()

    if threshold is None:
        threshold = load_calibrated_threshold(exp, default=0.5)

    # --- requested update ---
    if save_dir is None:
        save_dir = OUTPUTS_ROOT / "figures" / exp / "triplets" / str(crop_size)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = _get_dataset(exp, split=split)
    n_samples = len(dataset)

    # IMPORTANT: always generate for all images in the split
    indices = list(range(n_samples))

    crops_per_image = int(crops_per_image)
    if crops_per_image < 1:
        raise ValueError("crops_per_image must be >= 1")

    model = load_model_for_exp(exp, device=device)

    total_triplets = n_samples * crops_per_image
    print(
        f"Generating {n_samples} images × {crops_per_image} fragments each "
        f"({total_triplets} triplets) for exp='{exp}', "
        f"crop_size={crop_size}, split='{split}', threshold={threshold:.3f}"
    )

    rng = np.random.default_rng(seed)

    for i, idx in enumerate(indices):
        sample = dataset[idx]

        if exp == "mixed":
            img_t, mask_t, stain_t = sample
            stain_id = int(stain_t.item())
            stain_str = "tb" if stain_id == 0 else "ihc"
        else:
            img_t, mask_t = sample
            stain_id = None
            stain_str = None

        img_path = _get_image_path(exp, dataset, idx)
        basename = img_path.stem

        # Optional: group mixed outputs per stain
        if exp == "mixed" and stain_str is not None:
            out_dir = save_dir / f"stain_{stain_str}"
        else:
            out_dir = save_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- load original RGB image ---
        pil_img = Image.open(img_path).convert("RGB")
        rgb_full = np.asarray(pil_img)
        H, W, _ = rgb_full.shape

        mask_full = mask_t.squeeze(0).cpu().numpy().astype("uint8")

        # safety: if mask/image sizes differ, resize RGB to mask size
        if mask_full.shape != (H, W):
            pil_img = pil_img.resize(
                (mask_full.shape[1], mask_full.shape[0]),
                Image.BILINEAR,
            )
            rgb_full = np.asarray(pil_img)
            H, W, _ = rgb_full.shape

        # --- run model on full preprocessed image ---
        x = img_t.unsqueeze(0).to(device)  # [1,1,H,W]
        with torch.no_grad():
            if exp == "mixed":
                stains_batch = stain_t.view(1).to(device)
                logits = model(x, stains_batch)
            else:
                logits = model(x)

        prob_full = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_full = (prob_full >= float(threshold)).astype("uint8")

        # --- choose crop centres (per-image) ---
        centres = _get_crop_centres(H, W, crops_per_image, rng)
        s = min(crop_size, H, W)

        for j, (cy, cx) in enumerate(centres):
            y1, y2, x1, x2, _ = _crop_box_from_center(H, W, s, cy, cx)

            rgb_crop = rgb_full[y1:y2, x1:x2]
            gt_crop = mask_full[y1:y2, x1:x2]
            pred_crop = pred_full[y1:y2, x1:x2]

            gt_uint8 = (gt_crop * 255).astype("uint8")
            pred_uint8 = (pred_crop * 255).astype("uint8")
            gt_rgb = np.stack([gt_uint8] * 3, axis=-1)
            pred_rgb = np.stack([pred_uint8] * 3, axis=-1)

            # --- filenames (include original basename) ---
            base = f"{basename}_idx{idx:03d}_frag{j:02d}_cy{cy}_cx{cx}"
            if exp == "mixed" and stain_str is not None:
                base = f"{basename}_{stain_str}_idx{idx:03d}_frag{j:02d}_cy{cy}_cx{cx}"

            in_path = out_dir / f"{base}_input.png"
            gt_path = out_dir / f"{base}_gt.png"
            pred_path = out_dir / f"{base}_pred.png"

            Image.fromarray(rgb_crop).save(in_path)
            Image.fromarray(gt_rgb).save(gt_path)
            Image.fromarray(pred_rgb).save(pred_path)

        print(
            f"[{i+1}/{n_samples}] Saved {crops_per_image} fragments for {basename} (idx={idx})"
        )

    print("Done generating triplets.")
